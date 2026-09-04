# SPDX-License-Identifier: Apache-2.0
"""Offline greedy generation for Qwen3.5-MoE on Trainium.

Two architectures are registered and this script picks between them. Without ``--image`` it serves the
TEXT backbone, which is also what keeps image input out: the runner decides a model is multimodal from
its architecture name. With ``--image`` the checkpoint's own name is kept, the vision tower is built and
the encoder is compiled as its own graph, so the compile cost and the bucket set both change.
"""
import argparse
import os

from vllm import LLM, SamplingParams

# On single-EFA-card instances (e.g. trn2.3xlarge, the TP=4 target here) the Neuron EFA-affinity probe
# expects a co-located EFA under each NeuronCore's PCI path, which only holds on multi-card instances
# like trn2.48xlarge. Skip the affinity — it is a CPU-locality optimisation, not correctness.
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")


# Widths measured to be efficient on this model, in tokens. 128 is cheap in absolute terms for short
# prompts; everything from 2048 up is cheap per token. The gap is not an omission: 512 and 1024 cost MORE
# in absolute time than the 2048 bucket they exist to avoid, so a request is better off padded past them.
# Measured per bucket at max_model_len 2048 and again at 4096, with identical results, so the cost is a
# property of the width and not of max_model_len.
_EFFICIENT_PREFILL_WIDTHS = (128, 2048, 4096, 8192, 16384)


def _prefill_buckets(max_model_len):
    """Only the widths measured to be efficient, ending exactly at max_model_len.

    The last bucket must equal ``max_num_batched_tokens``, which this example sets to ``max_model_len``,
    so that value is always included even if it is not one of the measured widths. Retune the table for
    a different model: the widths that land on efficient kernels are a property of the shapes, and this
    one was measured for this architecture at TP=4 in bf16.
    """
    buckets = [w for w in _EFFICIENT_PREFILL_WIDTHS if w < max_model_len]
    return buckets + [max_model_len]


def _text_architecture(config):
    """Point the architecture at the text class.

    The checkpoint declares ``Qwen3_5MoeForConditionalGeneration``, and both names are registered, so
    this is a CHOICE of what to serve rather than a fix for a name the plugin lacks: the runner keys its
    multimodal path off the architecture, so naming the text class is also what keeps image input out.
    """
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    return config


def _vision_markers(llm):
    """``<vision_start><image><vision_end>`` for this checkpoint, decoded from its declared token ids.

    The processor expands the single image marker into one token per merged patch, so the prompt carries
    exactly one of them however large the image is.
    """
    config = llm.llm_engine.vllm_config.model_config.hf_config
    tokenizer = llm.get_tokenizer()
    ids = [config.vision_start_token_id, config.image_token_id, config.vision_end_token_id]
    markers = [tokenizer.decode([token_id]) for token_id in ids]
    if not all(markers):
        raise ValueError(
            f"the tokenizer decoded the vision marker ids {ids} to {markers!r}; an empty marker would "
            "leave the image span unmarked, and the image would be encoded and never read."
        )
    return "".join(markers)


def _load_image(reference):
    """An image from a path or a URL, as PIL, which is what the multimodal input field takes."""
    from io import BytesIO

    from PIL import Image

    if reference.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(reference) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    return Image.open(reference).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint", type=str, default="Qwen/Qwen3.6-35B-A3B",
        help="Path or HF id of the Qwen3.5-MoE checkpoint")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=4,
        help="35B-A3B (~72 GB bf16) needs TP=4 to fit the four NeuronCores of one trn2 chip")
    parser.add_argument(
        "--max-model-len", type=int, default=512,
        help="Keep this modest: 72 GB of weights leaves little of a 96 GB chip for anything else")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--prompt", action="append", default=None,
                        help="Repeatable; defaults to three short factual prompts")
    parser.add_argument(
        "--image", action="append", default=None,
        help="Path or URL of an image. Passing one selects the MULTIMODAL architecture, which is a "
             "different model class and a different set of compiled graphs (the vision encoder is its "
             "own NEFF). One image per prompt, in order.")
    parser.add_argument(
        "--vision-block-size", type=int, default=1024,
        help="Patches per vision attention block. An image must fit within one block, so this is a "
             "floor on the resolution the encoder accepts, not only a tuning knob")
    args = parser.parse_args()
    if args.image and args.prompt and len(args.image) != len(args.prompt):
        parser.error(
            f"got {len(args.image)} image(s) for {len(args.prompt)} prompt(s); pair them one to one "
            "so it is unambiguous which image belongs to which prompt")

    # The Gated DeltaNet conv and recurrent state are carried in place with a single per-layer buffer,
    # so max_num_seqs must be 1. Prefix caching is incompatible for the same reason (a reused prefix
    # would continue from the wrong recurrent state); the model's factory refuses both.
    llm = LLM(
        model=args.model_checkpoint,
        # The architecture name is the choice of what to serve. With --image the checkpoint's own name
        # is kept and the runner builds its multimodal path; without it the text class is named, and
        # that is what keeps image input out.
        #
        # Serving the TEXT backbone of a multimodal checkpoint still needs the vision path neutralised
        # rather than removed. The runner decides a model is multimodal by
        # `hasattr(hf_config, "vision_config")`, which is true for this config class even when the JSON
        # omits it — and it cannot simply be deleted, because vLLM's own config for this model_type
        # requires it. Changing `model_type` does not work either: vLLM resolves its own config class
        # from it and rejects the text variant. So the vision token buckets are given values that pass
        # validation against the prefill buckets, and nothing ever runs through them.
        hf_overrides=None if args.image else _text_architecture,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_config={
            "neuron_config": {
                # Measured, not guessed: TTFT per prefill bucket is NOT monotonic in bucket width.
                # The buckets cost 0.157 s (128), 0.510 s (512), 1.119 s (1024), 0.375 s (2048) and
                # 0.667 s (4096) — so a request landing in 512 or 1024 pays MORE than one padded all
                # the way to 2048. Per token the wide buckets are far better still (0.18 ms at 2048 and
                # 0.16 ms at 4096, against 1.0-1.2 ms below). A power-of-two ladder is therefore the
                # wrong default; see _EFFICIENT_PREFILL_WIDTHS for what to keep.
                "num_batched_tokens_buckets": _prefill_buckets(args.max_model_len),
            },
            "vision_neuron_config": (
                {
                    # Real values: one graph is compiled per bucket, and an image must fit inside one
                    # block, so the block size is a floor on the resolution the encoder accepts.
                    "num_vision_tokens_buckets": [args.vision_block_size,
                                                  args.vision_block_size * 2],
                    "vision_attention_block_size": args.vision_block_size,
                }
                if args.image else
                {
                    # Neutralised, not used: see the note above.
                    "num_vision_tokens_buckets": [args.max_model_len],
                    "vision_attention_block_size": args.max_model_len,
                }
            ),
        },
    )

    prompts = args.prompt or ["The capital of France is", "2 + 2 =", "日本の首都は"]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    if args.image:
        # The image markers have to appear in the prompt: they mark the span the encoder's output is
        # scattered into. Without them the image is encoded, cached and never read, and the answer comes
        # back fluent and unrelated.
        #
        # The marker STRINGS are decoded from the ids the checkpoint's config declares rather than
        # written out. Spelling them by hand is a guess that cannot be checked without the tokenizer,
        # and a wrong one is not an error: the text passes through as ordinary tokens and the image span
        # is never marked.
        prefix = _vision_markers(llm)
        requests = [
            {"prompt": f"{prefix}{text}",
             "multi_modal_data": {"image": _load_image(reference)}}
            for text, reference in zip(prompts, args.image)
        ]
        for reference, out in zip(args.image, llm.generate(requests, sampling_params)):
            print(f"[{reference}]", repr(out.outputs[0].text))
        return

    for out in llm.generate(prompts, sampling_params):
        print(repr(out.prompt), "->", repr(out.outputs[0].text))


if __name__ == "__main__":
    main()
