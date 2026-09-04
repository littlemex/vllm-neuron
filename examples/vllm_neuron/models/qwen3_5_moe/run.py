# SPDX-License-Identifier: Apache-2.0
"""Offline greedy generation for the Qwen3.5-MoE text backbone on Trainium.

Text-only: the checkpoint is multimodal but the vision tower is out of scope, so pass text prompts.
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
    """Point the architecture at the text class, which is what this plugin registers."""
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    return config


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
    args = parser.parse_args()

    # The Gated DeltaNet conv and recurrent state are carried in place with a single per-layer buffer,
    # so max_num_seqs must be 1. Prefix caching is incompatible for the same reason (a reused prefix
    # would continue from the wrong recurrent state); the model's factory refuses both.
    llm = LLM(
        model=args.model_checkpoint,
        # Serving the text backbone of a multimodal checkpoint needs two things.
        #
        # 1. `architectures` must name the text class, since that is what the plugin registers.
        # 2. The vision path must be neutralised rather than removed. The runner decides a model is
        #    multimodal by `hasattr(hf_config, "vision_config")`, which is true for this config class
        #    even when the JSON omits it — and it cannot simply be deleted, because vLLM's own config
        #    for this model_type requires it. Changing `model_type` does not work either: vLLM
        #    resolves its own config class from it and rejects the text variant. So instead the vision
        #    token buckets are given values that pass validation against the prefill buckets. Nothing
        #    ever runs through them, because no image or video input is accepted.
        hf_overrides=_text_architecture,
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
            "vision_neuron_config": {
                # Neutralised, not used: see the note above.
                "num_vision_tokens_buckets": [args.max_model_len],
                "vision_attention_block_size": args.max_model_len,
            },
        },
    )

    prompts = args.prompt or ["The capital of France is", "2 + 2 =", "日本の首都は"]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    for out in llm.generate(prompts, sampling_params):
        print(repr(out.prompt), "->", repr(out.outputs[0].text))


if __name__ == "__main__":
    main()
