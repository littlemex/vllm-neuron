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
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [args.max_model_len],
            }
        },
    )

    prompts = args.prompt or ["The capital of France is", "2 + 2 =", "日本の首都は"]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    for out in llm.generate(prompts, sampling_params):
        print(repr(out.prompt), "->", repr(out.outputs[0].text))


if __name__ == "__main__":
    main()
