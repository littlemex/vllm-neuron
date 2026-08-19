# SPDX-License-Identifier: Apache-2.0
import argparse
import os

from vllm import LLM, SamplingParams

# On single-EFA-card instances (e.g. trn2.3xlarge, the TP=4 target here) the Neuron EFA-affinity
# probe expects a co-located EFA under each NeuronCore's PCI path, which only holds on multi-card
# instances like trn2.48xlarge. Skip the affinity (a CPU-locality optimization, not correctness).
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Path or HF id of the NemotronH checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="30B-A3B (~60 GB bf16) needs TP=4 to fit the 4 NeuronCores of one trn2 chip",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32,
        help="The vectorized SSD prefill is O(l^2); keep the prefill bucket small",
    )
    args = parser.parse_args()

    # The Mamba2 recurrent state is carried in-place (batch=1), so max_num_seqs=1.
    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
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

    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "日本の首都は",
    ]
    sampling_params = SamplingParams(max_tokens=12, temperature=0.0, top_p=1.0)

    outputs = llm.generate(prompts, sampling_params)
    for out in outputs:
        print(repr(out.prompt), "->", repr(out.outputs[0].text))


if __name__ == "__main__":
    main()
