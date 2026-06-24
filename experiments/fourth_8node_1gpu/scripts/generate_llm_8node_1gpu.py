#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

SECOND_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "second_2node_4gpu" / "scripts"
sys.path.insert(0, str(SECOND_SCRIPT_DIR))

from generate_llm_2node_4gpu import (  # noqa: E402
    compute_times_us,
    generate_tp8,
    write_rank_trace,
    recv_node,
    send_node,
    add_full_stage,
)


def generate_pp8(path, microbatches, layers, activation_bytes, attention_us, mlp_us):
    prefix = f"pp8_mb{microbatches}"
    stage_ranks = [[0], [1], [2], [3], [4], [5], [6], [7]]
    layers_per_stage = layers // len(stage_ranks)
    for rank in range(8):
        stage = rank
        nodes = []
        node_id = 1
        prev = []
        for mb in range(microbatches):
            deps = prev
            if stage > 0:
                nodes.append(recv_node(node_id, f"rank{rank}_mb{mb}_recv_stage{stage}", rank - 1, rank, mb * 100 + stage, activation_bytes, deps))
                deps = [node_id]
                node_id += 1
            node_id, deps = add_full_stage(nodes, node_id, rank, mb, f"pp8_stage{stage}", layers_per_stage, attention_us, mlp_us, deps)
            if stage < 7:
                nodes.append(send_node(node_id, f"rank{rank}_mb{mb}_send_stage{stage}", rank, rank + 1, mb * 100 + stage + 1, activation_bytes, deps))
                deps = [node_id]
                node_id += 1
            prev = deps
        write_rank_trace(path, prefix, rank, nodes)
    return prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/fourth_8node_1gpu/workloads")
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=28672)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--gpu-peak-tflops", type=float, default=1000.0)
    parser.add_argument("--microbatches", default="1,2,4,8")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    activation_bytes = args.batch_size * args.sequence_length * args.hidden_size * args.bytes_per_element
    compute = compute_times_us(args.batch_size, args.sequence_length, args.hidden_size, args.intermediate_size, args.gpu_peak_tflops)
    microbatches = [int(value) for value in args.microbatches.split(",")]
    prefixes = []
    for mb in microbatches:
        prefixes.append(generate_tp8(output, mb, args.layers, activation_bytes, compute["attention"], compute["mlp"]))
        prefixes.append(generate_pp8(output, mb, args.layers, activation_bytes, compute["attention"], compute["mlp"]))
    (output / "MANIFEST.txt").write_text(
        "\n".join([
            "nodes=8",
            "gpus_per_node=1",
            "tp8=one TP group across all 8 nodes",
            "pp8_stages=[[0],[1],[2],[3],[4],[5],[6],[7]]",
            f"layers={args.layers}",
            f"activation_bytes={activation_bytes}",
            f"activation_mb={activation_bytes / 1024 / 1024:.2f}",
            f"full_attention_compute_us={compute['attention']}",
            f"full_mlp_compute_us={compute['mlp']}",
            f"microbatches={','.join(str(value) for value in microbatches)}",
            "prefixes=" + ",".join(prefixes),
        ]) + "\n"
    )


if __name__ == "__main__":
    main()
