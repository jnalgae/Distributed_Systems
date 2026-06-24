#!/usr/bin/env python3

import argparse
import math
from pathlib import Path


COMP_NODE = 4
COMM_SEND_NODE = 5
COMM_RECV_NODE = 6
COMM_COLL_NODE = 7
ALL_REDUCE = 0


def varint(value):
    out = bytearray()
    value = int(value)
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(0x80 | bits)
        else:
            out.append(bits)
            return bytes(out)


def key(field_number, wire_type):
    return varint((field_number << 3) | wire_type)


def field_varint(field_number, value):
    return key(field_number, 0) + varint(value)


def field_string(field_number, value):
    raw = value.encode("utf-8")
    return key(field_number, 2) + varint(len(raw)) + raw


def field_message(field_number, payload):
    return key(field_number, 2) + varint(len(payload)) + payload


def encode_record(out_file, payload):
    out_file.write(varint(len(payload)))
    out_file.write(payload)


def global_metadata(version="0.0.4"):
    return field_string(1, version)


def attr(name, value_field, value):
    payload = field_string(1, name)
    payload += field_varint(value_field, value)
    return payload


def attr_bool(name, value):
    return attr(name, 27, 1 if value else 0)


def attr_i32(name, value):
    return attr(name, 7, value)


def attr_i64(name, value):
    return attr(name, 9, value)


def comp_node(node_id, name, runtime_us, deps):
    return {
        "id": node_id,
        "name": name,
        "type": COMP_NODE,
        "duration_us": runtime_us,
        "deps": deps,
        "attrs": [attr_bool("is_cpu_op", False)],
    }


def all_reduce_node(node_id, name, comm_size, deps):
    return {
        "id": node_id,
        "name": name,
        "type": COMM_COLL_NODE,
        "deps": deps,
        "attrs": [
            attr_bool("is_cpu_op", False),
            attr_i64("comm_type", ALL_REDUCE),
            attr_i64("comm_size", comm_size),
        ],
    }


def send_node(node_id, name, src, dst, tag, comm_size, deps):
    return {
        "id": node_id,
        "name": name,
        "type": COMM_SEND_NODE,
        "deps": deps,
        "attrs": [
            attr_bool("is_cpu_op", False),
            attr_i32("comm_src", src),
            attr_i32("comm_dst", dst),
            attr_i32("comm_tag", tag),
            attr_i64("comm_size", comm_size),
        ],
    }


def recv_node(node_id, name, src, dst, tag, comm_size, deps):
    return {
        "id": node_id,
        "name": name,
        "type": COMM_RECV_NODE,
        "deps": deps,
        "attrs": [
            attr_bool("is_cpu_op", False),
            attr_i32("comm_src", src),
            attr_i32("comm_dst", dst),
            attr_i32("comm_tag", tag),
            attr_i64("comm_size", comm_size),
        ],
    }


def encode_node(node):
    payload = field_varint(1, node["id"])
    payload += field_string(2, node["name"])
    payload += field_varint(3, node["type"])
    for dep in node.get("deps", []):
        payload += field_varint(5, dep)
    if node.get("duration_us"):
        payload += field_varint(7, node["duration_us"])
    for attr_payload in node["attrs"]:
        payload += field_message(10, attr_payload)
    return payload


def write_rank_trace(path, prefix, rank, nodes):
    with open(path / f"{prefix}.{rank}.et", "wb") as et:
        encode_record(et, global_metadata())
        for node in nodes:
            encode_record(et, encode_node(node))


def calculate_compute_times_us(
    batch_size,
    sequence_length,
    hidden_size,
    intermediate_size,
    peak_tflops,
    tp_degree,
):
    b = batch_size
    s = sequence_length
    h = hidden_size
    i = intermediate_size

    attention_linear_flops = 8 * b * s * h * h
    attention_score_value_flops = 4 * b * s * s * h
    mlp_flops = 6 * b * s * h * i

    peak_flops = peak_tflops * 1e12
    full_attention_us = (attention_linear_flops + attention_score_value_flops) / peak_flops * 1e6
    full_mlp_us = mlp_flops / peak_flops * 1e6

    return {
        "full_attention_us": max(1, math.ceil(full_attention_us)),
        "full_mlp_us": max(1, math.ceil(full_mlp_us)),
        "tp_attention_us": max(1, math.ceil(full_attention_us / tp_degree)),
        "tp_mlp_us": max(1, math.ceil(full_mlp_us / tp_degree)),
    }


def generate_tp8(path, npus, layers, microbatches, activation_bytes, attention_us, mlp_us):
    prefix = f"tp8_mb{microbatches}"
    for rank in range(npus):
        nodes = []
        prev = []
        node_id = 1
        for mb in range(microbatches):
            for layer in range(layers):
                qkv = comp_node(
                    node_id,
                    f"rank{rank}_mb{mb}_layer{layer}_attn_compute",
                    attention_us,
                    prev,
                )
                nodes.append(qkv)
                prev = [node_id]
                node_id += 1

                attn_ar = all_reduce_node(
                    node_id,
                    f"rank{rank}_mb{mb}_layer{layer}_attn_allreduce",
                    activation_bytes,
                    prev,
                )
                nodes.append(attn_ar)
                prev = [node_id]
                node_id += 1

                mlp = comp_node(
                    node_id,
                    f"rank{rank}_mb{mb}_layer{layer}_mlp_compute",
                    mlp_us,
                    prev,
                )
                nodes.append(mlp)
                prev = [node_id]
                node_id += 1

                mlp_ar = all_reduce_node(
                    node_id,
                    f"rank{rank}_mb{mb}_layer{layer}_mlp_allreduce",
                    activation_bytes,
                    prev,
                )
                nodes.append(mlp_ar)
                prev = [node_id]
                node_id += 1

        write_rank_trace(path, prefix, rank, nodes)
    return prefix


def generate_pp8(
    path,
    npus,
    layers,
    microbatches,
    activation_bytes,
    full_attention_us,
    full_mlp_us,
):
    prefix = f"pp8_mb{microbatches}"
    layers_per_stage = layers // npus
    stage_compute_us = layers_per_stage * (full_attention_us + full_mlp_us)

    for rank in range(npus):
        nodes = []
        node_id = 1
        prev = []

        for mb in range(microbatches):
            deps = prev

            if rank > 0:
                recv = recv_node(
                    node_id,
                    f"rank{rank}_mb{mb}_recv_activation_from_rank{rank - 1}",
                    rank - 1,
                    rank,
                    mb * npus + rank - 1,
                    activation_bytes,
                    deps,
                )
                nodes.append(recv)
                deps = [node_id]
                node_id += 1

            comp = comp_node(
                node_id,
                f"rank{rank}_mb{mb}_stage_compute_{layers_per_stage}_layers",
                stage_compute_us,
                deps,
            )
            nodes.append(comp)
            deps = [node_id]
            node_id += 1

            if rank < npus - 1:
                send = send_node(
                    node_id,
                    f"rank{rank}_mb{mb}_send_activation_to_rank{rank + 1}",
                    rank,
                    rank + 1,
                    mb * npus + rank,
                    activation_bytes,
                    deps,
                )
                nodes.append(send)
                deps = [node_id]
                node_id += 1

            prev = deps

        write_rank_trace(path, prefix, rank, nodes)
    return prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/first_1node_8gpu/workloads")
    parser.add_argument("--npus", type=int, default=8)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=28672)
    parser.add_argument("--gpu-peak-tflops", type=float, default=1000.0)
    parser.add_argument("--microbatches", type=str, default="1,2,4,8")
    args = parser.parse_args()

    if args.layers % args.npus != 0:
        raise ValueError("--layers must be divisible by --npus for PP8")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    activation_bytes = (
        args.batch_size
        * args.sequence_length
        * args.hidden_size
        * args.bytes_per_element
    )
    compute_times = calculate_compute_times_us(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        args.intermediate_size,
        args.gpu_peak_tflops,
        args.npus,
    )

    microbatches = [int(value) for value in args.microbatches.split(",")]
    prefixes = []
    for microbatch_count in microbatches:
        if microbatch_count <= 0:
            raise ValueError("--microbatches values must be positive")
        tp_prefix = generate_tp8(
            output,
            args.npus,
            args.layers,
            microbatch_count,
            activation_bytes,
            compute_times["tp_attention_us"],
            compute_times["tp_mlp_us"],
        )
        pp_prefix = generate_pp8(
            output,
            args.npus,
            args.layers,
            microbatch_count,
            activation_bytes,
            compute_times["full_attention_us"],
            compute_times["full_mlp_us"],
        )
        prefixes.append((microbatch_count, tp_prefix, pp_prefix))

    manifest = output / "MANIFEST.txt"
    manifest.write_text(
        "\n".join(
            [
                f"npus={args.npus}",
                f"layers={args.layers}",
                f"batch_size={args.batch_size}",
                f"sequence_length={args.sequence_length}",
                f"hidden_size={args.hidden_size}",
                f"intermediate_size={args.intermediate_size}",
                f"bytes_per_element={args.bytes_per_element}",
                f"gpu_peak_tflops={args.gpu_peak_tflops}",
                f"activation_bytes={activation_bytes}",
                f"activation_mb={activation_bytes / 1024 / 1024:.2f}",
                f"full_attention_compute_us={compute_times['full_attention_us']}",
                f"full_mlp_compute_us={compute_times['full_mlp_us']}",
                f"tp_attention_compute_us={compute_times['tp_attention_us']}",
                f"tp_mlp_compute_us={compute_times['tp_mlp_us']}",
                f"microbatches={','.join(str(value) for value in microbatches)}",
                "prefixes="
                + ",".join(
                    f"mb{mb}:{tp_prefix}/{pp_prefix}"
                    for mb, tp_prefix, pp_prefix in prefixes
                ),
                "tp_pattern=80 layers x [attention_compute/8, AllReduce, mlp_compute/8, AllReduce]",
                "pp_pattern=8 pipeline stages x 10 full layers, 7 activation transfers",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
