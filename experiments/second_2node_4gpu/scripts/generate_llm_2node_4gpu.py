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
    return field_string(1, name) + field_varint(value_field, value)


def attr_bool(name, value):
    return attr(name, 27, 1 if value else 0)


def attr_i32(name, value):
    return attr(name, 7, value)


def attr_i64(name, value):
    return attr(name, 9, value)


def attr_string(name, value):
    return field_string(1, name) + field_string(29, value)


def comp_node(node_id, name, runtime_us, deps):
    return {
        "id": node_id,
        "name": name,
        "type": COMP_NODE,
        "duration_us": runtime_us,
        "deps": deps,
        "attrs": [attr_bool("is_cpu_op", False)],
    }


def all_reduce_node(node_id, name, comm_size, deps, pg_name=None):
    attrs = [
        attr_bool("is_cpu_op", False),
        attr_i64("comm_type", ALL_REDUCE),
        attr_i64("comm_size", comm_size),
    ]
    if pg_name is not None:
        attrs.append(attr_string("pg_name", str(pg_name)))
    return {
        "id": node_id,
        "name": name,
        "type": COMM_COLL_NODE,
        "deps": deps,
        "attrs": attrs,
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


def compute_times_us(batch_size, sequence_length, hidden_size, intermediate_size, peak_tflops):
    b = batch_size
    s = sequence_length
    h = hidden_size
    i = intermediate_size
    attention_flops = 8 * b * s * h * h + 4 * b * s * s * h
    mlp_flops = 6 * b * s * h * i
    peak_flops = peak_tflops * 1e12
    return {
        "attention": max(1, math.ceil(attention_flops / peak_flops * 1e6)),
        "mlp": max(1, math.ceil(mlp_flops / peak_flops * 1e6)),
    }


def add_tp_layers(nodes, node_id, rank, mb, stage_name, layers, tp_degree, activation_bytes, attention_us, mlp_us, pg_name, deps):
    attention_tp = max(1, math.ceil(attention_us / tp_degree))
    mlp_tp = max(1, math.ceil(mlp_us / tp_degree))
    for layer in range(layers):
        nodes.append(comp_node(node_id, f"rank{rank}_mb{mb}_{stage_name}_layer{layer}_attn", attention_tp, deps))
        deps = [node_id]
        node_id += 1
        nodes.append(all_reduce_node(node_id, f"rank{rank}_mb{mb}_{stage_name}_layer{layer}_attn_ar", activation_bytes, deps, pg_name))
        deps = [node_id]
        node_id += 1
        nodes.append(comp_node(node_id, f"rank{rank}_mb{mb}_{stage_name}_layer{layer}_mlp", mlp_tp, deps))
        deps = [node_id]
        node_id += 1
        nodes.append(all_reduce_node(node_id, f"rank{rank}_mb{mb}_{stage_name}_layer{layer}_mlp_ar", activation_bytes, deps, pg_name))
        deps = [node_id]
        node_id += 1
    return node_id, deps


def add_full_stage(nodes, node_id, rank, mb, stage_name, layers, attention_us, mlp_us, deps):
    runtime_us = layers * (attention_us + mlp_us)
    nodes.append(comp_node(node_id, f"rank{rank}_mb{mb}_{stage_name}_compute", runtime_us, deps))
    return node_id + 1, [node_id]


def generate_tp8(path, microbatches, layers, activation_bytes, attention_us, mlp_us):
    prefix = f"tp8_mb{microbatches}"
    for rank in range(8):
        nodes = []
        node_id = 1
        deps = []
        for mb in range(microbatches):
            node_id, deps = add_tp_layers(nodes, node_id, rank, mb, "tp8", layers, 8, activation_bytes, attention_us, mlp_us, None, deps)
        write_rank_trace(path, prefix, rank, nodes)
    return prefix


def generate_pp8(path, microbatches, layers, activation_bytes, attention_us, mlp_us):
    prefix = f"pp8_mb{microbatches}"
    stage_ranks = [[0], [1], [2], [3], [7], [6], [5], [4]]
    layers_per_stage = layers // len(stage_ranks)
    rank_to_stage = {ranks[0]: stage for stage, ranks in enumerate(stage_ranks)}
    for rank in range(8):
        stage = rank_to_stage[rank]
        nodes = []
        node_id = 1
        prev = []
        for mb in range(microbatches):
            deps = prev
            if stage > 0:
                src = stage_ranks[stage - 1][0]
                nodes.append(recv_node(node_id, f"rank{rank}_mb{mb}_recv_stage{stage}", src, rank, mb * 100 + stage, activation_bytes, deps))
                deps = [node_id]
                node_id += 1
            node_id, deps = add_full_stage(nodes, node_id, rank, mb, f"pp8_stage{stage}", layers_per_stage, attention_us, mlp_us, deps)
            if stage < len(stage_ranks) - 1:
                dst = stage_ranks[stage + 1][0]
                nodes.append(send_node(node_id, f"rank{rank}_mb{mb}_send_stage{stage}", rank, dst, mb * 100 + stage + 1, activation_bytes, deps))
                deps = [node_id]
                node_id += 1
            prev = deps
        write_rank_trace(path, prefix, rank, nodes)
    return prefix


def generate_hybrid(path, name, microbatches, layers, tp_degree, stage_groups, pg_names, activation_bytes, attention_us, mlp_us):
    prefix = f"{name}_mb{microbatches}"
    layers_per_stage = layers // len(stage_groups)
    rank_to_stage = {}
    rank_to_index = {}
    for stage, ranks in enumerate(stage_groups):
        for idx, rank in enumerate(ranks):
            rank_to_stage[rank] = stage
            rank_to_index[rank] = idx

    for rank in range(8):
        stage = rank_to_stage[rank]
        idx = rank_to_index[rank]
        nodes = []
        node_id = 1
        prev = []
        shard_bytes = activation_bytes // tp_degree
        for mb in range(microbatches):
            deps = prev
            if stage > 0:
                src = stage_groups[stage - 1][idx]
                nodes.append(recv_node(node_id, f"rank{rank}_mb{mb}_recv_stage{stage}", src, rank, mb * 100 + stage * 10 + idx, shard_bytes, deps))
                deps = [node_id]
                node_id += 1
            node_id, deps = add_tp_layers(
                nodes,
                node_id,
                rank,
                mb,
                f"{name}_stage{stage}",
                layers_per_stage,
                tp_degree,
                activation_bytes,
                attention_us,
                mlp_us,
                pg_names[stage],
                deps,
            )
            if stage < len(stage_groups) - 1:
                dst = stage_groups[stage + 1][idx]
                nodes.append(send_node(node_id, f"rank{rank}_mb{mb}_send_stage{stage}", rank, dst, mb * 100 + (stage + 1) * 10 + idx, shard_bytes, deps))
                deps = [node_id]
                node_id += 1
            prev = deps
        write_rank_trace(path, prefix, rank, nodes)
    return prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/second_2node_4gpu/workloads")
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
        prefixes.append((mb, generate_tp8(output, mb, args.layers, activation_bytes, compute["attention"], compute["mlp"])))
        prefixes.append((mb, generate_pp8(output, mb, args.layers, activation_bytes, compute["attention"], compute["mlp"])))
        prefixes.append((mb, generate_hybrid(
            output,
            "tp4pp2",
            mb,
            args.layers,
            4,
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [1, 2],
            activation_bytes,
            compute["attention"],
            compute["mlp"],
        )))
        prefixes.append((mb, generate_hybrid(
            output,
            "tp2pp4",
            mb,
            args.layers,
            2,
            [[0, 1], [2, 3], [6, 7], [4, 5]],
            [3, 4, 5, 6],
            activation_bytes,
            compute["attention"],
            compute["mlp"],
        )))

    (output / "MANIFEST.txt").write_text(
        "\n".join(
            [
                "nodes=2",
                "gpus_per_node=4",
                f"layers={args.layers}",
                f"batch_size={args.batch_size}",
                f"sequence_length={args.sequence_length}",
                f"hidden_size={args.hidden_size}",
                f"intermediate_size={args.intermediate_size}",
                f"activation_bytes={activation_bytes}",
                f"activation_mb={activation_bytes / 1024 / 1024:.2f}",
                f"full_attention_compute_us={compute['attention']}",
                f"full_mlp_compute_us={compute['mlp']}",
                "rank_address=rank -> [local_gpu, node]",
                "tp4pp2_stages=[[0,1,2,3],[4,5,6,7]]",
                "tp2pp4_stages=[[0,1],[2,3],[6,7],[4,5]]",
                "pp8_stages=[[0],[1],[2],[3],[7],[6],[5],[4]]",
                f"microbatches={','.join(str(value) for value in microbatches)}",
                "prefixes=" + ",".join(prefix for _, prefix in prefixes),
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
