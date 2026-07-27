#!/usr/bin/env python3
"""
Dump reference dequantized weights from rwkv-metal's fused sb6 Metal kernel,
for bit-exactness testing of the SwiftRWKV port.

The `.rwkvq` dequant path is one of the few places where "close enough" is
not acceptable: the REDUCTION preset is calibrated against exactly this
arithmetic (fp32 final combine, not half), so a Swift port that is merely
approximately right would silently add a second error source on top of a
calibration that assumes its absence.

Usage:
    cd ~/Develop/rwkv-metal
    .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_rwkvq_reference.py \
        --sidecar ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx \
        --out     ~/Develop/SwiftRWKV/.testdata/rwkvq_dequant_ref.safetensors
"""
import argparse
import json
import os
import sys

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True,
                    help="path WITHOUT .safetensors/.json suffix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensors", type=int, default=6,
                    help="how many tensors to dequantize (dense refs are big)")
    args = ap.parse_args()

    from rwkv_metal.lora.rwkvq_kernel import dequant_dense

    base = os.path.expanduser(args.sidecar)
    arrays = mx.load(base + ".safetensors")
    manifest = json.load(open(base + ".json"))

    # A deliberately mixed selection: square projections, the wide cmix pair
    # (different IN/OUT ratios exercise different kernel paths) and the huge
    # embedding (OUT far larger than anything else — catches indexing that
    # happens to work only for small OUT).
    preferred = [
        "blocks.0.att.key.weight",
        "blocks.0.att.receptance.weight",
        "blocks.0.ffn.key.weight",
        "blocks.0.ffn.value.weight",
        "blocks.5.att.output.weight",
        "head.weight",
    ]
    keys = [k for k in preferred if k in manifest["tensors"]]
    for k in manifest["tensors"]:
        if len(keys) >= args.tensors:
            break
        if k not in keys:
            keys.append(k)
    keys = keys[: args.tensors]

    out = {}
    meta = {}
    for key in keys:
        m = manifest["tensors"][key]
        OUT, IN = m["shape"]
        dense = dequant_dense(arrays[f"{key}::qblk"], arrays[f"{key}::qsqm"],
                              arrays[f"{key}::ddm"],
                              OUT, IN, m["gw_sb"], m["xbits"])
        mx.eval(dense)
        out[f"dequant/{key}"] = dense.astype(mx.float32)
        meta[key] = m
        print(f"  {key:38s} {tuple(dense.shape)} "
              f"absmax={mx.abs(dense).max().item():.6f}")

    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    mx.save_safetensors(dest, out)
    with open(dest + ".json", "w") as f:
        json.dump({"tensors": meta, "keys": keys}, f, indent=2)
    print(f"\nwrote {dest} ({len(out)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
