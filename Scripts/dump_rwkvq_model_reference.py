#!/usr/bin/env python3
"""
Reference logits for a model whose base weights come from a .rwkvq sidecar.

Why this exists separately from the dequant test: dequantization being
bit-exact says nothing about the WIRING. If `k_proj` were mapped to the
sidecar's `att.value` instead of `att.key`, every tensor would still
dequantize perfectly and the model would still produce plausible numbers —
just the wrong ones. Only an end-to-end reference catches that.

Usage:
    cd ~/Develop/rwkv-metal
    .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_rwkvq_model_reference.py \
        --model    world_0.1b_x070.safetensors \
        --sidecar  ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx \
        --out      ~/Develop/SwiftRWKV/.testdata/rwkvq_model_ref.safetensors
"""
import argparse
import json
import os
import sys

import mlx.core as mx

# world naming (.rwkvq) -> x070 naming (rwkv-metal / SwiftRWKV)
_TMIX = {"receptance": "r_proj", "key": "k_proj",
         "value": "v_proj", "output": "o_proj"}


def world_to_x070(key: str):
    if key in ("emb.weight", "head.weight"):
        return key
    parts = key.split(".")
    if len(parts) != 5 or parts[0] != "blocks" or parts[4] != "weight":
        return None
    layer, kind, name = parts[1], parts[2], parts[3]
    if kind == "att" and name in _TMIX:
        return f"blocks.{layer}.tmix.{_TMIX[name]}.weight"
    if kind == "ffn" and name in ("key", "value"):
        return f"blocks.{layer}.cmix.{name}.weight"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True, help="path WITHOUT suffix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    # Mirror SwiftRWKV's default attach options so the two sides substitute
    # exactly the same set of tensors.
    ap.add_argument("--quantize-embedding", action="store_true")
    ap.add_argument("--no-head", action="store_true")
    args = ap.parse_args()

    from rwkv_metal.model.rwkv7_x070 import RWKV7X070, lora_ranks
    from rwkv_metal.lora.rwkvq_kernel import dequant_dense

    weights = dict(mx.load(os.path.expanduser(args.model)))
    base = os.path.expanduser(args.sidecar)
    packed = mx.load(base + ".safetensors")
    manifest = json.load(open(base + ".json"))

    substituted = []
    for world_key, meta in manifest["tensors"].items():
        x = world_to_x070(world_key)
        if x is None or x not in weights:
            continue
        if x == "emb.weight" and not args.quantize_embedding:
            continue
        if x == "head.weight" and args.no_head:
            continue
        OUT, IN = meta["shape"]
        dense = dequant_dense(packed[f"{world_key}::qblk"],
                              packed[f"{world_key}::qsqm"],
                              packed[f"{world_key}::ddm"],
                              OUT, IN, meta["gw_sb"], meta["xbits"])
        mx.eval(dense)
        assert tuple(weights[x].shape) == (OUT, IN), \
            f"{x}: model {tuple(weights[x].shape)} vs sidecar {(OUT, IN)}"
        weights[x] = dense.astype(mx.bfloat16)
        substituted.append(x)

    print(f"substituted {len(substituted)} tensors from the sidecar")

    n_layer = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
    n_embd = weights["ln_out.weight"].shape[0]
    vocab = weights["head.weight"].shape[0]
    head_size = weights["blocks.0.tmix.k_k"].shape[1]

    class Cfg:
        pass

    cfg = Cfg()
    cfg.n_layer, cfg.n_embd, cfg.vocab_size = n_layer, n_embd, vocab
    cfg.head_size, cfg.n_head = head_size, n_embd // head_size

    model = RWKV7X070(cfg, lora_ranks(n_embd))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    ids = mx.random.randint(0, vocab, (2, args.seq_len),
                            key=mx.random.key(args.seed))
    mx.eval(ids)
    hidden = model.body(ids)
    logits = model.head(hidden)
    mx.eval(hidden, logits)

    out = {
        "ids": ids.astype(mx.int32),
        "hidden": hidden.astype(mx.float32),
        "logits": logits.astype(mx.float32),
    }
    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    mx.save_safetensors(dest, out)
    with open(dest + ".json", "w") as f:
        json.dump({"substituted": sorted(substituted),
                   "quantize_embedding": args.quantize_embedding,
                   "head": not args.no_head}, f, indent=2)
    print(f"wrote {dest}")
    print(f"logits absmax={mx.abs(logits).max().item():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
