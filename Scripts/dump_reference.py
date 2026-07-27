#!/usr/bin/env python3
"""
Dump reference activations from rwkv-metal (Python/MLX) for cross-language
parity testing of SwiftRWKV.

Everything in SwiftRWKV's test suite so far checks the Swift implementation
against *itself* — structural identities that must hold for any weights.
Those catch a lot, but they cannot catch a shared misreading of the
architecture: if a formula is ported wrong in a way that is internally
consistent, every structural test still passes.

This script closes that gap by producing ground truth from the reference
implementation on real pretrained weights. The Swift side loads the same
weight file and must reproduce these tensors.

Usage:
    cd ~/Develop/rwkv-metal
    .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_reference.py \
        --model world_0.1b_x070.safetensors \
        --out   ~/Develop/SwiftRWKV/.testdata/reference_0.1b.safetensors

The output is consumed by X070ParityTests, which skips itself when the file
is absent — the fixture is far too large to commit.
"""
import argparse
import os
import sys

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="x070 .safetensors (converted)")
    ap.add_argument("--out", required=True, help="where to write the reference dump")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from rwkv_metal.model.rwkv7_x070 import RWKV7X070, lora_ranks
    from rwkv_metal.model.state import build_mask

    weights = mx.load(os.path.expanduser(args.model))

    # geometry straight from the weight shapes — no configuration to get wrong
    n_layer = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
    n_embd = weights["ln_out.weight"].shape[0]
    vocab = weights["head.weight"].shape[0]
    head_size = weights["blocks.0.tmix.k_k"].shape[1]
    print(f"model: L={n_layer} D={n_embd} V={vocab} head={head_size}")

    class Cfg:
        pass

    cfg = Cfg()
    cfg.n_layer, cfg.n_embd, cfg.vocab_size = n_layer, n_embd, vocab
    cfg.head_size = head_size
    cfg.n_head = n_embd // head_size

    model = RWKV7X070(cfg, lora_ranks(n_embd))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    # Deterministic token ids, independent of any tokenizer: parity is about
    # the numerics, and pulling a tokenizer in would add a second thing to
    # disagree about.
    T = args.seq_len
    rng = mx.random.key(args.seed)
    ids = mx.random.randint(0, vocab, (2, T), key=rng)
    mx.eval(ids)

    out = {"ids": ids.astype(mx.int32)}

    # 1. full forward
    hidden = model.body(ids)
    logits = model.head(hidden)
    mx.eval(hidden, logits)
    out["hidden"] = hidden.astype(mx.float32)
    out["logits"] = logits.astype(mx.float32)

    # 2. state at the end of the sequence (no padding)
    st = model.states(ids)
    out["state_wkv"] = st.wkv.astype(mx.float32)
    out["state_tmix"] = st.tmix_shift.astype(mx.float32)
    out["state_cmix"] = st.cmix_shift.astype(mx.float32)

    # 3. split pass: first half, then continue — the prefix-cache path
    half = T // 2
    h1, s1 = model.body(ids[:, :half], return_state=True)
    h2, s2 = model.body(ids[:, half:], state=s1, return_state=True)
    mx.eval(h1, h2)
    out["split_hidden"] = mx.concatenate([h1, h2], axis=1).astype(mx.float32)
    out["split_state_wkv"] = s2.wkv.astype(mx.float32)

    # 4. right-padded batch with a mask — rows of different lengths
    lengths = [T, T // 2]
    mask = build_mask(lengths, T)
    end_idx = mx.array([l - 1 for l in lengths])
    st_masked = model.states(ids, mask=mask, end_idx=end_idx)
    out["masked_state_wkv"] = st_masked.wkv.astype(mx.float32)
    out["masked_state_tmix"] = st_masked.tmix_shift.astype(mx.float32)
    out["mask_lengths"] = mx.array(lengths, dtype=mx.int32)

    mx.eval(list(out.values()))
    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    mx.save_safetensors(dest, out)

    print(f"wrote {dest}")
    for k, v in out.items():
        print(f"  {k:20s} {tuple(v.shape)}")
    print(f"\nlogits: mean={logits.mean().item():.6f} "
          f"absmax={mx.abs(logits).max().item():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
