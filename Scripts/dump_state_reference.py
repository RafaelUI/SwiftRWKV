#!/usr/bin/env python3
"""
Эталон state-tuning из rwkv-metal для SwiftRWKV (StateTuningParityTests).

Пишет в .testdata:
  state_ref.safetensors      -- файл состояния, записанный `save_state` на
                                плотной World 0.1B (формат читает StateFile.load)
  state_ref_out.safetensors  -- что обязана воспроизвести Swift-сторона:
      ids [2, 48], targets     токены
      hidden                   body(ids) с подключённым состоянием (модель в fp32)
      loss, grad.0, grad.last  лосс LM и градиент WKV-состояния слоёв 0 и L-1
      метаданные               отпечатки базы: плотной, REDUCTION, COMPRESSION

    cd ~/Develop/rwkv-metal
    .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_state_reference.py
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.lora import load_rwkvq_model                               # noqa: E402
from rwkv_metal.model.convert import load_pretrained                       # noqa: E402
from rwkv_metal.model.state import RWKVState                               # noqa: E402
from rwkv_metal.model.initial_state import (                               # noqa: E402
    base_fingerprint, prepare_state_tuning, save_state, zero_state)

T = os.path.expanduser("~/Develop/SwiftRWKV/.testdata/")
PTH = T + "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"
PRESET = os.path.expanduser("~/Develop/WKV-kvant/files_0110/0p1b_{}.rwkvq")


def main():
    m = load_pretrained(PTH, verbose=False)[0]
    fps = {"dense": base_fingerprint(m)}      # в родном bf16 -- как загружает SwiftRWKV
    # Счёт эталона -- в fp32: в bf16 две реализации расходятся на решётку
    # (2^-8 на слой), и за ней не видно ошибок мельче. Отпечаток от типа не зависит.
    m.set_dtype("float32")
    assert base_fingerprint(m) == fps["dense"]
    rs = np.random.RandomState(3)
    z = zero_state(m)
    # сдвиги -- на решётке bf16: тип потока у двух реализаций один, а округление
    # произвольного fp32 в него сверять незачем
    f = lambda a, s: mx.array((rs.normal(size=a.shape) * s).astype(np.float32)) \
        .astype(mx.bfloat16).astype(mx.float32)
    state = RWKVState(wkv=f(z.wkv, 0.05), tmix_shift=f(z.tmix_shift, 0.5),
                      cmix_shift=f(z.cmix_shift, 0.5)).eval()
    ids = rs.randint(1, 60000, size=(2, 49)).astype(np.int32)
    x, y = mx.array(ids[:, :-1]), mx.array(ids[:, 1:])

    prepare_state_tuning(m, init=state)
    hidden = m.body(x).astype(mx.float32)
    loss, grads = nn.value_and_grad(m, lambda mm, a, b: mm.loss(a, b).astype(mx.float32))(m, x, y)
    g = dict(tree_flatten(grads))
    L = len(m.blocks)
    meta = save_state(T + "state_ref.safetensors", state, m, kind="tuned",
                      trained=("wkv",), note="World 0.1B v2.8, эталон rwkv-metal")
    for preset in ("reduction", "compression"):
        path = PRESET.format(preset)
        if os.path.exists(path):
            fps[preset] = base_fingerprint(load_rwkvq_model(path, rank=0, verbose=False)[0])
    out = {"ids": x, "targets": y, "hidden": hidden, "loss": loss.reshape(1),
           "grad.0": g["blocks.0.state0.wkv"], "grad.last": g[f"blocks.{L - 1}.state0.wkv"]}
    mx.eval(out)
    mx.save_safetensors(T + "state_ref_out.safetensors", out, metadata=fps)
    print("отпечатки:", fps)
    print("файл состояния:", meta["base_fingerprint"][:16], "лосс", loss.item())


if __name__ == "__main__":
    main()
