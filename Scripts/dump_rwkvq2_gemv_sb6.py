#!/usr/bin/env python3
"""
dump_rwkvq2_gemv_sb6.py -- эталон GEMV-ядра раскладки sb6 (ядро k3, путь декода Metal-бэкенда rwkv-quant) для порта.

Порт обязан совпасть ПОБИТНО: исходник ядра и порядок суммирования перенесены дословно.

    cd ~/Develop/rwkv-metal && .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_rwkvq2_gemv_sb6.py

Пишет ~/Develop/SwiftRWKV/.testdata/rwkvq2_compression_gemv.safetensors: на каждый тензор (имена .rwkvq, точки
заменены на "/") -- x/<ключ> [1, IN] fp32 и y/<ключ> [1, OUT] -- выход GwQuantLinear._gemv1 (ядро k3, умолчания
бэкенда: K3_HALF и K3_XSUM выключены).
"""
import os
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw

SRC = os.path.expanduser("~/Develop/WKV-kvant/files_0110/0p1b_compression.rwkvq")
OUT = os.path.expanduser("~/Develop/SwiftRWKV/.testdata/rwkvq2_compression_gemv.safetensors")
assert gw.K3 and not gw.K3_HALF and not gw.K3_XSUM, "эталон снимается на умолчаниях бэкенда"

ck = load_raw(SRC)
want = {}                      # по одному-два тензора на каждое сочетание (битность, форма)
for key, qt in ck.tensors.items():
    if getattr(qt, "gw_mode", None) != "sb6" or key == "emb.weight":
        continue
    lin = gw.GwQuantLinear(qt)
    sig = (lin.xbits, tuple(qt.shape))
    if len(want.setdefault(sig, [])) < 2:
        want[sig].append((key, lin))

rng = np.random.default_rng(20261011)
out = {}
for sig in sorted(want):
    for key, lin in want[sig]:
        x = mx.array(rng.standard_normal((1, lin.in_features)).astype(np.float32))
        tag = key.replace(".", "/")
        out["x/" + tag] = x
        out["y/" + tag] = lin._gemv1(x)
        print(key, "bits", 4 + lin.xbits, (lin.out_features, lin.in_features), "cfg", gw._k3_cfg(lin.in_features, lin.out_features, lin.xbits))
mx.eval(out)
mx.save_safetensors(OUT, out)
print("->", OUT, len(out) // 2, "тензоров")
