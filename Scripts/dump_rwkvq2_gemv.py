#!/usr/bin/env python3
"""
dump_rwkvq2_gemv.py -- эталон GEMV-ядра раскладки sym (путь декода Metal-бэкенда rwkv-quant) для порта.

Порт обязан совпасть ПОБИТНО: исходник ядра и порядок суммирования перенесены дословно.

    cd ~/Develop/rwkv-metal && .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_rwkvq2_gemv.py

Пишет ~/Develop/SwiftRWKV/.testdata/rwkvq2_reduction_gemv.safetensors: на каждый тензор (имена .rwkvq, точки заменены
на "/") -- x/<ключ> [4, IN] fp32 и y1 / y3 / y4/<ключ> -- выход SymQuantLinear._gemv на первых 1, 3 и 4 строках входа.
"""
import os
import numpy as np
import mlx.core as mx
from rwkv_quant.formats import codec
from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinear

SRC = os.path.expanduser("~/Develop/WKV-kvant/files_0110/0p1b_reduction.rwkvq")
OUT = os.path.expanduser("~/Develop/SwiftRWKV/.testdata/rwkvq2_reduction_gemv.safetensors")
KEYS = ["blocks.1.att.key.weight", "blocks.7.att.output.weight",      # sym@8, D x D
        "blocks.0.ffn.key.weight", "blocks.5.ffn.value.weight",       # sym@6, 4D x D и D x 4D
        "head.weight"]                                                # sym@8, V x D

man, buf = codec.open_rwkvq(SRC)
rng = np.random.default_rng(20261009)
out = {}
for key in KEYS:
    meta = man["tensors"][key]
    assert meta["kind"] == "sym", (key, meta["kind"])
    b = lambda f: buf.get(f"{key}::{f}")
    lin = SymQuantLinear.from_buffers(shape=tuple(meta["shape"]), bits=meta["bits"], qs=b("gw_qs"), d=b("gw_d"),
                                      codes=b("codes"), codes_packed=b("codes_packed"), qh=b("gw_qh"), qh2=b("gw_qh2"))
    x = mx.array(rng.standard_normal((4, meta["shape"][1])).astype(np.float32))
    tag = key.replace(".", "/")
    out["x/" + tag] = x
    for n in (1, 3, 4):
        out["y%d/%s" % (n, tag)] = lin._gemv(x[:n], n)
    print(key, meta["bits"], meta["shape"])
mx.eval(out)
mx.save_safetensors(OUT, out)
print("->", OUT)
