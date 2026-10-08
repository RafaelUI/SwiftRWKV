#!/usr/bin/env python3
"""
dump_rwkvq2_logits.py -- логиты rwkv-metal на .rwkvq нынешних пресетов для масштабов, на которые нет
полного эталона (tools/gen_rwkvq_fixtures.py в rwkv-metal пишет его только под одним именем, 0.1B).

Нужен ради одного свойства порта: модель, у которой ранги low-rank веток НЕ совпадают с формулой
(rwkv7-g1d-0.4b: g = 128, формула даёт 160), открывается и считает то же, что rwkv-metal.

    cd ~/Develop/rwkv-metal && .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_rwkvq2_logits.py 0p4b

Пишет ~/Develop/SwiftRWKV/.testdata/rwkvq2_<scale>_<preset>_logits.safetensors: input_ids [1,64],
logits_metal [64,V] (путь rwkv-metal по умолчанию: параметры fp16, база fp16) и ranks [4] (w, a, v, g).
"""
import os, sys
import numpy as np
import mlx.core as mx
from rwkv_metal.lora import add_rwkvq as AR

SCALE = sys.argv[1] if len(sys.argv) > 1 else "0p4b"
SRC = os.path.expanduser("~/Develop/WKV-kvant/files_0110/%s_{}.rwkvq" % SCALE)
OUT = os.path.expanduser("~/Develop/SwiftRWKV/.testdata/")
T = 64
IDS = mx.array((np.arange(1, T + 1, dtype=np.int64) * 7919 % 60000 + 100).astype(np.int32)[None])

for preset in ("reduction", "compression"):
    m, _, _ = AR.load_rwkvq_model(SRC.format(preset), rank=0, verbose=False)
    logits = m(IDS)[0].astype(mx.float32)
    t = m.blocks[1].tmix
    ranks = mx.array([t.w_lora_A.weight.shape[0], t.a_lora_A.weight.shape[0],
                      t.v_lora_A.weight.shape[0], t.g_lora_A.weight.shape[0]], dtype=mx.int32)
    mx.save_safetensors(OUT + "rwkvq2_%s_%s_logits.safetensors" % (SCALE, preset),
                        {"input_ids": IDS, "logits_metal": logits, "ranks": ranks})
    print(SCALE, preset, "ранги w/a/v/g", ranks.tolist(), "логиты", logits.shape)
