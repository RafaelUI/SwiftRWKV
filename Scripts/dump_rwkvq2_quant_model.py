#!/usr/bin/env python3
"""
dump_rwkvq2_quant_model.py -- эталон ИНФЕРЕНСА для порта: Metal-бэкенд rwkv-quant на нынешних пресетах (0.1B).

Зачем отдельно от rwkvq2_<пресет>_model.safetensors (эталон rwkv-metal). Решение владельца 12.10: rwkv-quant --
эталон инференса, rwkv-metal -- обучения. Путь генерации порта (ядра декода, префилл, загрузка) перенесён из
rwkv-quant, а сверялся с rwkv-metal, который COMPRESSION считает через родной контейнер MLX: это два разных счёта
одной модели. Здесь снимается то, что считает сам rwkv-quant, на ТЕХ ЖЕ входах, что у эталона rwkv-metal.

    cd ~/Develop/rwkv-metal && .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_rwkvq2_quant_model.py

Пишет ~/Develop/SwiftRWKV/.testdata/rwkvq2_<пресет>_quant.safetensors:
  logits_quant        [T, V]    параллельный проход (forward_stateful, покомандно), голова на всех позициях.
                                ВНИМАНИЕ: при T < 128 (здесь T = 64) бэкенд считает проекции ЯДРАМИ GEMV по сжатым
                                кодам (точный вес, сумма в fp32), а не «деквант + fp16-матмул», как на длинных
                                последовательностях (GEMM_MIN_BATCH_NB = 128)
  logits_quant_gemm   [T, V]    то же при пороге 1: «деквант + fp16-матмул» -- путь длинных последовательностей и
                                то, чем параллельный проход считает порт на любой длине
  hidden_quant        [T, D]    скрытое состояние после ln_out (forward_hidden) -- то, чем живут векторные модели
  state_wkv/tmix/cmix           состояние после промпта, по слоям
  decode_logits_quant [8, V]    продолжение по одному токену с этого состояния (forward_stateful, T = 1)
  decode_logits_quant_compiled  то же собранным шагом (m.step) -- так бэкенд считает в генерации
и rwkvq2_<пресет>_quant.json: версии, режим, KL к точному счёту (logits_ref из эталона rwkv-metal).
Входы (input_ids, decode_ids) берутся из rwkvq2_<пресет>_model.safetensors -- одна пара файлов, один текст.
"""
import json, os, subprocess
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
import rwkv_quant.backends.metal.quant_linear_sym as sym
from rwkv_quant.formats.reader import load_raw

TD = os.path.expanduser("~/Develop/SwiftRWKV/.testdata")


def kl(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    lp = p - (np.log(np.exp(p - p.max(-1, keepdims=True)).sum(-1, keepdims=True)) + p.max(-1, keepdims=True))
    lq = q - (np.log(np.exp(q - q.max(-1, keepdims=True)).sum(-1, keepdims=True)) + q.max(-1, keepdims=True))
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


def git(repo):
    return subprocess.run(["git", "-C", os.path.expanduser(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


for preset in ("reduction", "compression"):
    meta = json.load(open(f"{TD}/rwkvq2_{preset}.json"))
    ref = mx.load(f"{TD}/rwkvq2_{preset}_model.safetensors")
    ids, dec = ref["input_ids"], ref["decode_ids"]
    m = qm.QuantRWKV7(load_raw(meta["source"]))
    L = m.n_layer

    logits, st = m.forward_stateful(ids, m.init_state(1))
    hidden, st_h = m.forward_hidden(ids, m.init_state(1))
    mx.eval(logits, hidden, st)
    out = {
        "logits_quant": logits[0].astype(mx.float32),
        "hidden_quant": hidden[0].astype(mx.float32),
        "state_wkv": mx.stack([s[0][0] for s in st]).astype(mx.float32),
        "state_tmix": mx.stack([s[1].reshape(-1) for s in st]).astype(mx.float32),
        "state_cmix": mx.stack([s[2].reshape(-1) for s in st]).astype(mx.float32),
    }
    rows, rows_c = [], []
    s1, s2 = st, st
    for i in range(dec.shape[1]):
        lg, s1 = m.forward_stateful(dec[:, i:i + 1], s1)
        rows.append(lg[0, -1].astype(mx.float32))
        lg, s2 = m.step(dec[:, i:i + 1], s2)
        rows_c.append(lg[0, -1].astype(mx.float32))
    out["decode_logits_quant"] = mx.stack(rows)
    out["decode_logits_quant_compiled"] = mx.stack(rows_c)
    # путь длинных последовательностей на том же входе: порог GEMM снят
    nb = gw.GEMM_MIN_BATCH_NB
    gw.GEMM_MIN_BATCH_NB = sym.GEMM_MIN_BATCH_NB = 1
    try:
        lg, _ = m.forward_stateful(ids, m.init_state(1))
        out["logits_quant_gemm"] = lg[0].astype(mx.float32)
        mx.eval(out["logits_quant_gemm"])
    finally:
        gw.GEMM_MIN_BATCH_NB = sym.GEMM_MIN_BATCH_NB = nb
    mx.eval(out)
    mx.save_safetensors(f"{TD}/rwkvq2_{preset}_quant.safetensors", out)

    info = {
        "source": meta["source"], "preset": preset, "rwkv_quant": git("~/Develop/rwkv-quant"),
        "rwkv_metal": git("~/Develop/rwkv-metal"), "mlx": mx.__version__,
        "fast_ln": bool(m.fast_ln), "fast_ln_source": m.fast_ln_source,
        "emb": type(m.emb_weight).__name__,
        "kl_ref_quant": kl(ref["logits_ref"], out["logits_quant"]),
        "kl_ref_quant_gemm": kl(ref["logits_ref"], out["logits_quant_gemm"]),
        "gemm_min_batch": int(nb), "lora_q_decode": bool(m.lora_q), "lora_q_source": m.lora_q_source,
        "kl_metal_quant": kl(ref["logits_metal"], out["logits_quant"]),
        "kl_ref_quant_decode": kl(ref["decode_logits_ref"], out["decode_logits_quant"]),
        "kl_eager_compiled_decode": kl(out["decode_logits_quant"], out["decode_logits_quant_compiled"]),
        "top1_quant_vs_ref": float((np.asarray(out["logits_quant"]).argmax(-1)
                                    == np.asarray(ref["logits_ref"]).argmax(-1)).mean()),
    }
    json.dump(info, open(f"{TD}/rwkvq2_{preset}_quant.json", "w"), ensure_ascii=False, indent=1)
    print(preset, {k: (("%.3e" % v) if isinstance(v, float) else v) for k, v in info.items() if k != "source"})
    print("   формы:", {k: tuple(v.shape) for k, v in out.items()})
