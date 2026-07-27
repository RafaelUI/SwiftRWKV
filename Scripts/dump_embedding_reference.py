"""
Эталон дообучения эмбеддингов: снимает с rwkv-metal (Python/MLX) на реальной
0.1B то, против чего проверяется Swift.

Зачем нужен отдельный дамп, если паритет бэкбона уже доказан
────────────────────────────────────────────────────────────
Паритет body() показывает, что база перенесена верно. Он ничего не говорит о
том, что поверх неё построено: пулинг по poolIndex, голова, L2-нормировка,
температура, направление контрастного лосса, добивка пула кандидатов. Каждое
из этих мест можно перенести внутренне непротиворечиво и при этом неверно —
структурные тесты такое пропустят, потому что сравнивать им не с чем.

Веса головы РАНДОМИЗИРУЮТСЯ и кладутся в дамп. Штатная инициализация задаёт
fc2 = 0, то есть голова — тождество, и с ней паритет проверял бы только
LayerNorm поверх пулинга: ошибка в fc1/fc2 прошла бы незамеченной. Здесь оба
слоя ненулевые, поэтому проверяется весь путь.

Токенизация тоже кладётся в дамп (idx/pool), чтобы расхождение токенизатора
отделялось от расхождения модели: это разные дефекты и чинятся они в разных
местах.

Запуск:
    cd ~/Develop/rwkv-metal
    .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_embedding_reference.py \
        --model world_0.1b_x070.safetensors \
        --slice ~/Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl \
        --out   ~/Develop/SwiftRWKV/.testdata/embedding_ref.safetensors
"""
import argparse
import json
import os

import mlx.core as mx
import numpy as np

import rwkv_metal as rk
from rwkv_metal.embedding import (
    EmbeddingModel, encode_batch, parse_classification_candidates,
    retrieval_loss, sts_loss, classification_loss,
    evaluate_retrieval, evaluate_sts_pairwise,
)

# Закрытый пул эмоций LitRetrieval. Продублирован здесь намеренно: Swift
# держит свою копию, и дамп обязан проверить, что копии совпадают, а не
# импортировать одну и ту же и «доказать» тождество самой себе.
FULL_POOL = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "love",
    "shame", "guilt", "pride", "jealousy", "contempt",
    "frustration", "longing", "melancholy", "nostalgia", "loneliness",
    "hope", "despair", "resignation", "anxiety", "awe", "tenderness",
    "bitterness", "anticipation",
]

MAX_CHARS = 800
BATCH = 8
TEMPERATURE = 0.05
TERMINATOR = 0


def cut(s):
    return s[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="world_0.1b_x070.safetensors")
    ap.add_argument("--slice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.slice, encoding="utf-8") if l.strip()]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    print({k: len(v) for k, v in by_task.items()})

    # Геометрия берётся из форм весов, как в dump_reference.py: конфигурации,
    # которую можно задать неверно, тут просто нет.
    from rwkv_metal.model.rwkv7_x070 import RWKV7X070, lora_ranks

    weights = mx.load(os.path.expanduser(args.model))
    n_layer = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
    n_embd = weights["ln_out.weight"].shape[0]
    vocab = weights["head.weight"].shape[0]
    head_size = weights["blocks.0.tmix.k_k"].shape[1]
    print(f"model: L={n_layer} D={n_embd} V={vocab} head={head_size}")

    class Cfg:
        pass

    cfg = Cfg()
    cfg.n_layer, cfg.n_embd, cfg.vocab_size = n_layer, n_embd, vocab
    cfg.head_size, cfg.n_head = head_size, n_embd // head_size

    base = RWKV7X070(cfg, lora_ranks(n_embd))
    base.load_weights(list(weights.items()))
    mx.eval(base.parameters())

    tok = rk.WorldTokenizer()
    model = EmbeddingModel(base)

    # ── голова: оба слоя ненулевые, иначе паритет проверяет полголовы ──
    mx.random.seed(args.seed)
    D = cfg.n_embd
    fc1 = mx.random.normal([D, D]) * 0.05
    fc2 = mx.random.normal([D, D]) * 0.05
    nw = mx.ones([D]) + mx.random.normal([D]) * 0.02
    nb = mx.random.normal([D]) * 0.02
    model.head.fc1.weight = fc1
    model.head.fc2.weight = fc2
    model.head.norm.weight = nw
    model.head.norm.bias = nb
    mx.eval(fc1, fc2, nw, nb)

    out = {
        "head/fc1": fc1, "head/fc2": fc2,
        "head/norm.weight": nw, "head/norm.bias": nb,
    }

    # ── триплетный батч (первые BATCH строк retrieval) ────────────────
    tri = by_task["retrieval"][:BATCH]
    a_idx, a_pool = encode_batch(tok, [cut(r["anchor"]) for r in tri], TERMINATOR)
    p_idx, p_pool = encode_batch(tok, [cut(r["positive"]) for r in tri], TERMINATOR)
    n_idx, n_pool = encode_batch(tok, [cut(r["negative"]) for r in tri], TERMINATOR)
    batch = (a_idx, a_pool, p_idx, p_pool, n_idx, n_pool)

    for name, arr in [("a_idx", a_idx), ("a_pool", a_pool), ("p_idx", p_idx),
                      ("p_pool", p_pool), ("n_idx", n_idx), ("n_pool", n_pool)]:
        out["triplet/" + name] = arr.astype(mx.int32)

    out["triplet/a_emb"] = model.embed(a_idx, a_pool).astype(mx.float32)
    out["triplet/p_emb"] = model.embed(p_idx, p_pool).astype(mx.float32)
    out["triplet/n_emb"] = model.embed(n_idx, n_pool).astype(mx.float32)
    out["loss/retrieval"] = retrieval_loss(model, batch, TEMPERATURE).astype(mx.float32)
    out["loss/sts"] = sts_loss(model, batch, TEMPERATURE).astype(mx.float32)

    # сырой пулинг без головы — отделяет дефект головы от дефекта пулинга
    h = model.base.body(a_idx)
    pooled = mx.take_along_axis(h, a_pool.reshape(-1, 1, 1), axis=1).squeeze(1)
    out["triplet/a_pooled_raw"] = pooled.astype(mx.float32)

    # ── батч классификации ────────────────────────────────────────────
    cls_rows, cands, targets = [], [], []
    for r in by_task["classification"]:
        c = parse_classification_candidates(r["anchor"])
        if c is None or r["positive"] not in c:
            continue
        cls_rows.append(r)
        cands.append(c)
        targets.append(c.index(r["positive"]))
    cls_rows, cands, targets = cls_rows[:BATCH], cands[:BATCH], targets[:BATCH]

    K = max(len(c) for c in cands)
    flat_texts, mask = [], []
    for c in cands:
        for k in range(K):
            flat_texts.append(c[k] if k < len(c) else "")
            mask.append(1.0 if k < len(c) else 0.0)

    ca_idx, ca_pool = encode_batch(tok, [cut(r["anchor"]) for r in cls_rows], TERMINATOR)
    cc_idx, cc_pool = encode_batch(tok, flat_texts, TERMINATOR)
    B = len(cls_rows)
    cls_batch = (ca_idx, ca_pool,
                 cc_idx.reshape(B, K, -1), cc_pool.reshape(B, K),
                 mx.array(mask).reshape(B, K), mx.array(targets))
    out["cls/a_idx"] = ca_idx.astype(mx.int32)
    out["cls/a_pool"] = ca_pool.astype(mx.int32)
    out["cls/c_idx"] = cc_idx.reshape(B, K, -1).astype(mx.int32)
    out["cls/c_pool"] = cc_pool.reshape(B, K).astype(mx.int32)
    out["cls/mask"] = mx.array(mask).reshape(B, K).astype(mx.float32)
    out["cls/target"] = mx.array(targets).astype(mx.int32)
    out["loss/classification"] = classification_loss(model, cls_batch, TEMPERATURE).astype(mx.float32)

    # ── метрики на отложенных строках (весь срез задачи) ──────────────
    ret_rows = [{"anchor": cut(r["anchor"]), "positive": cut(r["positive"]),
                 "negative": cut(r["negative"])} for r in by_task["retrieval"]]
    m = evaluate_retrieval(model, tok, ret_rows, max_chars=MAX_CHARS, terminator=TERMINATOR)
    out["metric/retrieval"] = mx.array(
        [m["mrr"], m["recall@1"], m["recall@5"], m["recall@10"], m["ndcg@10"], float(m["n"])]
    ).astype(mx.float32)

    sts_rows = [{"anchor": cut(r["anchor"]), "positive": cut(r["positive"]),
                 "negative": cut(r["negative"])} for r in by_task["sts"]]
    s = evaluate_sts_pairwise(model, tok, sts_rows, max_chars=MAX_CHARS, terminator=TERMINATOR)
    out["metric/sts"] = mx.array(
        [s["pairwise_accuracy"], s["mean_sim_pos"], s["mean_sim_neg"], float(s["n"])]
    ).astype(mx.float32)

    # ── классификация на ПОЛНОМ пуле из 25 меток ──────────────────────
    # eval.py считает по семёрке из инструкции; полный пул — то, что нужно
    # Swift, поэтому он считается здесь явно, тем же способом.
    full_rows, full_targets = [], []
    for r in by_task["classification"]:
        lbl = r["positive"].strip()
        if lbl not in FULL_POOL:
            continue
        full_rows.append(cut(r["anchor"]))
        full_targets.append(FULL_POOL.index(lbl))
    av = _embed_all(model, tok, full_rows)
    lv = _embed_all(model, tok, FULL_POOL)
    pred = mx.argmax(av @ lv.T, axis=-1)
    mx.eval(pred)
    acc = float((np.array(pred) == np.array(full_targets)).mean())
    out["metric/classification_full_pool"] = mx.array(
        [acc, float(len(full_rows)), float(len(FULL_POOL))]).astype(mx.float32)
    # Сами предсказания, а не только их точность. С необученной головой
    # точность вырождается (здесь она 0.0), и сравнивать по ней нечего:
    # такое же число выдаст любая сломанная реализация. Вектор из 40
    # дискретных выборов вырожденным не бывает.
    out["metric/classification_predictions"] = pred.astype(mx.int32)
    out["metric/classification_targets"] = mx.array(full_targets).astype(mx.int32)

    mx.eval(list(out.values()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mx.save_safetensors(args.out, {k: v for k, v in out.items()})

    print(f"\nretrieval loss      {out['loss/retrieval'].item():.6f}")
    print(f"sts loss            {out['loss/sts'].item():.6f}")
    print(f"classification loss {out['loss/classification'].item():.6f}")
    print(f"retrieval metrics   {m}")
    print(f"sts metrics         {s}")
    print(f"cls full-pool acc   {acc:.4f} на {len(full_rows)} строках")
    print(f"\n-> {args.out}")


def _embed_all(model, tok, texts, batch_size=16):
    vecs = []
    for i in range(0, len(texts), batch_size):
        idx, pool = encode_batch(tok, texts[i:i + batch_size], TERMINATOR)
        vecs.append(model.embed(idx, pool))
    out = mx.concatenate(vecs, axis=0).astype(mx.float32)
    mx.eval(out)
    return out


if __name__ == "__main__":
    main()
