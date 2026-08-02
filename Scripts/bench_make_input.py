#!/usr/bin/env python3
"""
Общий вход для замера кодирования: Swift и Python получают ОДИН И ТОТ ЖЕ
файл — те же тексты, та же группировка по префиксам, те же параметры.

Иначе сравнивать нечего: разные срезы данных дают разные длины документов,
разное число уникальных префиксов и разное число пачек, и любая разница во
времени объясняется постановкой, а не реализацией.

Подготовка данных здесь НЕ засекается ни там, ни там, поэтому неважно, кто
её делает.

    python3 Scripts/bench_make_input.py \
        --data ~/Develop/reranker-triples-multi/train.jsonl \
        --prefixes 200 --candidates 8 --out /tmp/bench_input.json
"""
import argparse
import json
import random

INSTRUCT = ("Given a search query, retrieve relevant passages that answer "
            "the query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prefixes", type=int, default=200,
                    help="сколько уникальных документов кодировать")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--doc-batch", type=int, default=8)
    ap.add_argument("--query-batch", type=int, default=16)
    ap.add_argument("--max-doc-tokens", type=int, default=384)
    ap.add_argument("--max-query-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= a.rows:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d.get("query") or d.get("anchor")
            pos = d.get("positive")
            negs = d.get("negatives") or ([d["negative"]] if d.get("negative") else [])
            if q and pos:
                rows.append((q, pos, list(negs)))

    rng = random.Random(a.seed)
    pool, seen = [], {}

    def add(text):
        if text not in seen:
            seen[text] = len(pool)
            pool.append(text)
        return seen[text]

    for _, pos, negs in rows:
        add(pos)
        for n in negs:
            add(n)

    # Кандидаты: позитив + майненные негативы + добор из пула, позиция
    # позитива перемешана — как в обоих фреймворках.
    prefix_queries = {}
    for q, pos, negs in rows:
        ids = [add(pos)] + [add(n) for n in negs[:a.candidates - 1]]
        while len(ids) < a.candidates:
            cand = rng.randrange(len(pool))
            if cand not in ids:
                ids.append(cand)
        rng.shuffle(ids)
        for did in ids:
            prefix_queries.setdefault(did, []).append(q)

    # Берём первые N префиксов в порядке появления — детерминированно.
    chosen = sorted(prefix_queries.keys())[:a.prefixes]
    prefixes = [{"doc": pool[d], "queries": prefix_queries[d]} for d in chosen]

    out = {
        "instruct": INSTRUCT,
        "max_doc_tokens": a.max_doc_tokens,
        "max_query_tokens": a.max_query_tokens,
        "terminator": 0,
        "doc_batch": a.doc_batch,
        "query_batch": a.query_batch,
        "prefixes": prefixes,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    tails = sum(len(p["queries"]) for p in prefixes)
    print(f"префиксов {len(prefixes)}, хвостов {tails}, "
          f"пул {len(pool)}, файл {a.out}")


if __name__ == "__main__":
    main()
