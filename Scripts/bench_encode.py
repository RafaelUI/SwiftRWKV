#!/usr/bin/env python3
"""
Питоновская половина замера кодирования. Swift-половина — `rerank-run --bench`.

Обе читают ОДИН файл входа (`bench_make_input.py`) и делают ровно одно и то
же: токенизируют префиксы, гоняют их пачками через базу, продолжают хвостами
запросов, отбирают читаемые головой слои.

Что засекается по отдельности и почему:

  load        — веса и словарь. В разрыв «Swift против Python» не входит и
                вынесено, чтобы не входило.
  warmup      — первые батчи ОТБРАСЫВАЮТСЯ. На Metal первый запуск каждой
                новой формы тянет за собой сборку конвейера, и без прогрева
                этот разовый расход размазывается по замеру, притворяясь
                пропускной способностью.
  tokenize    — чистый CPU, без GPU вовсе.
  gpu_prefix  — batch_ids + проход базы + eval (синхронизация обязательна,
                иначе засекается постановка в очередь, а не работа).
  gpu_tail    — то же для хвостов поверх состояния префикса.
  select      — отбор слоёв и вынос в numpy.

Ряд по батчам печатается целиком: разовый расход виден только по форме
кривой, суммой его не отличить от медленного кода.

    .venv/bin/python bench_encode.py --model world_0.1b_x070.safetensors \
        --input /tmp/bench_input.json --warmup 3
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

# Рабочее ДЕРЕВО, а не установленная копия. Скрипт лежит в SwiftRWKV/Scripts,
# поэтому sys.path[0] указывает туда, и `import rwkv_metal` без этой строки
# подхватывает site-packages — то есть меряет не тот код, который лежит
# рядом с замером. Молчаливо: версии обычно близкие, а числа разные.
_REPO = os.environ.get("RWKV_METAL_REPO",
                       os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, _REPO)


# Память — ПО ДАННЫМ СИСТЕМЫ, а не по счётчикам MLX. `mx.get_peak_memory()`
# знает только про свой пул: ни про numpy-буферы, ни про веса, ни про то,
# ушла ли машина в своп. А если процесс свопится, врут и замеры времени.
def rss_gb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 / 1e9 if out else float("nan")


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (raw if sys.platform == "darwin" else raw * 1024) / 1e9


def footprint_gb() -> float:
    """Physical footprint: буферы Metal живут в IOAccelerator и в RSS
    попадают неполностью."""
    try:
        out = subprocess.run(["vmmap", "--summary", str(os.getpid())],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "Physical footprint:" in line and "Peak" not in line:
                v = line.split(":")[1].strip()
                n = float(v.rstrip("GMK"))
                return n if v.endswith("G") else n / 1024 if v.endswith("M") else n / 1e6
    except Exception:
        pass
    return float("nan")


def swap_counters():
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        ins = outs = 0
        for line in vm.splitlines():
            if "Swapins" in line:
                ins = int(line.split(":")[1].strip().rstrip("."))
            elif "Swapouts" in line:
                outs = int(line.split(":")[1].strip().rstrip("."))
        return ins, outs
    except Exception:
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--warmup", type=int, default=3,
                    help="сколько первых пачек префиксов отбросить")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--out", default=None, help="куда положить JSON с числами")
    ap.add_argument("--cache-limit", type=float, default=2.0,
                    help="потолок буферного кэша MLX, ГБ; 0 — не трогать")
    a = ap.parse_args()

    with open(a.input, encoding="utf-8") as f:
        inp = json.load(f)

    # Потолок буферного кэша Metal. БЕЗ него замер недействителен: формы
    # батчей плавают, переиспользовать буферы нечего, кэш растёт линейно по
    # числу пачек. Замерено: на восьмистах префиксах footprint дошёл до
    # 11.6 ГБ (RSS при этом показывал 0.4 — буферы живут в IOAccelerator), и
    # машина ушла в своп.
    #
    # Этого потолка нет в самом `encode_pairs` — то есть дефект здесь не
    # только у замера. В Swift он вылечен (`RerankEncodeConfig.cacheLimitGB`
    # плюс округление длины батча), в Python пока нет.
    if a.cache_limit > 0:
        mx.set_cache_limit(int(a.cache_limit * 1e9))

    swap0 = swap_counters()
    t = time.perf_counter()
    import rwkv_metal
    from rwkv_metal.model.rwkv7_x070 import RWKV7X070, lora_ranks
    from rwkv_metal.tokenizer import WorldTokenizer
    from rwkv_metal.reranker import Reranker, RerankerConfig
    from rwkv_metal.reranker.encode import (_batch_ids, _encode_prefix_ids,
                                            _encode_suffix_ids)
    from rwkv_metal.reranker.data import PairTemplate
    import_s = time.perf_counter() - t

    t = time.perf_counter()
    # Геометрия — из форм весов, как в dump_block_reference.py: конфигурации,
    # которую можно задать неверно, тут просто нет.
    weights = mx.load(os.path.expanduser(a.model))
    n_layer = max(int(k.split(".")[1])
                  for k in weights if k.startswith("blocks.")) + 1

    class Cfg:
        pass

    cfg = Cfg()
    cfg.n_layer = n_layer
    cfg.n_embd = weights["ln_out.weight"].shape[0]
    cfg.vocab_size = weights["head.weight"].shape[0]
    cfg.head_size = weights["blocks.0.tmix.k_k"].shape[1]
    cfg.n_head = cfg.n_embd // cfg.head_size
    base = RWKV7X070(cfg, lora_ranks(cfg.n_embd))
    base.load_weights(list(weights.items()))
    mx.eval(base.parameters())
    tok = WorldTokenizer()
    model = Reranker(base, RerankerConfig(layer_idx=(a.layers,)))
    load_s = time.perf_counter() - t
    print(f"модель: L={cfg.n_layer} D={cfg.n_embd} V={cfg.vocab_size}, "
          f"rwkv_metal из {_REPO}", flush=True)

    template = PairTemplate()
    prefixes = inp["prefixes"]
    doc_batch, query_batch = inp["doc_batch"], inp["query_batch"]

    def prefix_ids(doc):
        return _encode_prefix_ids(tok, template, inp["instruct"], doc,
                                  inp["max_doc_tokens"])

    def suffix_ids(doc, query):
        return _encode_suffix_ids(tok, template, doc, query,
                                  inp["max_query_tokens"],
                                  inp["max_doc_tokens"], inp["terminator"])

    acc = {"tokenize": 0.0, "gpu_prefix": 0.0, "gpu_tail": 0.0, "select": 0.0}
    series = []
    n_prefix_done = n_tail_done = 0
    shapes_prefix, shapes_tail = set(), set()

    wall0 = time.perf_counter()
    for bi, start in enumerate(range(0, len(prefixes), doc_batch)):
        chunk = prefixes[start:start + doc_batch]
        warm = bi < a.warmup
        b = {}

        t = time.perf_counter()
        seqs = [prefix_ids(p["doc"]) for p in chunk]
        b["tokenize"] = time.perf_counter() - t

        t = time.perf_counter()
        idx, mask, end_idx = _batch_ids(seqs)
        st = model.encode(idx, mask=mask, end_idx=end_idx)
        st.eval()
        b["gpu_prefix"] = time.perf_counter() - t
        shapes_prefix.add(tuple(idx.shape))

        jobs = [(local, q) for local, p in enumerate(chunk)
                for q in p["queries"]]
        b["gpu_tail"] = b["select"] = b["tok_tail"] = 0.0
        for qs in range(0, len(jobs), query_batch):
            part = jobs[qs:qs + query_batch]
            t = time.perf_counter()
            qseqs = [suffix_ids(chunk[l]["doc"], q) for l, q in part]
            b["tok_tail"] += time.perf_counter() - t

            t = time.perf_counter()
            locs = mx.array(np.array([p[0] for p in part], dtype=np.int32))
            sub = st[locs]
            qidx, qmask, qend = _batch_ids(qseqs)
            st_pair = model.encode(qidx, mask=qmask, end_idx=qend, state=sub)
            sel = model.select(st_pair)
            mx.eval(sel)
            b["gpu_tail"] += time.perf_counter() - t
            shapes_tail.add(tuple(qidx.shape))

            t = time.perf_counter()
            _ = np.array(sel.astype(mx.float32)).astype(np.float16)
            b["select"] += time.perf_counter() - t

        b["tokenize"] += b.pop("tok_tail")
        if not warm:
            for k in acc:
                acc[k] += b[k]
            n_prefix_done += len(chunk)
            n_tail_done += len(jobs)
        series.append({"batch": bi, "warmup": warm,
                       "prefixes": len(chunk), "tails": len(jobs),
                       "rss_gb": round(rss_gb(), 2),
                       **{k: round(v * 1000, 2) for k, v in b.items()}})
    wall = time.perf_counter() - wall0
    swap1 = swap_counters()

    measured = sum(acc.values())
    swapped_out = swap1[1] - swap0[1]
    res = {
        "side": "python", "cache_limit_gb": a.cache_limit,
        "valid": swapped_out == 0, "import_s": import_s, "load_s": load_s,
        "wall_s": wall, "measured_s": measured,
        "prefixes": n_prefix_done, "tails": n_tail_done,
        "warmup_batches": a.warmup,
        "ms_per_prefix": acc["gpu_prefix"] / max(1, n_prefix_done) * 1000,
        "ms_per_tail": acc["gpu_tail"] / max(1, n_tail_done) * 1000,
        "distinct_prefix_shapes": len(shapes_prefix),
        "distinct_tail_shapes": len(shapes_tail),
        "rss_gb": rss_gb(), "peak_rss_gb": peak_rss_gb(),
        "footprint_gb": footprint_gb(),
        "swapins": swap1[0] - swap0[0], "swapouts": swapped_out,
        "phases_s": acc, "series": series,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "series"},
                     ensure_ascii=False, indent=2))
    if swapped_out > 0:
        print(f"ЗАМЕР НЕДЕЙСТВИТЕЛЕН: {swapped_out} страниц ушло в своп. "
              "Числа выше меряют диск, а не код.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
