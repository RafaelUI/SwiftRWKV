"""
reranker-triples-multi: parquet → jsonl.

Зачем. Датасет лежит в parquet, а читать parquet из Swift нечем — ни в
стандартной библиотеке, ни среди зависимостей пакета. Заводить ради этого
Arrow-биндинг было бы несоразмерно: конвертация происходит ОДИН раз на
датасет, а не на каждый прогон, и её результат — обычный jsonl, который уже
умеют читать и эмбеддинги, и реранкер.

Формат на выходе — построчный JSON, одна строка на запрос:

    {"query": ..., "positive": ..., "negatives": [...], "language": "eng"}

то есть ровно поля исходника. Никакого приведения к формату LitRetrieval
здесь НЕ делается намеренно: у triples-multi пять майненных негативов на
строку против одного у LitRetrieval, и сплющивание потеряло бы четыре из
пяти. Разбираться с двумя формами кандидатов — работа загрузчика, там это
видно и проверяемо, а не спрятано в разовом скрипте.

Запуск:
    ~/Develop/tests/venv/bin/python \
        ~/Develop/SwiftRWKV/Scripts/convert_reranker_triples.py \
        --src ~/Develop/reranker-triples-multi/data \
        --out ~/Develop/reranker-triples-multi/train.jsonl

Срез для тестовой фикстуры (детерминированный, по строкам с начала каждого
языка — фикстуре нужна структура, а не репрезентативность):
    ... --out .testdata/reranker_slice.jsonl --per-language 6 --max-chars 600
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="каталог с train-*.parquet либо один файл")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-language", type=int, default=0,
                    help="сколько строк брать на язык; 0 — все")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="усечь тексты до N символов; 0 — не усекать")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("нужен pyarrow: ~/Develop/tests/venv/bin/pip install pyarrow")

    src = Path(os.path.expanduser(args.src))
    files = sorted(src.glob("*.parquet")) if src.is_dir() else [src]
    if not files:
        sys.exit(f"в {src} нет ни одного .parquet")

    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def cut(s):
        return s[:args.max_chars] if args.max_chars else s

    per_lang = {}
    written = 0
    skipped_empty = 0
    langs = set()

    with open(out_path, "w", encoding="utf-8") as fh:
        for path in files:
            pf = pq.ParquetFile(path)
            # По батчам, а не целиком: файлы по 230–380 МБ в parquet
            # разворачиваются в память кратно больше, а нужды держать их
            # целиком нет — пишем потоком.
            for batch in pf.iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    lang = row.get("language") or "unk"
                    langs.add(lang)
                    if args.per_language:
                        if per_lang.get(lang, 0) >= args.per_language:
                            continue
                        per_lang[lang] = per_lang.get(lang, 0) + 1

                    negs = [n for n in (row.get("negatives") or []) if n]
                    # Строка без единого негатива для listwise бесполезна:
                    # список кандидатов вырождается в одного позитива, и лосс
                    # на нём тождественно ноль. Такие пропускаем и считаем.
                    if not row.get("query") or not row.get("positive") or not negs:
                        skipped_empty += 1
                        continue

                    fh.write(json.dumps({
                        "query": cut(row["query"]),
                        "positive": cut(row["positive"]),
                        "negatives": [cut(n) for n in negs],
                        "language": lang,
                    }, ensure_ascii=False) + "\n")
                    written += 1
            print(f"  {path.name}: всего записано {written}")

    print(f"\nстрок: {written}, пропущено пустых/без негативов: {skipped_empty}")
    print(f"языков: {len(langs)} — {', '.join(sorted(langs))}")
    print(f"-> {out_path}  ({out_path.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
