"""
Эталонная таблица World-словаря: id → байты, разобранная САМИМ Python.

Зачем. Файл словаря хранит токены как Python-repr (`'слово'`, `b'\\xcc'`,
`'\\xa0'`), и разбор этого формата — отдельная задача со своими ловушками:
str-repr и bytes-repr трактуют `\\xNN` ПО-РАЗНОМУ (в первом это код-поинт
U+00NN, то есть два байта UTF-8; во втором — сырой байт), а у 57 токенов
repr начинается с кавычки, за которой идёт комбинирующий знак — и любой
разбор, работающий на уровне символов, а не байтов, склеит их в один
графемный кластер и потеряет кавычку.

Проверять такой разбор «на глаз» бессмысленно: 65 529 строк, и ошибка на
72 из них не мешает ни одному тексту без NBSP и ударений. Поэтому эталон
снимается ast.literal_eval'ом — тем самым разбором, которым словарь и был
записан, — и Swift сверяет с ним ВСЮ таблицу целиком.

Формат вывода — плоский бинарник, чтобы читаться без зависимостей:
    magic  "RWKVVOCB"            8 байт
    count  uint32 LE             число токенов
    затем count записей:  id uint32 LE | len uint32 LE | len байт

Запуск:
    python3 Scripts/dump_vocab_reference.py \
        --vocab .testdata/rwkv_vocab_v20230424.txt \
        --out   .testdata/world_vocab_bytes.bin
"""
import argparse
import ast
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.vocab, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            i = line.index(" ")
            j = line.rindex(" ")
            idx = int(line[:i])
            repr_s = line[i + 1:j]
            declared = int(line[j + 1:])
            x = ast.literal_eval(repr_s)
            b = x if isinstance(x, bytes) else x.encode("utf-8")
            # Заявленная длина — независимая проверка самого файла.
            assert len(b) == declared, f"строка {lineno}: {len(b)} != {declared}"
            rows.append((idx, b))

    with open(args.out, "wb") as f:
        f.write(b"RWKVVOCB")
        f.write(struct.pack("<I", len(rows)))
        for idx, b in rows:
            f.write(struct.pack("<II", idx, len(b)))
            f.write(b)

    print(f"{len(rows)} токенов -> {args.out}")


if __name__ == "__main__":
    main()
