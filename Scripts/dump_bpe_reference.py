#!/usr/bin/env python3
"""Эталон BPE-токенизатора HuggingFace (tokenizer.json) для BPETokenizerTests.

Кодирует набор крайних случаев (специальные токены, пробелы и переводы
строк, NFKC, цифры, эмодзи, апострофы, пустая строка) и тексты из корпуса
JSONL (поля anchor/positive/negative/query/text), пишет id, декодирование
с пропуском специальных токенов и без, байты каждого токена словаря.

    ~/Develop/rwkv-metal/.venv/bin/python Scripts/dump_bpe_reference.py \\
        --tokenizer .testdata/tokenizer_ru16k.json \\
        --corpus ~/Develop/swr_bench_keep/w1014/real.jsonl --n 1500 \\
        --out .testdata/bpe_ru16k_ref.json
"""
import argparse
import json

from tokenizers import Tokenizer

EDGE = [
    "", " ", "  ", "\n", "\n\n", "\t", " \n ", "a", "Привет, мир!", " Привет", "Привет  мир",
    "a<eos>b", "<eos>", "<eos><eos>", "x <bos> y", "<pad>текст<unk>", "<|doc_sep|>второй документ",
    "<eo s>", "<EOS>", "ﬁ ① ＡＢＣ ｶ ㍿", "  два  пробела\n\nи\tтаб  ", "123456 7.5 1e-9 3,14",
    "😀 эмодзи 👨‍👩‍👧", "don't I'll we've she'd they're It's", "Ёжик ёлка Й й",
    "café naïve résumé", "é й", "русский English Deutsch 中文 日本語 한국어",
    "end with space ", "   leading", "trailing\n", "\r\nwindows\r\n", " неразрывный пробел",
    "​нулевой​", "Ünïcödé — «кавычки» „немецкие“ …", "a" * 300, "абв" * 200,
    "http://example.com/path?q=1&b=2", "def f(x):\n    return x**2  # comment", "$100 €50 ₽1000",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    texts = list(EDGE)
    with open(a.corpus, encoding="utf-8") as f:
        for line in f:
            if len(texts) >= len(EDGE) + a.n:
                break
            d = json.loads(line)
            for k in ("anchor", "positive", "negative", "query", "text"):
                v = d.get(k)
                if isinstance(v, str) and v:
                    texts.append(v[: a.max_chars])
    texts = texts[: len(EDGE) + a.n]
    enc = tok.encode_batch(texts)
    ids = [e.ids for e in enc]
    out = {
        "tokenizer": a.tokenizer,
        "version": __import__("tokenizers").__version__,
        "texts": texts,
        "ids": ids,
        "decoded": [tok.decode(i) for i in ids],
        "decoded_special": [tok.decode(i, skip_special_tokens=False) for i in ids],
        "vocab_size": tok.get_vocab_size(),
        # байты каждого id через декодер (без пропуска специальных)
        "token_text": [tok.decode([i], skip_special_tokens=False) for i in range(tok.get_vocab_size())],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"текстов {len(texts)}, токенов {sum(map(len, ids))}, словарь {out['vocab_size']} -> {a.out}")


if __name__ == "__main__":
    main()
