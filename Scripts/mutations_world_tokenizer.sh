#!/bin/bash
# Мутационная проверка (25.10): плоское дерево WorldTokenizer против эталона
# словаря и Python (WorldTokenizerTests, GenerateTests).
#
#   ./Scripts/mutations_world_tokenizer.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
W=Sources/RWKVGen/WorldTokenizer.swift
FILTER='WorldTokenizerTests|GenerateTests'
run() {
    local desc="$1"; shift
    printf '%-52s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}
run "самый короткий токен вместо самого длинного" \
    "$W" '                    if t >= 0 { lastEnd = i; lastId = Int(t) }' '                    if t >= 0 { lastEnd = i; lastId = Int(t); break }'
run "ранний выход из списка не в ту сторону" \
    "$W" '                    if x > b { return 0 }' '                    if x < b { return 0 }'
run "список детей не отсортирован" \
    "$W" 'for (b, c) in ch.sorted(by: { $0.key < $1.key }) { edgeByte.append(b); edgeTo.append(c) }' 'for (b, c) in ch.sorted(by: { $0.key > $1.key }) { edgeByte.append(b); edgeTo.append(c) }'
run "плотная таблица не заполнена" \
    "$W" '                for (b, c) in ch { row[Int(b)] = c }' '                for (b, c) in ch where b < 0x80 { row[Int(b)] = c }'
run "id токена не записан в узел" \
    "$W" '            tok[n] = Int32(id)' '            tok[n] = -1'
run "decode пропускает последний токен" \
    "$W" '        for tok in tokens where tok >= 0 && tok < idx2token.count { bytes.append(contentsOf: idx2token[tok]) }' '        for tok in tokens.dropLast() where tok >= 0 && tok < idx2token.count { bytes.append(contentsOf: idx2token[tok]) }'
