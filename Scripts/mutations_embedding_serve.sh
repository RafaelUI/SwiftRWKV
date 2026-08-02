#!/bin/bash
# Мутационная проверка чекпоинта эмбеддера и контракта подачи текста.
#
#   ./Scripts/mutations_embedding_serve.sh [фильтр тестов]
#
# Дефекты здесь особого рода: вектор L2-нормирован ВСЕГДА — при любом
# пулинге, любой обрезке, с любым терминатором и даже с незагруженными
# весами. Ошибок формы не возникает нигде, отличить можно только по числам,
# и потому каждая мутация обязана ловиться.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
CKPT=Sources/RWKVEmbedding/EmbeddingCheckpoint.swift
EMB=Sources/RWKVEmbedding/Embedder.swift

FILTER="${1:-EmbeddingSmokeTests}"
run() {
    local desc="$1"; shift
    printf '%-52s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Обрезка ──"

# Раньше выдача не обрезала вовсе, а метрики обрезали на 512: измеренное
# качество описывало не то, что делает выдача.
run "обрезка снята" \
    "$EMB" 'if ids.count > room { ids = Array(ids.prefix(Swift.max(0, room))) }' \
           'if false { ids = Array(ids.prefix(Swift.max(0, room))) }'

# Обрезать после дописывания терминатора значит либо срезать его самого,
# либо оставить не на конце — тогда пулинг снимает вектор не с того места.
run "места под терминатор не оставлено" \
    "$EMB" 'let room = terminator == nil ? m : m - 1' \
           'let room = m'

echo
echo "── Чекпоинт ──"

run "веса из файла не загружаются" \
    "$CKPT" 'head.setParameters(names.map { weights[$0]! })' \
            '_ = names'

run "контракт в файл не пишется" \
    "$CKPT" 'var md = c.metadata' \
            'var md = [String: String]()'

run "пулинг при загрузке не сверяется" \
    "$CKPT" 'if want.pooling != pooling {' \
            'if false {'

run "ширина головы при загрузке не сверяется" \
    "$CKPT" 'if let h = md["hidden"], Int(h) != head.hidden {' \
            'if let h = md["hidden"], Int(h) != head.hidden, false {'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше. Каждая — либо пробел в наборе"
echo "тестов, либо утверждение о том, что мутация безвредна; второе"
echo "нужно обосновать разбором, а не оставлять как есть."
