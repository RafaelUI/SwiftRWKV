#!/bin/bash
# Мутационная проверка (18.10): оценка retrieval со слиянием одинаковых кандидатов.
#   ./Scripts/mutations_dedupe.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
MT=Sources/RWKVEmbedding/EmbeddingMetrics.swift
FT=Sources/RWKVEmbedding/EmbeddingFinetune.swift
FILTER='RetrievalDedupeTests'
run() { local d="$1"; shift; printf '%-52s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "слияние не применяется" \
    "$MT" '        if dedupeCandidates {' '        if false {'
run "пул без слияния тоже сливается" \
    "$MT" '        if dedupeCandidates {' '        if true {'
run "верный индекс — номер строки, а не текста" \
    "$MT" '            correct = positives.map { index[$0]! }' '            correct = positives.indices.map { Swift.min($0, unique.count - 1) }'
run "пул — позитивы без негативов" \
    "$MT" '            candidates = unique' '            candidates = Array(unique.prefix(Set(positives).count))'
run "стадия не передаёт выключатель" \
    "$FT" 'maxTokens: stage.maxTokens, dedupeCandidates: stage.dedupeEvaluationPool))' 'maxTokens: stage.maxTokens, dedupeCandidates: false))'
