#!/bin/bash
# Мутационная проверка (25.10): батчевый декод (BatchGenerateTests).
#
#   ./Scripts/mutations_batch_decode.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
G=Sources/RWKVGen/BatchGenerate.swift
FILTER='BatchGenerateTests'
run() {
    local desc="$1"; shift
    printf '%-50s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}
run "префилл без маски паддинга" \
    "$G" '            ? buildMask(lengths: lengths, total: T) : nil' '            ? Optional<MLXArray>.none : nil'
run "сдвиги — с последней позиции, а не с конца строки" \
    "$G" 'let (lnOut, state) = bodyWithState(ids, mask: mask, endIdx: endIdx)' 'let (lnOut, state) = bodyWithState(ids, mask: mask, endIdx: nil)'
run "логиты префилла с последней позиции" \
    "$G" 'let last = gatherLast(lnOut, endIdx).reshaped' 'let last = gatherLast(lnOut).reshaped'
run "выбывшие строки не сопоставлены промптам" \
    "$G" '                alive = keep.map { alive[$0] }' '                alive = keep'
run "сид строки не сдвигается" \
    "$G" '            c.seed = c.seed &+ UInt64(i)' '            c.seed = c.seed &+ 0'
run "история промпта не учитывается" \
    "$G" '            if config.penalizePrompt { for id in prompts[i] { s.observe(id) } }' '            if false { for id in prompts[i] { s.observe(id) } }'
run "стоп-строка не вырезается" \
    "$G" '                    bytes[i].removeSubrange(hit.start ..< bytes[i].count)' '                    _ = hit.start'
run "стоп-токен не останавливает строку" \
    "$G" 'if config.stopTokens.contains(id) { reasons[i] = .stopToken(id); continue }' 'if false { reasons[i] = .stopToken(id); continue }'
