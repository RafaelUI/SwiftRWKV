#!/bin/bash
# Мутационная проверка (20.10): маска лосса языковой модели (цель < 0 — вне лосса).
#   ./Scripts/mutations_loss_mask.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
LF=Sources/RWKVGen/LoRAFinetune.swift
SB=Sources/RWKVGen/Training/SequentialBackward.swift
FILTER='LossMaskTests'
run() { local d="$1"; shift; printf '%-56s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "делитель — все токены, а не вошедшие (кусками)" \
    "$LF" '            return total / Float(masked ? lossTokens(batch.y) : B * T0)' '            return total / Float(B * T0)'
run "делитель — все токены (целиком)" \
    "$LF" '                / Float(lossTokens(batch.y))' '                / Float(batch.y.size)'
run "исключённые токены входят в лосс (как цель 0)" \
    "$LF" '        let keep = (targets .>= 0).asType(.float32)' '        let keep = (targets .>= -1).asType(.float32)'
run "по блокам: делитель — все токены" \
    "$SB" 'let scale = MLXArray(Float(1) / Float(masked ? LoRAFinetune.lossTokens(batch.y) : B * T))' 'let scale = MLXArray(Float(1) / Float(B * T))'
run "по блокам: маска не применяется" \
    "$SB" $'                return [masked\n                    ? LoRAFinetune.maskedCrossEntropySum' $'                return [false\n                    ? LoRAFinetune.maskedCrossEntropySum'
# Мутант «маска включается и без отрицательных целей» равносилен: при всех целях
# >= 0 маскированная сумма / число токенов = среднее (на тесте — даже побитно), его
# не поймать и он ничего не ломает. Заменён на подсчёт вошедших токенов.
run "подсчёт вошедших токенов считает и исключённые" \
    "$LF" '    static func lossTokens(_ y: MLXArray) -> Int { (y .>= 0).asType(.int32).sum().item(Int.self) }' '    static func lossTokens(_ y: MLXArray) -> Int { (y .>= -1).asType(.int32).sum().item(Int.self) }'
