#!/bin/bash
# Мутационная проверка (25.10): умолчание приведения выхода WKV — по пути базы
# (плотная — вкл., норма .rwkvq — выкл.), явный выбор, восстановление флага
# после обучения состояния.
#
#   ./Scripts/mutations_wkv_cast_default.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
BB=Sources/RWKVGen/X070Backbone.swift
IS=Sources/RWKVGen/InitialState.swift

FILTER='WKVDTypeLeakTests|StateTuningCastTests'
run() {
    local desc="$1"; shift
    printf '%-62s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

run "умолчание — прежнее «выключено»" \
    "$BB" 'get { wkvOutputCast ?? (rwkvqBaseDType == nil) }' 'get { wkvOutputCast ?? false }'
run "умолчание — «включено» и на норме .rwkvq" \
    "$BB" 'get { wkvOutputCast ?? (rwkvqBaseDType == nil) }' 'get { wkvOutputCast ?? true }'
run "явная запись не действует" \
    "$BB" 'set { wkvOutputCast = newValue }' 'set { _ = newValue }'
run "обучение возвращает действующее значение, а не выбор" \
    "$IS" 'savedWKVCast = bb.wkvOutputCast' 'savedWKVCast = Optional(bb.castWKVOutputToComputeDType)'
run "обучение не возвращает приведение" \
    "$IS" 'bb.castDenseProjectionInput = savedProjectionCast; bb.wkvOutputCast = savedWKVCast' 'bb.castDenseProjectionInput = savedProjectionCast'
