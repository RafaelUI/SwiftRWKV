#!/bin/bash
# Мутационная проверка (23.10): типы в шаге state-tuning — приведение входа плотной проекции
# (castDenseProjectionInput) и выхода WKV (castWKVOutputToComputeDType), выставляемые на время
# StateTuning.run из StateTuneConfig.
#   ./Scripts/mutations_state_cast.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
BB=Sources/RWKVGen/X070Backbone.swift
IS=Sources/RWKVGen/InitialState.swift
FILTER='StateTuningCast'
run() { local d="$1"; shift; printf '%-58s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "умолчание castProjectionInput — включено" \
    "$IS" '    public var castProjectionInput = false' '    public var castProjectionInput = true'
run "умолчание castWKVOutput — выключено" \
    "$IS" '    public var castWKVOutput = true' '    public var castWKVOutput = false'
run "run не выставляет приведение проекций (остаётся флаг бэкбона)" \
    "$IS" '        bb.castDenseProjectionInput = cfg.castProjectionInput' '        bb.castDenseProjectionInput = savedProjectionCast'
run "приведение проекций берётся из поля WKV" \
    "$IS" '        bb.castDenseProjectionInput = cfg.castProjectionInput' '        bb.castDenseProjectionInput = cfg.castWKVOutput'
run "поля перепутаны (WKV из поля проекций)" \
    "$IS" '        bb.castWKVOutputToComputeDType = cfg.castWKVOutput' '        bb.castWKVOutputToComputeDType = cfg.castProjectionInput'
run "run не выставляет приведение выхода WKV" \
    "$IS" '        bb.castWKVOutputToComputeDType = cfg.castWKVOutput' '        bb.castWKVOutputToComputeDType = false'
run "флаги бэкбона не возвращаются после обучения" \
    "$IS" '            bb.castDenseProjectionInput = savedProjectionCast; bb.castWKVOutputToComputeDType = savedWKVCast' '            _ = savedProjectionCast; _ = savedWKVCast'
run "выход проекции не приводится к типу входа" \
    "$BB" '                return matmul(x.asType(W.dtype), W.transposed()).asType(x.dtype)' '                return matmul(x.asType(W.dtype), W.transposed())'
run "вход приводится к fp32, а не к типу веса" \
    "$BB" '                return matmul(x.asType(W.dtype), W.transposed()).asType(x.dtype)' '                return matmul(x.asType(.float32), W.transposed()).asType(x.dtype)'
run "условие по типу перевёрнуто" \
    "$BB" '            if castDenseProjectionInput && x.dtype != W.dtype {' '            if castDenseProjectionInput && x.dtype == W.dtype {'
run "флаг проекции не читается" \
    "$BB" '            if castDenseProjectionInput && x.dtype != W.dtype {' '            if false && x.dtype != W.dtype {'
