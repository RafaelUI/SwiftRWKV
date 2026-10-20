#!/bin/bash
# Мутационная проверка (20.10): банк начальных состояний (своё у каждой строки батча).
#   ./Scripts/mutations_state_bank.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
SB=Sources/RWKVGen/InitialStateBank.swift
BL=Sources/RWKVGen/RWKVBlock.swift
OB=Sources/RWKVEmbedding/EmbeddingObjective.swift
DS=Sources/RWKVEmbedding/EmbeddingDataset.swift
EM=Sources/RWKVEmbedding/EmbeddingMetrics.swift
IC=Sources/RWKVEmbedding/InstructionCache.swift
FILTER='StateBank'
run() { local d="$1"; shift; printf '%-56s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "WKV всех строк — из строки 0 банка" \
    "$SB" '        InitialLayerState(wkv: wkv.take(rows, axis: 0), tmixShift: tmixShift.take(rows, axis: 0),' '        InitialLayerState(wkv: wkv.take(rows * 0, axis: 0), tmixShift: tmixShift.take(rows, axis: 0),'
run "сдвиг tmix всех строк — из строки 0" \
    "$SB" 'wkv.take(rows, axis: 0), tmixShift: tmixShift.take(rows, axis: 0),' 'wkv.take(rows, axis: 0), tmixShift: tmixShift.take(rows * 0, axis: 0),'
run "при строках берётся одно подключённое, а не банк" \
    "$BL" '            return bank[layer].selecting(rows)' '            _ = bank; _ = rows; return backbone.initialLayers?[layer]'
run "приостановка не выключает банк" \
    "$BL" $'        if backbone.initialStateSuspended { return nil }\n        if let rows' $'        if let rows'
run "строки не снимаются после прохода" \
    "$SB" '        defer { initialRows = saved }' '        defer { _ = saved }'
run "inject пишет tmix в cmix" \
    "$SB" '                case .tmixShift: layers[l].tmixShift = ps[i]' '                case .tmixShift: layers[l].cmixShift = ps[i]'
run "куски по строкам берут строки банка с начала" \
    "$OB" 'st?[from ..< to]))' 'st?[0 ..< to - from]))'
run "GradCache берёт строки банка с начала" \
    "$OB" 'rows: batch.anchorRows?[start ..< end]),' 'rows: batch.anchorRows?[0 ..< end - start]),'
run "батчер переворачивает номера состояний" \
    "$DS" 'anchorRows: stateRows(rows))' 'anchorRows: stateRows(rows.reversed()))'
run "обрезка теряет номер состояния" \
    "$DS" '                        negative: String(negative.prefix(maxChars)), task: task, state: state)' '                        negative: String(negative.prefix(maxChars)), task: task, state: state.map { _ in 0 })'
run "пачки оценки берут строки банка с начала" \
    "$EM" 'MLXArray($0[i ..< end].map { Int32($0) })' 'MLXArray($0[0 ..< end - i].map { Int32($0) })'
run "банк свёрнутых инструкций в обратном порядке" \
    "$IC" '        let ss = prefixes.map { lookup($0) }' '        let ss = prefixes.reversed().map { lookup($0) }'
