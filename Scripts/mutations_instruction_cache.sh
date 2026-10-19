#!/bin/bash
# Мутационная проверка (19.10): кэш свёрнутых инструкций эмбеддера.
#   ./Scripts/mutations_instruction_cache.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
IC=Sources/RWKVEmbedding/InstructionCache.swift
IS=Sources/RWKVGen/InitialState.swift
EM=Sources/RWKVEmbedding/EmbeddingMetrics.swift
EF=Sources/RWKVEmbedding/EmbeddingFinetune.swift
EC=Sources/RWKVEmbedding/EmbeddingCurriculum.swift
FILTER='InstructionCacheTests'
run() { local d="$1"; shift; printf '%-52s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "разрез после «Query: », а не после «\\n»" \
    "$IC" '        let cut = anchor.index(after: r.lowerBound)' '        let cut = r.upperBound'
run "состояние не восстанавливается" \
    "$IS" '        defer { initialLayers = saved }' '        defer { _ = saved }'
run "восстановление копией через initialState" \
    "$IS" $'        let saved = initialLayers\n        defer { initialLayers = saved }' $'        let saved = initialState\n        defer { initialState = saved }'
run "нет сброса по отпечатку подключённого" \
    "$IC" '        if over != foldedOver { folded.removeAll(); foldedOver = over }' '        if over != foldedOver { foldedOver = over }'
run "строки группы в обратном порядке" \
    "$IC" '{ for (k, i) in idx.enumerated() { rows[i] = v[k] } }' '{ for (k, i) in idx.enumerated() { rows[i] = v[idx.count - 1 - k] } }'
run "свёртка с нуля, а не с подключённого" \
    "$IC" '        var s: RWKVBatchState? = nil                           // первый кусок — с подключённого' '        var s: RWKVBatchState? = bb.zeroInitialState()'
run "в хвосте остаётся инструкция" \
    "$IC" '            tails[i] = t' '            tails[i] = a'
run "без мемоизации" \
    "$IC" '        if let s = folded[prefix] { return s }' ''
run "пулинг .mean не отвергается" \
    "$IC" '        guard model.pooling == .last else { throw InstructionCacheError.meanPooling }' '        guard model.pooling == .last || model.pooling == .mean else { throw InstructionCacheError.meanPooling }'
run "оценка не смотрит на флаг" \
    "$EM" '        if foldInstructions {' '        if false {'
run "стадия не передаёт флаг" \
    "$EF" '                foldInstructions: stage.foldInstructions))' '                foldInstructions: false))'
run "флаг не входит в отпечаток плана" \
    "$EC" '                + (st.foldInstructions ? "|fold" : ""))' '                + "")'
run "кусок свёртки стартует с подключённого, а не с предыдущего" \
    "$IC" '            s = bb.foldPrompt(Array(ids[i ..< Swift.min(i + step, ids.count)]), from: s)' '            s = bb.foldPrompt(Array(ids[i ..< Swift.min(i + step, ids.count)]), from: nil)'
run "последний неполный кусок теряется" \
    "$IC" '        while i < ids.count {' '        while i + step <= ids.count || s == nil {'
