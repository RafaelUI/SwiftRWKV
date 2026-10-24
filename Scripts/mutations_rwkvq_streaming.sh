#!/bin/bash
# Мутационная проверка (24.10): подкачка весов собранной модели (.rwkvq) — enableWeightStreaming(cache:).
#   ./Scripts/mutations_rwkvq_streaming.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
RS=Sources/RWKVQuant/RwkvqSpill.swift
WS=Sources/RWKVGen/WeightStreaming.swift
FILTER='RwkvqWeightStreaming'
run() { local d="$1"; shift; printf '%-58s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "срезы сведённой группы не пересобираются" \
    "$RS" $'                for (i, key) in g.keys.enumerated() {\n                    a["\\(key)::\\(part)"] = v[(i * g.outPer) ..< ((i + 1) * g.outPer)]\n                }' ''
run "буфер группы (qs) не заменяется" \
    "$RS" 'qs: part == "qs" ? v : g.qs' 'qs: g.qs'
run "срезы выгружаются вместе с группой" \
    "$RS" '        var out = arrays.filter { !sliced.contains($0.key) }' '        var out = arrays; _ = sliced'
run "таблица эмбеддингов не заменяется" \
    "$WS" '        if !emb.isEmpty, let t = rwkvqEmbTable { rwkvqEmbTable = t.replacingSpilled(emb) }' '        _ = emb'
run "буферы базы не заменяются" \
    "$WS" '        if !sc.isEmpty, let s = rwkvqSidecar { rwkvqSidecar = s.replacingSpilled(sc) }' '        _ = sc'
run "слой определяется со сдвигом" \
    "$WS" '                byLayer[l].append(file)' '                byLayer[(l + 1) % cfg.nLayer].append(file)'
run "сведённая группа не относится к блоку" \
    "$WS" 'let owner = k.hasPrefix("fused::") ? k.components(separatedBy: "::")[1] : k' 'let owner = k'
run "после выгрузки прочитанное не отпускается" \
    "$WS" $'        arrays = [:]\n        releaseAllStreamedWeights()' $'        arrays = [:]'
run "родной контейнер не отвергается" \
    "$WS" '        guard rwkvqNative.isEmpty else { throw WeightStreamingError.nativeKernel }' ''
run "повторное включение не отвергается" \
    "$WS" '        guard weightStream == nil else { throw WeightStreamingError.alreadyStreaming }' ''
run "приведённая копия плотной цели не сбрасывается" \
    "$WS" '            for (k, v) in nw { all[k] = v; denseTargetCast[k] = nil }' '            for (k, v) in nw { all[k] = v }'
