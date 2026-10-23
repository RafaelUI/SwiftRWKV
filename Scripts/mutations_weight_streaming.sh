#!/bin/bash
# Мутационная проверка (23.10): подкачка весов с диска при обратном проходе по блокам.
#   ./Scripts/mutations_weight_streaming.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
WS=Sources/RWKVGen/WeightStreaming.swift
SB=Sources/RWKVGen/Training/SequentialBackward.swift
FILTER='WeightStreaming'
run() { local d="$1"; shift; printf '%-58s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "прямой проход не отпускает блоки" \
    "$SB" '                for j in unreleased { bb.releaseStreamed(layer: j) }' '                _ = unreleased'
run "обратный проход не отпускает блок" \
    "$SB" $'            settle(ready)\n            bb.releaseStreamed(layer: l)' $'            settle(ready)'
run "пересчёт с записью через блок не отпускает" \
    "$SB" $'                eval(xo)\n                bb.releaseStreamed(layer: j)' $'                eval(xo)'
run "голова не отпускается после лосса" \
    "$SB" $'        eval(dx, loss)\n        bb.releaseStreamedHead()' $'        eval(dx, loss)'
run "перечитанный вес без приведения к типу вычислений" \
    "$WS" '            out[k] = v.asType(dtype)' '            out[k] = v'
run "отпускается соседний блок" \
    "$WS" '        replace(s.fresh(s.byLayer[layer]))' '        replace(s.fresh(s.byLayer[(layer + 1) % s.byLayer.count]))'
run "подкачка не включается (поток не задан)" \
    "$WS" '        bb.weightStream = WeightStream(url: url, dtype: computeDType, byLayer: byLayer.map { $0.sorted() })' '        _ = byLayer'
run "не-модель принимается" \
    "$WS" '        else { throw WeightStreamingError.notAModel(url.lastPathComponent) }' '        else { return X070Backbone(weights: w, cfg: X070Config(nLayer: 1, nEmbd: 64, vocab: 64)) }'
