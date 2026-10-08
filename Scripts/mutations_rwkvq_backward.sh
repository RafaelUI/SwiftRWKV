#!/bin/bash
# Мутационная проверка обучения над квантованной базой из .rwkvq: адаптеры,
# обратный проход через базу (свой VJP при базе fp16), слияние.
#
#   ./Scripts/mutations_rwkvq_backward.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
BB=Sources/RWKVGen/X070Backbone.swift
LR=Sources/RWKVGen/LoRA.swift

FILTER="${1:-RwkvqBackwardTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── обратный проход ──"
run "обратный проход в типе базы (fp16 насквозь)" \
    "$BB" '[matmul(cotangents[0].asType(bwd), weight(bwd)).asType(primals[0].dtype)]' \
          '[matmul(cotangents[0].asType(base), weight(base)).asType(primals[0].dtype)]'
run "вес обратного прохода округлён в типе базы" \
    "$BB" '[matmul(cotangents[0].asType(bwd), weight(bwd)).asType(primals[0].dtype)]' \
          '[matmul(cotangents[0].asType(bwd), weight(base).asType(bwd)).asType(primals[0].dtype)]'
run "котангент не приводится к типу обратного прохода" \
    "$BB" '[matmul(cotangents[0].asType(bwd), weight(bwd)).asType(primals[0].dtype)]' \
          '[matmul(cotangents[0].asType(base).asType(bwd), weight(bwd)).asType(primals[0].dtype)]'
run "свой VJP заводится и при nil" \
    "$BB" 'guard base == .float16, let bwd = rwkvqBackwardDType else {' \
          'guard base == .float16, let bwd = rwkvqBackwardDType ?? DType.bfloat16 as DType? else {'
run "свой VJP не заводится вовсе" \
    "$BB" 'guard base == .float16, let bwd = rwkvqBackwardDType else {' \
          'guard base == .float64, let bwd = rwkvqBackwardDType else {'
run "прямой проход внутри VJP-обёртки в типе обратного" \
    "$BB" '[matmul(ins[0].asType(base), weight(base).transposed()).asType(ins[0].dtype)]' \
          '[matmul(ins[0].asType(bwd), weight(bwd).transposed()).asType(ins[0].dtype)]'

echo
echo "── адаптеры и слияние ──"
run "адаптеры ищут форму только среди плотных весов" \
    "$LR" 'guard let (outF, inF) = bb.projectionShape(wKey) else { return }' \
          'guard let W = bb.w[wKey] else { return }; let (outF, inF) = (W.shape[0], W.shape[1])'
run "адаптеры fp16-модели заводятся в fp16" \
    "$LR" 'if baseDType == .float16 { baseDType = .bfloat16 }' 'if baseDType == .float64 { baseDType = .bfloat16 }'
run "слияние не снимает квантованную подложку" \
    "$LR" 'if fromRwkvq { bb.releaseRwkvqBacking(wKey) }' 'if fromRwkvq, wKey.isEmpty { bb.releaseRwkvqBacking(wKey) }'
run "слияние пропускает квантованные цели" \
    "$LR" '} else if let dq = bb.denseWeight(wKey) {' '} else if wKey.isEmpty, let dq = bb.denseWeight(wKey) {'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
