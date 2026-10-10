#!/bin/bash
# Мутационная проверка (12.10): реранкер и эмбеддер на базе из .rwkvq,
# оценка и выдача прямым путём при обучаемой базе (withoutGradientPath),
# отказ от .full / .topLayers на квантованной базе.
#
#   ./Scripts/mutations_downstream.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
BB=Sources/RWKVGen/X070Backbone.swift
EM=Sources/RWKVEmbedding/EmbeddingMetrics.swift
ED=Sources/RWKVEmbedding/Embedder.swift
ET=Sources/RWKVEmbedding/EmbeddingTrainable.swift

FILTER="${1:-RwkvqDownstreamTests|EmbeddingFinetuneTests}"
run() {
    local desc="$1"; shift
    printf '%-60s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── прямой путь при обучаемой базе ──"
run "обёртка не снимает обучаемые слои" \
    "$BB" '        let saved = trainLayers
        trainLayers = []' \
          '        let saved = trainLayers'
run "обёртка не возвращает обучаемые слои" \
    "$BB" 'defer { trainLayers = saved }' 'defer { _ = saved }'
run "оценка идёт дифференцируемым ядром" \
    "$EM" 'let v = model.backbone.withoutGradientPath { model.embed(idx, poolIndex: pool) }' \
          'let v = model.embed(idx, poolIndex: pool)'
run "выдача идёт дифференцируемым ядром" \
    "$ED" 'return model.backbone.withoutGradientPath { model.embed(idx, poolIndex: poolIdx)[0] }' \
          'return model.embed(idx, poolIndex: poolIdx)[0]'

echo "── режимы обучения на квантованной базе ──"
run "отказ срабатывает на плотной базе, а не на квантованной" \
    "$ET" 'guard !model.backbone.rwkvqBackedKeys.isEmpty else { return nil }' \
          'guard model.backbone.rwkvqBackedKeys.isEmpty else { return nil }'
run ".full на квантованной базе разрешён" \
    "$ET" 'case .frozen, .lora: return nil' 'case .frozen, .lora, .full: return nil'
run ".topLayers на квантованной базе разрешён" \
    "$ET" 'case .frozen, .lora: return nil' 'case .frozen, .lora, .topLayers: return nil'
run "режим .lora оставляет базу на прямом ядре" \
    "$ET" '            bb.trainLayers = Set(0 ..< bb.cfg.nLayer)
            return CompositeTrainableSet([LoRATrainableSet(bb), headSet])' \
          '            bb.trainLayers = []
            return CompositeTrainableSet([LoRATrainableSet(bb), headSet])'
# (тип адаптеров на fp16-базе стережёт mutations_rwkvq_backward.sh)

echo "── счёт квантованной базы под реранкером и эмбеддером ──"
run "замороженный cmix без квадрата" \
    "$BB" 'return matmul(s * s, weights[1]().transposed()).asType(xk.dtype)' \
          'return matmul(s, weights[1]().transposed()).asType(xk.dtype)'
run "замороженный cmix: relu отрезает сверху" \
    "$BB" 'let s = maximum(h, MLXArray(Float(0)).asType(.float16))' \
          'let s = minimum(h, MLXArray(Float(0)).asType(.float16))'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
