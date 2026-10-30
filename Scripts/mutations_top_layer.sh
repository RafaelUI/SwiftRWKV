#!/bin/bash
# Мутационная проверка (30.10): TopLayerFinetune — верхние слои языковым лоссом
# (сильный контроль к обучению состояния в gen-run --distill --train top).
#
#   ./Scripts/mutations_top_layer.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
F=Sources/RWKVGen/TopLayerFinetune.swift
FILTER='TopLayerFinetuneTests'
run() {
    local desc="$1"; shift
    printf '%-56s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── что учится ──"
run "from не учитывается (учатся все слои)" \
    "$F" 'let keys = BackboneWeightsTrainableSet.topLayerKeys(bb, from: from)' 'let keys = BackboneWeightsTrainableSet.topLayerKeys(bb, from: 0)'
run "через WKV-ядро обучения идут не те слои" \
    "$F" 'bb.trainLayers = Set(from ..< bb.cfg.nLayer)' 'bb.trainLayers = []'
run "trainLayers не возвращается" \
    "$F" 'defer { bb.trainLayers = savedLayers; bb.useBlockCheckpoint = savedCheckpoint }' 'defer { bb.useBlockCheckpoint = savedCheckpoint }'
run "лосс без маски цели" \
    "$F" 'objective: { LoRAFinetune.languageModelLoss(bb, $0, chunk: cfg.lossChunkTokens) },' 'objective: { LoRAFinetune.languageModelLoss(bb, (x: $0.x, y: MLX.maximum($0.y, MLXArray(Int32(0)))), chunk: cfg.lossChunkTokens) },'
echo "── файл ──"
run "save пишет отпечаток после обучения" \
    "$F" 'StateFile.baseContractKey: result.baseFingerprint,' 'StateFile.baseContractKey: bb.baseFingerprint,'
run "load не сверяет базу" \
    "$F" 'if inFile != here { throw Error.otherBase(file: inFile, model: here) }' '_ = (inFile, here)'
run "load не сверяет форму" \
    "$F" 'guard old.shape == v.shape else { throw Error.shape(key: k, file: v.shape, model: old.shape) }' '_ = old'
run "load не проверяет формат файла" \
    "$F" 'guard md[formatKey] != nil else { throw Error.notTopLayers(url.lastPathComponent) }' '_ = md'
run "load подставляет до проверки всех ключей" \
    "$F" '        for (k, v) in w {
            guard let old = bb.w[k] else { throw Error.unknownKey(k) }' '        for (k, v) in w {
            if let o = bb.w[k], o.shape == v.shape { bb.w[k] = v.asType(o.dtype) }
            guard let old = bb.w[k] else { throw Error.unknownKey(k) }'
