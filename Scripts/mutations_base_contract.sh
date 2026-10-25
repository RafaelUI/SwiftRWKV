#!/bin/bash
# Мутационная проверка (25.10): отпечаток БАЗЫ в контрактах кэшей реранкера
# (кэш пар, индекс выдачи) и в чекпоинтах голов реранкера и эмбеддера и
# LoRA-адаптеров.
#
#   ./Scripts/mutations_base_contract.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
IS=Sources/RWKVGen/InitialState.swift
LO=Sources/RWKVGen/LoRA.swift
RE=Sources/RWKVRerank/RerankEncode.swift
RI=Sources/RWKVRerank/RerankInference.swift
RT=Sources/RWKVRerank/RerankTraining.swift
RR=Sources/RWKVRerank/Reranker.swift
EC=Sources/RWKVEmbedding/EmbeddingCheckpoint.swift
EM=Sources/RWKVEmbedding/Embedder.swift

FILTER='BaseContractTests'
run() {
    local desc="$1"; shift
    printf '%-66s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── отпечаток модели ──"
run "запомненный отпечаток не пересчитывается после записи в веса" \
    "$IS" 'if let m = baseFingerprintMemo, m.version == stepVersion { return m.value }' 'if let m = baseFingerprintMemo { return m.value }'
run "база QLoRA: строка не из кодов" \
    "$IS" 'if let q = bb.quant[name + ".weight"] {' 'if false, let q = bb.quant[name + ".weight"] {'

echo "── кэш пар и индекс ──"
run "кэш пар не пишет базу" \
    "$RE" 'contract[StateFile.baseContractKey] = model.base.baseFingerprint' '_ = contract'
run "база не входит в ключи префикса" \
    "$RI" 'StateFile.contractKey, StateFile.baseContractKey]' 'StateFile.contractKey]'
run "контракт префикса без базы модели" \
    "$RI" 'c[StateFile.baseContractKey] = model.base.baseFingerprint' '_ = c'
run "ожидание берёт базу из самого кэша" \
    "$RT" 'if contract?[StateFile.baseContractKey] == nil {' 'if contract?[StateFile.baseContractKey] == nil && c[StateFile.baseContractKey] == nil {'
run "ожидание без базы модели" \
    "$RT" 'c[StateFile.baseContractKey] = model.base.baseFingerprint' '_ = c'

echo "── голова реранкера ──"
run "голова не пишет базу" \
    "$RR" 'StateFile.baseContractKey: base.baseFingerprint,' ''
run "голова: база не сверяется" \
    "$RR" 'if inFile != here { throw RerankerError.otherBase(file: inFile, model: here) }' 'if false { throw RerankerError.otherBase(file: inFile, model: here) }'
run "голова: allowOtherBase не действует" \
    "$RR" 'if strict, !allowOtherBase, let inFile = md[StateFile.baseContractKey] {' 'if strict, let inFile = md[StateFile.baseContractKey] {'
run "голова: файл без ключа читается как чужая база" \
    "$RR" 'if strict, !allowOtherBase, let inFile = md[StateFile.baseContractKey] {' 'if strict, !allowOtherBase, let inFile = md[StateFile.baseContractKey] ?? Optional("none") {'
run "fromHead не передаёт allowOtherBase" \
    "$RR" 'try model.loadHead(from: url, allowOtherBase: allowOtherBase)' 'try model.loadHead(from: url)'

echo "── голова эмбеддера ──"
run "эмбеддер не пишет базу" \
    "$EC" 'md[StateFile.baseContractKey] = backbone.baseFingerprint' '_ = md'
run "эмбеддер: база не сверяется" \
    "$EC" '                if inFile != here {' '                if false {'
run "эмбеддер: allowOtherBase не действует" \
    "$EC" 'if !allowOtherBase, let inFile = md[StateFile.baseContractKey] {' 'if let inFile = md[StateFile.baseContractKey] {'
run "эмбеддер: fromHead не передаёт allowOtherBase" \
    "$EC" 'try model.loadHead(from: url, allowOtherBase: allowOtherBase)' 'try model.loadHead(from: url)'
run "Embedder.fromCheckpoint не передаёт allowOtherBase" \
    "$EM" '                                                            allowOtherBase: allowOtherBase)' '                                                            allowOtherBase: false)'

echo "── адаптеры ──"
run "адаптеры не пишут базу" \
    "$LO" 'try MLX.save(arrays: st, metadata: [StateFile.baseContractKey: bb.baseFingerprint], url: url)' 'try MLX.save(arrays: st, url: url)'
run "адаптеры: база не сверяется" \
    "$LO" 'if inFile != here { throw LoadError.otherBase(file: inFile, model: here) }' 'if false { throw LoadError.otherBase(file: inFile, model: here) }'
run "адаптеры: allowOtherBase не действует" \
    "$LO" 'if !allowOtherBase, let inFile = md[StateFile.baseContractKey] {' 'if let inFile = md[StateFile.baseContractKey] {'
