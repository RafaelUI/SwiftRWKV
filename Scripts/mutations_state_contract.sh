#!/bin/bash
# Мутационная проверка (17.10): отпечаток подключённого начального состояния в
# контрактах кэшей реранкера (индекс выдачи, кэш состояний пар, голова).
#
#   ./Scripts/mutations_state_contract.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
IS=Sources/RWKVGen/InitialState.swift
RI=Sources/RWKVRerank/RerankInference.swift
SC=Sources/RWKVRerank/StateCache.swift
RE=Sources/RWKVRerank/RerankEncode.swift
RT=Sources/RWKVRerank/RerankTraining.swift

FILTER='InitialStateContractTests'
run() {
    local desc="$1"; shift
    printf '%-66s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── отпечаток ──"
run "в отпечатке нет сдвига cmix" \
    "$IS" '("tmix_shift", s.tmixShift), ("cmix_shift", s.cmixShift)]' '("tmix_shift", s.tmixShift)]'
run "в отпечатке нет формы" \
    "$IS" '+ Data((shape.map(String.init).joined(separator: ",") + "\n").utf8))' '+ Data("\n".utf8))'
run "нулевое состояние не равно отсутствию" \
    "$IS" 'if values.allSatisfy({ $0.allSatisfy { $0 == 0 } }) { return noInitialState }' 'if false { return noInitialState }'
# Прежняя крышка «без .asType(.float32)» равносильна: asArray(Float.self)
# приводит тип сам (17.10, не поймана при зелёных тестах). Осмысленный дефект —
# канон через fp16: отпечаток перестаёт различать состояния ниже шага fp16.
run "отпечаток через решётку fp16" \
    "$IS" '.map { ($0.0, $0.1.asType(.float32).reshaped([-1]), $0.1.shape) }' '.map { ($0.0, $0.1.asType(.float16).reshaped([-1]), $0.1.shape) }'
run "модель не видит подключённое состояние" \
    "$IS" 'var initialStateFingerprint: String { StateFile.stateFingerprint(initialState) }' 'var initialStateFingerprint: String { StateFile.stateFingerprint(nil) }'

echo "── индекс выдачи ──"
run "состояние не входит в ключи префикса" \
    "$RI" '"max_doc_tokens",
                                    StateFile.contractKey]' '"max_doc_tokens"]'
run "контракт префикса без отпечатка модели" \
    "$RI" 'c[StateFile.contractKey] = model.base.initialStateFingerprint' '_ = c'
run "индекс без ключа не читается как none" \
    "$RI" '? (contract[key] ?? StateFile.noInitialState) : contract[key]' '? contract[key] : contract[key]'
run "ожидание без ключа не читается как none" \
    "$RI" 'let want = now[key] ?? (key == stateKey ? StateFile.noInitialState : "")' 'let want = now[key] ?? ""'
run "потоковый индекс без отпечатка" \
    "$RI" 'var md = prefixContract(instruct: ins)' 'var md = config.prefixContract'

echo "── кэш пар и голова ──"
run "кэш пар не пишет отпечаток" \
    "$RE" 'contract[StateFile.contractKey] = model.base.initialStateFingerprint' '_ = contract'
run "сверка кэша по сырому контракту" \
    "$SC" 'let stored = Self.withStateKey(index.contract)' 'let stored = index.contract'
run "слияние: свой контракт без нормировки" \
    "$SC" 'let mineC = Self.withStateKey(index.contract)' 'let mineC = index.contract'
run "слияние: чужой контракт без нормировки" \
    "$SC" 'let theirsC = Self.withStateKey(other.index.contract)' 'let theirsC = other.index.contract'
run "нормировка не подставляет none" \
    "$SC" 'if c[StateFile.contractKey] == nil { c[StateFile.contractKey] = StateFile.noInitialState }' 'if false { c[StateFile.contractKey] = StateFile.noInitialState }'
run "голова: отпечаток из метаданных не берётся" \
    "$RT" '"terminator", "instruct", StateFile.contractKey].contains($0.key)' '"terminator", "instruct"].contains($0.key)'
run "голова без ключа не читается как none" \
    "$RT" '        if want[StateFile.contractKey] == nil {' '        if false {'
