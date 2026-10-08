#!/bin/bash
# Мутационная проверка сборки модели из .rwkvq: проводка, плотная цель,
# ориентация low-rank веток и ТИПЫ (норма fp16).
#
#   ./Scripts/mutations_rwkvq_model.sh [фильтр тестов]
#
# Подмена типа — дефект, который не падает: модель считает, логиты
# правдоподобны, просто KL к исходной на несколько процентов хуже. Ловится
# только сквозным эталоном, и мутации проверяют, что эталон это ловит.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
BB=Sources/RWKVGen/X070Backbone.swift
RB=Sources/RWKVGen/RwkvqBase.swift
LD=Sources/RWKVGen/RwkvqStandaloneLoad.swift
FC=Sources/RWKVQuant/RwkvqFullConvert.swift
SC=Sources/RWKVQuant/RwkvqSidecar.swift
NA=Sources/RWKVQuant/RwkvqNative.swift

FILTER="${1:-RwkvqModelTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── проводка ──"
run "k_proj читает чужой тензор (att.value)" \
    "$SC" '"r_proj": "receptance", "k_proj": "key",' '"r_proj": "receptance", "k_proj": "value",'
run "low-rank ветки поворачиваются всегда (таблица имён)" \
    "$FC" 'let v = info.transposed ? raw.transposed() : raw
            guard v.ndim == 2' \
          'let v = raw.transposed()
            guard v.ndim == 2'
run "плотная цель не подключается" \
    "$RB" 'if info.kind == .dense {' 'if info.kind == .dense, key.isEmpty {'
run "плотная цель берёт вес соседней проекции" \
    "$RB" 'let raw = try? sidecar.dequantize(world, dtype: .bfloat16) else {' \
          'let raw = try? sidecar.dequantize(world.replacingOccurrences(of: "blocks.0.att.output", with: "blocks.1.att.output"), dtype: .bfloat16) else {'
run "удерживается вся база, а не нужные ключи" \
    "$RB" 'rwkvqSidecar = fullyNative ? nil : sidecar.pruned(keeping: needed)' \
          'rwkvqSidecar = fullyNative ? nil : sidecar'
run "цель вне файла не проверяется до сборки" \
    "$LD" 'if let first {' 'if let first, first.isEmpty {'

echo
echo "── типы ──"
run "плотные параметры по умолчанию bf16" \
    "$LD" 'paramDType: DType = .float16,' 'paramDType: DType = .bfloat16,'
run "emb деквантуется в bf16 (и лишь потом приводится)" \
    "$LD" 'try sidecar.dequantize("emb.weight", dtype: computeDType)' \
          'try sidecar.dequantize("emb.weight", dtype: .bfloat16)'
run "ветки и нормы деквантуются в bf16" \
    "$FC" 'try dequantize(worldKey, dtype: computeDType)' 'try dequantize(worldKey, dtype: .bfloat16)'
run "база по умолчанию — прежний путь (без приведения)" \
    "$LD" 'baseDType: DType? = .float16,
                            useNativeKernel: Bool = true) throws -> X070Backbone {
        try load(fromSidecar: try RwkvqSidecar.open(path)' \
          'baseDType: DType? = nil,
                            useNativeKernel: Bool = true) throws -> X070Backbone {
        try load(fromSidecar: try RwkvqSidecar.open(path)'
run "деквант базы округляется в bf16" \
    "$BB" 'if let dense = rwkvqWeight(wKey, base) {' 'if let dense = rwkvqWeight(wKey, .bfloat16) {'
run "родное ядро получает вход в fp16" \
    "$BB" 'let inType: DType = rwkvqBaseDType == nil ? .float16 : .float32' \
          'let inType: DType = .float16'
run "нормировка fp16 идёт ручной формулой" \
    "$BB" 'if x.dtype == .float16 { return MLX.layerNorm' 'if x.dtype == .float64 { return MLX.layerNorm'
run "зажим scale обнуляется всегда (fp16)" \
    "$NA" 'if exactScales, !arrayEqual' 'if false, !arrayEqual'
run "зажим scale: тип решает не сверка декванта" \
    "$NA" 'useHalf = arrayEqual(a, b).item(Bool.self)' 'useHalf = false'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
