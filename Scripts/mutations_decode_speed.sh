#!/bin/bash
# Мутационная проверка правок 10.10: собранный шаг и префилл (compile),
# фьюз r/k/v, перекладка в родной контейнер полосами, тип входа родного ядра
# на генерации, типы входов обратного ядра WKV.
#
#   ./Scripts/mutations_decode_speed.sh [фильтр тестов]
#
# Общее у этих правок: их поломка не падает. Устаревший собранный граф
# считает прежнюю модель, перепутанная строка фьюза — правдоподобные числа,
# выбитый конвейер ядра — NaN через раз.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
BB=Sources/RWKVGen/X070Backbone.swift
GN=Sources/RWKVGen/X070Generation.swift
GV=Sources/RWKVQuant/RwkvqSymGEMV.swift
NA=Sources/RWKVQuant/RwkvqNative.swift
WK=Sources/RWKVKernel/WKV7Train.swift

FILTER="${1:-CompiledStepTests|RwkvqSymGEMVTests|RwkvqModelTests|WKV7KernelHazardTests}"
run() {
    local desc="$1"; shift
    printf '%-60s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── собранный шаг и префилл ──"
run "запись в веса не меняет версию шага" \
    "$BB" '    var w: [String: MLXArray] { didSet { stepVersion &+= 1 } }' '    var w: [String: MLXArray]'
run "собранный шаг не сверяет версию" \
    "$GN" 'if compiledStepFn == nil || compiledStepVersion != stepVersion {' 'if compiledStepFn == nil {'
run "собранный префилл не сверяет версию" \
    "$GN" 'if compiledPrefillFn == nil || compiledPrefillVersion != stepVersion' 'if compiledPrefillFn == nil'
run "кеш длин префилла без потолка" \
    "$GN" '&& compiledPrefillLengths.count >= compiledPrefillMaxLengths) {' '&& compiledPrefillLengths.count >= Int.max) {'
run "короткий промпт уходит в собранный префилл" \
    "$GN" 'if useCompiledPrefill, ids.count >= compiledPrefillMinTokens,' 'if useCompiledPrefill, ids.count >= 1,'
run "bf16-модель признаётся собираемой" \
    "$BB" 'for (k, v) in w where v.dtype == .bfloat16 {' 'for (k, v) in w where v.dtype == .float64 {'
run "модель с адаптерами признаётся собираемой" \
    "$BB" '            && !loraA.values.contains { $0.dtype == .bfloat16 }
            && !loraB.values.contains { $0.dtype == .bfloat16 }' '            && true'

echo
echo "── фьюз r/k/v ──"
run "все строки фьюза читают первый вход" \
    "$GV" 'uint xi   = row0 / OUT_PER;          // при OUT_PER=OUT_C всегда 0' 'uint xi   = 0;'
run "срезы строк группы смотрят в первую матрицу" \
    "$GV" 'let rows = (i * OUT) ..< ((i + 1) * OUT)' 'let rows = 0 ..< OUT'
run "адаптер k считается от входа r" \
    "$BB" 'return (finish(0, xr), finish(1, xk), finish(2, xv))' 'return (finish(0, xr), finish(1, xr), finish(2, xv))'
run "фьюз не выключается флагом" \
    "$BB" 'guard rwkvqSymFuseRKV, rwkvqSymDecodeKernel, rwkvqBaseDType != nil,' 'guard rwkvqSymDecodeKernel, rwkvqBaseDType != nil,'

echo
echo "── перекладка полосами и тип входа родного ядра ──"
run "каждая полоса читает первые строки" \
    "$NA" 'let blk = qblk[rows].reshaped([R, NB, 16 + 4 * xbits])' 'let blk = qblk[0 ..< R].reshaped([R, NB, 16 + 4 * xbits])'
run "на генерации родное ядро получает fp32" \
    "$BB" 'let half = rwkvqBaseDType == nil || (decode && wKey != "head.weight")' 'let half = rwkvqBaseDType == nil'
run "голова на генерации считается в fp16" \
    "$BB" 'let half = rwkvqBaseDType == nil || (decode && wKey != "head.weight")' 'let half = rwkvqBaseDType == nil || decode'

echo
echo "── обратное ядро WKV ──"
run "примелы идут в обратное ядро в своих типах" \
    "$WK" '[r,w,k,v,a,b].map { $0.asType(.float32) }
                    + [hCkpts,saFwd,dOut.asType(.float32),dHOut],' '[r,w,k,v,a,b,hCkpts,saFwd,dOut.asType(.float32),dHOut],'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
echo "Память threadgroup обратного ядра (предел Metal 32768 байт) мутацией не"
echo "проверяется: превышение ловит только валидация —"
echo "    MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 swift test --filter WKV7KernelHazardTests"
