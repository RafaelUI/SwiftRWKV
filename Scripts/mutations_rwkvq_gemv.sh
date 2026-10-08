#!/bin/bash
# Мутационная проверка GEMV-ядра sym (декод без плотной матрицы).
#
#   ./Scripts/mutations_rwkvq_gemv.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
GV=Sources/RWKVQuant/RwkvqSymGEMV.swift
BB=Sources/RWKVGen/X070Backbone.swift
GN=Sources/RWKVGen/X070Generation.swift
LR=Sources/RWKVGen/LoRA.swift

FILTER="${1:-RwkvqSymGEMVTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── ядро GEMV ──"
run "8 бит: чужой суперблок у масштаба" \
    "$GV" '* (float)d[(row0+j)*NSB + p/16]);' '* (float)d[(row0+j)*NSB + p/8]);'
run "8 бит: масштаб без half-роундтрипа" \
    "$GV" 'half s = (half)((float)sc[(row0+j)*NB + p]' 'float s = ((float)sc[(row0+j)*NB + p]'
run "6 бит: сдвиг +32 не снят" \
    "$GV" 'acc[n*RS + j] += (float)s0 * (dv0 - 32.0f * b0)' 'acc[n*RS + j] += (float)s0 * (dv0 - 31.0f * b0)'
run "6 бит: второй блок пары берёт масштаб первого" \
    "$GV" 'half s1 = (half)((float)sc[(row0+j)*NB + 2*p + 1] * dsb);' \
          'half s1 = (half)((float)sc[(row0+j)*NB + 2*p] * dsb);'
run "6 бит: порядок регистров пары" \
    "$GV" 'float4 f2 = float4(h0.x, h0.y, h0.z, h0.w);' 'float4 f2 = float4(l2.x, l2.y, l2.z, l2.w);'
run "6 бит: xbsum не по блокам 16" \
    "$GV" 'x32.reshaped([N, IN / 16, 16]).sum(axis: 2)' 'x32.reshaped([N, IN / 16, 16]).max(axis: 2)'
run "строки входа перепутаны (N > 1)" \
    "$GV" '(device const float4*)(x + (n*NK + xi)*IN_C);' '(device const float4*)(x + xi*IN_C);'
run "число строк не входит в ключ кэша ядер" \
    "$GV" 'SymGemvKey(inFeatures: IN, outFeatures: OUT, bits: bits, rows: NN)' \
          'SymGemvKey(inFeatures: IN, outFeatures: OUT, bits: bits, rows: 1)'
run "ядро берёт больше 4 строк" \
    "$GV" '&& rows >= 1 && rows <= rwkvqSymGEMVMaxRows' '&& rows >= 1'

echo
echo "── где ядро включается ──"
run "рекуррентный шаг идёт мимо ядра" \
    "$GN" 'lora: String(wKey.dropLast(7)), decode: true)' 'lora: String(wKey.dropLast(7)), decode: false)'
run "ядро включается и на параллельном проходе" \
    "$BB" 'if decode, rwkvqSymDecodeKernel, let world' 'if rwkvqSymDecodeKernel, let world'
run "флаг ядра не читается" \
    "$BB" 'if decode, rwkvqSymDecodeKernel, let world' 'if decode, let world'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
