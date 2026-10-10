#!/bin/bash
# Мутационная проверка ядер декода 11.10: GEMV sb6, шаг WKV и хвост tmix
# одним запуском, и того, где они включаются.
#
#   ./Scripts/mutations_decode_kernels.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
SB=Sources/RWKVQuant/RwkvqSb6GEMV.swift
DK=Sources/RWKVKernel/WKV7DecodeKernels.swift
BB=Sources/RWKVGen/X070Backbone.swift
GN=Sources/RWKVGen/X070Generation.swift
LD=Sources/RWKVGen/RwkvqStandaloneLoad.swift

FILTER="${1:-RwkvqSb6GEMVTests|DecodeKernelTests|RwkvqModelTests}"
run() {
    local desc="$1"; shift
    printf '%-60s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── ядро GEMV sb6 ──"
run "qm прочитан беззнаковым" \
    "$SB" 'half  mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);' 'half  mn = (half)((float)sm.y * (float)dd.y);'
run "чужой суперблок у масштабов" \
    "$SB" 'half2  dd = dd2[(row0+j)*NSB + p/8];' 'half2  dd = dd2[(row0+j)*NSB + p/4];'
run "слагаемое min потеряно" \
    "$SB" 'acc[j] += (float)s * dv + (float)mn * xbs;' 'acc[j] += (float)s * dv;'
run "битплоскость бита 5 на месте бита 4" \
    "$SB" 'sb6Plane("hb2", shift: 5)' 'sb6Plane("hb2", shift: 4)'
run "xbsum не по блокам 32" \
    "$SB" 'x32.reshaped([1, IN / 32, 32]).sum(axis: 2)' 'x32.reshaped([1, IN / 32, 32]).max(axis: 2)'
run "порядок регистров: старшие нибблы на месте младших" \
    "$SB" '+ dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))' '+ dot(xb0, float4(l0.x, l0.y, l0.z, l0.w))'
run "таблица раскладок потоков не та" \
    "$SB" 'if OUT >= 8192 { return xbits == 0 ? (2, 2) : (4, 4) }' 'if OUT >= 8192 { return (4, 4) }'
run "ядро берёт больше одной строки входа" \
    "$SB" '&& superBlock == 8 && rows == 1' '&& superBlock == 8 && rows >= 1'

echo
echo "── шаг WKV и хвост tmix ──"
run "шаг: v берётся из нулевой колонки" \
    "$DK" 'float v_dv = v[base+dv];' 'float v_dv = v[base];'
run "шаг: состояние обновляется через a вместо b" \
    "$DK" 'h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*b[base+dk];' \
          'h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*a[base+dk];'
run "хвост: нормировка без вычитания среднего" \
    "$DK" 'float nrm = (wkv[base + i] - mean) * inv;' 'float nrm = wkv[base + i] * inv;'
run "хвост: bonus без умножения на v" \
    "$DK" 'float y   = nrm * lnw[hb + i] + lnb[hb + i] + dp * v[base + i];' \
          'float y   = nrm * lnw[hb + i] + lnb[hb + i] + dp;'
run "хвост: gate потерян" \
    "$DK" 'out[base + i] = y * g[base + i];' 'out[base + i] = y;'

echo
echo "── где включаются ──"
run "флаг ядра sb6 не читается" \
    "$BB" 'if rwkvqSb6DecodeKernel, let y = sc.sb6GEMV(world, x2d) {' 'if let y = sc.sb6GEMV(world, x2d) {'
run "ядра шага включаются у плотной модели" \
    "$GN" 'let fused = useStepKernels && rwkvqBaseDType != nil' 'let fused = useStepKernels'
run "ядра шага включаются при приведении выхода WKV" \
    "$GN" '            && !castWKVOutputToComputeDType && wkvTailApplies(headSize: S)' \
          '            && wkvTailApplies(headSize: S)'
run "умолчание загрузки — родной контейнер" \
    "$LD" 'useNativeKernel: Bool = false) throws -> X070Backbone {
        try load(fromSidecar: try RwkvqSidecar.open(path)' \
          'useNativeKernel: Bool = true) throws -> X070Backbone {
        try load(fromSidecar: try RwkvqSidecar.open(path)'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
echo
echo "ЗАМЕЧАНИЕ. Мутация «низкоранговые ветки шага считаются с повышением"
echo "матрицы до fp32» (narrow = false в tmixStep) здесь намеренно ОТСУТСТВУЕТ:"
echo "она возвращает счёт до правки, числа остаются в допусках (KL порядка"
echo "1e-6), меняется только время — 1.5B COMPRESSION 16.8 -> 13.7 мс/ток."
echo "Тесты скорость не меряют; её стережёт decode-bench --step-profile."
