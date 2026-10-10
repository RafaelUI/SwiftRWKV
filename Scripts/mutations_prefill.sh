#!/bin/bash
# Мутационная проверка префилла (12.10): замороженный WKV одним запуском на
# всю длину.
#
#   ./Scripts/mutations_prefill.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
WK=Sources/RWKVKernel/WKV7.swift

FILTER="${1:-WKV7SequenceTests}"
run() {
    local desc="$1"; shift
    printf '%-60s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── WKV одним запуском ──"
run "sa считается по b вместо a" \
    "$WK" 'for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];' \
          'for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*b_sh[dk];'
run "состояние обновляется через a вместо b" \
    "$WK" 'h_row[dk] = w_sh[dk]*h_row[dk] + v_dv*k_sh[dk] + sa*b_sh[dk];' \
          'h_row[dk] = w_sh[dk]*h_row[dk] + v_dv*k_sh[dk] + sa*a_sh[dk];'
run "последняя позиция не считается" \
    "$WK" 'uint T   = tlen[0];' 'uint T   = tlen[0] - 1;'
run "длина — не из входа: шаг по батчу чужой" \
    "$WK" 'uint base = ((bi*T+t)*H_C+hi)*HEAD_SIZE_C;' 'uint base = ((bi*(T+1)+t)*H_C+hi)*HEAD_SIZE_C;'
run "все головы читают голову 0" \
    "$WK" 'uint base = ((bi*T+t)*H_C+hi)*HEAD_SIZE_C;' 'uint base = ((bi*T+t)*H_C)*HEAD_SIZE_C;'
run "начальное состояние не доходит до ядра" \
    "$WK" 'let inputs = [r, w, k, v, a, b, h].map { $0.asType(.float32) } + [MLXArray([UInt32(T)])]' \
          'let inputs = [r, w, k, v, a, b, h * 0].map { $0.asType(.float32) } + [MLXArray([UInt32(T)])]'
run "выключатель не читается (всегда одним запуском)" \
    "$WK" 'if wkv7WholeSequenceForward { return wkv7SequenceForward(r, w, k, v, a, b, hIn) }' \
          'if true { return wkv7SequenceForward(r, w, k, v, a, b, hIn) }'
run "одним запуском не идёт никогда" \
    "$WK" 'if wkv7WholeSequenceForward { return wkv7SequenceForward(r, w, k, v, a, b, hIn) }' \
          'if false { return wkv7SequenceForward(r, w, k, v, a, b, hIn) }'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
