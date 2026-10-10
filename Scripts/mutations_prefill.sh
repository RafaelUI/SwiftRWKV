#!/bin/bash
# Мутационная проверка префилла (12.10): замороженный WKV одним запуском на
# всю длину, слитый счёт замороженного прохода (хвост tmix, нормировки,
# cmix в типе базы) и короткий проход ядрами GEMV.
#
#   ./Scripts/mutations_prefill.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
WK=Sources/RWKVKernel/WKV7.swift
DK=Sources/RWKVKernel/WKV7DecodeKernels.swift
BK=Sources/RWKVGen/RWKVBlock.swift
BB=Sources/RWKVGen/X070Backbone.swift
LD=Sources/RWKVGen/RwkvqStandaloneLoad.swift

FILTER="${1:-WKV7SequenceTests|FrozenPassKernelTests|ShortPassKernelTests}"
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
echo "── слитый счёт замороженного прохода ──"
run "слитый счёт действует вне нормы" \
    "$BK" 'backbone.rwkvqBaseDType != nil ? backbone.frozenPassKernels : []' 'backbone.frozenPassKernels'
run "хвост ядром у обучаемого слоя" \
    "$BK" 'if case .forward = ctx.wkvPath(T), ctx.frozenKernels.contains(.tail), !ctx.castWKVOutput,' \
          'if ctx.frozenKernels.contains(.tail), !ctx.castWKVOutput,'
run "флаг хвоста не читается" \
    "$BK" 'if case .forward = ctx.wkvPath(T), ctx.frozenKernels.contains(.tail), !ctx.castWKVOutput,' \
          'if case .forward = ctx.wkvPath(T), !ctx.castWKVOutput,'
run "нормировки штатным ядром у обучаемого слоя" \
    "$BK" 'if case .forward = ctx.wkvPath(T) { return ctx.frozenKernels.contains(.layerNorm) }
    return false' 'return ctx.frozenKernels.contains(.layerNorm)'
run "флаг нормировок не читается" \
    "$BK" 'if case .forward = ctx.wkvPath(T) { return ctx.frozenKernels.contains(.layerNorm) }' \
          'if case .forward = ctx.wkvPath(T) { return true }'
run "cmix слитно у обучаемого слоя" \
    "$BK" 'if case .forward = ctx.wkvPath(x.shape[1]), ctx.frozenKernels.contains(.cmix),' \
          'if ctx.frozenKernels.contains(.cmix),'
run "флаг cmix не читается" \
    "$BK" 'if case .forward = ctx.wkvPath(x.shape[1]), ctx.frozenKernels.contains(.cmix),' \
          'if case .forward = ctx.wkvPath(x.shape[1]),'
run "cmix: квадрат потерян" \
    "$BB" 'return matmul(s * s, weights[1]().transposed()).asType(xk.dtype)' \
          'return matmul(s, weights[1]().transposed()).asType(xk.dtype)'
run "cmix: relu потерян" \
    "$BB" 'let s = maximum(h, MLXArray(Float(0)).asType(.float16))' 'let s = h'
run "cmix с адаптером идёт слитно" \
    "$BB" 'guard quant[wKey] == nil, rwkvqNative[wKey] == nil, wOverride?[wKey] == nil,
                  loraA[String(wKey.dropLast(7))] == nil else { return nil }' \
          'guard quant[wKey] == nil, rwkvqNative[wKey] == nil, wOverride?[wKey] == nil else { return nil }'
run "хвост на N строк: сетка на одну строку" \
    "$DK" 'grid: (N * H * 32, 1, 1),' 'grid: (H * 32, 1, 1),'

echo
echo "── короткий проход ядрами GEMV ──"
run "короткий проход идёт и у обучаемого слоя" \
    "$BK" '            && !backbone.trainLayers.contains(layer)
            && x.shape.dropLast().reduce(1, *) <= backbone.shortPassKernelRows' \
          '            && x.shape.dropLast().reduce(1, *) <= backbone.shortPassKernelRows'
run "порог не действует: ядрами любая длина" \
    "$BK" 'x.shape.dropLast().reduce(1, *) <= backbone.shortPassKernelRows' 'x.shape.dropLast().reduce(1, *) >= 0'
run "порог считается по длине, а не по строкам батча" \
    "$BK" 'x.shape.dropLast().reduce(1, *) <= backbone.shortPassKernelRows' \
          'x.shape[x.ndim - 2] <= backbone.shortPassKernelRows'
run "cmix на коротком проходе разворачивает матрицы" \
    "$BK" 'isShort(xk) ? nil : backbone.cmixFrozen(xk, layer: layer)' 'backbone.cmixFrozen(xk, layer: layer)'
run "все куски строк читают начало входа" \
    "$BB" 'guard let y = kernel(x2d[start ..< min(start + maxRows, n)]) else { return nil }' \
          'guard let y = kernel(x2d[0 ..< min(maxRows, n - start)]) else { return nil }'
run "признак короткого прохода не доходит до проекции" \
    "$BB" 'let base = baseProj(x, wKey, decode: decode, shortKernel: shortKernel)' \
          'let base = baseProj(x, wKey, decode: decode, shortKernel: false)'
run "порог при загрузке не ставится" \
    "$LD" 'bb.shortPassKernelRows = X070Backbone.defaultShortPassRows(' '_ = X070Backbone.defaultShortPassRows('
run "порог ставится и при родном контейнере" \
    "$LD" 'if baseDType != nil, !useNativeKernel,
           let kind' 'if baseDType != nil,
           let kind'
run "sb6 получает порог sym" \
    "$LD" 'nEmbd: bb.cfg.nEmbd, sym: kind == .sym)' 'nEmbd: bb.cfg.nEmbd, sym: true)'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
