#!/bin/bash
# Мутационная проверка (12.10): ядра обучаемого WKV (прямое со стейджингом,
# обратное по схеме Бо Пэна, удержание выходов прямого ядра), лосс кусками и
# обратный проход по блокам.
#
#   ./Scripts/mutations_train_long.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
WK=Sources/RWKVKernel/WKV7Train.swift
AB=Sources/RWKVKernel/Ablation.swift
SB=Sources/RWKVGen/Training/SequentialBackward.swift
LF=Sources/RWKVGen/LoRAFinetune.swift
BB=Sources/RWKVGen/X070Backbone.swift

FILTER='WKV7TrainKernelTests|WKV7StateTests|WKV7KernelParityTests|WKV7KernelHazardTests'
run() {
    local desc="$1"; shift
    printf '%-62s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── обратное ядро по схеме Бо Пэна ──"
run "реконструкция состояния без члена sa*b" \
    "$WK" 'stT[j]   = (stT[j] - v_sh[j]*ki - sa_sh[j]*bb) * iwi;' 'stT[j]   = (stT[j] - v_sh[j]*ki) * iwi;'
run "строка градиента состояния: b вместо a" \
    "$WK" 'dst[j]  = dst[j]*w_sh[j] + dsb*a_sh[j];' 'dst[j]  = dst[j]*w_sh[j] + dsb*b_sh[j];'
run "колонка градиента состояния без члена dSb" \
    "$WK" 'dstT[j] = dstT[j]*wi + ai*dsb_sh[j];' 'dstT[j] = dstT[j]*wi;'
run "dw по строке вместо колонки" \
    "$WK" 'dw0  += tA*sA;         dw1  += tB*sB;' 'dw0  += uA*sA;         dw1  += uB*sB;'
run "dk: k вместо v" \
    "$WK" 'dkv0 += tA*v_sh[j];    dkv1 += tB*v_sh[j+1];' 'dkv0 += tA*k_sh[j];    dkv1 += tB*k_sh[j+1];'
run "db: a вместо sa" \
    "$WK" 'db0  += tA*sa_sh[j];   db1  += tB*sa_sh[j+1];' 'db0  += tA*a_sh[j];   db1  += tB*a_sh[j+1];'
run "dv: v вместо k" \
    "$WK" 'dvv0 += uA*k_sh[j];    dvv1 += uB*k_sh[j+1];' 'dvv0 += uA*v_sh[j];    dvv1 += uB*v_sh[j+1];'
run "котангент состояния: строка вместо колонки" \
    "$WK" 'dstT[j] = d_h_out[hbase + j*HEAD_SIZE_C + i];' 'dstT[j] = d_h_out[hrow + j];'
run "чекпоинт состояния: строка вместо колонки" \
    "$WK" 'stT[j] = h_ckpts[ckb + j*HEAD_SIZE_C + i];' 'stT[j] = h_ckpts[ckb + i*HEAD_SIZE_C + j];'
run "вклад dy в строку градиента: k вместо r" \
    "$WK" 'dst[j]  += dyi*r_sh[j];' 'dst[j]  += dyi*k_sh[j];'
run "переключатель ядра не действует" \
    "$WK" 'let kern = wkv7BackwardV2 ? ckptBwdKernelV2(H: H, T: T) : ckptBwdKernel(H: H, T: T)' \
          'let kern = ckptBwdKernel(H: H, T: T)'
run "примелы идут в обратное ядро в своих типах" \
    "$WK" 'primals[0 ..< 6].map { $0.asType(.float32) } + [hCkpts, saFwd, dOut.asType(.float32), dH],' \
          'Array(primals[0 ..< 6]) + [hCkpts, saFwd, dOut.asType(.float32), dH],'

echo "── выходы прямого ядра держатся ──"
run "режимы удержания и пересчёта перепутаны" \
    "$WK" 'if RWKVAblation.skipWKVBackward || !wkv7KeepForwardOutputs {' \
          'if RWKVAblation.skipWKVBackward || wkv7KeepForwardOutputs {'
run "котангент по конечному состоянию потерян" \
    "$WK" 'let g = backward(primals, saFwd: primals[9], hCkpts: primals[10], dOut: cotangents[0],
                             dHOut: cotangents.count > 1 ? cotangents[1] : nil)' \
          'let g = backward(primals, saFwd: primals[9], hCkpts: primals[10], dOut: cotangents[0],
                             dHOut: nil)'
run "прямое ядро снаружи — без stopGradient" \
    "$WK" 'let f = forwardKernel([r, w, k, v, a, b, h0].map { stopGradient($0.asType(.float32)) })' \
          'let f = forwardKernel([r, w, k, v, a, b, h0].map { $0.asType(.float32) })'
run "абляция обратного WKV включена по умолчанию" \
    "$AB" 'public nonisolated(unsafe) static var skipWKVBackward = false' \
          'public nonisolated(unsafe) static var skipWKVBackward = true'

echo "── прямое ядро со стейджингом ──"
run "sa по b вместо a" \
    "$WK" 'for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];' \
          'for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*b_sh[dk];'
run "состояние обновляется через a вместо b" \
    "$WK" 'h_row[dk] = w_sh[dk]*h_row[dk] + vv*k_sh[dk] + sa*b_sh[dk];' \
          'h_row[dk] = w_sh[dk]*h_row[dk] + vv*k_sh[dk] + sa*a_sh[dk];'

FILTER='SequentialBackwardTests'
echo "── обратный проход по блокам ──"
run "котангент по v_first в слой 0 не доходит" \
    "$SB" 'if outputsV { cotangents.append(dv ?? MLXArray.zeros(vFirst!.shape, dtype: vFirst!.dtype)) }' \
          'if outputsV { cotangents.append(MLXArray.zeros(vFirst!.shape, dtype: vFirst!.dtype)) }'
run "котангент по v_first не копится по слоям" \
    "$SB" 'if hasV { dv = dv.map { $0 + g[1] } ?? g[1]; ready.append(dv!) }' \
          'if hasV { dv = g[1]; ready.append(dv!) }'
run "котангент по входу не доходит до нижнего блока" \
    "$SB" 'let needInput = l > first' 'let needInput = l > first + 1'
run "лосс нормируется без батча" \
    "$SB" 'let scale = MLXArray(Float(1) / Float(B * T))' 'let scale = MLXArray(Float(1) / Float(T))'
run "куски котангента головы склеены в обратном порядке" \
    "$SB" 'let dh = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)' \
          'let dh = pieces.count == 1 ? pieces[0] : concatenated(pieces.reversed(), axis: 1)'
run "параметры слоя подставляются не свои" \
    "$SB" 'for (j, i) in idx.enumerated() { current[i] = ins[nData + j] }' \
          'for (j, i) in idx.enumerated() { current[i] = ins[nData + (j + 1) % idx.count] }'
run "режим включён, а градиент идёт автоградом" \
    "$LF" 'let provider: GradientProvider<LoRABatch>? = cfg.sequentialBackward' \
          'let provider: GradientProvider<LoRABatch>? = !cfg.sequentialBackward'

echo "── лосс кусками ──"
run "куски и при обучаемой голове" \
    "$LF" 'if chunk > 0, T0 > chunk, bb.headIsConstant {' 'if chunk > 0, T0 > chunk {'
run "голова считается константой при обучаемой подмене" \
    "$BB" 'var headIsConstant: Bool { loraA["head"] == nil && wOverride?["head.weight"] == nil }' \
          'var headIsConstant: Bool { loraA["head"] == nil }'
run "лосс кусками нормируется без батча" \
    "$LF" 'return total / Float(B * T0)' 'return total / Float(T0)'
run "в лосс идёт только последний кусок" \
    "$LF" 'total = total + piece([h[0..., from ..< to]])[0]' 'total = piece([h[0..., from ..< to]])[0]'
run "цели куска берутся с начала последовательности" \
    "$LF" 'let targets = batch.y[0..., from ..< to].reshaped([-1]).asType(.int32)
                let piece' 'let targets = batch.y[0..., 0 ..< (to - from)].reshaped([-1]).asType(.int32)
                let piece'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
