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
run "кусок лосса не делится на батч (по блокам)" \
    "$SB" 'let step = lossPieceTokens(chunk: lossChunk, batch: B, length: T)' \
          'let step = lossPieceTokens(chunk: lossChunk, batch: 1, length: T)'
run "кусок лосса считается на строку" \
    "$LF" 'chunk > 0 ? Swift.max(1, chunk / Swift.max(1, batch)) : length' 'chunk > 0 ? chunk : length'

FILTER='EmbeddingSequentialTests'
EO=Sources/RWKVEmbedding/EmbeddingObjective.swift
EF=Sources/RWKVEmbedding/EmbeddingFinetune.swift
ED=Sources/RWKVEmbedding/Embedder.swift
echo "── эмбеддер: обратный проход по блокам ──"
run "параметры основания (ln0) считаются параметрами над телом" \
    "$SB" '} else if name.hasPrefix("ln0.") || name.hasPrefix("emb.") {' '} else if name.hasPrefix("emb.") {'
run "градиент основания не считается" \
    "$SB" 'for (j, i) in idx.enumerated() { add(i, g[j]) }' 'for (j, i) in idx.enumerated() { _ = (i, g[j]) }'
run "градиенты пачек не складываются, а затирают друг друга" \
    "$SB" 'func add(_ i: Int, _ g: MLXArray) { grads[i] = grads[i].map { $0 + g } ?? g }' \
          'func add(_ i: Int, _ g: MLXArray) { grads[i] = g }'
run "котангент пачки взят у соседней" \
    "$EO" 'SequentialBody.backward(bb, trace: trace, dTop: g[k], parameters: parameters, plan: plan,' \
          'SequentialBody.backward(bb, trace: trace, dTop: g[(k + 1) % n], parameters: parameters, plan: plan,'
run "градиенты параметров над телом перепутаны" \
    "$EO" 'for (j, i) in above.enumerated() { grads[i] = g[n + j] }' \
          'for (j, i) in above.enumerated() { grads[i] = g[n + (j + 1) % above.count] }'
run "параметры над телом в лосс не подставляются" \
    "$EO" 'for (j, i) in above.enumerated() { current[i] = ins[n + j] }' \
          'for (j, i) in above.enumerated() { _ = (i, ins[n + j]) }'
run "вектор над телом без ln_out" \
    "$ED" 'let h = backbone.finalNorm(x, T: x.shape[1])' 'let h = x'
run "пулинг пачки взят у якорей" \
    "$EO" 'byGroup[part.group].append(model.embedFromTop(ins[c], poolIndex: part.pool))' \
          'byGroup[part.group].append(model.embedFromTop(ins[c], poolIndex: parts[0].pool))'
run "кандидаты классификации не собраны по строкам" \
    "$EO" 'zeroShotClassificationLoss(anchor: v[0], candidates: v[1].reshaped([B, K, -1]),' \
          'zeroShotClassificationLoss(anchor: v[0], candidates: v[1].reshaped([K, B, -1]).transposed(1, 0, 2),'
run "стадия: режим по умолчанию выключен" \
    "$EF" 'public var sequentialBackward: Bool = true' 'public var sequentialBackward: Bool = false'
run "стадия: режим действует и под GradCache наоборот" \
    "$EF" ': stage.sequentialBackward
                ? EmbeddingObjective.sequentialProvider(' ': !stage.sequentialBackward
                ? EmbeddingObjective.sequentialProvider('

echo "── эмбеддер: тело по строкам, запись входов с шагом, потолок кэша до оценки ──"
run "куски по строкам не режутся" \
    "$EO" 'let rows = chunkTokens > 0 ? Swift.max(1, chunkTokens / Swift.max(1, T)) : B' 'let rows = B'
run "неполный последний кусок потерян" \
    "$EO" 'while from < B {' 'while from + rows <= B {'
run "пулинг куска взят с первых строк пачки" \
    "$EO" 'out.append((gi, grp.idx[from ..< to], grp.pool[from ..< to]))' \
          'out.append((gi, grp.idx[from ..< to], grp.pool[0 ..< to - from]))'
run "векторы кусков собраны в обратном порядке" \
    "$EO" 'byGroup[part.group].append(model.embedFromTop(ins[c], poolIndex: part.pool))' \
          'byGroup[part.group].insert(model.embedFromTop(ins[c], poolIndex: part.pool), at: 0)'
run "стадия не передаёт bodyChunkTokens" \
    "$EF" 'symmetric: symmetric,
                    chunkTokens: stage.bodyChunkTokens, traceStride: stage.traceStride)' 'symmetric: symmetric,
                    chunkTokens: 0, traceStride: stage.traceStride)'
run "стадия не передаёт traceStride" \
    "$EF" 'symmetric: symmetric,
                    chunkTokens: stage.bodyChunkTokens, traceStride: stage.traceStride)' 'symmetric: symmetric,
                    chunkTokens: stage.bodyChunkTokens, traceStride: 1)'
run "запись с шагом хранит все входы" \
    "$SB" 'if l >= first && (l - first) % Swift.max(1, stride) == 0 { trace.blockInput[l] = x }' \
          'if l >= first { trace.blockInput[l] = x }'
run "пересчёт входа идёт без v_first" \
    "$SB" 'let (xo, _) = bb.blockForward(x, j > 0 ? vFirst : nil, j)' \
          'let (xo, _) = bb.blockForward(x, j > 0 ? vFirst.map { MLXArray.zeros($0.shape, dtype: $0.dtype) } : nil, j)'
run "пересчёт входа идёт от самого нижнего записанного" \
    "$SB" 'blockInput.keys.filter({ $0 < l }).max()' 'blockInput.keys.filter({ $0 < l }).min()'
run "пересчитанный вход положен блоку ниже" \
    "$SB" 'blockInput[j + 1] = x' 'blockInput[j + 1] = blockInput[j]!'
run "потолок кэша ставится только в цикле обучения" \
    "$EF" '        applyCacheLimit(stage.config)

        let before' '        let before'
run "потолок кэша 0 понимается как «без кэша»" \
    "$EF" 'if cfg.cacheLimitGB > 0 { MLX.Memory.cacheLimit' 'if cfg.cacheLimitGB >= 0 { MLX.Memory.cacheLimit'

echo "── эмбеддер: верхние слои базы из .rwkvq плотными ──"
XB=Sources/RWKVGen/X070Backbone.swift
ET=Sources/RWKVEmbedding/EmbeddingTrainable.swift
run "разворачиваются нижние слои вместо верхних" \
    "$XB" 'for key in (rwkvqBackedKeys + rwkvqDenseTargets.sorted()) where (layer(key) ?? -1) >= from {' \
          'for key in (rwkvqBackedKeys + rwkvqDenseTargets.sorted()) where (layer(key) ?? cfg.nLayer) < n {'
run "развёрнутый вес остаётся на сжатой подложке" \
    "$XB" '            w[key] = value
            releaseRwkvqBacking(key)' '            w[key] = value'
run "развёрнут вес соседней проекции" \
    "$XB" 'guard let dense = denseWeight(key) else { continue }' \
          'guard let dense = denseWeight(key.replacingOccurrences(of: "k_proj", with: "v_proj")) else { continue }'
run "верхние слои на сжатой базе разрешены без разворачивания" \
    "$ET" 'if packed.isEmpty { return nil }' 'if packed.isEmpty || n > 0 { return nil }'
run "проверка сжатых слоёв смотрит на нижние" \
    "$ET" '(from ..< bb.cfg.nLayer).contains { key.hasPrefix("blocks.\($0).") }' \
          '(0 ..< from).contains { key.hasPrefix("blocks.\($0).") }'
run "куски по строкам по умолчанию выключены" \
    "$EF" 'public var bodyChunkTokens: Int = 2048' 'public var bodyChunkTokens: Int = 0'

echo "── эмбеддер: оценка батчем по токенам ──"
EM=Sources/RWKVEmbedding/EmbeddingMetrics.swift
run "бюджет токенов батча оценки не действует" \
    "$EM" 'if batchTokens > 0, (end - i + 1) * grown > batchTokens { break }' \
          'if batchTokens < 0, (end - i + 1) * grown > batchTokens { break }'
run "бюджет считается по первой строке батча, а не по самой длинной" \
    "$EM" 'if batchTokens > 0, (end - i + 1) * grown > batchTokens { break }' \
          'if batchTokens > 0, (end - i + 1) * rows[i].count > batchTokens { break }'
run "последняя строка батча оценки потеряна" \
    "$EM" 'let (idx, pool) = padBatch(Array(rows[i ..< end]), terminator: terminator)' \
          'let (idx, pool) = padBatch(Array(rows[i ..< Swift.max(i + 1, end - 1)]), terminator: terminator)'
run "батч оценки не ограничен числом строк" \
    "$EM" 'while end < rows.count, end - i < batchSize {' 'while end < rows.count {'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
