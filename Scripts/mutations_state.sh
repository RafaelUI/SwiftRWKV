#!/bin/bash
# Мутационная проверка (14.10): начальное состояние как подключаемый объект,
# его обучение (state-tuning), файл и отпечаток базы.
#
#   ./Scripts/mutations_state.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
IS=Sources/RWKVGen/InitialState.swift
RB=Sources/RWKVGen/RWKVBlock.swift
TR=Sources/RWKVGen/Training/Trainer.swift
TS=Sources/RWKVGen/Training/TrainableSet.swift
ET=Sources/RWKVEmbedding/EmbeddingTrainable.swift

FILTER='StateTuningTests|StateTuningParityTests'
run() {
    local desc="$1"; shift
    printf '%-66s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── блок: откуда берётся состояние ──"
run "обычный проход не берёт подключённую матрицу WKV" \
    "$RB" 'vFirst, ctx, shiftPrev: s0?.tmixShift, hIn: s0?.wkv)' 'vFirst, ctx, shiftPrev: s0?.tmixShift, hIn: nil)'
run "обычный проход: сдвиг tmix взят у cmix" \
    "$RB" 'vFirst, ctx, shiftPrev: s0?.tmixShift, hIn: s0?.wkv)' 'vFirst, ctx, shiftPrev: s0?.cmixShift, hIn: s0?.wkv)'
run "обычный проход не берёт сдвиг cmix" \
    "$RB" '        shiftPrev: s0?.cmixShift)' '        shiftPrev: nil)'
run "явная матрица WKV не вытесняет подключённую" \
    "$RB" 'hIn: hIn ?? s0?.wkv, mask: mask' 'hIn: s0?.wkv ?? hIn, mask: mask'
run "явный сдвиг tmix не вытесняет подключённый" \
    "$RB" 'shiftPrev: tmixPrev ?? s0?.tmixShift' 'shiftPrev: s0?.tmixShift ?? tmixPrev'
run "явный сдвиг cmix не вытесняет подключённый" \
    "$RB" 'shiftPrev: cmixPrev ?? s0?.cmixShift)' 'shiftPrev: s0?.cmixShift ?? cmixPrev)'
run "проход с состоянием не берёт подключённую матрицу WKV" \
    "$RB" 'hIn: hIn ?? s0?.wkv, mask: mask' 'hIn: hIn, mask: mask'
run "проход с состоянием не берёт подключённый сдвиг tmix" \
    "$RB" 'shiftPrev: tmixPrev ?? s0?.tmixShift' 'shiftPrev: tmixPrev'
run "проход с состоянием не берёт подключённый сдвиг cmix" \
    "$RB" 'shiftPrev: cmixPrev ?? s0?.cmixShift)' 'shiftPrev: cmixPrev)'
run "каждый слой берёт состояние слоя 0" \
    "$RB" 'backbone.initialLayers?[layer]' 'backbone.initialLayers?[0]'

echo "── подключение, старт декода, свёртка промпта ──"
run "подключение кладёт во все слои матрицу слоя 0" \
    "$IS" 'InitialLayerState(wkv: state.layerWKV(l)[0].asType(.float32),' 'InitialLayerState(wkv: state.layerWKV(0)[0].asType(.float32),'
run "проверка формы состояния выключена" \
    "$IS" 'for (name, got, shape) in want where got != shape {' 'for (name, got, shape) in want where got != shape && name.isEmpty {'
run "пошаговый декод стартует с нуля при подключённом состоянии" \
    "$IS" 'guard let s = initialState else { return RWKVState(cfg: cfg, dtype: dtype) }' \
          'guard let s = initialState, s.batch == 0 else { return RWKVState(cfg: cfg, dtype: dtype) }'
run "свёртка промпта теряет последний токен" \
    "$IS" 'let x = MLXArray(ids.map { Int32($0) }, [1, ids.count])' \
          'let x = MLXArray(ids.dropLast().map { Int32($0) }, [1, ids.count - 1])'
run "свёртка промпта не берёт явный старт" \
    "$IS" 'return withoutGradientPath { states(x, state: state) }' 'return withoutGradientPath { states(x, state: nil) }'

echo "── обучаемое множество и обучение ──"
run "слои не переведены на дифференцируемое ядро" \
    "$IS" '        bb.trainLayers = Set(0 ..< bb.cfg.nLayer)' '        bb.trainLayers = []'
run "матрица WKV подставляется в слой 0" \
    "$IS" 'case .wkv: layers[l].wkv = ps[i]' 'case .wkv: layers[0].wkv = ps[i]'
run "параметры состояния названы не по блокам" \
    "$IS" 'parts.map { "blocks.\(l).state0.\($0.rawValue)" }' 'parts.map { "state0.\(l).\($0.rawValue)" }'
run "обучаются все части, а не заказанные" \
    "$IS" 'self.parts = Part.allCases.filter(train.contains)' 'self.parts = Part.allCases'
run "после обучения trainLayers не возвращён" \
    "$IS" 'defer { bb.trainLayers = savedLayers; bb.useBlockCheckpoint = savedCheckpoint }' \
          'defer { bb.useBlockCheckpoint = savedCheckpoint }'
run "state-tuning идёт одним графом, а не по блокам" \
    "$IS" 'nextBatch: source, config: cfg.training, gradient: provider)' 'nextBatch: source, config: cfg.training, gradient: nil)'
run "стартовое состояние обучения потеряно" \
    "$IS" 'let set = StateTrainableSet(bb, train: cfg.train, initial: initial)' 'let set = StateTrainableSet(bb, train: cfg.train, initial: nil)'
run "множитель шага по параметрам не действует" \
    "$TR" 'params[i] = params[i] - (scales.map { lr * $0[i] } ?? lr) * upd' 'params[i] = params[i] - lr * upd'
run "множитель шага состояния не отдаётся" \
    "$IS" 'scale.map { [Float](repeating: $0, count: bb.cfg.nLayer * parts.count) }' 'nil'
run "композиция: множитель частей без своего -- ноль" \
    "$TS" 's ?? [Float](repeating: 1, count: part.parameterNames.count)' 's ?? [Float](repeating: 0, count: part.parameterNames.count)'

FILTER='EmbeddingSequentialTests'
run "эмбеддер: состояние в множество не входит" \
    "$ET" 'return CompositeTrainableSet([stateSet, headSet])' 'return CompositeTrainableSet([headSet])'
run "эмбеддер: множитель шага состояния теряется" \
    "$ET" 'learningRateScale: scale == 1 ? nil : scale)' 'learningRateScale: nil)'
run "эмбеддер: заказанные части состояния не доходят" \
    "$ET" 'let stateSet = StateTrainableSet(bb, train: parts,' 'let stateSet = StateTrainableSet(bb, train: [.wkv],'
FILTER='StateTuningTests|StateTuningParityTests'

echo "── отпечаток базы ──"
run "в отпечаток не идут значения" \
    "$IS" 'canon.asArray(Float.self).withUnsafeBytes { h.update(bufferPointer: $0) }' '_ = canon'
run "отпечаток считается без приведения к решётке" \
    "$IS" 'let canon = a.asType(lattice).asType(.float32).reshaped([-1])' 'let canon = a.asType(.float32).reshaped([-1])'
run "в отпечаток идёт только нижний слой" \
    "$IS" 'for l in 0 ..< cfg.nLayer { names += fingerprintBlock' 'for l in 0 ..< 1 { names += fingerprintBlock'
run "в отпечаток не идут строки проекций" \
    "$IS" 'feed(name, row, lattice: row.dtype == .float16 ? .float16 : .bfloat16)' '_ = row'
run "строки сжатых проекций -- через bf16, а не fp16" \
    "$IS" 'feed(name, row, lattice: row.dtype == .float16 ? .float16 : .bfloat16)' 'feed(name, row, lattice: .bfloat16)'
run "контракт не входит в отпечаток" \
    "$IS" 'h.update(data: Data("rwkv-state-base-v1\n\(cfg.nLayer),\(cfg.nEmbd),\(cfg.nHead),\(cfg.headSize)\n".utf8))' \
          'h.update(data: Data("rwkv-state-base-v1\n".utf8))'

echo "── файл ──"
run "чужая база принимается" \
    "$IS" 'if info.baseFingerprint != fp && !allowOtherBase {' 'if info.baseFingerprint != fp && allowOtherBase {'
run "контракт из метаданных не проверяется" \
    "$IS" 'where file != model {' 'where file != model && name.isEmpty {'
run "формат файла не проверяется" \
    "$IS" 'guard md["format"] == format else' 'guard md["format"] != nil else'
run "версия формата не проверяется" \
    "$IS" 'guard md["version"] == version else' 'guard md["version"] != nil else'
run "отсутствие тензора не замечается" \
    "$IS" 'where arrays[t] == nil { throw Failure.missingTensor(t) }' 'where arrays[t] == nil && t.isEmpty { throw Failure.missingTensor(t) }'
run "в файл пишется пустой отпечаток" \
    "$IS" '"base_fingerprint": info.baseFingerprint, "note": note,' '"base_fingerprint": "", "note": note,'
run "в файл не пишется, что обучалось" \
    "$IS" '"trained": info.trained.joined(separator: ","),' '"trained": "",'
run "сдвиг cmix пишется из tmix" \
    "$IS" '"cmix_shift": state.cmixShift.asType(.float32)]' '"cmix_shift": state.tmixShift.asType(.float32)]'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
