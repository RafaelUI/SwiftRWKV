import Foundation
import MLX
import MLXNN          // crossEntropy

// ───────────────────────────────────────────────────────────────────────
//  Обратный проход ПО БЛОКАМ: градиент адаптеров при памяти одного блока.
//
//  Зачем. Блочный чекпоинт (`useBlockCheckpoint`) держит между проходами
//  только входы блоков, но весь обратный проход остаётся ОДНИМ ленивым
//  графом MLX, и порядок его вычисления выбирает обход графа, а не мы:
//  пересчитанные активации нескольких блоков оказываются живыми сразу.
//  Замерено на 1.5B (QLoRA, батч 1): пик MLX растёт на ~1.9 ГБ на каждые
//  1024 токена при ~0.4 ГБ входов блоков — T = 2048 пикует 6.3 ГБ, T = 4096
//  в 16 ГБ не входит. Делить чекпоинт мельче (tmix / cmix) и считать лосс
//  кусками почти не помогает (−0.6 ГБ на T = 2048): дело не в размере куска,
//  а в том, что куски вычисляются не по очереди.
//
//  Здесь очередь задана явно. Прямой проход без градиента сохраняет вход
//  каждого блока; затем голова и лосс, затем блоки сверху вниз — и после
//  каждого блока граф ВЫЧИСЛЯЕТСЯ (eval вне трассировки), так что живёт
//  один блок: его пересчитанные активации и котангенты. Математика та же,
//  что у автограда по всему графу: цепное правило по блокам, котангент по
//  v_first копится со всех слоёв и уходит в слой 0, где v_first рождается.
//
//  Цена по времени та же, что у блочного чекпоинта: каждый блок считается
//  дважды (прямой проход и пересчёт внутри vjp).
// ───────────────────────────────────────────────────────────────────────

extension LoRAFinetune {

    /// Сколько раз градиент посчитан обратным проходом по блокам (счётчик для
    /// тестов: градиент побитно равен автограду, и по числам режим не виден).
    nonisolated(unsafe) static var sequentialGradientCalls = 0

    /// (лосс, градиенты в порядке `set.parameterNames`) обратным проходом по
    /// блокам. `parameters` — fp32-мастер адаптеров; подстановка в модель идёт
    /// внутри каждого grad-замыкания (контракт `GradientProvider`).
    static func sequentialGradients(
        _ bb: X070Backbone, set: LoRATrainableSet, parameters: [MLXArray],
        batch: LoRABatch, lossChunk: Int
    ) -> (loss: MLXArray, gradients: [MLXArray]) {
        sequentialGradientCalls += 1
        let L = bb.cfg.nLayer
        let T = batch.x.shape[1], B = batch.x.shape[0]

        // параметры по слоям
        var byLayer: [Int: [Int]] = [:]
        for (i, name) in set.parameterNames.enumerated() {
            let parts = name.split(separator: ".")
            guard parts.count > 2, parts[0] == "blocks", let l = Int(parts[1]) else {
                preconditionFailure("обратный проход по блокам: параметр \(name) не принадлежит блоку")
            }
            byLayer[l, default: []].append(i)
        }
        guard let first = byLayer.keys.min() else { preconditionFailure("нет адаптеров") }
        precondition(Set(first ..< L).isSubset(of: bb.trainLayers),
                     "слои \(first)..<\(L) обязаны идти дифференцируемым ядром (trainLayers)")

        // ── 1. прямой проход без градиента: входы обучаемых блоков ──
        set.inject(parameters)
        var x = bb.stem(batch.x)
        var blockInput: [Int: MLXArray] = [:]
        var vFirst: MLXArray? = nil
        for l in 0 ..< L {
            if l >= first { blockInput[l] = x }
            let (xo, vf) = bb.blockForward(x, vFirst, l)
            eval(xo, vf)
            x = xo
            vFirst = vf
        }

        // ── 2. ln_out, голова и лосс: по кускам, каждый — свой vjp и eval ──
        let h = bb.finalNorm(x, T: T)
        eval(h)
        let V = bb.cfg.vocab
        let scale = MLXArray(Float(1) / Float(B * T))
        let step = lossChunk > 0 ? lossChunk : T
        var total = MLXArray(Float(0))
        var pieces: [MLXArray] = []
        var from = 0
        while from < T {
            let to = Swift.min(from + step, T)
            let targets = batch.y[0..., from ..< to].reshaped([-1]).asType(.int32)
            let (value, g) = vjp({ ins in
                let lg = bb.projectForBlock(ins[0], "head.weight", lora: nil)
                return [crossEntropy(logits: lg.reshaped([-1, V]), targets: targets,
                                     reduction: .sum).asType(.float32)]
            }, primals: [h[0..., from ..< to]], cotangents: [scale])
            eval(value[0], g[0])
            total = total + value[0]
            pieces.append(g[0])
            from = to
        }
        let dh = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
        var dx = vjp({ ins in [bb.finalNorm(ins[0], T: T)] }, primals: [x], cotangents: [dh]).1[0]
        let loss = total * scale
        eval(dx, loss)
        pieces = []

        // ── 3. блоки сверху вниз ──
        var grads = [MLXArray?](repeating: nil, count: parameters.count)
        var dv: MLXArray? = nil            // накопленный котангент по v_first
        for l in stride(from: L - 1, through: first, by: -1) {
            let idx = byLayer[l] ?? []
            let xin = blockInput[l]!
            let hasV = l > 0
            // У нижнего обучаемого блока котангенты по входу никому не нужны:
            // ниже него нет параметров. Вход тогда — константа замыкания, и
            // обратный проход через проекции этого блока по x не считается.
            let needInput = l > first
            var primals: [MLXArray] = []
            if needInput { primals.append(xin); if hasV { primals.append(vFirst!) } }
            let nData = primals.count
            primals += idx.map { parameters[$0] }
            let outputsV = !hasV            // слой 0 сам рождает v_first
            let f: ([MLXArray]) -> [MLXArray] = { ins in
                var current = parameters
                for (j, i) in idx.enumerated() { current[i] = ins[nData + j] }
                set.inject(current)
                let xi = needInput ? ins[0] : xin
                let vi: MLXArray? = hasV ? (needInput ? ins[1] : vFirst!) : nil
                let (xo, vo) = bb.blockForward(xi, vi, l)
                return outputsV ? [xo, vo] : [xo]
            }
            var cotangents = [dx]
            if outputsV { cotangents.append(dv ?? MLXArray.zeros(vFirst!.shape, dtype: vFirst!.dtype)) }
            let g = vjp(f, primals: primals, cotangents: cotangents).1
            set.inject(parameters)         // в модели — значения, а не трассеры
            var ready: [MLXArray] = []
            if needInput {
                dx = g[0]
                ready.append(dx)
                if hasV { dv = dv.map { $0 + g[1] } ?? g[1]; ready.append(dv!) }
            }
            for (j, i) in idx.enumerated() { grads[i] = g[nData + j]; ready.append(g[nData + j]) }
            eval(ready)
            blockInput[l] = nil
        }
        return (loss, grads.enumerated().map { i, g in
            g ?? MLXArray.zeros(parameters[i].shape, dtype: parameters[i].dtype)
        })
    }
}
