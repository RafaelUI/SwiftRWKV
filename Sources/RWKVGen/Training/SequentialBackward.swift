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
//
//  `SequentialBody` — общая часть: прямой проход с записью входов блоков и
//  обратный проход по блокам от котангента на выходе последнего блока. Что
//  стоит НАД телом (ln_out, голова, лосс), решает вызывающий: языковая
//  модель (ниже), эмбеддер (RWKVEmbedding). Параметры различаются по именам:
//  `blocks.N.…` принадлежат блоку N, `ln0.…` / `emb.…` — основанию под
//  блоками, остальные — тому, что над телом.
// ───────────────────────────────────────────────────────────────────────

public enum SequentialBody {

    /// Отметки фаз для приборов памяти: вызывается ВНЕ трассировки градиента,
    /// сразу после очередного eval (после прямого прохода пачки, после
    /// каждого блока обратного прохода, после основания). По умолчанию nil.
    /// Прибор снаружи читает активную память и пик MLX между отметками —
    /// так пик шага раскладывается по статьям без eval внутри трассировки.
    nonisolated(unsafe) public static var probe: ((String) -> Void)? = nil

    /// ОПЫТ (по умолчанию выключено): после блока — `asyncEval` вместо `eval`,
    /// то есть CPU строит граф следующего блока, пока GPU считает этот (как
    /// `DecodePipeline` на декоде). Счёт тот же; меняется только, ждёт ли CPU.
    nonisolated(unsafe) public static var pipelined = false

    @inline(__always) static func settle(_ arrays: [MLXArray]) {
        if pipelined { asyncEval(arrays) } else { eval(arrays) }
    }

    /// Куда относится каждый параметр обучаемого множества.
    public struct Plan {
        public let nLayer: Int
        /// Индексы параметров по слоям.
        public let byLayer: [Int: [Int]]
        /// Параметры основания (ln0, emb) — под блоками.
        public let stem: [Int]
        /// Всё остальное — над телом (ln_out, голова): их градиент считает вызывающий.
        public let above: [Int]
        /// Нижний блок, через который идёт градиент; nil — тело не обучается.
        public var first: Int? { stem.isEmpty ? byLayer.keys.min() : 0 }

        public init(names: [String], nLayer: Int) {
            var by: [Int: [Int]] = [:], st: [Int] = [], ab: [Int] = []
            for (i, name) in names.enumerated() {
                let parts = name.split(separator: ".")
                if parts.count > 2, parts[0] == "blocks", let l = Int(parts[1]) {
                    by[l, default: []].append(i)
                } else if name.hasPrefix("ln0.") || name.hasPrefix("emb.") {
                    st.append(i)
                } else {
                    ab.append(i)
                }
            }
            self.nLayer = nLayer; byLayer = by; stem = st; above = ab
        }
    }

    /// Запись прямого прохода одной пачки.
    public final class Trace {
        let ids: MLXArray
        var blockInput: [Int: MLXArray] = [:]
        var vFirst: MLXArray? = nil
        /// Выход последнего блока — до ln_out.
        public internal(set) var top: MLXArray
        init(ids: MLXArray, top: MLXArray) { self.ids = ids; self.top = top }

        /// Вход блока `l`, которого в записи нет (запись с шагом, см. `forward`):
        /// блоки от ближайшего записанного ниже считаются заново, их входы
        /// кладутся в запись и уходят по мере обратного прохода. Тот же счёт
        /// на тех же числах, что в первом прямом проходе, — вход побитно тот же.
        func restoreInput(_ l: Int, _ bb: X070Backbone) {
            guard blockInput[l] == nil else { return }
            guard let from = blockInput.keys.filter({ $0 < l }).max() else {
                preconditionFailure("нет записанного входа ниже блока \(l)")
            }
            var x = blockInput[from]!
            for j in from ..< l {
                let (xo, _) = bb.blockForward(x, j > 0 ? vFirst : nil, j)
                eval(xo)
                x = xo
                blockInput[j + 1] = x
                SequentialBody.probe?("retrace")
            }
        }
    }

    /// Прямой проход без градиента; хранит входы блоков, через которые пойдёт
    /// градиент. Модель должна уже видеть текущие параметры (`inject`).
    ///
    /// `stride > 1` — запись с шагом: хранится вход каждого `stride`-го блока
    /// (считая от нижнего обучаемого), остальные восстанавливаются в обратном
    /// проходе пересчётом отрезка (`Trace.restoreInput`). Память входов блоков
    /// (L · токены · D · 4 байта на каждую пачку — у 0.4B при 8 триплетах по
    /// 1024 токена это 2.4 ГБ) делится на `stride`; цена — (stride − 1) / stride
    /// ещё одного прямого прохода.
    public static func forward(_ bb: X070Backbone, ids: MLXArray, plan: Plan, stride: Int = 1) -> Trace {
        let first = plan.first ?? plan.nLayer
        precondition(Set(first ..< plan.nLayer).isSubset(of: bb.trainLayers),
                     "слои \(first)..<\(plan.nLayer) обязаны идти дифференцируемым ядром (trainLayers)")
        var x = bb.stem(ids)
        eval(x)
        let trace = Trace(ids: ids, top: x)
        var vFirst: MLXArray? = nil
        for l in 0 ..< plan.nLayer {
            if l >= first && (l - first) % Swift.max(1, stride) == 0 { trace.blockInput[l] = x }
            let (xo, vf) = bb.blockForward(x, vFirst, l)
            settle([xo, vf])
            x = xo
            vFirst = vf
        }
        trace.vFirst = vFirst
        trace.top = x
        probe?("forward")
        return trace
    }

    /// Блоки сверху вниз от котангента `dTop` на выходе последнего блока;
    /// градиенты ПРИБАВЛЯЮТСЯ к `grads` (несколько пачек на один шаг).
    /// `inject` — подстановка полного списка параметров в модель; вызывается
    /// внутри каждого grad-замыкания и после него (вернуть значения вместо
    /// трассеров). Входы блоков из `trace` освобождаются по ходу.
    public static func backward(
        _ bb: X070Backbone, trace: Trace, dTop: MLXArray, parameters: [MLXArray],
        plan: Plan, inject: @escaping ([MLXArray]) -> Void, grads: inout [MLXArray?]
    ) {
        guard let first = plan.first else { return }
        let stemNeeded = !plan.stem.isEmpty
        var dx = dTop
        var dv: MLXArray? = nil            // накопленный котангент по v_first
        let vFirst = trace.vFirst
        func add(_ i: Int, _ g: MLXArray) { grads[i] = grads[i].map { $0 + g } ?? g }

        for l in stride(from: plan.nLayer - 1, through: first, by: -1) {
            let idx = plan.byLayer[l] ?? []
            trace.restoreInput(l, bb)
            let xin = trace.blockInput[l]!
            let hasV = l > 0
            // У нижнего обучаемого блока котангенты по входу никому не нужны:
            // ниже него нет параметров. Вход тогда — константа замыкания, и
            // обратный проход через проекции этого блока по x не считается.
            let needInput = l > first || stemNeeded
            var primals: [MLXArray] = []
            if needInput { primals.append(xin); if hasV { primals.append(vFirst!) } }
            let nData = primals.count
            primals += idx.map { parameters[$0] }
            if primals.isEmpty { trace.blockInput[l] = nil; continue }
            let outputsV = !hasV            // слой 0 сам рождает v_first
            let f: ([MLXArray]) -> [MLXArray] = { ins in
                var current = parameters
                for (j, i) in idx.enumerated() { current[i] = ins[nData + j] }
                inject(current)
                let xi = needInput ? ins[0] : xin
                let vi: MLXArray? = hasV ? (needInput ? ins[1] : vFirst!) : nil
                let (xo, vo) = bb.blockForward(xi, vi, l)
                return outputsV ? [xo, vo] : [xo]
            }
            var cotangents = [dx]
            if outputsV { cotangents.append(dv ?? MLXArray.zeros(vFirst!.shape, dtype: vFirst!.dtype)) }
            let g = vjp(f, primals: primals, cotangents: cotangents).1
            inject(parameters)             // в модели — значения, а не трассеры
            var ready: [MLXArray] = []
            if needInput {
                dx = g[0]
                ready.append(dx)
                if hasV { dv = dv.map { $0 + g[1] } ?? g[1]; ready.append(dv!) }
            }
            for (j, i) in idx.enumerated() { add(i, g[nData + j]); ready.append(grads[i]!) }
            settle(ready)
            trace.blockInput[l] = nil
            probe?("block \(l)")
        }

        if stemNeeded {
            let idx = plan.stem
            let ids = trace.ids
            let g = vjp({ ins in
                var current = parameters
                for (j, i) in idx.enumerated() { current[i] = ins[j] }
                inject(current)
                return [bb.stem(ids)]
            }, primals: idx.map { parameters[$0] }, cotangents: [dx]).1
            inject(parameters)
            for (j, i) in idx.enumerated() { add(i, g[j]) }
            eval(idx.map { grads[$0]! })
            probe?("stem")
        }
    }
}

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
        let T = batch.x.shape[1], B = batch.x.shape[0]
        let plan = SequentialBody.Plan(names: set.parameterNames, nLayer: bb.cfg.nLayer)
        precondition(plan.above.isEmpty && plan.stem.isEmpty && plan.first != nil,
                     "обратный проход по блокам: у LoRA все параметры принадлежат блокам")

        // ── 1. прямой проход без градиента: входы обучаемых блоков ──
        set.inject(parameters)
        let trace = SequentialBody.forward(bb, ids: batch.x, plan: plan)
        let x = trace.top

        // ── 2. ln_out, голова и лосс: по кускам, каждый — свой vjp и eval ──
        let h = bb.finalNorm(x, T: T)
        eval(h)
        let V = bb.cfg.vocab
        let scale = MLXArray(Float(1) / Float(B * T))
        let step = lossPieceTokens(chunk: lossChunk, batch: B, length: T)
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
            SequentialBody.probe?("loss")
            total = total + value[0]
            pieces.append(g[0])
            from = to
        }
        let dh = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
        let dx = vjp({ ins in [bb.finalNorm(ins[0], T: T)] }, primals: [x], cotangents: [dh]).1[0]
        let loss = total * scale
        eval(dx, loss)
        pieces = []

        // ── 3. блоки сверху вниз ──
        var grads = [MLXArray?](repeating: nil, count: parameters.count)
        SequentialBody.backward(bb, trace: trace, dTop: dx, parameters: parameters, plan: plan,
                                inject: { set.inject($0) }, grads: &grads)
        return (loss, grads.enumerated().map { i, g in
            g ?? MLXArray.zeros(parameters[i].shape, dtype: parameters[i].dtype)
        })
    }
}
