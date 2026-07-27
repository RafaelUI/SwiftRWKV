import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Общий цикл обучения: AdamW, расписание LR, grad-accumulation, клип по
//  глобальной норме, чекпоинты. Ничего не знает ни о модели, ни о лоссе.
//
//  Три оси задаются снаружи:
//    • TrainableSet — что дифференцируется;
//    • objective    — чем меряется (замыкание batch -> скаляр);
//    • nextBatch    — откуда берутся данные.
//
//  Порядок операций на шаге зафиксирован и воспроизводит дорефакторный
//  LoRA-путь БИТ-В-БИТ (это проверяется характеризационными тестами):
//     накопить микрошаги → усреднить → клип по глобальной норме → AdamW.
//  Клип ПОСЛЕ усреднения, а не до: иначе порог означал бы разное при разном
//  gradAccum.
//
//  Почему AdamW руками, а не готовый оптимизатор: он обязан работать с
//  плоским [MLXArray] и fp32-мастером, поверх которого в forward идёт
//  приведённая копия (см. TrainableSet.inject).
// ───────────────────────────────────────────────────────────────────────

/// Чем считается пара (лосс, градиенты) на одном микробатче.
///
/// Существует, потому что не всякая задача укладывается в один
/// `valueAndGrad`: GradCache считает лосс на ПОЛНОМ батче векторов, а
/// градиенты — тремя фазами по чанкам, и снаружи это по-прежнему просто
/// «лосс и градиенты в порядке параметров».
///
/// Контракт: реализация ОБЯЗАНА вызвать `TrainableSet.inject` внутри своего
/// grad-замыкания. Подстановка снаружи разорвала бы цепь к fp32-мастеру —
/// градиенты пришли бы нулевыми, и притом молча.
public typealias GradientProvider<Batch> =
    (_ parameters: [MLXArray], _ batch: Batch) -> (loss: MLXArray, gradients: [MLXArray])

public final class Trainer<Batch> {

    private let trainable: TrainableSet
    private let objective: (Batch) -> MLXArray
    private let nextBatch: () -> Batch
    private let cfg: TrainingConfig
    /// nil ⇒ встроенный valueAndGrad поверх `objective` (обычный путь).
    private let gradient: GradientProvider<Batch>?

    /// Текущий микробатч для grad-замыкания.
    ///
    /// Свойство ЭКЗЕМПЛЯРА, а не file-private глобаль (как было в
    /// LoRAFinetune): глобаль означала, что два тренера в одном процессе
    /// молча затирают батчи друг друга. Здесь каждый тренер изолирован.
    private var current: Batch?

    // Состояние оптимизатора
    private var params: [MLXArray] = []
    private var m: [MLXArray] = []
    private var v: [MLXArray] = []
    private var t = 0
    private var startStep = 0

    /// - gradient: подмена способа получить градиент (см. GradientProvider).
    ///   По умолчанию nil — тогда `objective` дифференцируется обычным
    ///   `valueAndGrad`, и путь остаётся ровно тем, что был до появления
    ///   этого параметра (проверяется характеризационными тестами).
    ///   Когда провайдер задан, `objective` не вызывается вообще.
    public init(trainable: TrainableSet,
                objective: @escaping (Batch) -> MLXArray,
                nextBatch: @escaping () -> Batch,
                config: TrainingConfig,
                gradient: GradientProvider<Batch>? = nil) {
        self.trainable = trainable
        self.objective = objective
        self.nextBatch = nextBatch
        self.cfg = config
        self.gradient = gradient
    }

    // ── Цикл ─────────────────────────────────────────────────────────

    @discardableResult
    public func run(isCancelled: () -> Bool = { false },
                    onStep: (TrainingStep) -> Void = { _ in }) -> TrainingResult {

        if cfg.cacheLimitGB > 0 {
            MLX.GPU.set(cacheLimit: Int(cfg.cacheLimitGB * 1e9))
        }

        if params.isEmpty {
            params = trainable.initialParameters()
            m = params.map { MLXArray.zeros($0.shape, dtype: .float32) }
            v = params.map { MLXArray.zeros($0.shape, dtype: .float32) }
        }

        let builtIn = valueAndGrad({ [unowned self] (ps: [MLXArray]) -> [MLXArray] in
            self.trainable.inject(ps)
            return [self.objective(self.current!).asType(.float32)]
        }, argumentNumbers: Array(params.indices))

        // Единая точка вызова: дальше цикл не знает, откуда взялся градиент.
        // Форма ([лосс], градиенты) сохранена ради того, чтобы порядок
        // операций ниже (eval → накопление → усреднение → клип → AdamW)
        // остался буква в букву прежним.
        let vg: ([MLXArray]) -> ([MLXArray], [MLXArray]) = { [unowned self] ps in
            guard let provider = self.gradient else { return builtIn(ps) }
            let r = provider(ps, self.current!)
            return ([r.loss.asType(.float32)], r.gradients)
        }

        var lastLoss: Float = .nan
        var step = startStep

        while step < cfg.maxSteps {
            if isCancelled() { break }
            let lr = cfg.learningRate(at: step)

            // ── накопление градиентов ──
            current = nextBatch()
            var (vals, grads) = vg(params)
            eval(vals + grads)
            var lossVal = vals[0]

            for _ in 1 ..< cfg.gradAccum {
                current = nextBatch()
                let (vi, gi) = vg(params)
                eval(vi + gi)
                lossVal = lossVal + vi[0]
                for i in grads.indices { grads[i] = grads[i] + gi[i] }
                eval(grads)
            }
            if cfg.gradAccum > 1 {
                let inv = 1.0 / Float(cfg.gradAccum)
                grads = grads.map { $0 * inv }
                lossVal = lossVal * inv
            }

            // ── клип по глобальной норме ──
            let (clipped, norm) = clipByGlobalNorm(grads)

            // ── AdamW (decoupled weight decay) ──
            t += 1
            let c1 = 1 - Float(pow(Double(cfg.beta1), Double(t)))
            let c2 = 1 - Float(pow(Double(cfg.beta2), Double(t)))
            for i in params.indices {
                m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * clipped[i]
                v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * (clipped[i] * clipped[i])
                let mhat = m[i] / c1, vhat = v[i] / c2
                var upd = mhat / (sqrt(vhat) + cfg.adamEps)
                if cfg.weightDecay > 0 { upd = upd + cfg.weightDecay * params[i] }
                params[i] = params[i] - lr * upd
            }
            eval(params + m + v)

            lastLoss = lossVal.item(Float.self)
            step += 1
            if cfg.logEvery > 0 && (step % cfg.logEvery == 0 || step == cfg.maxSteps) {
                onStep(TrainingStep(step: step, loss: lastLoss,
                                    gradNorm: norm.item(Float.self),
                                    learningRate: lr,
                                    peakMemoryMB: residentMemoryMB()))
            }
        }

        trainable.commit(params)
        startStep = step
        return TrainingResult(finalLoss: lastLoss, steps: step)
    }

    /// Обрезка по глобальной норме; возвращает (обрезанные, исходная норма).
    /// gradClip <= 0 ⇒ без обрезки, но норма всё равно считается и сообщается.
    private func clipByGlobalNorm(_ grads: [MLXArray]) -> ([MLXArray], MLXArray) {
        var sq = MLXArray(Float(0))
        for g in grads { sq = sq + (g * g).sum() }
        let norm = sqrt(sq)
        guard cfg.gradClip > 0 else { return (grads, norm) }
        let s = minimum(MLXArray(Float(1)), MLXArray(cfg.gradClip) / (norm + 1e-6))
        return (grads.map { $0 * s }, norm)
    }

    // ── Чекпоинты ────────────────────────────────────────────────────

    /// Сохранить параметры, моменты Adam и счётчики. Без моментов возобновление
    /// даёт всплеск лосса на первых шагах: Adam стартует с нулевой истории.
    public func saveCheckpoint(to url: URL) throws {
        var d: [String: MLXArray] = [:]
        let names = trainable.parameterNames
        for (i, p) in params.enumerated() {
            let key = i < names.count ? names[i] : "p\(i)"
            d["param/" + key] = p
            d["m/" + key] = m[i]
            d["v/" + key] = v[i]
        }
        d["_meta/step"] = MLXArray([Int32(startStep), Int32(t)])
        eval(Array(d.values))
        try MLX.save(arrays: d, url: url)
    }

    /// Восстановить параметры, моменты и счётчики. Порядок берётся из
    /// parameterNames — тот же, что при сохранении.
    public func loadCheckpoint(from url: URL) throws {
        let d = try loadArrays(url: url)
        let names = trainable.parameterNames
        var ps: [MLXArray] = [], ms: [MLXArray] = [], vs: [MLXArray] = []
        for (i, name) in names.enumerated() {
            guard let p = d["param/" + name] else {
                throw TrainerError.checkpointMissing("param/" + name)
            }
            ps.append(p.asType(.float32))
            ms.append(d["m/" + name]?.asType(.float32)
                      ?? MLXArray.zeros(p.shape, dtype: .float32))
            vs.append(d["v/" + name]?.asType(.float32)
                      ?? MLXArray.zeros(p.shape, dtype: .float32))
            _ = i
        }
        params = ps; m = ms; v = vs
        if let meta = d["_meta/step"] {
            eval(meta)
            startStep = Int(meta[0].item(Int32.self))
            t = Int(meta[1].item(Int32.self))
        }
        eval(params + m + v)
        trainable.inject(params)
    }

    /// Текущие параметры (fp32-мастер) — для диагностики и тестов.
    public var currentParameters: [MLXArray] { params }
    public var completedSteps: Int { startStep }
}

public enum TrainerError: Error, CustomStringConvertible {
    case checkpointMissing(String)

    public var description: String {
        switch self {
        case .checkpointMissing(let k): return "в чекпоинте нет тензора \(k)"
        }
    }
}
