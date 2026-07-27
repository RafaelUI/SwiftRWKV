import Foundation
import MLX
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Предобучение RWKV-7 с нуля.
//
//  Это не отдельный тренировочный движок, а конфигурация общего:
//    • обучаемое множество — ВСЕ веса backbone (BackboneWeightsTrainableSet);
//    • лосс — LM cross-entropy (тот же, что у LoRA-пути);
//    • источник — поток токенов из .bin.
//  Плюс то, чего нет у дообучения: инициализация с нуля (X070Init),
//  косинусный спад LR и возобновление с чекпоинта.
//
//  Первоначальная гипотеза была «предобучение = обучение верхних N слоёв,
//  где N = все». Это неверно: X070PartialFinetune — классификационный путь
//  с дисковым кэшем границы, который при freeze = 0 вырождается. Донором
//  оказался LM-цикл LoRA-пути, а после выноса общего тренера — сам тренер.
// ───────────────────────────────────────────────────────────────────────

public enum PretrainError: Error, CustomStringConvertible {
    case outOfVocabulary(maxToken: Int, vocabSize: Int, issues: [String])
    case badContextLength(Int)

    public var description: String {
        switch self {
        case .outOfVocabulary(let maxToken, let vocabSize, let issues):
            return "в данных есть токены вне словаря (max=\(maxToken), "
                 + "vocabSize=\(vocabSize)). Это даст NaN на обучении. "
                 + "Исправь vocabSize или перетокенизируй.\n" + issues.joined(separator: "\n")
        case .badContextLength(let t):
            return "ctxLen (\(t)) должна делиться на CHUNK \(WKV7_CHUNK): "
                 + "обучаемые слои идут через дифференцируемое WKV-ядро"
        }
    }
}

public struct PretrainConfig: Sendable {

    // ── Архитектура ──────────────────────────────────────────────────
    public var nLayer: Int
    public var nEmbd: Int
    public var headSize: Int
    public var vocab: Int

    // ── Данные ───────────────────────────────────────────────────────
    public var trainData: String
    public var valData: String?
    public var ctxLen: Int
    public var batchSize: Int

    // ── Сколько учить ────────────────────────────────────────────────
    /// Задаётся ЛИБО шагами, либо токенами; maxSteps имеет приоритет.
    public var maxSteps: Int?
    public var maxTokens: Int?

    // ── Оптимизатор ──────────────────────────────────────────────────
    public var lr: Float
    public var lrMin: Float
    public var schedule: LRSchedule
    public var warmupSteps: Int
    public var gradAccum: Int
    public var gradClip: Float
    public var weightDecay: Float
    public var beta1: Float
    public var beta2: Float
    public var adamEps: Float

    // ── Железо и память ──────────────────────────────────────────────
    public var computeDType: DType
    public var useBlockCheckpoint: Bool
    public var cacheLimitGB: Double

    // ── Валидация, чекпоинты, логи ───────────────────────────────────
    public var evalEvery: Int
    public var evalBatches: Int
    public var checkpointDir: String?
    public var saveEvery: Int
    public var resume: Bool
    public var logEvery: Int
    public var initSeed: UInt64

    public init(nLayer: Int = 12, nEmbd: Int = 768, headSize: Int = 64,
                vocab: Int = 65536,
                trainData: String = "data/train.bin", valData: String? = nil,
                ctxLen: Int = 512, batchSize: Int = 8,
                maxSteps: Int? = nil, maxTokens: Int? = 3_000_000_000,
                lr: Float = 1.5e-3, lrMin: Float = 1e-4,
                schedule: LRSchedule = .cosine, warmupSteps: Int = 200,
                gradAccum: Int = 1, gradClip: Float = 1.0,
                weightDecay: Float = 0.0, beta1: Float = 0.9, beta2: Float = 0.95,
                adamEps: Float = 1e-18,
                computeDType: DType = .bfloat16, useBlockCheckpoint: Bool = false,
                cacheLimitGB: Double = 1.5,
                evalEvery: Int = 500, evalBatches: Int = 20,
                checkpointDir: String? = nil, saveEvery: Int = 500,
                resume: Bool = true, logEvery: Int = 50, initSeed: UInt64 = 0) {
        self.nLayer = nLayer; self.nEmbd = nEmbd; self.headSize = headSize
        self.vocab = vocab
        self.trainData = trainData; self.valData = valData
        self.ctxLen = ctxLen; self.batchSize = batchSize
        self.maxSteps = maxSteps; self.maxTokens = maxTokens
        self.lr = lr; self.lrMin = lrMin; self.schedule = schedule
        self.warmupSteps = warmupSteps; self.gradAccum = gradAccum
        self.gradClip = gradClip; self.weightDecay = weightDecay
        self.beta1 = beta1; self.beta2 = beta2; self.adamEps = adamEps
        self.computeDType = computeDType
        self.useBlockCheckpoint = useBlockCheckpoint
        self.cacheLimitGB = cacheLimitGB
        self.evalEvery = evalEvery; self.evalBatches = evalBatches
        self.checkpointDir = checkpointDir; self.saveEvery = saveEvery
        self.resume = resume; self.logEvery = logEvery; self.initSeed = initSeed
    }

    public var modelConfig: X070Config {
        X070Config(nLayer: nLayer, nEmbd: nEmbd, headSize: headSize, vocab: vocab)
    }

    /// Явные шаги либо пересчёт из бюджета токенов.
    public func resolvedMaxSteps() -> Int {
        if let s = maxSteps { return s }
        if let t = maxTokens {
            let perStep = batchSize * ctxLen * gradAccum
            return Int((Double(t) / Double(perStep)).rounded(.up))
        }
        return 1000
    }

    var training: TrainingConfig {
        TrainingConfig(lr: lr, lrMin: lrMin, schedule: schedule,
                       warmupSteps: warmupSteps, gradClip: gradClip,
                       weightDecay: weightDecay, beta1: beta1, beta2: beta2,
                       adamEps: adamEps, maxSteps: resolvedMaxSteps(),
                       gradAccum: gradAccum, cacheLimitGB: cacheLimitGB,
                       logEvery: logEvery)
    }
}

public struct PretrainResult: Sendable {
    public let finalLoss: Float
    public let bestValLoss: Float?
    public let steps: Int
}

public enum Pretrain {

    /// Предобучение с нуля (или продолжение с чекпоинта).
    ///
    /// - onStep: прогресс обучения.
    /// - onEval: (шаг, val-лосс) на каждой валидации.
    @discardableResult
    public static func run(
        config cfg: PretrainConfig,
        isCancelled: () -> Bool = { false },
        onStep: (TrainingStep) -> Void = { _ in },
        onEval: (_ step: Int, _ valLoss: Float) -> Void = { _, _ in }
    ) throws -> PretrainResult {

        guard cfg.ctxLen % WKV7_CHUNK == 0 else {
            throw PretrainError.badContextLength(cfg.ctxLen)
        }

        // ── Данные ───────────────────────────────────────────────────
        let train = try BinTokenStream(path: cfg.trainData, ctxLen: cfg.ctxLen)
        try train.validateOrThrow(vocabSize: cfg.vocab)
        let val = try cfg.valData.map { try BinTokenStream(path: $0, ctxLen: cfg.ctxLen) }
        try val?.validateOrThrow(vocabSize: cfg.vocab)

        // ── Модель ───────────────────────────────────────────────────
        let model = cfg.modelConfig
        let bb = X070Init.makeBackbone(cfg: model,
                                       init: X070InitConfig(seed: cfg.initSeed),
                                       computeDType: cfg.computeDType)
        bb.useBlockCheckpoint = cfg.useBlockCheckpoint
        // Все слои обучаются ⇒ все идут через дифференцируемое WKV-ядро.
        bb.trainLayers = Set(0 ..< model.nLayer)

        let trainable = BackboneWeightsTrainableSet(bb, keys: Array(bb.w.keys))

        // ── Тренер ───────────────────────────────────────────────────
        let source = train.source(batchSize: cfg.batchSize)
        let trainer = Trainer<LoRABatch>(
            trainable: trainable,
            objective: { LoRAFinetune.languageModelLoss(bb, $0) },
            nextBatch: source,
            config: cfg.training)

        let ckptURL = cfg.checkpointDir.map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
                .appendingPathComponent("rwkv7_\(cfg.nLayer)l\(cfg.nEmbd)d.safetensors")
        }
        if cfg.resume, let u = ckptURL, FileManager.default.fileExists(atPath: u.path) {
            try trainer.loadCheckpoint(from: u)
        }

        // ── Валидация ────────────────────────────────────────────────
        // Считается ВНЕ grad-тейпа и по фиксированным позициям: val-лосс
        // должен быть сравним между эпохами, а не плавать вместе с данными.
        var bestVal: Float? = nil
        func evaluate() -> Float {
            var total: Float = 0
            guard let val else { return .nan }
            for i in 0 ..< cfg.evalBatches {
                let b = val.batch(batchSize: cfg.batchSize, step: i)
                let l = LoRAFinetune.languageModelLoss(bb, b)
                eval(l)
                total += l.item(Float.self)
            }
            return total / Float(cfg.evalBatches)
        }

        var lastStep = 0
        let res = trainer.run(isCancelled: isCancelled) { s in
            onStep(s)
            lastStep = s.step
            if cfg.evalEvery > 0 && s.step % cfg.evalEvery == 0 && val != nil {
                let v = evaluate()
                onEval(s.step, v)
                if bestVal == nil || v < bestVal! { bestVal = v }
            }
            if let u = ckptURL, cfg.saveEvery > 0, s.step % cfg.saveEvery == 0 {
                try? FileManager.default.createDirectory(
                    at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? trainer.saveCheckpoint(to: u)
            }
        }

        if let u = ckptURL {
            try? FileManager.default.createDirectory(
                at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try trainer.saveCheckpoint(to: u)
        }
        _ = lastStep
        return PretrainResult(finalLoss: res.finalLoss, bestValLoss: bestVal,
                              steps: res.steps)
    }
}
