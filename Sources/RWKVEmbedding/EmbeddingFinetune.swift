import Foundation
import MLX
import RWKVGen
import RWKVKernel   // WKV7_CHUNK

// ───────────────────────────────────────────────────────────────────────
//  Дообучение эмбеддингов: одна стадия и curriculum из стадий.
//
//  Порт связки rwkv_metal/embedding/train.py + tools/run_embedding_curriculum.py,
//  но БЕЗ собственного цикла обучения: цикл общий (RWKVGen.Trainer), здесь
//  остаётся только специфика задачи — какое множество обучается, каким лоссом
//  меряется, откуда берутся батчи и чем оценивается результат.
//
//  Curriculum — это не отдельный механизм, а тот же код-путь, вызванный по
//  разу на задачу над одной и той же моделью. Ровно так устроен sft_curriculum
//  у EmbeddingRWKV, и держать под это второй цикл незачем.
//
//  Оптимизатор между стадиями НЕ переносится: каждая стадия поднимает свой
//  Trainer, то есть свои моменты Adam с нуля. Это осознанно и совпадает с
//  питоновским поведением (новый AdamW на каждый вызов finetune_embedding):
//  у стадий разные лоссы, и импульс, накопленный на retrieval, для sts —
//  история чужой задачи, а не полезная инерция.
// ───────────────────────────────────────────────────────────────────────

/// Результат оценки — своя форма у каждой задачи.
public enum EmbeddingEvaluation: Sendable {
    case ranking(RankingMetrics)
    case pairwise(PairwiseMetrics)
    case accuracy(AccuracyMetrics)

    public var description: String {
        switch self {
        case .ranking(let m):  return m.description
        case .pairwise(let m): return m.description
        case .accuracy(let m): return m.description
        }
    }

    /// Одно число, по которому стадии сравнимы между собой: MRR / точность.
    public var headline: Double {
        switch self {
        case .ranking(let m):  return m.mrr
        case .pairwise(let m): return m.accuracy
        case .accuracy(let m): return m.accuracy
        }
    }
}

/// Одна стадия curriculum.
public struct EmbeddingStage {
    public var task: EmbeddingTask
    /// Обучающие строки. Должны быть той же задачи, что `task`.
    public var train: [EmbeddingSample]
    /// Отложенные строки для оценки до/после. Пусто ⇒ оценка пропускается.
    public var heldOut: [EmbeddingSample]
    public var config: TrainingConfig
    public var batchSize: Int
    /// >0 ⇒ триплетные стадии идут под GradCache с чанком такого размера.
    /// Для классификации игнорируется (см. EmbeddingObjective).
    public var gradCacheChunk: Int
    public var temperature: Float
    public var maxTokens: Int?
    public var maxChars: Int
    public var seed: UInt64
    /// Оценивать классификацию на полном пуле из 25 меток.
    public var evaluateOnFullPool: Bool

    public init(task: EmbeddingTask,
                train: [EmbeddingSample],
                heldOut: [EmbeddingSample] = [],
                config: TrainingConfig = TrainingConfig(lr: 2e-5, lrMin: 1e-6,
                                                        schedule: .cosine,
                                                        warmupSteps: 100,
                                                        maxSteps: 1500),
                batchSize: Int = 8,
                gradCacheChunk: Int = 0,
                temperature: Float = 0.05,
                maxTokens: Int? = 512,
                maxChars: Int = 800,
                seed: UInt64 = 0,
                evaluateOnFullPool: Bool = true) {
        self.task = task
        self.train = train
        self.heldOut = heldOut
        self.config = config
        self.batchSize = batchSize
        self.gradCacheChunk = gradCacheChunk
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.maxChars = maxChars
        self.seed = seed
        self.evaluateOnFullPool = evaluateOnFullPool
    }
}

public struct StageResult: Sendable {
    public let task: EmbeddingTask
    public let before: EmbeddingEvaluation?
    public let after: EmbeddingEvaluation?
    public let finalLoss: Float
    public let steps: Int
    public let seconds: Double

    /// Изменение ведущей метрики. nil, если оценки не было.
    public var delta: Double? {
        guard let b = before, let a = after else { return nil }
        return a.headline - b.headline
    }
}

public enum EmbeddingFinetuneError: Error, CustomStringConvertible {
    case emptyStage(EmbeddingTask)
    case taskMismatch(EmbeddingTask, Int)
    case vocabOverflow(maxId: Int, vocab: Int)

    public var description: String {
        switch self {
        case .emptyStage(let t):
            return "стадия \(t.rawValue): нет обучающих строк"
        case .taskMismatch(let t, let n):
            return "стадия \(t.rawValue): \(n) строк относятся к другой задаче"
        case .vocabOverflow(let maxId, let vocab):
            return "токенизатор выдал id \(maxId) при словаре модели \(vocab) — "
                 + "модель и токенизатор от разных моделей"
        }
    }
}

public enum EmbeddingFinetune {

    /// Прогнать одну стадию: оценка до, обучение, оценка после.
    @discardableResult
    public static func runStage(model: EmbeddingModel,
                                tokenizer: WorldTokenizer,
                                trainable: TrainableSet,
                                stage: EmbeddingStage,
                                terminator: Int = 0,
                                isCancelled: () -> Bool = { false },
                                onStep: (TrainingStep) -> Void = { _ in }) throws -> StageResult {

        guard !stage.train.isEmpty else { throw EmbeddingFinetuneError.emptyStage(stage.task) }
        let wrong = stage.train.filter { $0.task != stage.task }.count
        guard wrong == 0 else { throw EmbeddingFinetuneError.taskMismatch(stage.task, wrong) }

        // Один раз на стадию: не выдаёт ли токенизатор id за пределами словаря
        // модели. Выборка из нескольких строк — не доказательство, но ловит
        // единственный реальный сценарий: модель и токенизатор от РАЗНЫХ
        // моделей (например World-токенизатор к модели со своим BPE). Без
        // проверки выборка из таблицы эмбеддингов уходит за границу, лосс
        // становится NaN, и обучение молча крутится вхолостую до конца прогона.
        // Стоимость — одна синхронизация на стадию.
        try checkVocab(model: model, tokenizer: tokenizer, stage: stage,
                       terminator: terminator)

        let before = stage.heldOut.isEmpty ? nil
            : evaluate(model: model, tokenizer: tokenizer, stage: stage, terminator: terminator)

        let t0 = Date()
        let result = train(model: model, tokenizer: tokenizer, trainable: trainable,
                           stage: stage, terminator: terminator,
                           isCancelled: isCancelled, onStep: onStep)
        let seconds = Date().timeIntervalSince(t0)

        let after = stage.heldOut.isEmpty ? nil
            : evaluate(model: model, tokenizer: tokenizer, stage: stage, terminator: terminator)

        return StageResult(task: stage.task, before: before, after: after,
                           finalLoss: result.finalLoss, steps: result.steps,
                           seconds: seconds)
    }

    /// Curriculum: стадии подряд над одной моделью, обучаемое множество общее.
    @discardableResult
    public static func run(model: EmbeddingModel,
                           tokenizer: WorldTokenizer,
                           mode: BaseTrainingMode,
                           stages: [EmbeddingStage],
                           terminator: Int = 0,
                           isCancelled: () -> Bool = { false },
                           onStage: (StageResult) -> Void = { _ in },
                           onStep: (EmbeddingTask, TrainingStep) -> Void = { _, _ in }) throws -> [StageResult] {

        // Множество собирается ОДИН раз на весь curriculum: у него внутри
        // fp32-мастер параметров, и пересборка на каждой стадии means взять
        // мастер заново из модели — то есть потерять точность, накопленную
        // предыдущей стадией, откатив её к bf16-представлению в весах.
        let trainable = EmbeddingTrainable.make(model: model, mode: mode)

        var out: [StageResult] = []
        for stage in stages {
            if isCancelled() { break }
            let r = try runStage(model: model, tokenizer: tokenizer,
                                 trainable: trainable, stage: stage,
                                 terminator: terminator, isCancelled: isCancelled,
                                 onStep: { onStep(stage.task, $0) })
            out.append(r)
            onStage(r)
        }
        return out
    }

    static func checkVocab(model: EmbeddingModel, tokenizer: WorldTokenizer,
                           stage: EmbeddingStage, terminator: Int) throws {
        let probe = stage.train.prefix(4).flatMap {
            [String($0.anchor.prefix(stage.maxChars)),
             String($0.positive.prefix(stage.maxChars)),
             String($0.negative.prefix(stage.maxChars))]
        }
        guard !probe.isEmpty else { return }
        let (idx, _) = encodeBatch(tokenizer: tokenizer, texts: probe,
                                   terminator: terminator, maxTokens: stage.maxTokens)
        let maxId = Int(idx.max().item(Int32.self))
        guard maxId < model.backbone.cfg.vocab else {
            throw EmbeddingFinetuneError.vocabOverflow(maxId: maxId,
                                                       vocab: model.backbone.cfg.vocab)
        }
    }

    // ── Обучение одной стадии ────────────────────────────────────────

    private static func train(model: EmbeddingModel, tokenizer: WorldTokenizer,
                              trainable: TrainableSet, stage: EmbeddingStage,
                              terminator: Int,
                              isCancelled: () -> Bool,
                              onStep: (TrainingStep) -> Void) -> TrainingResult {

        // Выравнивание T нужно ровно тогда, когда через дифференцируемое
        // ядро идёт хоть один слой. При замороженной базе (обучается одна
        // голова) forward-ядро работает при любой длине, и добивать батч до
        // кратности 16 значило бы считать лишние токены задаром.
        let pad = model.backbone.trainLayers.isEmpty ? nil : WKV7_CHUNK
        let truncated = stage.train.map {
            EmbeddingSample(anchor: String($0.anchor.prefix(stage.maxChars)),
                            positive: String($0.positive.prefix(stage.maxChars)),
                            negative: String($0.negative.prefix(stage.maxChars)),
                            task: $0.task)
        }

        switch stage.task {
        case .retrieval, .sts:
            let symmetric = stage.task == .sts
            var batcher = TripletBatcher(samples: truncated, tokenizer: tokenizer,
                                         batchSize: stage.batchSize,
                                         terminator: terminator,
                                         maxTokens: stage.maxTokens,
                                         seed: stage.seed, padMultiple: pad)
            let provider: GradientProvider<TripletBatch>? =
                stage.gradCacheChunk > 0
                ? EmbeddingObjective.gradCacheProvider(
                    model: model, trainable: trainable,
                    chunkSize: stage.gradCacheChunk,
                    temperature: stage.temperature, symmetric: symmetric)
                : nil

            let trainer = Trainer<TripletBatch>(
                trainable: trainable,
                objective: { EmbeddingObjective.tripletLoss(model, $0,
                                                            temperature: stage.temperature,
                                                            symmetric: symmetric) },
                nextBatch: { batcher.next() },
                config: stage.config,
                gradient: provider)
            return trainer.run(isCancelled: isCancelled, onStep: onStep)

        case .classification:
            var batcher = ClassificationBatcher(samples: truncated, tokenizer: tokenizer,
                                                batchSize: stage.batchSize,
                                                terminator: terminator,
                                                maxTokens: stage.maxTokens,
                                                seed: stage.seed,
                                                useFullPool: false,
                                                padMultiple: pad)
            let trainer = Trainer<ClassificationBatch>(
                trainable: trainable,
                objective: { EmbeddingObjective.classificationLoss(model, $0,
                                                                   temperature: stage.temperature) },
                nextBatch: { batcher.next() },
                config: stage.config)
            return trainer.run(isCancelled: isCancelled, onStep: onStep)
        }
    }

    // ── Оценка ───────────────────────────────────────────────────────

    public static func evaluate(model: EmbeddingModel, tokenizer: WorldTokenizer,
                                stage: EmbeddingStage,
                                terminator: Int = 0) -> EmbeddingEvaluation {
        switch stage.task {
        case .retrieval:
            return .ranking(EmbeddingMetrics.evaluateRetrieval(
                model: model, tokenizer: tokenizer, rows: stage.heldOut,
                maxChars: stage.maxChars, terminator: terminator,
                maxTokens: stage.maxTokens))
        case .sts:
            return .pairwise(EmbeddingMetrics.evaluateSTS(
                model: model, tokenizer: tokenizer, rows: stage.heldOut,
                maxChars: stage.maxChars, terminator: terminator,
                maxTokens: stage.maxTokens))
        case .classification:
            return .accuracy(EmbeddingMetrics.evaluateClassification(
                model: model, tokenizer: tokenizer, rows: stage.heldOut,
                maxChars: stage.maxChars, useFullPool: stage.evaluateOnFullPool,
                terminator: terminator, maxTokens: stage.maxTokens))
        }
    }
}
