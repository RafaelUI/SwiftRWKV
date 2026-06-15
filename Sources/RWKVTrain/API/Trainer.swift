import Foundation
import MLX
import RWKVKernel

/// Параметры файнтюна верхних слоёв + головы.
public struct TrainingConfig: Sendable {
    /// Число замороженных нижних слоёв. Остальные `nLayer - freeze` слоёв
    /// обучаются вместе с головой.
    public var freeze: Int
    /// Длина контекста (паддинг/обрезка). Должна делиться на размер чанка ядра.
    public var contextSize: Int
    public var epochs: Int
    public var batchSize: Int
    public var learningRate: Float
    /// Размер пачки текстов при извлечении граничных фич на диск.
    public var featureBatch: Int
    public var pooling: RWKVPooling

    public init(
        freeze: Int,
        contextSize: Int = 128,
        epochs: Int = 5,
        batchSize: Int = 8,
        learningRate: Float = 1e-4,
        featureBatch: Int = 5,
        pooling: RWKVPooling = .mean
    ) {
        self.freeze = freeze
        self.contextSize = contextSize
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.featureBatch = featureBatch
        self.pooling = pooling
    }
}

/// Результат обучения.
public struct TrainingResult: Sendable {
    /// Точность на валидации (доля верных), 0…1.
    public let validationAccuracy: Float
    /// Обученные тензоры (веса верхних слоёв + `head.weight`/`head.bias`).
    let parameters: [String: MLXArray]
    /// Итоговый конфиг обученной модели (для сохранения).
    public let config: ModelConfig
}

/// Источник сигнала отмены обучения. Потокобезопасен.
public final class CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    public init() {}
    public var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    public func cancel() { lock.lock(); flag = true; lock.unlock() }
}

/// Фасад обучения: частичный файнтюн верхних слоёв RWKV-7 + головы
/// классификатора поверх базовой ``RWKVModel``.
///
/// Пайплайн (как в исходной реализации):
/// 1. извлечение граничных фич замороженной части на диск (mmap-кэш);
/// 2. обучение верхних слоёв + головы по кэшу.
public final class Trainer {
    private let baseModel: RWKVModel
    private let classes: [String]
    private let logger: RWKVLogger

    /// - Parameters:
    ///   - model: базовая (предобученная) модель — папка с config/weights/tokenizer.
    ///   - classes: имена классов в порядке индексов ярлыков.
    ///   - logger: канал логов/метрик (по умолчанию — без вывода).
    public init(model: RWKVModel, classes: [String], logger: RWKVLogger = NoopLogger()) {
        self.baseModel = model
        self.classes = classes
        self.logger = logger
    }

    /// Обучает модель на данных провайдера. Блокирующий вызов — запускайте
    /// на фоновой очереди. Бросает ``RWKVError`` при некорректных входных данных
    /// или ``RWKVError/cancelled`` при отмене.
    public func train(
        data: DataProvider,
        config: TrainingConfig,
        cancellation: CancellationToken = CancellationToken()
    ) throws -> TrainingResult {

        // ── Валидация входа ──
        let rcfg = baseModel.config.rwkvConfig
        guard config.contextSize % WKV7_CHUNK == 0 else {
            throw RWKVError.invalidGeometry(
                reason: "contextSize (\(config.contextSize)) must be divisible by CHUNK (\(WKV7_CHUNK))")
        }
        guard config.freeze >= 0, config.freeze < rcfg.nLayer else {
            throw RWKVError.invalidGeometry(
                reason: "freeze (\(config.freeze)) must be in 0..<nLayer (\(rcfg.nLayer))")
        }

        let train = try data.trainExamples()
        let val = try data.validationExamples()
        guard !train.isEmpty else {
            throw RWKVError.invalidDataset(reason: "training set is empty")
        }
        let numClasses = classes.count
        guard numClasses >= 2 else {
            throw RWKVError.invalidDataset(reason: "need at least 2 classes, got \(numClasses)")
        }

        let backbone = baseModel.backbone
        let isCancelled = { cancellation.isCancelled }

        MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)   // ограничить рост буферного кэша Metal
        PartialFinetune.clearCache()
        defer {
            PartialFinetune.clearCache()             // удалить диск-кэш
            MLX.GPU.clearCache()                     // вернуть буферы Metal в ОС
        }

        // ── 1) Извлечение граничных фич train/val на диск ──
        logger.info("Extracting boundary features (train=\(train.count), val=\(val.count))…")
        let tExtract = Date()
        let tc = PartialFinetune.buildBoundaryCache(
            backbone: backbone, examples: train, freeze: config.freeze, name: "train",
            ctxLen: config.contextSize, batch: config.featureBatch,
            isCancelled: isCancelled,
            progress: { [logger] pct, mb in
                let tps = pct * Double(train.count) * Double(config.contextSize)
                    / max(Date().timeIntervalSince(tExtract), 0.001)
                logger.metric(.extractionProgress, value: pct, step: 0)
                logger.metric(.tokensPerSecond, value: tps, step: 0)
                logger.metric(.peakMemoryMB, value: mb, step: 0)
            })
        if isCancelled() { throw RWKVError.cancelled }

        let vc = PartialFinetune.buildBoundaryCache(
            backbone: backbone, examples: val, freeze: config.freeze, name: "val",
            ctxLen: config.contextSize, batch: config.featureBatch,
            isCancelled: isCancelled)
        if isCancelled() { throw RWKVError.cancelled }

        // ── 2) Обучение верхних слоёв + головы ──
        logger.info("Training layers \(config.freeze)..<\(rcfg.nLayer) + head…")
        let res = PartialFinetune.train(
            backbone: backbone, trainCache: tc, valCache: vc, cfg: rcfg,
            numClasses: numClasses, freeze: config.freeze,
            epochs: config.epochs, batchSize: config.batchSize, lr: config.learningRate,
            isCancelled: isCancelled,
            onStep: { [logger] step, loss, mb in
                logger.metric(.loss, value: Double(loss), step: step)
                logger.metric(.peakMemoryMB, value: mb, step: step)
            },
            onEpoch: { [logger] epoch, acc in
                logger.metric(.epoch, value: Double(epoch), step: epoch)
                logger.metric(.valAccuracy, value: Double(acc), step: epoch)
                logger.info("epoch \(epoch): valAcc=\(acc)")
            })
        if isCancelled() { throw RWKVError.cancelled }

        // ── 3) Итоговый конфиг обученной модели ──
        let base = baseModel.config
        let trainedConfig = ModelConfig(
            arch: base.arch, nLayer: base.nLayer, nEmbd: base.nEmbd,
            headSize: base.headSize, vocab: base.vocab, contextSize: config.contextSize,
            language: base.language, tokenizer: base.tokenizer,
            task: .classification, pooling: config.pooling,
            numClasses: numClasses, classes: classes, freeze: config.freeze,
            parent: base.parent, valAcc: res.valAcc,
            createdAt: Date().timeIntervalSince1970)

        logger.info("Training done. final valAcc=\(res.valAcc)")
        return TrainingResult(validationAccuracy: res.valAcc,
                              parameters: res.params, config: trainedConfig)
    }

    /// Сохраняет обученную модель полным пакетом в `directory`:
    /// `config.json` + `model.safetensors` (веса базы + обученные тензоры) +
    /// `tokenizer.json` (копия из базовой модели).
    public func save(_ result: TrainingResult, to directory: URL) throws {
        let fm = FileManager.default
        try? fm.removeItem(at: directory)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        // веса = база + обученные тензоры (bf16, как база)
        var weights = baseModel.weights
        for (k, v) in result.parameters { weights[k] = v.asType(.bfloat16) }
        eval(Array(weights.values))
        do {
            try MLX.save(arrays: weights,
                         url: directory.appendingPathComponent("model.safetensors"))
        } catch {
            throw RWKVError.weightsLoadFailed(reason: "save failed: \(error)")
        }

        // config.json
        do {
            try JSONEncoder().encode(result.config)
                .write(to: directory.appendingPathComponent("config.json"))
        } catch {
            throw RWKVError.invalidConfig(reason: "encode failed: \(error)")
        }

        // tokenizer.json — копия из базовой модели
        let srcTok = baseModel.directory.appendingPathComponent("tokenizer.json")
        let dstTok = directory.appendingPathComponent("tokenizer.json")
        try? fm.copyItem(at: srcTok, to: dstTok)
    }
}
