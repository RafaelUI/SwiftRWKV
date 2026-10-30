import Foundation
import MLX
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Обучение верхних слоёв языковым лоссом.
//
//  Веса слоёв [from..<nLayer] и ln_out обучаются целиком (fp32-мастер,
//  общий Trainer), остальная модель заморожена. Нужен как «сильный контроль»
//  к обучению начального состояния (gen-run --distill --train top): те же
//  данные, тот же цикл и лосс, но параметров на порядки больше. Тем же
//  способом верхние слои учатся в эмбеддере (EmbeddingTrainable), здесь —
//  с головой языковой модели вместо пулинга.
//
//  Только плотная база: у .rwkvq веса верхних слоёв квантованы, и «обучить
//  их целиком» без декванта не выйдет.
// ───────────────────────────────────────────────────────────────────────

public enum TopLayerFinetune {

    public struct Result: Sendable {
        public let finalLoss: Float
        public let steps: Int
        /// Обученные ключи (слои [from..<nLayer] и ln_out), в порядке сортировки.
        public let keys: [String]
        /// Отпечаток базы ДО обучения — его и пишет `save`: после commit веса
        /// уже другие, и отпечаток обученной модели с исходной базой не совпал бы.
        public let baseFingerprint: String
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case notTopLayers(String)
        case otherBase(file: String, model: String)
        case shape(key: String, file: [Int], model: [Int])
        case unknownKey(String)
        public var description: String {
            switch self {
            case .notTopLayers(let f): return "\(f): не файл верхних слоёв (нет ключа \(TopLayerFinetune.formatKey))"
            case .otherBase(let f, let m):
                return "верхние слои обучены на другой базе (base_fingerprint): \(f.prefix(12))… в файле, \(m.prefix(12))… у модели"
            case .shape(let k, let f, let m): return "\(k): форма \(f) в файле, \(m) у модели"
            case .unknownKey(let k): return "\(k): такого веса в модели нет"
            }
        }
    }

    public static let formatKey = "swiftrwkv_top_layers"

    /// Обучить верхние слои начиная с `from` (0 ≤ from < nLayer). Веса в модели
    /// по окончании — обученные (Trainer коммитит мастер в тип модели).
    @discardableResult
    public static func run(
        _ bb: X070Backbone,
        from: Int,
        nextBatch: () -> LoRABatch,
        config cfg: LoRAConfig,
        isCancelled: () -> Bool = { false },
        onStep: (_ step: Int, _ loss: Float, _ gradNorm: Float, _ peakMB: Double) -> Void = { _, _, _, _ in }
    ) -> Result {
        precondition(from >= 0 && from < bb.cfg.nLayer, "from \(from): слоёв \(bb.cfg.nLayer)")
        precondition(!bb.w.isEmpty, "нужна плотная база (веса в bb.w)")
        let savedLayers = bb.trainLayers, savedCheckpoint = bb.useBlockCheckpoint
        bb.trainLayers = Set(from ..< bb.cfg.nLayer)
        // Обратный проход по блокам (sequentialGradients) принимает только параметры блоков, а здесь
        // учится и ln_out — поэтому всегда обычный valueAndGrad: градиент идёт лишь через верхние слои,
        // нижние в графе без градиента, так что память и так невелика. cfg.sequentialBackward не действует.
        bb.useBlockCheckpoint = cfg.useBlockCheckpoint
        defer { bb.trainLayers = savedLayers; bb.useBlockCheckpoint = savedCheckpoint }

        let fingerprint = bb.baseFingerprint
        let keys = BackboneWeightsTrainableSet.topLayerKeys(bb, from: from)
        let set = BackboneWeightsTrainableSet(bb, keys: keys)
        var checked = false
        let res = withoutActuallyEscaping(nextBatch) { escapingNext -> TrainingResult in
            let source: () -> LoRABatch = {
                let b = escapingNext()
                if !checked {
                    precondition(b.x.shape[1] % WKV7_CHUNK == 0,
                                 "T (\(b.x.shape[1])) должна делиться на CHUNK \(WKV7_CHUNK)")
                    checked = true
                }
                return b
            }
            let trainer = Trainer<LoRABatch>(
                trainable: set,
                objective: { LoRAFinetune.languageModelLoss(bb, $0, chunk: cfg.lossChunkTokens) },
                nextBatch: source,
                config: cfg.training)
            return trainer.run(isCancelled: isCancelled) { s in
                onStep(s.step, s.loss, s.gradNorm, s.peakMemoryMB)
            }
        }
        return Result(finalLoss: res.finalLoss, steps: res.steps, keys: keys, baseFingerprint: fingerprint)
    }

    /// Сохранить обученные веса (`result.keys`) с отпечатком исходной базы.
    public static func save(_ bb: X070Backbone, result: Result, to url: URL, note: String = "") throws {
        var d: [String: MLXArray] = [:]
        for k in result.keys {
            guard let v = bb.w[k] else { throw Error.unknownKey(k) }
            d[k] = v
        }
        eval(Array(d.values))
        try MLX.save(arrays: d, metadata: [formatKey: "1", StateFile.baseContractKey: result.baseFingerprint, "note": note],
                     url: url)
    }

    /// Подставить сохранённые веса в модель (тип — как у модели). База
    /// сверяется по отпечатку: верхние слои — поправка к ЭТИМ нижним слоям.
    @discardableResult
    public static func load(_ bb: X070Backbone, from url: URL, allowOtherBase: Bool = false) throws -> [String] {
        let (w, md) = try loadArraysAndMetadata(url: url)
        guard md[formatKey] != nil else { throw Error.notTopLayers(url.lastPathComponent) }
        if !allowOtherBase, let inFile = md[StateFile.baseContractKey] {
            let here = bb.baseFingerprint
            if inFile != here { throw Error.otherBase(file: inFile, model: here) }
        }
        for (k, v) in w {
            guard let old = bb.w[k] else { throw Error.unknownKey(k) }
            guard old.shape == v.shape else { throw Error.shape(key: k, file: v.shape, model: old.shape) }
        }
        for (k, v) in w { bb.w[k] = v.asType(bb.w[k]!.dtype) }
        eval(Array(bb.w.values))
        return w.keys.sorted()
    }
}
