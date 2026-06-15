import Foundation
import MLX
import MLXNN

/// Результат классификации одного текста.
public struct Prediction: Sendable {
    /// Имя класса-победителя.
    public let label: String
    /// Все классы с вероятностями, отсортированы по убыванию.
    public let probabilities: [(label: String, probability: Float)]

    public init(label: String, probabilities: [(label: String, probability: Float)]) {
        self.label = label
        self.probabilities = probabilities
    }
}

/// Инференс-классификатор поверх обученной ``RWKVModel``.
///
/// Модель должна содержать обученную голову (`head.weight`) и список классов
/// в ``ModelConfig/classes``. Создание бросает ``RWKVError``, если этого нет.
public final class Classifier {
    private let model: RWKVModel
    private let head: Linear
    private let classes: [String]
    private let pooling: RWKVPooling
    private let maxLen: Int

    public init(model: RWKVModel) throws {
        self.model = model

        guard let classes = model.config.classes, !classes.isEmpty else {
            throw RWKVError.invalidConfig(reason: "model has no classes for classification")
        }
        self.classes = classes
        self.pooling = model.config.pooling ?? .mean
        self.maxLen = model.config.contextSize

        guard let hw = model.weights["head.weight"] else {
            throw RWKVError.missingWeight(key: "head.weight")
        }
        let numClasses = model.config.numClasses ?? classes.count
        let h = Linear(model.config.nEmbd, numClasses)
        var params: [String: MLXArray] = ["weight": hw.asType(.float32)]
        if let hb = model.weights["head.bias"] { params["bias"] = hb.asType(.float32) }
        _ = h.update(parameters: NestedDictionary.unflattened(params))
        eval(h.parameters())
        self.head = h
    }

    /// Классифицирует текст: возвращает метку-победителя и распределение
    /// вероятностей (softmax) по всем классам.
    public func classify(_ text: String) -> Prediction {
        var ids = model.tokenizer.encode(text)
        if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }
        if ids.isEmpty { ids = [0] }

        let idsArr = MLXArray(ids, [1, ids.count])
        let lnOut = model.backbone.forward(idsArr)
        let feat = Pooling.pool(lnOut, lengths: [ids.count], kind: pooling.kind)
        let logits = head(feat)
        let probs = softmax(logits, axis: -1)
        probs.eval()

        let p = probs[0].asArray(Float.self)
        var pairs = zip(classes, p).map { (label: $0, probability: $1) }
        pairs.sort { $0.probability > $1.probability }
        return Prediction(label: pairs[0].label, probabilities: pairs)
    }
}

/// Извлечение векторных признаков (без головы): текст → вектор `[nEmbd]`.
public final class FeatureExtractor {
    private let model: RWKVModel
    private let pooling: RWKVPooling
    private let maxLen: Int

    public init(model: RWKVModel) {
        self.model = model
        self.pooling = model.config.pooling ?? .mean
        self.maxLen = model.config.contextSize
    }

    /// Возвращает вектор признаков длины `nEmbd` для одного текста.
    public func features(_ text: String) -> [Float] {
        var ids = model.tokenizer.encode(text)
        if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }
        if ids.isEmpty { ids = [0] }

        let idsArr = MLXArray(ids, [1, ids.count])
        let lnOut = model.backbone.forward(idsArr)
        let feat = Pooling.pool(lnOut, lengths: [ids.count], kind: pooling.kind)
        feat.eval()
        return feat[0].asArray(Float.self)
    }
}

extension RWKVPooling {
    /// Мостик к внутреннему `PoolKind`.
    var kind: PoolKind {
        switch self {
        case .mean: return .mean
        case .last: return .last
        }
    }
}
