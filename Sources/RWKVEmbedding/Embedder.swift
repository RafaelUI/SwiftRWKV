import Foundation
import MLX
import MLXNN
import MLXRandom
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Векторы текста из RWKV-7.
//
//  Порт rwkv_metal/embedding/{embed,heads}.py.
//
//  Дешёвая половина работает СРАЗУ, без всякого дообучения: RWKV — RNN, он
//  по конструкции сворачивает последовательность в состояние фиксированного
//  размера, поэтому пулинг скрытого состояния уже даёт рабочий (пусть и
//  неотшлифованный) эмбеддинг. Дообучение контрастными лоссами его улучшает,
//  но не является условием применимости.
// ───────────────────────────────────────────────────────────────────────

@inline(__always)
func l2Normalize(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}

/// Как сворачивать [B,T,D] в [B,D].
public enum Pooling: String, Sendable {
    /// Состояние на последней (реальной) позиции. Естественный выбор для RNN:
    /// там весь контекст уже свёрнут.
    case last
    /// Среднее по реальным позициям.
    case mean
}

/// Косинусная матрица между наборами L2-нормированных векторов: [N,D]×[M,D] → [N,M].
public func cosineSimilarity(_ a: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
    matmul(a, (b ?? a).transposed())
}

// ───────────────────────────────────────────────────────────────────────
//  Голова
// ───────────────────────────────────────────────────────────────────────

/// Остаточная голова с нулевой инициализацией: на нулевом шаге — тождество.
///
/// Нулевая инициализация fc2 здесь не косметика: до обучения голова НЕ должна
/// трогать геометрию, которую база уже выучила. Иначе первый же шаг сдвигает
/// векторы случайным преобразованием, и контрастное дообучение начинается не
/// с предобученного пространства, а с испорченного.
public final class EmbeddingHead {

    public var fc1: MLXArray        // [hidden, D]
    public var fc2: MLXArray        // [D, hidden]
    public var normWeight: MLXArray // [D]
    public var normBias: MLXArray   // [D]

    public let dim: Int
    public let hidden: Int

    public init(dim: Int, hidden: Int? = nil, seed: UInt64 = 0) {
        let h = hidden ?? dim
        self.dim = dim
        self.hidden = h
        MLXRandom.seed(seed)
        let scale = Float(1.0 / sqrt(Double(dim)))
        self.fc1 = MLXRandom.uniform(low: -scale, high: scale, [h, dim])
        self.fc2 = MLXArray.zeros([dim, h])        // ← тождество на старте
        self.normWeight = MLXArray.ones([dim])
        self.normBias = MLXArray.zeros([dim])
        eval(fc1, fc2, normWeight, normBias)
    }

    /// Имена параметров в порядке упаковки — он же порядок в TrainableSet.
    public static let parameterNames = ["head.fc1", "head.fc2",
                                        "head.norm.weight", "head.norm.bias"]

    public var parameters: [MLXArray] { [fc1, fc2, normWeight, normBias] }

    public func setParameters(_ ps: [MLXArray]) {
        precondition(ps.count == 4, "ожидались 4 параметра, получено \(ps.count)")
        fc1 = ps[0]; fc2 = ps[1]; normWeight = ps[2]; normBias = ps[3]
    }

    /// x [.., D] → [.., D]. Форма как у NonlinearHead в EmbeddingRWKV.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        apply(x, parameters: parameters)
    }

    /// Вариант с явными параметрами — нужен, чтобы градиент тёк к ним внутри
    /// grad-замыкания (см. TrainableSet: подстановка обязана быть внутри).
    public func apply(_ x: MLXArray, parameters ps: [MLXArray]) -> MLXArray {
        let h = matmul(relu(matmul(x, ps[0].transposed())), ps[1].transposed())
        return layerNormLast(h + x, weight: ps[2], bias: ps[3])
    }

    private func layerNormLast(_ x: MLXArray, weight: MLXArray,
                               bias: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let varc = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) / sqrt(varc + eps) * weight + bias
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Модель
// ───────────────────────────────────────────────────────────────────────

/// База + голова.
///
/// Голова — СИБЛИНГ базы, а не её часть. Это требование корректности, а не
/// вкусовщина: заморозка или квантование базы обходит её дерево параметров,
/// и голова, живущая внутри, была бы молча заморожена вместе с ней. Здесь
/// они хранятся раздельно, поэтому голова остаётся обучаемой независимо от
/// того, что происходит с базой — full-FT, frozen, LoRA или QLoRA.
public final class EmbeddingModel {

    public let backbone: X070Backbone
    public let head: EmbeddingHead
    public let pooling: Pooling

    public init(backbone: X070Backbone, head: EmbeddingHead? = nil,
                pooling: Pooling = .last) {
        self.backbone = backbone
        self.head = head ?? EmbeddingHead(dim: backbone.cfg.nEmbd)
        self.pooling = pooling
    }

    /// idx [B,T], poolIdx [B] — позиция, с которой снимать вектор (для .last)
    /// либо длина строки (для .mean). Возвращает [B,D], L2-нормировано.
    public func embed(_ idx: MLXArray, poolIndex: MLXArray,
                      headParameters: [MLXArray]? = nil) -> MLXArray {
        let h = backbone.body(idx)                       // [B,T,D]
        let pooled = pool(h, poolIndex: poolIndex)
        return l2Normalize(head.apply(pooled, parameters: headParameters ?? head.parameters))
    }

    /// Пулинг без головы — сырой вектор базы.
    public func pooledOnly(_ idx: MLXArray, poolIndex: MLXArray) -> MLXArray {
        l2Normalize(pool(backbone.body(idx), poolIndex: poolIndex))
    }

    func pool(_ h: MLXArray, poolIndex: MLXArray) -> MLXArray {
        switch pooling {
        case .last:
            return takeAlong(h, poolIndex.reshaped([-1, 1, 1]), axis: 1).squeezed(axis: 1)
        case .mean:
            // poolIndex здесь — ДЛИНА строки; пад-позиции не должны попасть
            // в среднее, иначе вектор короткой строки поедет к нулю.
            let T = h.shape[1]
            let pos = MLXArray(Array(0 ..< T).map { Int32($0) }).reshaped([1, T, 1])
            let mask = (pos .< poolIndex.reshaped([-1, 1, 1])).asType(h.dtype)
            let summed = (h * mask).sum(axis: 1)
            let counts = mask.sum(axis: 1)
            return summed / maximum(counts, MLXArray(Float(1)))
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Инференс
// ───────────────────────────────────────────────────────────────────────

/// Векторизация текстов.
public struct Embedder {

    public let model: EmbeddingModel
    public let tokenizer: WorldTokenizer
    /// Контракт подачи текста: терминатор и обрезка.
    ///
    /// Пулинг живёт в модели и в контракт входит для сверки при загрузке
    /// чекпоинта — здесь он не дублируется, чтобы не разъехался с моделью.
    public let contract: EmbeddingContract

    /// Токен, дописываемый в конец перед пулингом.
    ///
    /// 0 — зарезервированный id в World-вокабе (ни одной byte-строке не
    /// сопоставлен), поэтому для НЕдообученной базы он естественный
    /// терминатор: модель его не видела в тексте и он не тянет за собой
    /// смысл. nil ⇒ не дописывать.
    public var terminator: Int? { contract.terminator }

    /// Обрезка входа по токенам.
    ///
    /// Умолчание 512 — ТО ЖЕ, что у метрик и у стадий обучения, и это
    /// главное в нём. Раньше выдача не обрезала вовсе: на длинном тексте
    /// измеренное качество описывало не то, что делает выдача, и заметить
    /// это по формам было невозможно — вектор нормирован в обоих случаях.
    /// Расхождение замерено, а не предположено (см. `EmbeddingSmokeTests`).
    public var maxTokens: Int? { contract.maxTokens }

    public init(model: EmbeddingModel, tokenizer: WorldTokenizer,
                contract: EmbeddingContract? = nil) {
        self.model = model
        self.tokenizer = tokenizer
        // Пулинг берётся у МОДЕЛИ: она источник истины, а контракт — то, что
        // записывается рядом с ней.
        var c = contract ?? EmbeddingContract()
        c.pooling = model.pooling
        self.contract = c
    }

    /// Собрать по чекпоинту: контракт читается ИЗ ФАЙЛА.
    ///
    /// Рекомендуемый способ. Обрезка, терминатор и пулинг — часть того, на
    /// чём голова обучалась; подать текст иначе не ошибка формы, а тихая
    /// потеря качества.
    public static func fromCheckpoint(backbone: X070Backbone,
                                      tokenizer: WorldTokenizer,
                                      head url: URL) throws -> Embedder {
        let (model, contract) = try EmbeddingModel.fromHead(backbone: backbone,
                                                            url: url)
        return Embedder(model: model, tokenizer: tokenizer, contract: contract)
    }

    public func encode(_ text: String) -> [Int] {
        var ids = tokenizer.encode(text)
        // Обрезка ДО терминатора: он обязан остаться последним, иначе
        // позиция пулинга укажет на обычный токен, и вектор снимется не с
        // того места.
        if let m = maxTokens {
            let room = terminator == nil ? m : m - 1
            if ids.count > room { ids = Array(ids.prefix(Swift.max(0, room))) }
        }
        if let t = terminator { ids.append(t) }
        return ids
    }

    /// Вектор одного текста [D].
    public func embed(_ text: String) -> MLXArray {
        let ids = encode(text)
        let idx = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let poolIdx = MLXArray([Int32(model.pooling == .last ? ids.count - 1 : ids.count)])
        return model.embed(idx, poolIndex: poolIdx)[0]
    }

    /// Векторы списка текстов [N,D].
    ///
    /// Каждый текст идёт ОТДЕЛЬНЫМ проходом, без паддинга. Так короткие и
    /// длинные строки заведомо не влияют друг на друга через общий батч.
    /// Батчевый путь возможен (см. buildMask/lastRealIndex в RWKVGen), но
    /// требует аккуратной маски, и для инференса выигрыш редко стоит риска
    /// тихо испортить вектор короткой строки.
    public func embed(_ texts: [String]) -> MLXArray {
        let vecs = texts.map { embed($0).reshaped([1, -1]) }
        let out = concatenated(vecs, axis: 0)
        eval(out)
        return out
    }
}
