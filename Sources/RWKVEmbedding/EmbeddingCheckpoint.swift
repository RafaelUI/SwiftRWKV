import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Чекпоинт головы и контракт подачи текста.
//
//  Без этого файла модуль был наполовину: голову можно было обучить и
//  нельзя было вынести из процесса. Здесь то же решение, что в реранкере, и
//  по той же причине — голова и УСЛОВИЯ, при которых её применяют, обязаны
//  ехать в одном файле.
//
//  Что именно расходится молча. Вектор нормирован при любом пулинге, при
//  любой обрезке и с любым терминатором: формы сходятся всегда, ошибок нет
//  нигде, а числа получаются от другого текста и другой геометрии. Обучили
//  с `.last`, применили с `.mean` — качество упало, и искать причину негде.
// ───────────────────────────────────────────────────────────────────────

/// Контракт подачи текста для эмбеддера.
///
/// Ровно те величины, от которых зависит вектор, и ничего сверх: всё
/// лишнее здесь запрещало бы законное переиспользование чекпоинта.
public struct EmbeddingContract: Sendable, Equatable {

    /// Откуда снимается вектор. Меняет геометрию целиком, а не по краям.
    public var pooling: Pooling
    /// Токен, дописываемый в конец перед пулингом. nil ⇒ не дописывать.
    public var terminator: Int?
    /// Обрезка входа по токенам. nil ⇒ не обрезать.
    ///
    /// Входит в контракт потому, что вектор длинного текста при обрезке и
    /// без неё — разные векторы. Раньше это расходилось: оценка обрезала на
    /// 512, выдача не обрезала вовсе, и измеренное качество описывало не то,
    /// что делает выдача. Расхождение было замерено, а не предположено.
    public var maxTokens: Int?

    public init(pooling: Pooling = .last, terminator: Int? = 0,
                maxTokens: Int? = 512) {
        self.pooling = pooling
        self.terminator = terminator
        self.maxTokens = maxTokens
    }

    public var metadata: [String: String] {
        ["pooling": pooling.rawValue,
         "terminator": terminator.map(String.init) ?? "none",
         "max_tokens": maxTokens.map(String.init) ?? "none"]
    }

    /// Собрать из метаданных чекпоинта.
    ///
    /// Отсутствующий ключ берётся из умолчания, а не считается ошибкой:
    /// чекпоинты, записанные до появления контракта, обязаны читаться.
    /// Молчаливой подмены при этом не возникает — сверять их всё равно не с
    /// чем, и `checkCompatible` об этом говорит.
    public init(metadata md: [String: String]) {
        self.init()
        if let v = md["pooling"], let p = Pooling(rawValue: v) { pooling = p }
        if let v = md["terminator"] { terminator = v == "none" ? nil : Int(v) }
        if let v = md["max_tokens"] { maxTokens = v == "none" ? nil : Int(v) }
    }
}

public enum EmbeddingCheckpointError: Error, CustomStringConvertible {
    case notAnEmbeddingHead(String)
    case shapeMismatch(String)
    case missingParameters([String])
    case contractMismatch(key: String, file: String, now: String)

    public var description: String {
        switch self {
        case .notAnEmbeddingHead(let p):
            return "\(p): не чекпоинт головы эмбеддера"
        case .shapeMismatch(let s):
            return "форма головы не сходится: \(s)"
        case .missingParameters(let n):
            return "в чекпоинте нет параметров: \(n.joined(separator: ", "))"
        case .contractMismatch(let key, let file, let now):
            return """
                \(key): голова обучена с '\(file)', применяется с '\(now)'. \
                Вектор нормирован в обоих случаях, поэтому ошибки формы \
                здесь не будет — будут правдоподобные числа от другой \
                геометрии.
                """
        }
    }
}

extension EmbeddingModel {

    public static let checkpointFormat = "swiftrwkv-embedding-head-v1"

    /// Сохранить ТОЛЬКО голову: база по определению не изменилась, если её
    /// не обучали. Обучали — сохранять надо и её, отдельно; здесь речь про
    /// голову и про условия, при которых её применяют.
    ///
    /// - contract: как подавать текст. Не украшение: разойтись он может
    ///   молча, и единственное место, где это ловится, — загрузка.
    public func saveHead(to url: URL, contract: EmbeddingContract? = nil,
                         extra: [String: String]? = nil) throws {
        let c = contract ?? EmbeddingContract(pooling: pooling)
        var md = c.metadata
        md["format"] = Self.checkpointFormat
        md["dim"] = String(head.dim)
        md["hidden"] = String(head.hidden)
        if let extra { for (k, v) in extra { md[k] = v } }

        var arrays: [String: MLXArray] = [:]
        for (name, p) in zip(EmbeddingHead.parameterNames, head.parameters) {
            arrays[name] = p
        }
        eval(Array(arrays.values))
        try save(arrays: arrays, metadata: md, url: url)
    }

    /// Метаданные чекпоинта без загрузки весов.
    public static func readHeadMetadata(_ url: URL) throws -> [String: String] {
        try loadArraysAndMetadata(url: url).1
    }

    /// Контракт, записанный в чекпоинте.
    public static func readContract(_ url: URL) throws -> EmbeddingContract {
        EmbeddingContract(metadata: try readHeadMetadata(url))
    }

    @discardableResult
    public func loadHead(from url: URL, strict: Bool = true) throws -> EmbeddingModel {
        let (weights, md) = try loadArraysAndMetadata(url: url)
        if strict {
            guard (md["format"] ?? "").hasPrefix("swiftrwkv-embedding-head") else {
                throw EmbeddingCheckpointError.notAnEmbeddingHead(url.path)
            }
            // Формы сверяются ЯВНО. Голова с другим `hidden` имеет другие
            // формы и упала бы сама, но сообщение было бы про матмул, а не
            // про то, что взят чужой чекпоинт.
            if let d = md["dim"], Int(d) != head.dim {
                throw EmbeddingCheckpointError.shapeMismatch(
                    "dim в файле \(d), у головы \(head.dim)")
            }
            if let h = md["hidden"], Int(h) != head.hidden {
                throw EmbeddingCheckpointError.shapeMismatch(
                    "hidden в файле \(h), у головы \(head.hidden)")
            }
            // Пулинг — часть МОДЕЛИ, а не головы, поэтому сверяется здесь:
            // загрузить голову, обученную на `.last`, в модель с `.mean`
            // технически можно, и получится тихая бессмыслица.
            let want = EmbeddingContract(metadata: md)
            if want.pooling != pooling {
                throw EmbeddingCheckpointError.contractMismatch(
                    key: "pooling", file: want.pooling.rawValue,
                    now: pooling.rawValue)
            }
        }
        let names = EmbeddingHead.parameterNames
        let missing = names.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw EmbeddingCheckpointError.missingParameters(missing)
        }
        head.setParameters(names.map { weights[$0]! })
        eval(head.parameters)
        return self
    }

    /// Собрать модель по конфигурации из САМОГО чекпоинта.
    ///
    /// Рекомендуемый способ загрузки. Пулинг и размеры головы не нужно
    /// помнить — а помнить пришлось бы: модель со сходными формами примет
    /// чужие веса молча, если не сверять.
    public static func fromHead(backbone: X070Backbone, url: URL)
        throws -> (model: EmbeddingModel, contract: EmbeddingContract) {
        let md = try readHeadMetadata(url)
        guard (md["format"] ?? "").hasPrefix("swiftrwkv-embedding-head") else {
            throw EmbeddingCheckpointError.notAnEmbeddingHead(url.path)
        }
        let contract = EmbeddingContract(metadata: md)
        let dim = md["dim"].flatMap { Int($0) } ?? backbone.cfg.nEmbd
        let hidden = md["hidden"].flatMap { Int($0) } ?? dim
        let head = EmbeddingHead(dim: dim, hidden: hidden)
        let model = EmbeddingModel(backbone: backbone, head: head,
                                   pooling: contract.pooling)
        try model.loadHead(from: url)
        return (model, contract)
    }
}
