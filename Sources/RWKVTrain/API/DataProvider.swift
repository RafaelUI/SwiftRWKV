import Foundation

/// Один обучающий пример: уже токенизированный текст + целочисленный ярлык.
///
/// `ids` — токены без паддинга (обрезка/паддинг делает пайплайн обучения).
/// `label` — индекс класса (0..<numClasses) для классификации; для
/// извлечения признаков может быть проигнорирован.
public struct Example: Sendable {
    public let ids: [Int]
    public let label: Int

    public init(ids: [Int], label: Int) {
        self.ids = ids
        self.label = label
    }
}

/// Поставщик данных для обучения. Абстрагирует источник (файл, БД, сеть,
/// массив в памяти) от пайплайна — никакой привязки к `Bundle.main`.
///
/// Реализация отвечает за токенизацию: возвращает уже готовые `Example`.
/// Для удобства есть готовые реализации `InMemoryDataProvider` и
/// `JSONLDataProvider`.
public protocol DataProvider: Sendable {
    /// Обучающая выборка.
    func trainExamples() throws -> [Example]
    /// Валидационная выборка (может быть пустой).
    func validationExamples() throws -> [Example]
    /// Имена классов в порядке индексов ярлыков. Пустой массив — если задача
    /// без классов (извлечение признаков).
    var classes: [String] { get }
}

// ───────────────────────── In-memory ─────────────────────────

/// Готовые `Example` в памяти (например, разработчик сам токенизировал).
public struct InMemoryDataProvider: DataProvider {
    public let train: [Example]
    public let validation: [Example]
    public let classes: [String]

    public init(train: [Example], validation: [Example] = [], classes: [String] = []) {
        self.train = train
        self.validation = validation
        self.classes = classes
    }

    public func trainExamples() throws -> [Example] { train }
    public func validationExamples() throws -> [Example] { validation }
}

// ───────────────────────── JSONL ─────────────────────────

/// Поставщик из файлов JSONL вида `{"text": "...", "label": 0}`.
///
/// Токенизация выполняется переданным `BPETokenizer`. Тексты обрезаются до
/// `maxLen` токенов. Пустые тексты заменяются на `[0]` (защита от пустого входа).
public struct JSONLDataProvider: DataProvider {
    private let trainURL: URL
    private let validationURL: URL?
    private let tokenizer: BPETokenizer
    private let maxLen: Int
    public let classes: [String]

    public init(
        trainURL: URL,
        validationURL: URL? = nil,
        tokenizer: BPETokenizer,
        maxLen: Int,
        classes: [String]
    ) {
        self.trainURL = trainURL
        self.validationURL = validationURL
        self.tokenizer = tokenizer
        self.maxLen = maxLen
        self.classes = classes
    }

    public func trainExamples() throws -> [Example] {
        try load(trainURL)
    }

    public func validationExamples() throws -> [Example] {
        guard let url = validationURL else { return [] }
        return try load(url)
    }

    private func load(_ url: URL) throws -> [Example] {
        guard let raw = try? String(contentsOf: url, encoding: .utf8) else {
            throw RWKVError.invalidDataset(reason: "cannot read \(url.lastPathComponent)")
        }
        var out: [Example] = []
        out.reserveCapacity(raw.count / 80)
        for line in raw.split(separator: "\n") {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let text = obj["text"] as? String,
                  let label = obj["label"] as? Int
            else { continue }
            var ids = tokenizer.encode(text)
            if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }
            if ids.isEmpty { ids = [0] }
            out.append(Example(ids: ids, label: label))
        }
        if out.isEmpty {
            throw RWKVError.invalidDataset(reason: "no valid rows in \(url.lastPathComponent)")
        }
        return out
    }
}
