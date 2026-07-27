import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Данные для обучения эмбеддингов.
//
//  Порт rwkv_metal/embedding/dataset.py, форма — под LitRetrieval:
//  {anchor, positive, negative, task} в JSONL, три задачи в одном файле.
// ───────────────────────────────────────────────────────────────────────

public enum EmbeddingTask: String, Sendable, CaseIterable {
    case retrieval
    case sts
    case classification
}

public struct EmbeddingSample: Sendable {
    public let anchor: String
    public let positive: String
    public let negative: String
    public let task: EmbeddingTask

    public init(anchor: String, positive: String, negative: String,
                task: EmbeddingTask) {
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.task = task
    }
}

public enum EmbeddingDataError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case noSamples(String)

    public var description: String {
        switch self {
        case .fileNotFound(let p): return "файл не найден: \(p)"
        case .noSamples(let why): return "не набралось примеров: \(why)"
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Загрузка
// ───────────────────────────────────────────────────────────────────────

public enum EmbeddingDataset {

    /// Прочитать JSONL. `limit` ограничивает число строк — полный
    /// LitRetrieval это 2.6 ГБ, и для теста нужен срез, а не весь файл.
    ///
    /// Строки с битым UTF-8 и незнакомым `task` пропускаются молча: в корпусе
    /// такого размера одиночный мусор — норма, и падать из-за него посреди
    /// загрузки хуже, чем пропустить.
    public static func loadJSONL(path: String, limit: Int? = nil,
                                 tasks: Set<EmbeddingTask>? = nil) throws -> [EmbeddingSample] {
        let expanded = (path as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            throw EmbeddingDataError.fileNotFound(expanded)
        }
        guard let stream = InputStream(fileAtPath: expanded) else {
            throw EmbeddingDataError.fileNotFound(expanded)
        }
        stream.open()
        defer { stream.close() }

        var out: [EmbeddingSample] = []
        var buffer = Data()
        let chunkSize = 1 << 20
        var raw = [UInt8](repeating: 0, count: chunkSize)

        func drain(_ finalize: Bool) {
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer[buffer.startIndex ..< nl]
                buffer.removeSubrange(buffer.startIndex ... nl)
                if let s = parse(line) { out.append(s) }
                if let l = limit, out.count >= l { return }
            }
            if finalize, !buffer.isEmpty, let s = parse(buffer) { out.append(s) }
        }

        func parse(_ data: Data) -> EmbeddingSample? {
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let anchor = obj["anchor"] as? String,
                  let positive = obj["positive"] as? String,
                  let negative = obj["negative"] as? String,
                  let taskRaw = obj["task"] as? String,
                  let task = EmbeddingTask(rawValue: taskRaw) else { return nil }
            if let filter = tasks, !filter.contains(task) { return nil }
            return EmbeddingSample(anchor: anchor, positive: positive,
                                   negative: negative, task: task)
        }

        while stream.hasBytesAvailable {
            let n = stream.read(&raw, maxLength: chunkSize)
            if n <= 0 { break }
            buffer.append(contentsOf: raw[0 ..< n])
            drain(false)
            if let l = limit, out.count >= l { break }
        }
        drain(true)
        if let l = limit, out.count > l { out = Array(out.prefix(l)) }
        guard !out.isEmpty else {
            throw EmbeddingDataError.noSamples("\(expanded): ни одной валидной строки")
        }
        return out
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Метки классификации
// ───────────────────────────────────────────────────────────────────────

public enum ClassificationLabels {

    /// Закрытый пул эмоций LitRetrieval — 25 меток.
    ///
    /// Проверено по данным: 134 165 строк classification, ровно 25 уникальных
    /// меток, все встречаются, и positive/negative всегда внутри пула. Каждая
    /// строка предъявляет СВОИ 7 из этих 25 (наборы почти всегда различны:
    /// сочетаний C(25,7) = 480 700), поэтому кандидаты парсятся построчно, а
    /// не берутся фиксированным списком.
    ///
    /// Заметьте: README датасета называет другой набор из семи эмоций — он
    /// не соответствует данным. Источник истины здесь — сами данные.
    public static let pool = [
        "joy", "sadness", "anger", "fear", "surprise", "disgust", "love",
        "shame", "guilt", "pride", "jealousy", "contempt",
        "frustration", "longing", "melancholy", "nostalgia", "loneliness",
        "hope", "despair", "resignation", "anxiety", "awe", "tenderness",
        "bitterness", "anticipation",
    ]

    /// Вытащить список кандидатов из инструкции внутри anchor:
    /// "...categories: joy, fear, shame\nQuery: ..." → [joy, fear, shame].
    public static func parseCandidates(from anchor: String) -> [String]? {
        guard let r = anchor.range(of: "categories:") else { return nil }
        let rest = anchor[r.upperBound...]
        let line = rest.prefix(while: { $0 != "\n" })
        let items = line.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
              .trimmingCharacters(in: CharacterSet(charactersIn: "."))
        }.filter { !$0.isEmpty }
        return items.isEmpty ? nil : items
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Токенизация батча
// ───────────────────────────────────────────────────────────────────────

/// Токенизировать, дописать терминатор, добить справа до максимума в батче.
///
/// Возвращает (idx [B,T], poolIndex [B]), где poolIndex — позиция
/// терминатора строки, то есть откуда снимать её вектор. Добивка идёт ТЕМ ЖЕ
/// терминатором — на пулинг она не влияет, потому что читается ровно
/// poolIndex, а не конец тензора.
///
/// - padMultiple: додвинуть T вверх до кратности. Нужно для ОБУЧЕНИЯ:
///   дифференцируемое ядро `wkv7Train` требует `T % WKV7_CHUNK == 0`, а
///   максимум длины в батче произвольный. Добивка здесь безвредна по той же
///   причине, по которой безвредна обычная: RWKV причинен, позиция t зависит
///   только от позиций ≤ t, а вектор снимается с poolIndex — то есть ни один
///   добитый токен в него не входит. Это РАВЕНСТВО, а не приближение, и оно
///   проверяется тестом. nil ⇒ не выравнивать (путь инференса не меняется).
public func encodeBatch(tokenizer: WorldTokenizer, texts: [String],
                        terminator: Int = 0,
                        maxTokens: Int? = nil,
                        padMultiple: Int? = nil) -> (idx: MLXArray, poolIndex: MLXArray) {
    var seqs = texts.map { text -> [Int] in
        var ids = tokenizer.encode(text)
        if let m = maxTokens, ids.count > m - 1 { ids = Array(ids.prefix(m - 1)) }
        ids.append(terminator)
        return ids
    }
    let poolIdx = seqs.map { Int32($0.count - 1) }
    var maxLen = seqs.map(\.count).max() ?? 1
    if let mult = padMultiple, mult > 1, maxLen % mult != 0 {
        maxLen += mult - (maxLen % mult)
    }
    for i in seqs.indices {
        seqs[i].append(contentsOf: Array(repeating: terminator,
                                         count: maxLen - seqs[i].count))
    }
    let flat = seqs.flatMap { $0.map { Int32($0) } }
    return (MLXArray(flat, [seqs.count, maxLen]), MLXArray(poolIdx))
}

// ───────────────────────────────────────────────────────────────────────
//  Батчеры
// ───────────────────────────────────────────────────────────────────────

/// Детерминированный обход с перемешиванием: одинаковый seed — одинаковый
/// порядок, иначе сравнивать два прогона невозможно.
struct CyclingSampler {
    private var order: [Int]
    private var cursor = 0
    private var rng: SplitMix64
    private let shuffle: Bool

    init(count: Int, shuffle: Bool, seed: UInt64) {
        self.order = Array(0 ..< count)
        self.shuffle = shuffle
        self.rng = SplitMix64(state: seed &* 0x9E3779B97F4A7C15 &+ 1)
        if shuffle { self.order = Self.shuffled(order, &rng) }
    }

    static func shuffled(_ a: [Int], _ rng: inout SplitMix64) -> [Int] {
        var arr = a
        guard arr.count > 1 else { return arr }
        for i in stride(from: arr.count - 1, to: 0, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            arr.swapAt(i, j)
        }
        return arr
    }

    mutating func next(_ n: Int) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(n)
        while out.count < n {
            if cursor >= order.count {
                cursor = 0
                if shuffle { order = Self.shuffled(order, &rng) }
            }
            out.append(order[cursor])
            cursor += 1
        }
        return out
    }
}

struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// Батч триплетов (retrieval / sts).
public struct TripletBatch {
    public let anchorIdx: MLXArray, anchorPool: MLXArray
    public let positiveIdx: MLXArray, positivePool: MLXArray
    public let negativeIdx: MLXArray, negativePool: MLXArray
    public var batchSize: Int { anchorIdx.shape[0] }
}

public struct TripletBatcher {
    private let samples: [EmbeddingSample]
    private let tokenizer: WorldTokenizer
    private let batchSize: Int
    private let terminator: Int
    private let maxTokens: Int?
    private let padMultiple: Int?
    private var sampler: CyclingSampler

    /// - padMultiple: выравнивание T (см. `encodeBatch`). Для обучения с
    ///   размороженными слоями базы обязано быть WKV7_CHUNK.
    public init(samples: [EmbeddingSample], tokenizer: WorldTokenizer,
                batchSize: Int, terminator: Int = 0, maxTokens: Int? = 512,
                shuffle: Bool = true, seed: UInt64 = 0,
                padMultiple: Int? = nil) {
        self.samples = samples
        self.tokenizer = tokenizer
        self.batchSize = batchSize
        self.terminator = terminator
        self.maxTokens = maxTokens
        self.padMultiple = padMultiple
        self.sampler = CyclingSampler(count: samples.count,
                                      shuffle: shuffle, seed: seed)
    }

    public mutating func next() -> TripletBatch {
        let idx = sampler.next(batchSize)
        let rows = idx.map { samples[$0] }
        let a = encodeBatch(tokenizer: tokenizer, texts: rows.map(\.anchor),
                            terminator: terminator, maxTokens: maxTokens,
                            padMultiple: padMultiple)
        let p = encodeBatch(tokenizer: tokenizer, texts: rows.map(\.positive),
                            terminator: terminator, maxTokens: maxTokens,
                            padMultiple: padMultiple)
        let n = encodeBatch(tokenizer: tokenizer, texts: rows.map(\.negative),
                            terminator: terminator, maxTokens: maxTokens,
                            padMultiple: padMultiple)
        return TripletBatch(anchorIdx: a.idx, anchorPool: a.poolIndex,
                            positiveIdx: p.idx, positivePool: p.poolIndex,
                            negativeIdx: n.idx, negativePool: n.poolIndex)
    }
}

/// Батч zero-shot классификации.
public struct ClassificationBatch {
    public let anchorIdx: MLXArray, anchorPool: MLXArray
    /// [B, K, T] — пул кандидатов СВОЙ у каждой строки.
    public let candidateIdx: MLXArray
    public let candidatePool: MLXArray   // [B, K]
    public let mask: MLXArray            // [B, K], 1 — реальный кандидат
    public let targetIndex: MLXArray     // [B]
    public var batchSize: Int { anchorIdx.shape[0] }
    public var candidateCount: Int { candidateIdx.shape[1] }
}

public struct ClassificationBatcher {
    private let samples: [EmbeddingSample]
    private let candidates: [[String]]
    private let targets: [Int]
    private let tokenizer: WorldTokenizer
    private let batchSize: Int
    private let terminator: Int
    private let maxTokens: Int?
    private let padMultiple: Int?
    private var sampler: CyclingSampler

    /// - useFullPool: предъявлять все 25 меток вместо семи из инструкции.
    ///   Труднее и честнее для оценки; для обучения оставляйте false, чтобы
    ///   совпадать с тем, как построен датасет.
    /// - padMultiple: выравнивание T (см. `encodeBatch`).
    public init(samples: [EmbeddingSample], tokenizer: WorldTokenizer,
                batchSize: Int, terminator: Int = 0, maxTokens: Int? = 512,
                shuffle: Bool = true, seed: UInt64 = 0,
                useFullPool: Bool = false,
                padMultiple: Int? = nil) {
        var kept: [EmbeddingSample] = []
        var cands: [[String]] = []
        var tgts: [Int] = []
        for s in samples where s.task == .classification {
            let pool = useFullPool
                ? ClassificationLabels.pool
                : (ClassificationLabels.parseCandidates(from: s.anchor) ?? [])
            let label = s.positive.trimmingCharacters(in: .whitespacesAndNewlines)
            // Строка без разобранных кандидатов или с ответом вне набора
            // бесполезна: цель обязана быть достижима, иначе лосс учит
            // выбирать из множества, где верного варианта нет.
            guard let t = pool.firstIndex(of: label) else { continue }
            kept.append(s); cands.append(pool); tgts.append(t)
        }
        self.samples = kept
        self.candidates = cands
        self.targets = tgts
        self.tokenizer = tokenizer
        self.batchSize = batchSize
        self.terminator = terminator
        self.maxTokens = maxTokens
        self.padMultiple = padMultiple
        self.sampler = CyclingSampler(count: kept.count, shuffle: shuffle, seed: seed)
    }

    public var count: Int { samples.count }

    public mutating func next() -> ClassificationBatch {
        let idx = sampler.next(batchSize)
        let rows = idx.map { samples[$0] }
        let a = encodeBatch(tokenizer: tokenizer, texts: rows.map(\.anchor),
                            terminator: terminator, maxTokens: maxTokens,
                            padMultiple: padMultiple)

        // K различается между строками ⇒ добиваем до максимума в батче и
        // отмечаем маской; иначе добивка конкурировала бы с настоящими метками.
        let perRow = idx.map { candidates[$0] }
        let K = perRow.map(\.count).max() ?? 1
        var flatTexts: [String] = []
        var maskVals: [Float] = []
        for row in perRow {
            for k in 0 ..< K {
                flatTexts.append(k < row.count ? row[k] : "")
                maskVals.append(k < row.count ? 1 : 0)
            }
        }
        let c = encodeBatch(tokenizer: tokenizer, texts: flatTexts,
                            terminator: terminator, maxTokens: maxTokens,
                            padMultiple: padMultiple)
        let B = rows.count
        return ClassificationBatch(
            anchorIdx: a.idx, anchorPool: a.poolIndex,
            candidateIdx: c.idx.reshaped([B, K, -1]),
            candidatePool: c.poolIndex.reshaped([B, K]),
            mask: MLXArray(maskVals, [B, K]),
            targetIndex: MLXArray(idx.map { Int32(targets[$0]) }))
    }
}
