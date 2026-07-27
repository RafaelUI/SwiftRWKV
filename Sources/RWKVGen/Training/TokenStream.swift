import Foundation
import MLX
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Поток токенов для предобучения: .bin (uint16), читаемый через mmap.
//
//  Порт rwkv_metal/pretrain/dataset.py.
//
//  Файл не грузится в память целиком — страницы подтягиваются по мере
//  обращения. Корпус на десятки гигабайт при этом остаётся работоспособным
//  на машине с 16 ГБ: батч — это несколько килобайт, а не весь корпус.
// ───────────────────────────────────────────────────────────────────────

public enum TokenStreamError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case oddByteCount(String, Int)
    case tooShort(needed: Int, got: Int)

    public var description: String {
        switch self {
        case .fileNotFound(let p):
            return "файл данных не найден: \(p)"
        case .oddByteCount(let p, let n):
            return "\(p): \(n) байт — не кратно 2, это не uint16-поток"
        case .tooShort(let needed, let got):
            return "в потоке \(got) токенов, нужно минимум \(needed) (ctxLen + 1)"
        }
    }
}

/// Токенизированный корпус в формате uint16, отображённый в память.
public final class BinTokenStream {

    private let mapped: Data
    public let count: Int          // число токенов
    public let ctxLen: Int
    public let path: String

    /// stride = ctxLen + 1: на батч нужен ctxLen входов плюс один сдвинутый
    /// таргет, иначе последний токен окна остался бы без цели.
    private var stride: Int { ctxLen + 1 }

    public init(path: String, ctxLen: Int) throws {
        let expanded = (path as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            throw TokenStreamError.fileNotFound(expanded)
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: expanded),
                            options: .alwaysMapped)
        guard data.count % 2 == 0 else {
            throw TokenStreamError.oddByteCount(expanded, data.count)
        }
        let n = data.count / 2
        guard n > ctxLen + 1 else {
            throw TokenStreamError.tooShort(needed: ctxLen + 1, got: n)
        }
        self.mapped = data
        self.count = n
        self.ctxLen = ctxLen
        self.path = expanded
    }

    /// Токен по индексу. uint16 little-endian — как пишет numpy на Apple
    /// Silicon, порядок байт совпадает с нативным.
    @inline(__always)
    private func token(at i: Int) -> Int32 {
        mapped.withUnsafeBytes { raw -> Int32 in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let lo = UInt16(base[i * 2])
            let hi = UInt16(base[i * 2 + 1])
            return Int32(lo | (hi << 8))
        }
    }

    /// Батч (x, y) для шага `step`; y сдвинут на один токен вперёд.
    ///
    /// Позиции детерминированы по step — перемешивания нет: корпус для
    /// претрейна и так перемешан при подготовке, а детерминированный обход
    /// делает возобновление с чекпоинта точным (шаг N читает то же, что
    /// читал бы непрерывный прогон).
    public func batch(batchSize: Int, step: Int) -> LoRABatch {
        let span = Swift.max(1, count - stride)
        var xs = [Int32](repeating: 0, count: batchSize * ctxLen)
        var ys = [Int32](repeating: 0, count: batchSize * ctxLen)

        mapped.withUnsafeBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            @inline(__always) func tok(_ i: Int) -> Int32 {
                Int32(UInt16(base[i * 2]) | (UInt16(base[i * 2 + 1]) << 8))
            }
            for b in 0 ..< batchSize {
                let s = ((step * batchSize + b) * stride) % span
                for t in 0 ..< ctxLen {
                    xs[b * ctxLen + t] = tok(s + t)
                    ys[b * ctxLen + t] = tok(s + t + 1)
                }
            }
        }
        return (x: MLXArray(xs, [batchSize, ctxLen]),
                y: MLXArray(ys, [batchSize, ctxLen]))
    }

    /// Замыкание-источник для Trainer: сам ведёт счётчик шагов.
    public func source(batchSize: Int, startStep: Int = 0) -> () -> LoRABatch {
        var step = startStep
        return { [self] in
            defer { step += 1 }
            return batch(batchSize: batchSize, step: step)
        }
    }

    // ── Проверка на OOV ──────────────────────────────────────────────

    public struct Validation: Sendable {
        public let ok: Bool
        public let maxToken: Int
        public let issues: [String]
    }

    /// Ищет токены >= vocabSize.
    ///
    /// Зачем это отдельным шагом: OOV-токен даёт NaN не сразу, а на первом
    /// же батче, где он встретится, — то есть посреди многочасового прогона.
    /// Дешевле выяснить заранее. Проверка выборочная: читаются равномерно
    /// разбросанные куски, а не весь корпус, иначе она сама стоила бы полного
    /// прохода по диску.
    public func validate(vocabSize: Int, sampleChunks: Int = 10,
                         maxSampled: Int = 10_000_000) -> Validation {
        let total = Swift.min(maxSampled, count)
        let per = Swift.max(1, total / sampleChunks)
        var issues: [String] = []
        var maxToken = 0

        mapped.withUnsafeBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            @inline(__always) func tok(_ i: Int) -> Int {
                Int(UInt16(base[i * 2]) | (UInt16(base[i * 2 + 1]) << 8))
            }
            let lastStart = Swift.max(0, count - per)
            for c in 0 ..< sampleChunks {
                let start = sampleChunks == 1 ? 0
                    : lastStart * c / (sampleChunks - 1)
                var chunkMax = 0
                var oov = 0
                for i in start ..< Swift.min(start + per, count) {
                    let t = tok(i)
                    if t > chunkMax { chunkMax = t }
                    if t >= vocabSize { oov += 1 }
                }
                if chunkMax > maxToken { maxToken = chunkMax }
                if oov > 0 {
                    issues.append("OOV в районе позиции \(start): \(oov) шт, max=\(chunkMax)")
                }
            }
        }
        return Validation(ok: issues.isEmpty, maxToken: maxToken, issues: issues)
    }

    /// Бросает, если найдены OOV. Вызывать перед обучением: молчаливый NaN
    /// посреди прогона дороже, чем отказ на старте.
    public func validateOrThrow(vocabSize: Int) throws {
        let v = validate(vocabSize: vocabSize)
        guard v.ok else {
            throw PretrainError.outOfVocabulary(maxToken: v.maxToken,
                                                vocabSize: vocabSize,
                                                issues: v.issues)
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Подготовка .bin
// ───────────────────────────────────────────────────────────────────────

public enum TokenStreamWriter {

    /// Записать токены в .bin (uint16 little-endian).
    ///
    /// Формат намеренно совпадает с rwkv-metal, чтобы корпус, подготовленный
    /// одним, читался другим без конвертации.
    public static func write(tokens: [Int], to url: URL) throws {
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokens.count * 2)
        for t in tokens {
            precondition(t >= 0 && t <= 0xFFFF,
                         "токен \(t) не влезает в uint16 — формат .bin рассчитан "
                         + "на словарь не больше 65536")
            bytes.append(UInt8(t & 0xFF))
            bytes.append(UInt8((t >> 8) & 0xFF))
        }
        try Data(bytes).write(to: url)
    }

    /// Токенизировать текст World-токенизатором и записать train/val.
    ///
    /// Разбиение по документам (пустая строка — разделитель), каждый
    /// `valEvery`-й документ уходит в val. Разделять именно по документам,
    /// а не по токенам, важно: иначе конец train-документа и начало
    /// val-документа окажутся в одном окне контекста, и val перестанет быть
    /// held-out.
    @discardableResult
    public static func tokenizeText(
        at input: URL, tokenizer: WorldTokenizer,
        trainOut: URL, valOut: URL,
        valEvery: Int = 200, documentDelimiter: Int? = 0
    ) throws -> (trainTokens: Int, valTokens: Int) {
        let text = try String(contentsOf: input, encoding: .utf8)
        let docs = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var train: [Int] = [], val: [Int] = []
        for (i, doc) in docs.enumerated() {
            var ids = tokenizer.encode(doc)
            if let d = documentDelimiter { ids.append(d) }
            if valEvery > 0 && i % valEvery == 0 { val += ids } else { train += ids }
        }
        try write(tokens: train, to: trainOut)
        try write(tokens: val, to: valOut)
        return (train.count, val.count)
    }
}
