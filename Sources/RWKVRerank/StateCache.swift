import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Кэш состояний. Порт StateCache из rwkv_metal/reranker/encode.py.
//
//  Зачем он есть. База заморожена, значит отображение «текст пары →
//  состояние» ФИКСИРОВАНО. Тогда обучение головы не обязано каждый шаг
//  пересчитывать длинный проход по базе: пары сворачиваются ОДИН раз, а
//  дальше голова учится на готовых состояниях. Голова крошечная (один-два
//  блока на одном токене), поэтому эпоха по десяткам тысяч пар занимает
//  секунды вместо десятков минут — и становится возможным то, ради чего это
//  всё: большие батчи, много эпох, честный подбор гиперпараметров.
//
//  Хранятся не все слои, а только читаемые головой (`uniqueSources`). Для
//  умолчания это 1 слой из 12: 98 КБ на пару в fp16 против 2.4 МБ на полное
//  состояние в fp32.
//
//  Почему СВОЙ формат файла, а не safetensors
//  ──────────────────────────────────────────
//  Ради mmap. Загрузчик safetensors в mlx-swift материализует массив в
//  память целиком, а весь смысл кэша в том, что он НЕ обязан помещаться:
//  кэш на 2.4 ГБ при 16 ГБ памяти плюс база плюс транзиент — это своп, а
//  своп портит не только скорость, но и все замеры, снятые в это время.
//  Здесь состояния лежат сырым блоком, отображаются в память страницами, и
//  в RAM живёт только то, что реально трогали.
//
//  Формат:
//      <base>.states   — сырой little-endian блок [nPairs, nSrc, H, S, S]
//      <base>.idx.json — формы, тип, pairIndex, labels, контракт текста
// ───────────────────────────────────────────────────────────────────────

public enum StateCacheDType: String, Sendable, Codable {
    case float16
    case float32

    var itemSize: Int { self == .float16 ? 2 : 4 }
    var mlx: DType { self == .float16 ? .float16 : .float32 }
}

public enum StateCacheError: Error, CustomStringConvertible {
    case shapeMismatch(String)
    case fp16Overflow(Float)
    case fileNotFound(String)
    case corrupt(String)

    public var description: String {
        switch self {
        case .shapeMismatch(let s): return "форма не сходится: \(s)"
        case .fp16Overflow(let m):
            return """
                состояния доходят до \(Int(m)), fp16 обрежется на 65504. \
                Пересобери кэш с dtype: .float32 (вдвое больше места).
                """
        case .fileNotFound(let p): return "нет файла кэша: \(p)"
        case .corrupt(let s): return "кэш повреждён: \(s)"
        }
    }
}

/// Индексы кэша: какие строки состояний относятся к какому примеру.
struct StateCacheIndex: Codable {
    var shape: [Int]                 // [nPairs, nSrc, H, S, S]
    var dtype: StateCacheDType
    var pairIndex: [[Int]]           // [nSamples][nCand] → строка в states
    var labels: [Int]                // [nSamples]
    var hardNegs: [[Int]]            // [nSamples] позиции майненных негативов
    /// Контракт, при котором кэш собран. Состояние — функция ровно этого
    /// префикса, и при смене чего угодно из перечисленного кэш становится
    /// неверным МОЛЧА.
    var contract: [String: String]
}

public final class StateCache {

    /// Сырые состояния: `[nPairs, nSrc, H, S, S]`. Отображённый в память
    /// файл либо буфер в памяти — снаружи разницы нет.
    let storage: Data
    let index: StateCacheIndex

    public var shape: [Int] { index.shape }
    public var dtype: StateCacheDType { index.dtype }
    public var nPairs: Int { index.shape[0] }
    public var nSamples: Int { index.pairIndex.count }
    public var nCandidates: Int { index.pairIndex.first?.count ?? 0 }
    public var labels: [Int] { index.labels }
    public var hardNegs: [[Int]] { index.hardNegs }
    public var pairIndex: [[Int]] { index.pairIndex }
    public var contract: [String: String] { index.contract }

    /// Байт на одну пару.
    public var rowBytes: Int {
        index.shape.dropFirst().reduce(1, *) * index.dtype.itemSize
    }
    public var byteCount: Int { storage.count }

    init(storage: Data, index: StateCacheIndex) {
        self.storage = storage
        self.index = index
    }

    // ── Чтение ──

    /// Строки кэша → MLXArray `[rows.count, nSrc, H, S, S]`.
    ///
    /// Единственное место, где данные попадают в MLX, и попадают ровно в
    /// размере батча. Копирование здесь неизбежно и желательно: строки
    /// разбросаны по файлу, а MLX нужен непрерывный буфер.
    public func gather(_ rows: [Int]) -> MLXArray {
        let rb = rowBytes
        var buf = Data(count: rows.count * rb)
        buf.withUnsafeMutableBytes { dst in
            storage.withUnsafeBytes { src in
                for (i, r) in rows.enumerated() {
                    precondition(r >= 0 && r < nPairs,
                                 "строка \(r) вне кэша из \(nPairs) пар")
                    let from = src.baseAddress!.advanced(by: r * rb)
                    let to = dst.baseAddress!.advanced(by: i * rb)
                    memcpy(to, from, rb)
                }
            }
        }
        let outShape = [rows.count] + Array(index.shape.dropFirst())
        return dtype == .float16
            ? MLXArray(buf, outShape, type: Float16.self)
            : MLXArray(buf, outShape, type: Float.self)
    }

    /// Батч примеров → (состояния `[b·nCand, nSrc, H, S, S]`, метки `[b]`).
    ///
    /// Кандидаты уложены подряд по примерам: голова считает их одним
    /// проходом, а лосс потом смотрит на `[b, nCand]`.
    public func batch(_ samples: [Int]) -> (states: MLXArray, labels: MLXArray) {
        let rows = samples.flatMap { index.pairIndex[$0] }
        let lbl = samples.map { Int32(index.labels[$0]) }
        return (gather(rows), MLXArray(lbl))
    }

    // ── Диск ──

    /// `base` — путь без расширения; рядом лягут `.states` и `.idx.json`.
    public func save(to base: URL) throws {
        try storage.write(to: Self.statesURL(base))
        try JSONEncoder().encode(index).write(to: Self.indexURL(base))
    }

    /// `mapped: true` (умолчание) — состояния читаются страницами с диска.
    public static func load(_ base: URL, mapped: Bool = true) throws -> StateCache {
        let sURL = statesURL(base), iURL = indexURL(base)
        guard FileManager.default.fileExists(atPath: sURL.path),
              FileManager.default.fileExists(atPath: iURL.path) else {
            throw StateCacheError.fileNotFound(base.path)
        }
        let index = try JSONDecoder().decode(StateCacheIndex.self,
                                             from: Data(contentsOf: iURL))
        let storage = try Data(contentsOf: sURL,
                               options: mapped ? [.mappedIfSafe] : [])
        let want = index.shape.reduce(1, *) * index.dtype.itemSize
        guard storage.count == want else {
            throw StateCacheError.corrupt(
                "\(sURL.lastPathComponent): \(storage.count) байт, ожидалось \(want)")
        }
        return StateCache(storage: storage, index: index)
    }

    static func statesURL(_ base: URL) -> URL {
        base.appendingPathExtension("states")
    }
    static func indexURL(_ base: URL) -> URL {
        base.appendingPathExtension("idx.json")
    }

    /// Совместим ли кэш с этой головой и этим контрактом подачи текста.
    ///
    /// Проверять обязательно, и не из педантизма: кэш, собранный для другого
    /// набора слоёв или другого шаблона, имеет ПРАВДОПОДОБНУЮ форму и
    /// обучение на нём пойдёт как ни в чём не бывало — просто научит не тому.
    /// Единственное, чего проверка не увидит, — подменённый чекпоинт базы.
    public func checkCompatible(head: RerankerHead,
                                contract: [String: String]) throws {
        let wantSrc = head.uniqueSources.count
        guard shape[1] == wantSrc else {
            throw StateCacheError.shapeMismatch(
                "в кэше \(shape[1]) слоёв состояния, голова читает \(wantSrc)")
        }
        for (key, want) in contract {
            if let have = index.contract[key], have != want {
                throw StateCacheError.shapeMismatch(
                    "\(key): кэш собран с '\(have)', сейчас '\(want)'")
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Запись
// ───────────────────────────────────────────────────────────────────────

/// Построение кэша построчно.
///
/// Строки приходят ВРАЗБРОС: пары группируются по общему префиксу документа,
/// а нумеруются по (пример, кандидат). Поэтому запись идёт по смещению, а не
/// последовательно, — и поэтому же файл создаётся сразу нужного размера.
public final class StateCacheWriter {

    private let shape: [Int]
    private let dtype: StateCacheDType
    private let rowBytes: Int
    private var handle: FileHandle?
    private var memory: Data?
    private let base: URL?
    private(set) var maxAbs: Float = 0
    private(set) var rowsWritten = 0

    /// - path: писать сразу на диск. nil ⇒ держать в памяти (тесты, мелкие
    ///   прогоны). Для кэшей, сравнимых с объёмом памяти, диск — разница
    ///   между «работает» и «машина ушла в своп».
    public init(shape: [Int], dtype: StateCacheDType = .float16,
                path: URL? = nil) throws {
        precondition(shape.count == 5, "ожидалась форма [nPairs,nSrc,H,S,S]")
        self.shape = shape
        self.dtype = dtype
        self.rowBytes = shape.dropFirst().reduce(1, *) * dtype.itemSize
        self.base = path

        let total = shape[0] * rowBytes
        if let path {
            let url = StateCache.statesURL(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            let h = try FileHandle(forWritingTo: url)
            try h.truncate(atOffset: UInt64(total))
            self.handle = h
            self.memory = nil
        } else {
            self.handle = nil
            self.memory = Data(count: total)
        }
    }

    /// Записать состояния `values` `[n, nSrc, H, S, S]` в строки `rows`.
    public func write(rows: [Int], _ values: MLXArray) throws {
        precondition(rows.count == values.shape[0],
                     "строк \(rows.count), значений \(values.shape[0])")
        let f32 = values.asType(.float32)
        eval(f32)
        maxAbs = Swift.max(maxAbs, MLX.abs(f32).max().item(Float.self))

        // disambiguate: true выбирает перегрузку, отдающую именно Data,
        // а не обёртку MLXArrayData.
        let payload = (dtype == .float16 ? f32.asType(.float16) : f32)
            .asData(disambiguate: true)
        payload.withUnsafeBytes { src in
            for (i, r) in rows.enumerated() {
                precondition(r >= 0 && r < shape[0],
                             "строка \(r) вне кэша из \(shape[0]) пар")
                let chunk = Data(bytes: src.baseAddress!.advanced(by: i * rowBytes),
                                 count: rowBytes)
                if let handle {
                    try? handle.seek(toOffset: UInt64(r * rowBytes))
                    handle.write(chunk)
                } else {
                    memory!.replaceSubrange(r * rowBytes ..< (r + 1) * rowBytes,
                                            with: chunk)
                }
            }
        }
        rowsWritten += rows.count
    }

    /// Закрыть кэш. Проверка на переполнение fp16 — здесь, а не при записи:
    /// узнать «состояния не влезли» лучше один раз в конце, чем на середине
    /// многоминутного кодирования.
    public func finish(pairIndex: [[Int]], labels: [Int], hardNegs: [[Int]],
                       contract: [String: String]) throws -> StateCache {
        if dtype == .float16 && maxAbs > 60000 {
            throw StateCacheError.fp16Overflow(maxAbs)
        }
        let index = StateCacheIndex(shape: shape, dtype: dtype,
                                    pairIndex: pairIndex, labels: labels,
                                    hardNegs: hardNegs, contract: contract)
        if let handle {
            try handle.close()
            self.handle = nil
            let base = self.base!
            try JSONEncoder().encode(index).write(to: StateCache.indexURL(base))
            // Перечитываем ЧЕРЕЗ mmap, а не оставляем то, что писали: так
            // кэш сразу живёт в том же режиме, в каком его увидит следующий
            // запуск, и разницы между «только что собрали» и «загрузили с
            // диска» не возникает.
            return try StateCache.load(base)
        }
        return StateCache(storage: memory!, index: index)
    }
}
