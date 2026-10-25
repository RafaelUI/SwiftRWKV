import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Свёрнутые инструкции: кэш инструкции запроса.
//
//  Запрос эмбеддера с инструкцией — «Instruct: X\nQuery: q». RWKV — RNN, и
//  всё, что инструкция даёт запросу, — состояние после её токенов. Поэтому
//  инструкцию можно свернуть в состояние ОДИН раз (`foldPrompt`) и
//  подключать его к каждому запросу с этой инструкцией: запрос платит только
//  за свои токены. На длинных инструкциях (политики агентов ~1000 токенов)
//  это и есть основная цена запроса.
//
//  Где резать. Префикс — «Instruct: X\n», хвост — «Query: q». Так
//  токенизация кусков совпадает со сплошной (проверено на 3000 строк двух
//  наборов: 100 %); при разрезе после «Query: » она расходится почти всегда —
//  жадный словарь World склеивает пробел со следующим словом. Совпадение
//  токенов делает свёрнутый путь тем же счётом, что инструкция в строке, до
//  порядка сложения на стыке (тест держит расхождение векторов).
//
//  Чем отличается от инструкции в строке:
//  - обрезка `maxTokens` действует на хвост, а не на всю строку: длинная
//    инструкция больше не вытесняет запрос;
//  - пулинг только `.last` — среднее по хвосту не равно среднему по всей
//    строке;
//  - подключённое состояние модели (обученное) — старт свёртки:
//    «обученное → инструкция → запрос».
//
//  Кэш привязан к модели в её нынешнем виде: свёрнутое состояние зависит от
//  весов и подключённого состояния. Подключённое меняется — кэш узнаёт это
//  по отпечатку и сворачивает заново; веса меняются (после обучения) —
//  создайте новый кэш (оценка внутри библиотеки так и делает).
// ───────────────────────────────────────────────────────────────────────

public enum InstructionCacheError: Error, CustomStringConvertible {
    case meanPooling
    public var description: String {
        "свёрнутая инструкция требует пулинга .last: среднее по хвосту запроса не равно среднему по всей строке"
    }
}

public final class InstructionCache {

    public static let instructPrefix = "Instruct: "
    public static let queryMarker = "\nQuery: "

    public let model: EmbeddingModel
    public let tokenizer: any RWKVTokenizer
    private var folded: [String: RWKVBatchState] = [:]
    private var foldedOver = StateFile.noInitialState

    /// Сколько инструкций свёрнуто и сколько их токенов (для отчёта о цене).
    /// Пересвёртка после смены подключённого состояния тоже считается.
    public private(set) var foldedCount = 0
    public private(set) var foldedTokens = 0

    /// Свёртка кусками по столько токенов, сцепленными состоянием (0 — одним
    /// проходом). `foldPrompt` строит граф всех слоёв на весь промпт сразу, и
    /// его пик растёт с длиной инструкции: на Tool_Use (до ~3000 токенов,
    /// 0.1B) свёртка одним куском подняла пик оценки с 0.69 до 1.27 ГБ.
    /// Счёт тот же до порядка сложения.
    public var foldChunkTokens = 512

    public init(model: EmbeddingModel, tokenizer: any RWKVTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// «Instruct: X\nQuery: q» → (префикс «Instruct: X\n», хвост «Query: q»);
    /// nil — в строке нет инструкции этого вида. Режется по ПЕРВОМУ
    /// «\nQuery: » — как его снимает вторая стадия embed-run.
    public static func split(_ anchor: String) -> (prefix: String, tail: String)? {
        guard anchor.hasPrefix(instructPrefix), let r = anchor.range(of: queryMarker) else { return nil }
        let cut = anchor.index(after: r.lowerBound)            // после «\n»
        return (String(anchor[..<cut]), String(anchor[cut...]))
    }

    /// Свёрнутое сброшено, если подключённое состояние модели сменилось.
    private func refresh() {
        let over = model.backbone.initialStateFingerprint
        if over != foldedOver { folded.removeAll(); foldedOver = over }
    }

    private func lookup(_ prefix: String) -> RWKVBatchState {
        if let s = folded[prefix] { return s }
        let ids = tokenizer.encode(prefix)
        let s = fold(ids)
        folded[prefix] = s
        foldedCount += 1
        foldedTokens += ids.count
        return s
    }

    private func fold(_ ids: [Int]) -> RWKVBatchState {
        let bb = model.backbone
        let step = foldChunkTokens > 0 ? foldChunkTokens : ids.count
        var s: RWKVBatchState? = nil                           // первый кусок — с подключённого
        var i = 0
        while i < ids.count {
            s = bb.foldPrompt(Array(ids[i ..< Swift.min(i + step, ids.count)]), from: s)
            i += step
        }
        return s!
    }

    /// Состояние после префикса — свёрнутое один раз на кэш (и на
    /// подключённое состояние модели).
    public func state(forPrefix prefix: String) -> RWKVBatchState {
        refresh()
        return lookup(prefix)
    }

    /// Банк свёрнутых инструкций (батч K): строка k — состояние после
    /// `prefixes[k]`. Старт для банка состояний на инструкцию
    /// (`StateBankTrainableSet`): с ним запрос «Query: q» со строкой k считается
    /// так же, как «prefixes[k]Query: q» в строке.
    public func bank(_ prefixes: [String]) -> RWKVBatchState {
        precondition(!prefixes.isEmpty, "bank: пустой список инструкций")
        refresh()
        let ss = prefixes.map { lookup($0) }
        return RWKVBatchState(wkv: concatenated(ss.map(\.wkv), axis: 1),
                              tmixShift: concatenated(ss.map(\.tmixShift), axis: 1),
                              cmixShift: concatenated(ss.map(\.cmixShift), axis: 1)).evaluated()
    }

    /// Векторы запросов [N, D] со свёрнутыми инструкциями. Строки без
    /// инструкции векторизуются как есть. Порядок строк сохраняется.
    /// Подключённое состояние модели после вызова — прежнее (те же массивы).
    /// `maxTokens` обрезает хвост, а не всю строку.
    public func embedQueries(_ anchors: [String], terminator: Int = 0, maxTokens: Int? = 512,
                             batchTokens: Int = EmbeddingMetrics.evaluationBatchTokens) throws -> MLXArray {
        guard model.pooling == .last else { throw InstructionCacheError.meanPooling }
        precondition(!anchors.isEmpty, "embedQueries: пустой список")
        refresh()
        var groups: [String: [Int]] = [:], order: [String] = []
        var plain: [Int] = []
        var tails = [String](repeating: "", count: anchors.count)
        for (i, a) in anchors.enumerated() {
            guard let (p, t) = Self.split(a) else { plain.append(i); continue }
            if groups[p] == nil { order.append(p) }
            groups[p, default: []].append(i)
            tails[i] = t
        }
        var rows = [MLXArray?](repeating: nil, count: anchors.count)
        func put(_ idx: [Int], _ v: MLXArray) { for (k, i) in idx.enumerated() { rows[i] = v[k] } }
        if !plain.isEmpty {
            put(plain, EmbeddingMetrics.embedAll(model: model, tokenizer: tokenizer, texts: plain.map { anchors[$0] },
                                                 terminator: terminator, maxTokens: maxTokens,
                                                 batchTokens: batchTokens, role: .query))
        }
        let bb = model.backbone
        for p in order {
            let idx = groups[p]!
            let s = lookup(p)
            put(idx, bb.withInitialState(s) {
                EmbeddingMetrics.embedAll(model: model, tokenizer: tokenizer, texts: idx.map { tails[$0] },
                                          terminator: terminator, maxTokens: maxTokens,
                                          batchTokens: batchTokens, role: .query)
            })
        }
        let out = stacked(rows.map { $0! }, axis: 0)
        eval(out)
        return out
    }
}
