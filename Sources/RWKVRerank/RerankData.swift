import Foundation

// ───────────────────────────────────────────────────────────────────────
//  Данные реранкера. Порт rwkv_metal/reranker/data.py, расширенный вторым
//  форматом строк.
//
//  Ключевое решение — порядок в шаблоне. Документ идёт ДО запроса:
//
//      Instruct: {инструкция}
//      Document: {документ}
//      Query: {запрос}
//
//  Причина не стилистическая. RWKV — RNN, состояние после префикса зависит
//  ТОЛЬКО от префикса. Поставив документ первым, мы делаем состояние
//  «Instruct + Document» кэшируемым: посчитали один раз на документ, а
//  каждый следующий запрос стоит только своих токенов. Поставив запрос
//  первым, эту возможность пришлось бы выбросить.
//
//  Отсюда же берётся почти бесплатное расширение набора негативов: префиксы
//  документов внутри шага уже посчитаны, поэтому спарить запрос со ВСЕМИ
//  документами шага стоит только хвостов запроса.
//
//  Плата за такой порядок есть, и она честная: документ кодируется, ещё не
//  зная запроса. «Слепым» он при этом не остаётся — голова читает состояние
//  ПОСЛЕ того, как в него свернулись токены запроса, так что обусловленность
//  запросом происходит на считывании, а не при кодировании документа.
// ───────────────────────────────────────────────────────────────────────

public let defaultRerankInstruct =
    "Given a search query, retrieve relevant passages that answer the query"

/// Шаблон пары, разбитый на префикс (кэшируемый) и суффикс.
public struct PairTemplate: Sendable, Equatable {

    /// `true` → «Instruct/Document/Query»: префикс кэшируется по паре
    /// (инструкция, документ). `false` → «Instruct/Query/Document»:
    /// кэшировать нечего, зато документ читается уже зная запрос.
    /// Оставлено для честного сравнения, умолчание — `true`.
    public var docFirst: Bool

    public init(docFirst: Bool = true) {
        self.docFirst = docFirst
    }

    /// Кэшируемая часть. При `docFirst == false` от документа не зависит
    /// ничего, кроме инструкции, — кэшировать нечего, и это видно прямо
    /// в результате.
    public func prefix(instruct: String, document: String) -> String {
        docFirst ? "Instruct: \(instruct)\nDocument: \(document)\n"
                 : "Instruct: \(instruct)\n"
    }

    public func suffix(document: String, query: String) -> String {
        docFirst ? "Query: \(query)"
                 : "Query: \(query)\nDocument: \(document)"
    }

    public func full(instruct: String, document: String, query: String) -> String {
        prefix(instruct: instruct, document: document)
            + suffix(document: document, query: query)
    }

    /// Строка, однозначно определяющая контракт подачи текста. Пишется в
    /// метаданные чекпоинта и индекса: состояние — функция ровно этого
    /// префикса, и при смене шаблона любой кэш становится неверным молча.
    public var contract: String { docFirst ? "doc_first" : "query_first" }
}

// ───────────────────────────────────────────────────────────────────────
//  Детерминированный ГПСЧ
// ───────────────────────────────────────────────────────────────────────

/// SplitMix64 — намеренно СВОЙ, а не системный `SystemRandomNumberGenerator`.
///
/// Системный сидом не управляется, и тогда два прогона с одним `seed`
/// получали бы разные наборы кандидатов. Любое сравнение двух прогонов
/// (детерминизм обучения, сравнение конфигураций на одном кэше) сломалось бы
/// из-за данных, а не из-за кода, — и выглядело бы как настоящее расхождение.
///
/// С питоновским `random.Random` побитового совпадения НЕТ и не будет: там
/// Mersenne Twister. Поэтому наборы кандидатов между реализациями
/// сравнивать бессмысленно, и паритет здесь проверяется по СВОЙСТВАМ
/// (позитив на месте, кандидаты различны, позиция перемешана), а не по
/// совпадению последовательностей.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed &* 0x2545_F491_4F6C_DD1D &+ 0x9E37_79B9_7F4A_7C15
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Равномерное целое в `0 ..< n`. Через отбрасывание смещённого хвоста,
    /// а не через остаток: `next() % n` перекошен в пользу малых значений,
    /// и на маленьком пуле документов этот перекос виден.
    public mutating func below(_ n: Int) -> Int {
        precondition(n > 0)
        let bound = UInt64(n)
        let limit = UInt64.max - (UInt64.max % bound) - 1
        var r = next()
        while r > limit { r = next() }
        return Int(r % bound)
    }

    /// Перемешивание Фишера — Йетса.
    public mutating func shuffled<T>(_ xs: [T]) -> [T] {
        var a = xs
        guard a.count > 1 else { return a }
        for i in stride(from: a.count - 1, to: 0, by: -1) {
            a.swapAt(i, below(i + 1))
        }
        return a
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Строки датасета
// ───────────────────────────────────────────────────────────────────────

/// Нормализованная строка: запрос, его позитив и майненные негативы.
///
/// Негативов может быть НЕСКОЛЬКО. У LitRetrieval он один, у
/// reranker-triples-multi — до пяти, и сплющивать вторую форму к первой
/// значило бы выбросить четыре пятых самого ценного, что в датасете есть.
public struct RerankRow: Sendable, Equatable {
    public var instruct: String
    public var query: String
    public var positive: String
    public var negatives: [String]
    public var language: String?

    public init(instruct: String = defaultRerankInstruct, query: String,
                positive: String, negatives: [String] = [],
                language: String? = nil) {
        self.instruct = instruct
        self.query = query
        self.positive = positive
        self.negatives = negatives
        self.language = language
    }
}

public enum RerankDataError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case noRows(String)
    case poolTooSmall(needed: Int, have: Int)

    public var description: String {
        switch self {
        case .fileNotFound(let p): return "файл не найден: \(p)"
        case .noRows(let why): return "не набралось строк: \(why)"
        case .poolTooSmall(let needed, let have):
            return """
                не удалось набрать \(needed) различных кандидатов: в пуле \
                всего \(have) документов. Уменьши nCandidates или возьми \
                больше строк.
                """
        }
    }
}

public enum RerankDataset {

    /// «Instruct: X\nQuery: Y» → (X, Y). Без префикса — весь текст запрос.
    public static func parseAnchor(_ anchor: String) -> (String, String) {
        guard anchor.hasPrefix("Instruct:"),
              let q = anchor.range(of: "\nQuery:") else {
            return (defaultRerankInstruct, anchor)
        }
        let instruct = anchor[anchor.index(anchor.startIndex, offsetBy: 9) ..< q.lowerBound]
        let query = anchor[q.upperBound...]
        return (instruct.trimmingCharacters(in: .whitespacesAndNewlines),
                query.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Прочитать JSONL с РЕЗЕРВУАРНОЙ выборкой (алгоритм R).
    ///
    /// Полный LitRetrieval — 2.6 ГБ на 554 тысячи строк, и читать его
    /// целиком в память нельзя. Но и «первые N» — не выборка: у собранного
    /// по источникам корпуса начало файла систематически отличается от
    /// середины. Здесь в памяти живут ровно `limit` строк, и они —
    /// равномерная подвыборка всего файла.
    ///
    /// Цена честности: файл всё равно прочитывается ЦЕЛИКОМ. Для 2.6 ГБ это
    /// десятки секунд один раз; «первые N» вернулись бы мгновенно, но
    /// сравнивать результаты на них было бы не с чем.
    ///
    /// Формат строки определяется по полям, а не по флагу:
    ///   * `{anchor, positive, negative, task}` — LitRetrieval;
    ///   * `{query, positive, negatives[], language}` — reranker-triples-multi.
    ///
    /// Строки с битым UTF-8, незнакомой формой и без единого негатива
    /// пропускаются молча: в корпусе такого размера одиночный мусор — норма,
    /// а строка без негативов для listwise бесполезна (список кандидатов
    /// вырождается в одного позитива).
    public static func loadJSONL(path: String, limit: Int? = nil,
                                 seed: UInt64 = 0, task: String? = "retrieval",
                                 language: String? = nil) throws -> [RerankRow] {
        let expanded = (path as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded),
              let stream = InputStream(fileAtPath: expanded) else {
            throw RerankDataError.fileNotFound(expanded)
        }
        stream.open()
        defer { stream.close() }

        var rng = SplitMix64(seed: seed)
        var reservoir: [RerankRow] = []
        var seen = 0

        func offer(_ row: RerankRow) {
            guard let limit else { reservoir.append(row); return }
            seen += 1
            if reservoir.count < limit {
                reservoir.append(row)
            } else {
                let j = rng.below(seen)
                if j < limit { reservoir[j] = row }
            }
        }

        func parse(_ data: Data) -> RerankRow? {
            guard let obj = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                  let positive = obj["positive"] as? String, !positive.isEmpty
            else { return nil }

            let lang = obj["language"] as? String
            if let language, lang != language { return nil }

            if let anchor = obj["anchor"] as? String {
                // LitRetrieval
                if let task, (obj["task"] as? String) != task { return nil }
                let (instruct, query) = parseAnchor(anchor)
                let neg = obj["negative"] as? String
                let negs = (neg?.isEmpty == false) ? [neg!] : []
                guard !negs.isEmpty else { return nil }
                return RerankRow(instruct: instruct, query: query,
                                 positive: positive, negatives: negs,
                                 language: lang)
            }
            if let query = obj["query"] as? String, !query.isEmpty {
                // reranker-triples-multi
                let negs = (obj["negatives"] as? [Any] ?? [])
                    .compactMap { $0 as? String }.filter { !$0.isEmpty }
                guard !negs.isEmpty else { return nil }
                return RerankRow(query: query, positive: positive,
                                 negatives: negs, language: lang)
            }
            return nil
        }

        var buffer = Data()
        let chunkSize = 1 << 20
        var raw = [UInt8](repeating: 0, count: chunkSize)

        func drain(_ finalize: Bool) {
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer[buffer.startIndex ..< nl]
                buffer.removeSubrange(buffer.startIndex ... nl)
                if let r = parse(line) { offer(r) }
            }
            if finalize, !buffer.isEmpty, let r = parse(buffer) { offer(r) }
        }

        while stream.hasBytesAvailable {
            let n = stream.read(&raw, maxLength: chunkSize)
            if n <= 0 { break }
            buffer.append(contentsOf: raw[0 ..< n])
            drain(false)
        }
        drain(true)

        guard !reservoir.isEmpty else {
            throw RerankDataError.noRows("\(expanded): ни одной валидной строки")
        }
        return reservoir
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Кандидаты
// ───────────────────────────────────────────────────────────────────────

/// Один запрос со своим набором кандидатов.
public struct RerankSample: Sendable, Equatable {
    public var instruct: String
    public var query: String
    /// Индексы документов в общем пуле; порядок = порядок кандидатов.
    public var docIds: [Int]
    /// Позиция правильного документа внутри `docIds`.
    public var label: Int
    /// Позиции МАЙНЕННЫХ негативов внутри `docIds`.
    ///
    /// Нужны для честной метрики: случайный документ из корпуса отличить
    /// легко, а выбранный майнером — нет, и общий MRR по восьми кандидатам
    /// это различие размывает. Именно колонка «против майненного негатива»
    /// показывает, за что реранкер вообще нужен.
    public var hardNegs: [Int]

    public init(instruct: String, query: String, docIds: [Int], label: Int,
                hardNegs: [Int] = []) {
        self.instruct = instruct
        self.query = query
        self.docIds = docIds
        self.label = label
        self.hardNegs = hardNegs
    }
}

public enum RerankCandidates {

    /// Строки → (пул документов, примеры).
    ///
    /// Кандидаты каждого запроса:
    ///   1. его позитив,
    ///   2. его майненные негативы (сколько влезет),
    ///   3. остальные — случайные документы из общего пула.
    ///
    /// Пул дедуплицирован: документ, встретившийся у нескольких запросов,
    /// кодируется ОДИН раз. Именно поэтому «добавить негативов» стоит почти
    /// ничего — платим только за хвост запроса.
    ///
    /// Позиция позитива ПЕРЕМЕШИВАЕТСЯ. При listwise-лоссе фиксированная
    /// позиция — ярлык, который голова выучит вместо задачи, и заметить это
    /// по лоссу нельзя: он будет исправно падать.
    ///
    /// Честная оговорка про добранные негативы: документ из пула — это
    /// позитив ЧУЖОГО запроса, и никто не проверяет, не отвечает ли он
    /// заодно на этот. На литературных пассажах столкновения редки, но
    /// метрика оптимистична на неизмеренную величину.
    public static func build(_ rows: [RerankRow], nCandidates: Int = 8,
                             seed: UInt64 = 0)
        throws -> (pool: [String], samples: [RerankSample]) {
        precondition(nCandidates >= 2, "кандидатов должно быть хотя бы два")
        var rng = SplitMix64(seed: seed)

        var pool: [String] = []
        var docToId: [String: Int] = [:]
        func add(_ doc: String) -> Int {
            if let i = docToId[doc] { return i }
            let i = pool.count
            docToId[doc] = i
            pool.append(doc)
            return i
        }

        // Пул набирается ЦЕЛИКОМ до раздачи кандидатов: иначе ранние запросы
        // добирали бы негативы из куцего пула, а поздние — из полного, и
        // сложность примера зависела бы от номера строки в файле.
        let prepared: [(RerankRow, Int, [Int])] = rows.map { row in
            let posId = add(row.positive)
            let negIds = row.negatives.map { add($0) }.filter { $0 != posId }
            return (row, posId, negIds)
        }

        guard pool.count >= nCandidates else {
            throw RerankDataError.poolTooSmall(needed: nCandidates, have: pool.count)
        }

        var samples: [RerankSample] = []
        samples.reserveCapacity(prepared.count)

        for (row, posId, negIds) in prepared {
            var cand: [Int] = [posId]
            var inCand: Set<Int> = [posId]
            for n in negIds where cand.count < nCandidates {
                if inCand.insert(n).inserted { cand.append(n) }
            }
            let mined = Set(cand.dropFirst())

            // Добор из пула. Ограничитель попыток нужен, потому что
            // случайная выборка без повторов на почти исчерпанном пуле может
            // долго не попадать в свободный документ; при pool.count >=
            // nCandidates он гарантированно завершается, но время не
            // ограничено сверху.
            var guardCount = 0
            while cand.count < nCandidates && guardCount < 50 * nCandidates {
                guardCount += 1
                let j = rng.below(pool.count)
                if inCand.insert(j).inserted { cand.append(j) }
            }
            guard cand.count == nCandidates else {
                throw RerankDataError.poolTooSmall(needed: nCandidates,
                                                   have: pool.count)
            }

            let shuffled = rng.shuffled(cand)
            var position: [Int: Int] = [:]
            for (i, d) in shuffled.enumerated() { position[d] = i }

            samples.append(RerankSample(
                instruct: row.instruct, query: row.query,
                docIds: shuffled, label: position[posId]!,
                hardNegs: mined.compactMap { position[$0] }.sorted()))
        }
        return (pool, samples)
    }

    /// Непересекающийся held-out ПО ЗАПРОСАМ.
    ///
    /// Документы при этом пул делят, и это осознанно: меряется ранжирование,
    /// а не запоминание того, какие документы существуют. Оговорка: документ,
    /// бывший позитивом в обучении, может встретиться негативом в оценке.
    /// Ярлык «этот документ вообще-то позитивный» предсказательной силы почти
    /// не имеет (каждый документ — позитив ровно одного запроса), но это НЕ
    /// измерено, и разбиение с непересекающимися документами вопрос бы закрыло.
    public static func splitTrainEval(_ samples: [RerankSample], nEval: Int,
                                      seed: UInt64 = 0)
        -> (train: [RerankSample], eval: [RerankSample]) {
        precondition(nEval >= 0 && nEval <= samples.count,
                     "nEval=\(nEval) вне 0...\(samples.count)")
        var rng = SplitMix64(seed: seed)
        let idx = rng.shuffled(Array(0 ..< samples.count))
        return (idx.dropFirst(nEval).map { samples[$0] },
                idx.prefix(nEval).map { samples[$0] })
    }
}
