import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Выдача: применить обученную голову.
//
//  Порт rwkv_metal/reranker/rerank.py. Два режима, и выбор между ними —
//  не про скорость вообще, а про то, повторяются ли документы.
//
//  Прямой путь (`score` / `rank`)
//      Каждая пара считается целиком: O(L_doc + L_query) на кандидата.
//      Ничего не хранит, годится когда документы каждый раз новые.
//
//  С индексом (`buildIndex` / `scoreIndexed`)
//      Состояние префикса «Instruct + Document» считается ОДИН раз и
//      хранится. Дальше запрос стоит O(L_query) на кандидата. Питоновский
//      замер на M4 Air (0.1B): 73 мс против 3.5 мс, то есть примерно 20×.
//      В Swift эти числа не воспроизводились и здесь не заявляются — но
//      асимметрия та же, и она структурная, а не про константы.
//
//      Цена — память: индекс держит ПОЛНОЕ состояние (все слои), потому
//      что продолжать префикс запросом нужно по всей глубине. Для 0.1B это
//      ~2.4 МБ на документ в fp32. Индекс на тысячу документов — 2.4 ГБ,
//      поэтому он для «горячего» подмножества (например, top-100 от
//      эмбеддера), а не для всего корпуса.
//
//  Главное, что здесь можно сделать не так, — подать текст иначе, чем при
//  обучении. Ошибок формы при этом не возникает нигде: скоры выходят
//  правдоподобные, просто хуже. Поэтому контракт подачи текста здесь не
//  комментарий, а проверяемая величина, и рекомендуемый способ построения —
//  `fromCheckpoint`, где он берётся из файла головы, а не из памяти
//  человека.
// ───────────────────────────────────────────────────────────────────────

/// Контракт подачи текста при выдаче.
///
/// Обёртка над `RerankEncodeConfig`, а не собственный набор полей — и это
/// главное решение в файле. Токенизация при выдаче идёт через ТЕ ЖЕ
/// `prefixIds`/`suffixIds` с тем же типом конфигурации, что и при
/// кодировании обучающих пар. Разойтись им негде по построению; будь здесь
/// вторая копия правил подачи, она обязана была бы совпадать с первой — то
/// есть разошлась бы.
///
/// `instruct` живёт отдельно потому, что при обучении это поле ПРИМЕРА
/// (в корпусе инструкций несколько), а при выдаче — часть замороженного
/// префикса, одна на индекс.
public struct RerankServingConfig: Sendable {

    public var encode: RerankEncodeConfig
    public var instruct: String

    public init(encode: RerankEncodeConfig = RerankEncodeConfig(),
                instruct: String = defaultRerankInstruct) {
        self.encode = encode
        self.instruct = instruct
    }

    /// То, что записано в чекпоинт головы и в индекс.
    public var contract: [String: String] {
        encode.contract.merging(["instruct": instruct]) { a, _ in a }
    }

    /// Ключи, относящиеся к ПРЕФИКСУ, то есть к тому, что попадает в индекс.
    ///
    /// Обрезка запроса и терминатор сюда не входят: они в хвосте, префикс от
    /// них не зависит, и требовать их совпадения значило бы запрещать
    /// законное — переиндексировать корпус ради смены длины запроса.
    public static let prefixKeys = ["template", "instruct", "max_doc_tokens"]

    public var prefixContract: [String: String] {
        contract.filter { Self.prefixKeys.contains($0.key) }
    }

    /// Собрать из метаданных чекпоинта головы.
    ///
    /// Значения берутся ИЗ ФАЙЛА, а не из аргументов вызова. Шаблон,
    /// обрезки, терминатор и инструкция — часть того, на чём голова
    /// обучалась; подать текст иначе не ошибка формы, а тихая потеря
    /// качества.
    public init(metadata md: [String: String],
                base: RerankEncodeConfig = RerankEncodeConfig()) {
        var enc = base
        if let t = md["template"] { enc.template = PairTemplate(docFirst: t != "query_first") }
        if let v = md["max_doc_tokens"], let n = Int(v) { enc.maxDocTokens = n }
        if let v = md["max_query_tokens"], let n = Int(v) { enc.maxQueryTokens = n }
        if let v = md["terminator"] { enc.terminator = (v == "none" || v.isEmpty) ? nil : Int(v) }
        self.encode = enc
        self.instruct = md["instruct"] ?? defaultRerankInstruct
    }
}

public enum RerankServeError: Error, CustomStringConvertible {
    case contractMismatch(key: String, index: String, now: String)
    case indexRequiresDocFirst
    case emptyIndex
    case docIdOutOfRange(Int, count: Int)
    case corruptIndex(String)

    public var description: String {
        switch self {
        case .contractMismatch(let key, let a, let b):
            return """
                \(key): индекс построен с '\(a)', скорится с '\(b)'. \
                Состояние есть функция ровно префикса, поэтому \
                несовпадение здесь не ошибка формы — это правдоподобные, \
                но неверные числа.
                """
        case .indexRequiresDocFirst:
            return """
                индекс имеет смысл только при docFirst = true: при обратном \
                порядке префикс зависит от запроса, и кэшировать нечего.
                """
        case .emptyIndex: return "индекс пуст"
        case .docIdOutOfRange(let i, let n):
            return "документ \(i) вне индекса из \(n)"
        case .corruptIndex(let s): return "индекс повреждён: \(s)"
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Индекс префиксов документов
// ───────────────────────────────────────────────────────────────────────

/// Состояния префиксов «Instruct + Document», по одному на документ.
///
/// Поля кроме `state` — не документация, а КОНТРАКТ. Индекс, построенный с
/// другой инструкцией или другой обрезкой документа, содержит совершенно
/// правдоподобные числа не от того текста, и `scoreIndexed` обязан упасть,
/// а не вернуть их.
///
/// Единица здесь — ДОКУМЕНТ, а не пара, поэтому это не `StateCache`: там
/// хранится финальное состояние пары и только читаемые головой слои, здесь —
/// промежуточное состояние префикса и обязательно все слои, потому что
/// продолжать его запросом придётся по всей глубине.
public struct DocIndex {

    public let state: RWKVBatchState     // [L, nDocs, ...]
    public let docs: [String]
    /// `template`, `instruct`, `max_doc_tokens` — то, от чего зависит префикс.
    public let contract: [String: String]

    public init(state: RWKVBatchState, docs: [String],
                contract: [String: String]) {
        precondition(state.batch == docs.count,
                     "состояний \(state.batch), документов \(docs.count)")
        self.state = state
        self.docs = docs
        self.contract = contract
    }

    public var count: Int { docs.count }
    public var nbytes: Int { state.nbytes }

    /// Совместим ли индекс с этой конфигурацией выдачи.
    public func check(_ cfg: RerankServingConfig) throws {
        let now = cfg.prefixContract
        for key in RerankServingConfig.prefixKeys {
            guard let have = contract[key] else { continue }
            let want = now[key] ?? ""
            guard have == want else {
                throw RerankServeError.contractMismatch(key: key, index: have,
                                                        now: want)
            }
        }
    }

    // ── Диск ──
    //
    // Формат: safetensors с тремя тензорами состояния; контракт и сами
    // документы — в metadata. Документы хранятся ВНУТРИ файла намеренно:
    // индекс без текстов бесполезен (ранжировать надо что-то), а разъехаться
    // двум файлам ничего не мешает.

    public static let format = "swiftrwkv-docindex-v1"

    public func save(to url: URL) throws {
        var md = contract
        md["format"] = Self.format
        md["n_docs"] = String(docs.count)
        md["docs"] = String(data: try JSONEncoder().encode(docs),
                            encoding: .utf8) ?? "[]"
        let arrays = ["wkv": state.wkv, "tmix_shift": state.tmixShift,
                      "cmix_shift": state.cmixShift]
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, metadata: md, url: url)
    }

    public static func load(_ url: URL) throws -> DocIndex {
        let (arrays, md) = try loadArraysAndMetadata(url: url)
        guard (md["format"] ?? "") == format else {
            throw RerankServeError.corruptIndex(
                "\(url.lastPathComponent): не индекс документов")
        }
        guard let wkv = arrays["wkv"], let tmix = arrays["tmix_shift"],
              let cmix = arrays["cmix_shift"] else {
            throw RerankServeError.corruptIndex("нет тензоров состояния")
        }
        guard let raw = md["docs"]?.data(using: .utf8),
              let docs = try? JSONDecoder().decode([String].self, from: raw) else {
            throw RerankServeError.corruptIndex("не разобрались тексты документов")
        }
        let state = RWKVBatchState(wkv: wkv, tmixShift: tmix, cmixShift: cmix)
        guard state.batch == docs.count else {
            throw RerankServeError.corruptIndex(
                "состояний \(state.batch), текстов \(docs.count)")
        }
        var contract = md
        for k in ["format", "n_docs", "docs"] { contract.removeValue(forKey: k) }
        return DocIndex(state: state, docs: docs, contract: contract)
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Выдача
// ───────────────────────────────────────────────────────────────────────

public final class RerankerInference {

    public let model: Reranker
    public let tokenizer: WorldTokenizer
    public let config: RerankServingConfig

    public init(model: Reranker, tokenizer: WorldTokenizer,
                config: RerankServingConfig = RerankServingConfig()) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    }

    /// Рекомендуемый способ: конфигурация берётся ИЗ чекпоинта головы.
    ///
    /// Голова и контракт подачи текста едут в одном файле, и разъехаться им
    /// негде. Собранная вручную конфигурация может отличаться от обучения
    /// молча — формы сойдутся, качество упадёт, причину искать будет негде.
    public static func fromCheckpoint(
        base: X070Backbone, tokenizer: WorldTokenizer, head url: URL,
        overrides: ((inout RerankServingConfig) -> Void)? = nil
    ) throws -> RerankerInference {
        let md = try Reranker.readHeadMetadata(url)
        let model = try Reranker.fromHead(base: base, url: url)
        var cfg = RerankServingConfig(metadata: md)
        overrides?(&cfg)
        return RerankerInference(model: model, tokenizer: tokenizer, config: cfg)
    }

    /// Контракт, который стоит класть в `saveHead(extra:)`.
    public var servingMetadata: [String: String] { config.contract }

    // ── Прямой путь ────────────────────────────────────────────────────

    /// Скоры `[docs.count]`. Больше — релевантнее.
    ///
    /// Величина СЫРАЯ (логит) и сравнима только внутри одного запроса: голова
    /// обучалась listwise-лоссом, который фиксирует лишь порядок. Абсолютной
    /// шкалы у неё нет, и sigmoid от этого числа означал бы вероятность
    /// только при обучении с ненулевым BCE-членом.
    public func score(query: String, docs: [String],
                      instruct: String? = nil) throws -> [Float] {
        guard !docs.isEmpty else { return [] }
        let ins = instruct ?? config.instruct
        let enc = config.encode
        var out: [Float] = []
        out.reserveCapacity(docs.count)
        var start = 0
        while start < docs.count {
            let end = Swift.min(start + enc.docBatch, docs.count)
            let seqs = docs[start ..< end].map { d in
                RerankEncoder.prefixIds(tokenizer, enc, instruct: ins, document: d)
                + RerankEncoder.suffixIds(tokenizer, enc, document: d, query: query)
            }
            if start == 0 { try RerankEncoder.checkVocab(model, Array(seqs)) }
            let (idx, mask, endIdx) = RerankEncoder.batchIds(
                Array(seqs), bucket: enc.lengthBucket,
                minT: enc.lengthBucketMinTokens)
            let s = model(idx, mask: mask, endIdx: endIdx).asType(.float32)
            eval(s)
            out.append(contentsOf: s.asArray(Float.self))
            start = end
        }
        return out
    }

    /// `[(индекс документа, скор)]` по убыванию скора.
    public func rank(query: String, docs: [String], topK: Int? = nil,
                     instruct: String? = nil) throws -> [(index: Int, score: Float)] {
        let s = try score(query: query, docs: docs, instruct: instruct)
        return Self.order(s, ids: Array(0 ..< s.count), topK: topK)
    }

    // ── Путь с индексом ────────────────────────────────────────────────

    /// Посчитать и сохранить состояния префиксов документов.
    ///
    /// - dtype: `.float16` ополовинит память индекса. Рекуррентная часть в
    ///   продолжении всё равно считается в fp32; на совпадение со сплошным
    ///   проходом это влияет, и насколько — замерено тестом, а не обещано.
    public func buildIndex(docs: [String], instruct: String? = nil,
                           dtype: DType? = nil,
                           progress: ((Int, Int) -> Void)? = nil) throws -> DocIndex {
        guard !docs.isEmpty else { throw RerankServeError.emptyIndex }
        guard config.encode.template.docFirst else {
            throw RerankServeError.indexRequiresDocFirst
        }
        let ins = instruct ?? config.instruct
        let enc = config.encode
        if enc.cacheLimitGB > 0 {
            MLX.GPU.set(cacheLimit: Int(enc.cacheLimitGB * 1e9))
        }
        var parts: [RWKVBatchState] = []
        var start = 0
        while start < docs.count {
            let end = Swift.min(start + enc.docBatch, docs.count)
            let seqs = docs[start ..< end].map {
                RerankEncoder.prefixIds(tokenizer, enc, instruct: ins, document: $0)
            }
            if start == 0 { try RerankEncoder.checkVocab(model, Array(seqs)) }
            let (idx, mask, endIdx) = RerankEncoder.batchIds(
                Array(seqs), bucket: enc.lengthBucket,
                minT: enc.lengthBucketMinTokens)
            var st = model.encode(idx, mask: mask, endIdx: endIdx)
            if let dtype { st = st.asType(dtype) }
            parts.append(st.evaluated())
            start = end
            progress?(end, docs.count)
        }
        let state = parts.count == 1 ? parts[0]
                                     : RWKVBatchState.concatenated(parts)
        var contract = config.prefixContract
        contract["instruct"] = ins
        return DocIndex(state: state, docs: docs, contract: contract)
    }

    /// То же, но состояния уходят на диск ПО ХОДУ, а не копятся в памяти.
    ///
    /// `buildIndex` держит все пачки и склеивает их в конце — то есть пик
    /// памяти равен размеру всего индекса плюс копия при склейке. Индекс
    /// это ~2.4 МБ на документ в fp32, так что тысяча документов упирается
    /// в память ровно там, где индекс и становится интересен.
    ///
    /// Здесь пачка пишется в файл и отпускается. Пик — одна пачка, а не весь
    /// индекс. Цена: результат приходится перечитать с диска, поэтому
    /// возвращается загруженный `DocIndex`, а не построенный.
    ///
    /// Формат тот же, что у `DocIndex.save`, — иначе получилось бы два
    /// формата, обязанных совпадать.
    @discardableResult
    public func buildIndexToDisk(docs: [String], at url: URL,
                                 instruct: String? = nil, dtype: DType? = nil,
                                 progress: ((Int, Int) -> Void)? = nil)
        throws -> DocIndex {
        guard !docs.isEmpty else { throw RerankServeError.emptyIndex }
        guard config.encode.template.docFirst else {
            throw RerankServeError.indexRequiresDocFirst
        }
        let ins = instruct ?? config.instruct
        let enc = config.encode
        if enc.cacheLimitGB > 0 {
            MLX.GPU.set(cacheLimit: Int(enc.cacheLimitGB * 1e9))
        }

        var wkv: [MLXArray] = [], tmix: [MLXArray] = [], cmix: [MLXArray] = []
        var start = 0
        while start < docs.count {
            let end = Swift.min(start + enc.docBatch, docs.count)
            let seqs = docs[start ..< end].map {
                RerankEncoder.prefixIds(tokenizer, enc, instruct: ins, document: $0)
            }
            if start == 0 { try RerankEncoder.checkVocab(model, Array(seqs)) }
            let (idx, mask, endIdx) = RerankEncoder.batchIds(
                Array(seqs), bucket: enc.lengthBucket,
                minT: enc.lengthBucketMinTokens)
            var st = model.encode(idx, mask: mask, endIdx: endIdx)
            if let dtype { st = st.asType(dtype) }
            st = st.evaluated()
            // Пачка выносится в CPU-буфер и отпускается: держать её в MLX
            // означало бы ровно ту память, от которой здесь и уходим.
            wkv.append(st.wkv); tmix.append(st.tmixShift); cmix.append(st.cmixShift)
            start = end
            progress?(end, docs.count)
        }

        var contract = config.prefixContract
        contract["instruct"] = ins
        var md = contract
        md["format"] = DocIndex.format
        md["n_docs"] = String(docs.count)
        md["docs"] = String(data: try JSONEncoder().encode(docs),
                            encoding: .utf8) ?? "[]"
        // Склейка по оси документов (1), а не по слоям: [L, B, ...].
        let arrays = ["wkv": concatenated(wkv, axis: 1),
                      "tmix_shift": concatenated(tmix, axis: 1),
                      "cmix_shift": concatenated(cmix, axis: 1)]
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, metadata: md, url: url)
        return try DocIndex.load(url)
    }

    /// Скоры запроса против документов индекса (всех или подмножества).
    public func scoreIndexed(query: String, index: DocIndex,
                             docIds: [Int]? = nil,
                             instruct: String? = nil) throws -> [Float] {
        var cfg = config
        if let instruct { cfg.instruct = instruct }
        try index.check(cfg)

        let ids = docIds ?? Array(0 ..< index.count)
        for i in ids where i < 0 || i >= index.count {
            throw RerankServeError.docIdOutOfRange(i, count: index.count)
        }
        guard !ids.isEmpty else { return [] }

        let enc = config.encode
        // Хвост от документа не зависит (docFirst проверен при построении
        // индекса), поэтому токенизируется ОДИН раз на запрос, а не на
        // кандидата. Это и есть та экономия, ради которой индекс существует.
        let qIds = RerankEncoder.suffixIds(tokenizer, enc, document: "",
                                           query: query)
        var out: [Float] = []
        out.reserveCapacity(ids.count)
        var start = 0
        while start < ids.count {
            let end = Swift.min(start + enc.queryBatch, ids.count)
            let part = Array(ids[start ..< end])
            let sub = index.state.gather(part).asType(.float32)
            let (idx, mask, endIdx) = RerankEncoder.batchIds(
                Array(repeating: qIds, count: part.count),
                bucket: enc.lengthBucket, minT: enc.lengthBucketMinTokens)
            let s = model(idx, mask: mask, endIdx: endIdx, state: sub)
                .asType(.float32)
            eval(s)
            out.append(contentsOf: s.asArray(Float.self))
            start = end
        }
        return out
    }

    /// Много запросов против ОДНОГО индекса, одним проходом.
    ///
    /// Обычный `scoreIndexed` гоняет один запрос: хвосты батчатся между
    /// кандидатами, но не между запросами. При оффлайновом переранжировании
    /// (переоценить тысячу запросов по готовому индексу) это оставляет
    /// батчи неполными ровно там, где их нечем заполнить, — а заполнить есть
    /// чем, соседним запросом.
    ///
    /// Возвращает скоры по запросам, в порядке `queries`.
    public func scoreIndexedBatch(queries: [String], index: DocIndex,
                                  docIds: [Int]? = nil,
                                  instruct: String? = nil) throws -> [[Float]] {
        guard !queries.isEmpty else { return [] }
        var cfg = config
        if let instruct { cfg.instruct = instruct }
        try index.check(cfg)

        let ids = docIds ?? Array(0 ..< index.count)
        for i in ids where i < 0 || i >= index.count {
            throw RerankServeError.docIdOutOfRange(i, count: index.count)
        }
        guard !ids.isEmpty else { return queries.map { _ in [] } }

        let enc = config.encode
        // Хвост зависит только от запроса, поэтому токенизируется по разу
        // на запрос, а не на пару.
        let qIds = queries.map {
            RerankEncoder.suffixIds(tokenizer, enc, document: "", query: $0)
        }
        // Плоский список работ (запрос, документ) — и режется он по общему
        // размеру пачки, не оглядываясь на границы запросов. В этом весь
        // смысл: последний неполный батч бывает один на всё, а не на каждый
        // запрос.
        var jobs: [(q: Int, d: Int)] = []
        jobs.reserveCapacity(queries.count * ids.count)
        for qi in 0 ..< queries.count { for d in ids { jobs.append((qi, d)) } }

        var out = [[Float]](repeating: [Float](repeating: 0, count: ids.count),
                            count: queries.count)
        let position = Dictionary(uniqueKeysWithValues:
            ids.enumerated().map { ($0.element, $0.offset) })
        var start = 0
        while start < jobs.count {
            let end = Swift.min(start + enc.queryBatch, jobs.count)
            let part = Array(jobs[start ..< end])
            let sub = index.state.gather(part.map { $0.d }).asType(.float32)
            let (idx, mask, endIdx) = RerankEncoder.batchIds(
                part.map { qIds[$0.q] }, bucket: enc.lengthBucket,
                minT: enc.lengthBucketMinTokens)
            let s = model(idx, mask: mask, endIdx: endIdx, state: sub)
                .asType(.float32)
            eval(s)
            for (k, v) in s.asArray(Float.self).enumerated() {
                out[part[k].q][position[part[k].d]!] = v
            }
            start = end
        }
        return out
    }

    public func rankIndexed(query: String, index: DocIndex, topK: Int? = nil,
                            docIds: [Int]? = nil, instruct: String? = nil)
        throws -> [(index: Int, score: Float)] {
        let ids = docIds ?? Array(0 ..< index.count)
        let s = try scoreIndexed(query: query, index: index, docIds: ids,
                                 instruct: instruct)
        return Self.order(s, ids: ids, topK: topK)
    }

    /// Порядок по убыванию скора.
    ///
    /// Ничьи разрешаются по возрастанию индекса — не «как получится».
    /// Сортировка в Swift не стабильна, а у необученной (zero-init) головы
    /// все скоры РАВНЫ, и тогда порядок был бы произвольным от запуска к
    /// запуску. Воспроизводимость выдачи дороже этой мелочи.
    public static func order(_ scores: [Float], ids: [Int], topK: Int?)
        -> [(index: Int, score: Float)] {
        var pairs = Array(zip(ids, scores)).enumerated().map {
            (rank: $0.offset, id: $0.element.0, score: $0.element.1)
        }
        pairs.sort { a, b in
            a.score != b.score ? a.score > b.score : a.rank < b.rank
        }
        let take = topK.map { Swift.min($0, pairs.count) } ?? pairs.count
        return pairs.prefix(take).map { (index: $0.id, score: $0.score) }
    }
}
