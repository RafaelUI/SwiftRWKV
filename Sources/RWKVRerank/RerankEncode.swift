import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Свёртка пар (документ, запрос) в состояния.
//
//  Порядок работы задан устройством данных: документов много, запросов на
//  документ мало.
//
//    1. кодируем ПРЕФИКС «Instruct + Document» (дорого, O(L_doc)), пачками;
//    2. продолжаем каждый префикс всеми нужными хвостами «Query: …»
//       (дёшево, O(L_query)) — префикс уже свёрнут в состояние;
//    3. от финального состояния оставляем только читаемые головой слои.
//
//  Питоновский замер на M4 Air (0.1B): шаг 1 — ~73 мс на документ из 512
//  токенов, шаг 2 — ~3.5 мс на пару. То есть добавить запросу ещё восемь
//  кандидатов почти бесплатно, если документы уже в пуле. В Swift эти числа
//  НЕ воспроизводились и здесь не заявляются.
// ───────────────────────────────────────────────────────────────────────

public struct RerankEncodeConfig: Sendable {
    /// Обрезка документа. Обрезается ХВОСТ: начало пассажа обычно
    /// информативнее, а состояние всё равно копится слева направо.
    public var maxDocTokens: Int
    public var maxQueryTokens: Int
    /// Токен, дописываемый в конец пары. 0 — зарезервированный id в
    /// World-вокабе, ни одной байтовой строке не сопоставлен.
    public var terminator: Int?
    /// Префиксов за раз. Они длинные и держат много активаций.
    public var docBatch: Int
    /// Хвостов за раз. Они короткие.
    public var queryBatch: Int
    public var template: PairTemplate
    public var dtype: StateCacheDType

    public init(maxDocTokens: Int = 384, maxQueryTokens: Int = 96,
                terminator: Int? = 0, docBatch: Int = 8, queryBatch: Int = 16,
                template: PairTemplate = PairTemplate(),
                dtype: StateCacheDType = .float16) {
        self.maxDocTokens = maxDocTokens
        self.maxQueryTokens = maxQueryTokens
        self.terminator = terminator
        self.docBatch = docBatch
        self.queryBatch = queryBatch
        self.template = template
        self.dtype = dtype
    }

    /// Контракт подачи текста — то, что записывается в кэш и в чекпоинт.
    /// Состояние есть функция ровно этих величин, и расходятся они молча.
    public var contract: [String: String] {
        ["template": template.contract,
         "max_doc_tokens": String(maxDocTokens),
         "max_query_tokens": String(maxQueryTokens),
         "terminator": terminator.map(String.init) ?? "none"]
    }
}

public enum RerankEncodeError: Error, CustomStringConvertible {
    case vocabOverflow(maxId: Int, vocab: Int)

    public var description: String {
        switch self {
        case .vocabOverflow(let maxId, let vocab):
            return """
                токенизатор выдал id \(maxId) при словаре модели \(vocab) — \
                модель и токенизатор от разных моделей. Кодирование \
                остановлено: выборка из таблицы эмбеддингов ушла бы за \
                границу, и кэш молча заполнился бы мусором.
                """
        }
    }
}

public enum RerankEncoder {

    /// Проверка «токенизатор от этой ли модели» — ОДИН раз на прогон.
    ///
    /// Нужна потому, что промах здесь абсолютно молчаливый: выборка строки
    /// эмбеддинга за границей таблицы не падает, а возвращает мусор (в
    /// эмбеддинг-пути та же ошибка давала NaN и обучение крутилось вхолостую
    /// до конца прогона). Кодирование пар — минуты, а на выходе получился бы
    /// правдоподобный кэш, на котором голова честно обучилась бы шуму.
    ///
    /// Найдено не рассуждением, а тестом: сверка кэшированного пути со
    /// сплошным разошлась на 3–5% там, где обязана была совпасть, — крошечный
    /// бэкбон со словарём 64 против World-токенизатора с id до 65 тысяч.
    /// Мусор при этом зависел от формы батча, отчего и выглядел как ошибка
    /// батчинга.
    static func checkVocab(_ model: Reranker, _ ids: [[Int]]) throws {
        guard let maxId = ids.compactMap({ $0.max() }).max() else { return }
        guard maxId < model.base.cfg.vocab else {
            throw RerankEncodeError.vocabOverflow(maxId: maxId,
                                                  vocab: model.base.cfg.vocab)
        }
    }

    /// Токены префикса.
    ///
    /// Куски токенизируются ПО ОТДЕЛЬНОСТИ, чтобы обрезка документа была
    /// ровно по токенам, а не по символам: обрезав строку, легко разрубить
    /// многобайтовый символ или изменить границы токенов у соседей.
    static func prefixIds(_ tok: WorldTokenizer, _ cfg: RerankEncodeConfig,
                          instruct: String, document: String) -> [Int] {
        guard cfg.template.docFirst else {
            return tok.encode("Instruct: \(instruct)\n")
        }
        let head = tok.encode("Instruct: \(instruct)\nDocument: ")
        let body = Array(tok.encode(document).prefix(cfg.maxDocTokens))
        return head + body + tok.encode("\n")
    }

    static func suffixIds(_ tok: WorldTokenizer, _ cfg: RerankEncodeConfig,
                          document: String, query: String) -> [Int] {
        var ids = tok.encode("Query: ")
                + Array(tok.encode(query).prefix(cfg.maxQueryTokens))
        if !cfg.template.docFirst {
            ids += tok.encode("\nDocument: ")
                 + Array(tok.encode(document).prefix(cfg.maxDocTokens))
        }
        if let t = cfg.terminator { ids.append(t) }
        return ids
    }

    /// Right-padding + маска + позиции последних реальных токенов.
    static func batchIds(_ seqs: [[Int]], pad: Int = 0)
        -> (idx: MLXArray, mask: MLXArray, endIdx: MLXArray) {
        let lens = seqs.map { $0.count }
        let T = lens.max() ?? 1
        var flat = [Int32](repeating: Int32(pad), count: seqs.count * T)
        for (i, s) in seqs.enumerated() {
            for (j, v) in s.enumerated() { flat[i * T + j] = Int32(v) }
        }
        return (MLXArray(flat, [seqs.count, T]),
                buildMask(lengths: lens, total: T),
                lastRealIndex(lengths: lens))
    }

    /// Свернуть все пары (кандидат, запрос) в кэш состояний.
    ///
    /// - path: писать состояния сразу на диск. nil ⇒ держать в памяти.
    /// - progress: зовётся после каждой пачки префиксов; кодирование —
    ///   минуты, и прогон без признаков жизни неотличим от зависшего.
    public static func encodePairs(
        _ model: Reranker, tokenizer: WorldTokenizer,
        pool: [String], samples: [RerankSample],
        config: RerankEncodeConfig = RerankEncodeConfig(),
        path: URL? = nil,
        progress: ((_ prefixesDone: Int, _ prefixesTotal: Int,
                    _ pairsDone: Int, _ pairsTotal: Int) -> Void)? = nil
    ) throws -> StateCache {
        precondition(!samples.isEmpty, "нечего кодировать")
        let nCand = samples[0].docIds.count
        precondition(samples.allSatisfy { $0.docIds.count == nCand },
                     "у всех примеров должно быть одинаковое число кандидатов")

        // Ключ префикса — (инструкция, документ). Инструкций в корпусе
        // единицы, поэтому дедупликация по паре почти ничего не стоит, а при
        // docFirst == false документ живёт в ХВОСТЕ и в ключ не входит вовсе
        // (иначе получилось бы N одинаковых префиксов вместо одного).
        struct Job { var instruct: String; var docId: Int }
        var prefixKey: [String: Int] = [:]
        var prefixJobs: [Job] = []
        var pending: [[(row: Int, query: String, docId: Int)]] = []

        var pairIndex = [[Int]](repeating: [Int](repeating: 0, count: nCand),
                                count: samples.count)
        var nPairs = 0
        for (si, s) in samples.enumerated() {
            for (ci, did) in s.docIds.enumerated() {
                let key = "\(s.instruct)\u{0}\(config.template.docFirst ? did : -1)"
                var pi = prefixKey[key]
                if pi == nil {
                    pi = prefixJobs.count
                    prefixKey[key] = pi
                    prefixJobs.append(Job(instruct: s.instruct, docId: did))
                    pending.append([])
                }
                pending[pi!].append((nPairs, s.query, did))
                pairIndex[si][ci] = nPairs
                nPairs += 1
            }
        }

        let cfgModel = model.base.cfg
        let nSrc = model.head.uniqueSources.count
        let writer = try StateCacheWriter(
            shape: [nPairs, nSrc, cfgModel.nHead, cfgModel.headSize, cfgModel.headSize],
            dtype: config.dtype, path: path)

        var pairsDone = 0
        var start = 0
        while start < prefixJobs.count {
            let end = Swift.min(start + config.docBatch, prefixJobs.count)
            let chunk = Array(prefixJobs[start ..< end])
            let seqs = chunk.map {
                prefixIds(tokenizer, config, instruct: $0.instruct,
                          document: pool[$0.docId])
            }
            if start == 0 { try checkVocab(model, seqs) }
            let (idx, mask, endIdx) = batchIds(seqs)
            let prefixState = model.encode(idx, mask: mask, endIdx: endIdx)
            prefixState.evaluated()

            // Все хвосты для этой пачки префиксов.
            var jobs: [(local: Int, row: Int, query: String, docId: Int)] = []
            for local in 0 ..< chunk.count {
                for p in pending[start + local] {
                    jobs.append((local, p.row, p.query, p.docId))
                }
            }

            var qs = 0
            while qs < jobs.count {
                let qe = Swift.min(qs + config.queryBatch, jobs.count)
                let part = Array(jobs[qs ..< qe])
                let sub = prefixState.gather(part.map { $0.local })
                let qseqs = part.map {
                    suffixIds(tokenizer, config, document: pool[$0.docId],
                              query: $0.query)
                }
                let (qidx, qmask, qend) = batchIds(qseqs)
                let pairState = model.encode(qidx, mask: qmask, endIdx: qend,
                                             state: sub)
                try writer.write(rows: part.map { $0.row },
                                 model.select(pairState))
                pairsDone += part.count
                qs = qe
            }

            progress?(end, prefixJobs.count, pairsDone, nPairs)
            start = end
        }

        return try writer.finish(pairIndex: pairIndex,
                                 labels: samples.map { $0.label },
                                 hardNegs: samples.map { $0.hardNegs },
                                 contract: config.contract)
    }

    /// Сплошной путь БЕЗ кэша префиксов: (instruct, doc, query) → состояния
    /// `[n, nSrc, H, S, S]`.
    ///
    /// Нужен там, где кэшировать нечего (разовый скоринг), и — важнее — как
    /// независимая проверка `encodePairs`: два пути обязаны сходиться, и
    /// расхождение между ними означает, что перенос состояния через границу
    /// префикса что-то теряет.
    public static func encodePairsDirect(
        _ model: Reranker, tokenizer: WorldTokenizer,
        pairs: [(instruct: String, document: String, query: String)],
        config: RerankEncodeConfig = RerankEncodeConfig(),
        batchSize: Int = 8
    ) throws -> MLXArray {
        var outs: [MLXArray] = []
        var start = 0
        while start < pairs.count {
            let end = Swift.min(start + batchSize, pairs.count)
            let seqs = pairs[start ..< end].map { p in
                prefixIds(tokenizer, config, instruct: p.instruct,
                          document: p.document)
                + suffixIds(tokenizer, config, document: p.document,
                            query: p.query)
            }
            if start == 0 { try checkVocab(model, Array(seqs)) }
            let (idx, mask, endIdx) = batchIds(Array(seqs))
            let st = model.encode(idx, mask: mask, endIdx: endIdx)
            let sel = model.select(st)
            eval(sel)
            outs.append(sel)
            start = end
        }
        return concatenated(outs, axis: 0)
    }
}
