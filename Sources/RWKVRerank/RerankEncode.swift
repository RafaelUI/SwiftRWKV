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

    /// Потолок буферного кэша Metal, ГБ. `<= 0` — не трогать.
    ///
    /// Без него кодирование съедает ДЕСЯТКИ гигабайт, и это не утечка:
    /// MLX держит освобождённые буферы для переиспользования, а
    /// переиспользовать их здесь почти нечего. Каждая пачка префиксов
    /// паддится до СВОЕЙ максимальной длины, поэтому формы буферов почти
    /// не повторяются, и кэш растёт линейно по числу пачек.
    ///
    /// Замерено на прогоне 1000 запросов × 8 кандидатов: без лимита
    /// physical footprint дошёл до 13 ГБ (из них IOAccelerator 11.3 ГБ) и
    /// машина ушла в своп — то есть все замеры времени с этого момента
    /// становились фикцией. См. также `lengthBucket`.
    public var cacheLimitGB: Double

    /// Округлять длину батча ВВЕРХ до кратной этому числу. `<= 1` — не
    /// округлять.
    ///
    /// Сокращает число разных форм, отчего буферы Metal начинают
    /// переиспользоваться. Пад-позиции для рекуррентности нейтральны
    /// (w←1, k←0, b←0) и на состояние не влияют вовсе.
    ///
    /// Цена — ЛИШНЯЯ РАБОТА, и она не мелкая. Замерено на 0.1B, 200
    /// префиксов, три прогона: округление всего подряд до 64 стоит 38%
    /// полного времени кодирования (28.1 с против 20.4 с). Бьёт оно по
    /// хвостам: те короткие, около двадцати токенов, и округление до 64
    /// утраивает работу — 14.2 мс на хвост против 6.4 мс.
    ///
    /// Поэтому округление применяется НЕ ко всем батчам, см.
    /// `lengthBucketMinTokens`.
    public var lengthBucket: Int

    /// Округлять только батчи ДЛИННЕЕ этого. Короткие оставлять как есть.
    ///
    /// Добивка стоит `(bucket − 1) / T` лишней работы: для префикса в 400
    /// токенов это 16%, для хвоста в 20 токенов — 320%. Порог 4×bucket
    /// держит накладные ниже четверти и ровно поэтому выбран: он не подобран
    /// под данные, а выведен из того, сколько добивки допустимо.
    ///
    /// Замерено: с порогом округляются префиксы и не округляются хвосты,
    /// время возвращается к уровню «без округления», а пик памяти не растёт —
    /// его держит `cacheLimitGB`, и это тоже замер, а не расчёт: при
    /// выставленном потолке пик одинаков (2.9 ГБ) при любом bucket.
    public var lengthBucketMinTokens: Int

    public init(maxDocTokens: Int = 384, maxQueryTokens: Int = 96,
                terminator: Int? = 0, docBatch: Int = 8, queryBatch: Int = 16,
                template: PairTemplate = PairTemplate(),
                dtype: StateCacheDType = .float16,
                cacheLimitGB: Double = 2.0,
                lengthBucket: Int = 64,
                lengthBucketMinTokens: Int? = nil) {
        self.maxDocTokens = maxDocTokens
        self.maxQueryTokens = maxQueryTokens
        self.terminator = terminator
        self.docBatch = docBatch
        self.queryBatch = queryBatch
        self.template = template
        self.dtype = dtype
        self.cacheLimitGB = cacheLimitGB
        self.lengthBucket = lengthBucket
        self.lengthBucketMinTokens = lengthBucketMinTokens ?? (4 * lengthBucket)
    }

    /// Контракт подачи текста — то, что записывается в кэш и в чекпоинт.
    /// Состояние есть функция ровно этих величин, и расходятся они молча.
    ///
    /// Выравнивание длины СЮДА НЕ ВХОДИТ намеренно: пад-позиции для
    /// рекуррентности нейтральны, и состояние от них не зависит — это
    /// отдельный тест (`testLengthBucketingDoesNotChangeStates`, расхождение
    /// 2.3e-7, того же порядка, что между кэшированным и сплошным путём).
    /// Требовать совпадения выравнивания значило бы запрещать
    /// переиспользовать кэш после смены параметра производительности.
    public var contract: [String: String] {
        ["template": template.contract,
         "max_doc_tokens": String(maxDocTokens),
         "max_query_tokens": String(maxQueryTokens),
         "terminator": terminator.map(String.init) ?? "none"]
    }
}

public enum RerankEncodeError: Error, CustomStringConvertible {
    case vocabOverflow(maxId: Int, vocab: Int)
    case badSource(Int, nLayer: Int)

    public var description: String {
        switch self {
        case .vocabOverflow(let maxId, let vocab):
            return """
                токенизатор выдал id \(maxId) при словаре модели \(vocab) — \
                модель и токенизатор от разных моделей. Кодирование \
                остановлено: выборка из таблицы эмбеддингов ушла бы за \
                границу, и кэш молча заполнился бы мусором.
                """
        case .badSource(let i, let n):
            return "слой \(i) вне базы из \(n) слоёв"
        }
    }
}

public enum RerankEncoder {

    /// Какие слои базы класть в кэш.
    ///
    /// nil ⇒ ровно то, что читает голова, — обычный случай «один кэш, одна
    /// конфигурация». Явный список — НАДМНОЖЕСТВО: одно кодирование (минуты)
    /// обслуживает сколько угодно конфигураций головы, каждая берёт из кэша
    /// свой срез. Проверять, что нужный срез там есть, — работа
    /// `StateCache.slots(for:)`, а не эта: слои, которых текущая голова не
    /// читает, здесь совершенно законны, ради них всё и затевается.
    ///
    /// Результат всегда по возрастанию и без повторов — тем же порядком,
    /// каким `RerankerHead.uniqueSources` адресует слоты. Порядок здесь
    /// несущий: он связывает слот в файле со слоем базы.
    public static func resolveSources(_ sources: [Int]?, head: RerankerHead,
                                      nLayer: Int) throws -> [Int] {
        guard let sources else { return head.uniqueSources }
        var out: [Int] = []
        for i in sources {
            let a = i < 0 ? nLayer + i : i
            guard a >= 0 && a < nLayer else {
                throw RerankEncodeError.badSource(i, nLayer: nLayer)
            }
            out.append(a)
        }
        precondition(!out.isEmpty, "список слоёв кэша пуст")
        return Array(Set(out)).sorted()
    }

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
    public static func prefixIds(_ tok: WorldTokenizer, _ cfg: RerankEncodeConfig,
                          instruct: String, document: String) -> [Int] {
        guard cfg.template.docFirst else {
            return tok.encode("Instruct: \(instruct)\n")
        }
        let head = tok.encode("Instruct: \(instruct)\nDocument: ")
        let body = Array(tok.encode(document).prefix(cfg.maxDocTokens))
        return head + body + tok.encode("\n")
    }

    public static func suffixIds(_ tok: WorldTokenizer, _ cfg: RerankEncodeConfig,
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
    ///
    /// `bucket` округляет общую длину вверх до кратной себе, но ТОЛЬКО если
    /// длина уже не меньше `minT`. Добитые позиции для рекуррентности
    /// нейтральны, поэтому на состояние округление не влияет; влияет оно на
    /// ВРЕМЯ, и на коротких батчах катастрофически: добивка с 20 токенов до
    /// 64 — это втрое больше работы ради формы, которая и так встречается
    /// часто. Замеры в `RerankEncodeConfig.lengthBucket`.
    ///
    /// `minT = 0` ⇒ округлять всё подряд (прежнее поведение, оставлено для
    /// сверки со старыми кэшами).
    public static func batchIds(_ seqs: [[Int]], pad: Int = 0, bucket: Int = 0,
                                minT: Int = 0)
        -> (idx: MLXArray, mask: MLXArray, endIdx: MLXArray) {
        let lens = seqs.map { $0.count }
        var T = lens.max() ?? 1
        if bucket > 1 && T >= minT { T = ((T + bucket - 1) / bucket) * bucket }
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
    /// - sources: какие слои базы класть в кэш. nil ⇒ те, что читает голова.
    ///   Явный список делает кэш надмножеством: кодирование стоит минуты и
    ///   не зависит от того, сколько конфигураций головы потом на нём
    ///   обучится, а обучение — секунды. Ради этой асимметрии всё и есть.
    /// - progress: зовётся после каждой пачки префиксов; кодирование —
    ///   минуты, и прогон без признаков жизни неотличим от зависшего.
    /// - onBatch: форма каждого поданного батча. Существует потому, что
    ///   выравнивание длины НЕ меняет чисел (доказано отдельно) — значит,
    ///   «доехал ли порог округления до батчера» никаким сравнением
    ///   состояний не проверить, только структурно. Заодно годится для
    ///   отладки: видно, сколько разных форм создаёт прогон.
    public static func encodePairs(
        _ model: Reranker, tokenizer: WorldTokenizer,
        pool: [String], samples: [RerankSample],
        config: RerankEncodeConfig = RerankEncodeConfig(),
        path: URL? = nil, sources: [Int]? = nil,
        progress: ((_ prefixesDone: Int, _ prefixesTotal: Int,
                    _ pairsDone: Int, _ pairsTotal: Int) -> Void)? = nil,
        onBatch: ((_ kind: String, _ shape: [Int]) -> Void)? = nil
    ) throws -> StateCache {
        precondition(!samples.isEmpty, "нечего кодировать")
        if config.cacheLimitGB > 0 {
            MLX.GPU.set(cacheLimit: Int(config.cacheLimitGB * 1e9))
        }
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
        let srcs = try resolveSources(sources, head: model.head,
                                      nLayer: cfgModel.nLayer)
        let nSrc = srcs.count
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
            let (idx, mask, endIdx) = batchIds(seqs, bucket: config.lengthBucket,
                                               minT: config.lengthBucketMinTokens)
            onBatch?("prefix", idx.shape)
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
                let (qidx, qmask, qend) = batchIds(qseqs, bucket: config.lengthBucket,
                                                   minT: config.lengthBucketMinTokens)
                onBatch?("tail", qidx.shape)
                let pairState = model.encode(qidx, mask: qmask, endIdx: qend,
                                             state: sub)
                try writer.write(rows: part.map { $0.row },
                                 RerankerHead.select(pairState, sources: srcs))
                pairsDone += part.count
                qs = qe
            }

            progress?(end, prefixJobs.count, pairsDone, nPairs)
            start = end
        }

        // Инструкция попадает в контракт, ТОЛЬКО если она в корпусе одна.
        // При обучении это поле примера (инструкций может быть несколько), а
        // при выдаче — часть замороженного префикса, одна на индекс. Когда
        // она одна, кэш имеет право её зафиксировать, и тогда голова,
        // обученная на нём, унесёт её в свой чекпоинт. Когда их несколько,
        // писать нечего — и молчание здесь честнее любого умолчания.
        var contract = config.contract
        let instructs = Set(samples.map { $0.instruct })
        if instructs.count == 1 { contract["instruct"] = instructs.first! }

        return try writer.finish(pairIndex: pairIndex,
                                 labels: samples.map { $0.label },
                                 hardNegs: samples.map { $0.hardNegs },
                                 contract: contract,
                                 sources: srcs)
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
        batchSize: Int = 8, sources: [Int]? = nil
    ) throws -> MLXArray {
        let srcs = try resolveSources(sources, head: model.head,
                                      nLayer: model.base.cfg.nLayer)
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
            let (idx, mask, endIdx) = batchIds(Array(seqs),
                                               bucket: config.lengthBucket,
                                               minT: config.lengthBucketMinTokens)
            let st = model.encode(idx, mask: mask, endIdx: endIdx)
            let sel = RerankerHead.select(st, sources: srcs)
            eval(sel)
            outs.append(sel)
            start = end
        }
        return concatenated(outs, axis: 0)
    }
}
