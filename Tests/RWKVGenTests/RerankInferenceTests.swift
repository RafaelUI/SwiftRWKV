//
//  RerankInferenceTests.swift
//  Выдача реранкера: прямой путь, индекс префиксов, контракт подачи текста.
//
//  Главный инвариант — ИНДЕКСНЫЙ путь равен ПРЯМОМУ. Индекс сворачивает
//  «Instruct + Document» один раз и продолжает его хвостом «Query: …»;
//  прямой путь гоняет всё одним куском. Расхождение здесь означает, что
//  выдача ранжирует не тем, чем обучалась голова, — и заметить это по
//  скорам невозможно, они и так сырые логиты без абсолютной шкалы.
//
//  Второй мотив половины тестов: голова с ZERO-INIT `fc2` даёт тождественно
//  нулевой скор при ЛЮБОЙ, в том числе полностью сломанной, реализации
//  всего, что до неё. Поэтому там, где сверяются числа, `fc2`
//  рандомизируется — иначе тест зелен по построению.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank

final class RerankInferenceTests: XCTestCase {

    // ─────────────────────────────────────────────────────────────────
    //  Оснастка
    // ─────────────────────────────────────────────────────────────────

    /// Словарь ПОЛНОГО размера: World-токенизатор выдаёт id до 65 тысяч, а
    /// выход за таблицу эмбеддингов в MLX не падает, а возвращает мусор,
    /// зависящий от формы батча.
    func makeModel(nLayer: Int = 4, layerIdx: [Int] = [-1])
        throws -> (Reranker, X070Backbone) {
        let (bb, _) = TinyBackbone.make(nLayer: nLayer, vocab: 65536)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: layerIdx))
        return (model, bb)
    }

    /// Сделать голову НЕвырожденной.
    ///
    /// Штатный `fc2 = 0` — самое ценное свойство головы при обучении
    /// (стартовый лосс ровно `ln C`) и самое вредное при проверке выдачи:
    /// скор тождественно ноль, все сравнения сходятся сами собой.
    func randomiseHead(_ model: Reranker, seed: Int = 17) {
        MLXRandom.seed(UInt64(seed))
        model.head.fc2Weight = MLXRandom.normal(model.head.fc2Weight.shape) * 0.3
        eval(model.head.fc2Weight)
    }

    private func worldTokenizer() throws -> WorldTokenizer {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? home.appendingPathComponent(
                "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path),
                          "нет словаря World — тест пропущен")
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: path)) else {
            throw XCTSkip("словарь World не разобрался")
        }
        return tok
    }

    let docs = [
        "Пчёлы опыляют растения и производят мёд из нектара цветов.",
        "Паровая машина Уатта запустила промышленную революцию в Англии.",
        "Медоносная пчела танцем сообщает улью направление к взятку.",
        "Железные дороги девятнадцатого века связали города и порты.",
    ]
    let query = "как пчёлы делают мёд"

    func serving(_ maxDoc: Int = 48, docBatch: Int = 2, queryBatch: Int = 3)
        -> RerankServingConfig {
        RerankServingConfig(
            encode: RerankEncodeConfig(maxDocTokens: maxDoc, maxQueryTokens: 16,
                                       docBatch: docBatch, queryBatch: queryBatch),
            instruct: "найди документ, отвечающий на вопрос")
    }

    func maxDiff(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).map { Swift.abs($0 - $1) }.max() ?? 0
    }

    // ─────────────────────────────────────────────────────────────────
    //  Главный инвариант
    // ─────────────────────────────────────────────────────────────────

    /// Скоры по индексу совпадают со сплошным путём.
    ///
    /// Допуск, а не равенство: индексный путь режет последовательность на
    /// префикс и хвост, отчего меняются формы входа матмулов и те
    /// раскладываются на GPU иначе. Сама причинность точная — она проверена
    /// на уровне ядра (`WKV7StateTests`), где разрез даёт РОВНО ноль.
    ///
    /// Граница взята с запасом к ЗАМЕРУ, а не назначена заранее: измеренное
    /// расхождение 2.4e-7 при границе 1e-4, то есть запас в четыреста раз.
    /// Назначать границу до замера здесь особенно вредно — слабая прошла бы
    /// мимо настоящего дефекта, а строгая падала бы на шуме раскладки.
    func testIndexedPathMatchesDirectPath() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        let inf = RerankerInference(model: model, tokenizer: tok,
                                    config: serving())

        let direct = try inf.score(query: query, docs: docs)
        let index = try inf.buildIndex(docs: docs)
        let indexed = try inf.scoreIndexed(query: query, index: index)

        // Различающее утверждение: скоры обязаны РАЗЛИЧАТЬСЯ между
        // документами, иначе совпадение путей ничего не значит — сойтись
        // могли бы и четыре одинаковых нуля.
        XCTAssertGreaterThan(direct.max()! - direct.min()!, 1e-3,
                             "скоры вырождены — голова не рандомизирована")
        XCTAssertLessThan(maxDiff(direct, indexed), 1e-4,
                          "индексный путь разошёлся со сплошным")
    }

    /// Скор не зависит от разбиения на пачки — ни в прямом пути, ни в
    /// индексном.
    ///
    /// Пачка меняет форму батча, а с ней и раскладку матмулов. Проверка
    /// нужна потому, что размер пачки — параметр производительности, и
    /// молчаливое влияние на ЧИСЛА сделало бы его параметром качества.
    ///
    /// Замерено 7.2e-7 при границе 1e-4. Это НЕ ноль и нулём быть не может:
    /// причинность точна, а порядок сложения в матмуле при другой форме
    /// входа другой. Практический вывод тот же, что и для выравнивания
    /// длины при обучении: воспроизводимость требует ОДИНАКОВОГО разбиения
    /// на пачки, а не просто достаточного.
    func testScoresDoNotDependOnBatching() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)

        let small = RerankerInference(model: model, tokenizer: tok,
                                      config: serving(48, docBatch: 1, queryBatch: 1))
        let big = RerankerInference(model: model, tokenizer: tok,
                                    config: serving(48, docBatch: 8, queryBatch: 8))
        XCTAssertLessThan(maxDiff(try small.score(query: query, docs: docs),
                                  try big.score(query: query, docs: docs)), 1e-4)

        let idx = try big.buildIndex(docs: docs)
        XCTAssertLessThan(
            maxDiff(try small.scoreIndexed(query: query, index: idx),
                    try big.scoreIndexed(query: query, index: idx)), 1e-4)
    }

    /// Подмножество документов индекса скорится так же, как всё сразу.
    func testSubsetOfIndexScoresIdentically() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        let index = try inf.buildIndex(docs: docs)

        let all = try inf.scoreIndexed(query: query, index: index)
        let some = try inf.scoreIndexed(query: query, index: index, docIds: [2, 0])
        XCTAssertLessThan(maxDiff(some, [all[2], all[0]]), 1e-5)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Порядок
    // ─────────────────────────────────────────────────────────────────

    /// `rank` сортирует по УБЫВАНИЮ и совпадает со скорами.
    func testRankOrdersByDescendingScore() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())

        let s = try inf.score(query: query, docs: docs)
        let r = try inf.rank(query: query, docs: docs)
        XCTAssertEqual(r.count, docs.count)
        for i in 1 ..< r.count {
            XCTAssertGreaterThanOrEqual(r[i - 1].score, r[i].score)
        }
        for (id, sc) in r { XCTAssertEqual(sc, s[id], accuracy: 1e-6) }

        let top2 = try inf.rank(query: query, docs: docs, topK: 2)
        XCTAssertEqual(top2.count, 2)
        XCTAssertEqual(top2.map { $0.index }, Array(r.prefix(2)).map { $0.index })
    }

    /// Ничьи разрешаются по возрастанию индекса, а не «как получится».
    ///
    /// У необученной головы все скоры РОВНО равны, и `sort` в Swift не
    /// стабилен: без явного правила порядок выдачи менялся бы от запуска к
    /// запуску на одних и тех же данных.
    func testTiesResolveByIndexDeterministically() throws {
        let tied = [Float](repeating: 0, count: 6)
        let ids = Array(0 ..< 6)
        let a = RerankerInference.order(tied, ids: ids, topK: nil)
        XCTAssertEqual(a.map { $0.index }, ids)

        // И на «частичной» ничьей: равные держат исходный порядок между собой.
        let partial: [Float] = [1, 0, 1, 0, 2]
        let b = RerankerInference.order(partial, ids: [10, 11, 12, 13, 14],
                                        topK: nil)
        XCTAssertEqual(b.map { $0.index }, [14, 10, 12, 11, 13])

        // Индексы кандидатов, а не позиции в массиве.
        let c = RerankerInference.order([0.5, 0.9], ids: [7, 3], topK: 1)
        XCTAssertEqual(c.map { $0.index }, [3])
    }

    /// Индексный и прямой путь дают ОДИН порядок.
    func testIndexedAndDirectAgreeOnOrder() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        let index = try inf.buildIndex(docs: docs)
        XCTAssertEqual(try inf.rank(query: query, docs: docs).map { $0.index },
                       try inf.rankIndexed(query: query, index: index).map { $0.index })
    }

    // ─────────────────────────────────────────────────────────────────
    //  Контракт
    // ─────────────────────────────────────────────────────────────────

    /// Индекс, построенный с другой ИНСТРУКЦИЕЙ, отвергается.
    ///
    /// Формы у него совершенно правильные, скоры правдоподобные — просто от
    /// другого текста. Это ровно тот отказ, который без проверки не
    /// проявляется никак.
    func testIndexBuiltWithOtherInstructIsRejected() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        let index = try inf.buildIndex(docs: docs, instruct: "совсем другая задача")

        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: index),
                             "индекс с чужой инструкцией принят")
        // А со СВОЕЙ — проходит.
        XCTAssertNoThrow(try inf.scoreIndexed(query: query, index: index,
                                              instruct: "совсем другая задача"))
    }

    /// Индекс, построенный с другой ОБРЕЗКОЙ документа, отвергается.
    func testIndexBuiltWithOtherDocTruncationIsRejected() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        let short = RerankerInference(model: model, tokenizer: tok,
                                      config: serving(16))
        let long = RerankerInference(model: model, tokenizer: tok,
                                     config: serving(256))
        let index = try short.buildIndex(docs: docs)
        XCTAssertNoThrow(try short.scoreIndexed(query: query, index: index))
        XCTAssertThrowsError(try long.scoreIndexed(query: query, index: index),
                             "индекс с обрезкой 16 принят при обрезке 256")
    }

    /// Обрезка ЗАПРОСА к индексу не относится и переиндексации не требует.
    ///
    /// Утверждение о границе контракта, а не о числах: хвост в индекс не
    /// входит, и требовать совпадения `max_query_tokens` значило бы
    /// запрещать законное.
    func testQueryTruncationIsNotPartOfIndexContract() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        var cfgA = serving(); cfgA.encode.maxQueryTokens = 8
        var cfgB = serving(); cfgB.encode.maxQueryTokens = 64
        let a = RerankerInference(model: model, tokenizer: tok, config: cfgA)
        let b = RerankerInference(model: model, tokenizer: tok, config: cfgB)
        let index = try a.buildIndex(docs: docs)
        XCTAssertNoThrow(try b.scoreIndexed(query: query, index: index))
    }

    /// Индекс при `docFirst == false` не строится вовсе.
    func testIndexRequiresDocFirst() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        var cfg = serving()
        cfg.encode.template = PairTemplate(docFirst: false)
        let inf = RerankerInference(model: model, tokenizer: tok, config: cfg)
        XCTAssertThrowsError(try inf.buildIndex(docs: docs))
        // Прямой путь при этом работает: кэшировать нечего, считать есть что.
        XCTAssertEqual(try inf.score(query: query, docs: docs).count, docs.count)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Чекпоинт
    // ─────────────────────────────────────────────────────────────────

    /// `fromCheckpoint` берёт контракт ИЗ ФАЙЛА, а не из умолчаний.
    ///
    /// Это единственный способ не разъехаться: обученная голова и то, как ей
    /// подавали текст, едут вместе. Умолчания здесь — самый вероятный
    /// источник тихой потери качества, потому что они правдоподобны.
    func testFromCheckpointTakesContractFromFile() throws {
        let tok = try worldTokenizer()
        let (model, bb) = try makeModel()
        randomiseHead(model)

        var cfg = serving(32)
        cfg.encode.maxQueryTokens = 12
        cfg.encode.terminator = nil
        cfg.instruct = "инструкция, записанная в чекпоинт"
        let inf = RerankerInference(model: model, tokenizer: tok, config: cfg)

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("head_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.saveHead(to: url, extra: inf.servingMetadata)

        let loaded = try RerankerInference.fromCheckpoint(
            base: bb, tokenizer: tok, head: url)
        XCTAssertEqual(loaded.config.encode.maxDocTokens, 32)
        XCTAssertEqual(loaded.config.encode.maxQueryTokens, 12)
        XCTAssertNil(loaded.config.encode.terminator)
        XCTAssertEqual(loaded.config.instruct, "инструкция, записанная в чекпоинт")
        XCTAssertTrue(loaded.config.encode.template.docFirst)

        // Умолчания отличаются от записанного — иначе тест ничего не значит.
        let fallback = RerankServingConfig()
        XCTAssertNotEqual(fallback.encode.maxDocTokens, 32)
        XCTAssertNotEqual(fallback.instruct, loaded.config.instruct)

        // И числа те же, что у исходной выдачи.
        XCTAssertLessThan(maxDiff(try inf.score(query: query, docs: docs),
                                  try loaded.score(query: query, docs: docs)),
                          1e-5)
    }

    /// Явная поправка поверх чекпоинта возможна, но требует сказать об этом.
    func testOverridesApplyOnTopOfCheckpoint() throws {
        let tok = try worldTokenizer()
        let (model, bb) = try makeModel()
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving(32))
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("head_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.saveHead(to: url, extra: inf.servingMetadata)

        let loaded = try RerankerInference.fromCheckpoint(
            base: bb, tokenizer: tok, head: url,
            overrides: { $0.encode.docBatch = 16 })
        XCTAssertEqual(loaded.config.encode.docBatch, 16)
        XCTAssertEqual(loaded.config.encode.maxDocTokens, 32, "поправка задела не своё")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Индекс на диске
    // ─────────────────────────────────────────────────────────────────

    /// Круг «сохранить → загрузить» даёт те же скоры и тот же контракт.
    func testIndexDiskRoundTrip() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("index_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let built = try inf.buildIndex(docs: docs)
        try built.save(to: url)
        let loaded = try DocIndex.load(url)

        XCTAssertEqual(loaded.docs, docs, "тексты документов не пережили круг")
        XCTAssertEqual(loaded.contract, built.contract)
        XCTAssertEqual(loaded.count, built.count)
        XCTAssertLessThan(
            maxDiff(try inf.scoreIndexed(query: query, index: built),
                    try inf.scoreIndexed(query: query, index: loaded)), 1e-6)
    }

    /// Загруженный индекс проверяет контракт так же, как построенный.
    func testLoadedIndexKeepsContractCheck() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving(16))
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("index_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try inf.buildIndex(docs: docs).save(to: url)

        let other = RerankerInference(model: model, tokenizer: tok,
                                      config: serving(256))
        XCTAssertThrowsError(
            try other.scoreIndexed(query: query, index: try DocIndex.load(url)),
            "контракт потерялся при записи на диск")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Потоковый индекс и пакетная выдача
    // ─────────────────────────────────────────────────────────────────

    /// Индекс, собранный ПОТОКОМ на диск, равен собранному в памяти.
    ///
    /// Разница между ними только в том, где живут пачки по ходу сборки, —
    /// значит числа обязаны совпасть побитово. Расхождение означало бы, что
    /// склейка по документам перепутана с чем-то ещё, а это молчаливо: формы
    /// сойдутся при любом порядке.
    func testStreamedIndexEqualsInMemoryIndex() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        // Пачка МЕНЬШЕ числа документов — иначе потока не возникает вовсе
        // и тест сравнивал бы один путь сам с собой.
        let inf = RerankerInference(model: model, tokenizer: tok,
                                    config: serving(48, docBatch: 2))
        XCTAssertLessThan(inf.config.encode.docBatch, docs.count,
                          "тест вырожден: всё влезло в одну пачку")

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("idx_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let inMemory = try inf.buildIndex(docs: docs)
        let streamed = try inf.buildIndexToDisk(docs: docs, at: url)

        XCTAssertEqual(streamed.docs, docs)
        XCTAssertEqual(streamed.contract, inMemory.contract)
        XCTAssertEqual(maxDiff(try inf.scoreIndexed(query: query, index: inMemory),
                               try inf.scoreIndexed(query: query, index: streamed)),
                       0, "потоковый индекс разошёлся с собранным в памяти")
    }

    /// Потоковый индекс проверяет то же, что обычный.
    func testStreamedIndexKeepsGuards() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        var cfg = serving()
        cfg.encode.template = PairTemplate(docFirst: false)
        let inf = RerankerInference(model: model, tokenizer: tok, config: cfg)
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("idx_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try inf.buildIndexToDisk(docs: docs, at: url))
        let ok = RerankerInference(model: model, tokenizer: tok, config: serving())
        XCTAssertThrowsError(try ok.buildIndexToDisk(docs: [], at: url))
    }

    /// Пакетная выдача даёт то же, что запросы по одному.
    ///
    /// Плоский список работ режется по границам ПАЧКИ, а не запросов,
    /// поэтому один и тот же запрос может оказаться разложенным по двум
    /// батчам — и разложить результат обратно надо по (запрос, документ), а
    /// не по порядку. Ошибка здесь даёт полный набор правдоподобных чисел,
    /// приписанных не тем запросам.
    ///
    /// Допуск, а не ноль, и это не уступка: в пачку попадают хвосты РАЗНЫХ
    /// запросов, значит длина батча другая, значит матмулы раскладываются
    /// иначе. Замерено 6e-7 при границе 1e-5; ровного нуля здесь не бывает
    /// по той же причине, что и у разбиения на пачки вообще. Зато ПОРЯДОК
    /// кандидатов обязан совпасть точно — это и проверяется отдельно.
    func testBatchedScoringMatchesOneByOne() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        // queryBatch намеренно НЕ кратен числу документов: иначе границы
        // пачек совпали бы с границами запросов и перепутать было бы нечего.
        let inf = RerankerInference(model: model, tokenizer: tok,
                                    config: serving(48, queryBatch: 3))
        let index = try inf.buildIndex(docs: docs)
        let queries = ["как пчёлы делают мёд", "кто изобрёл паровую машину",
                       "чем питаются пчёлы"]

        let batched = try inf.scoreIndexedBatch(queries: queries, index: index)
        XCTAssertEqual(batched.count, queries.count)
        for (i, q) in queries.enumerated() {
            let one = try inf.scoreIndexed(query: q, index: index)
            XCTAssertLessThan(maxDiff(batched[i], one), 1e-5,
                              "запрос \(i) разошёлся с одиночным путём")
            XCTAssertEqual(
                RerankerInference.order(batched[i], ids: Array(0 ..< one.count),
                                        topK: nil).map { $0.index },
                RerankerInference.order(one, ids: Array(0 ..< one.count),
                                        topK: nil).map { $0.index },
                "запрос \(i): порядок кандидатов разошёлся")
        }
        // Различающее утверждение: запросы обязаны давать РАЗНЫЕ скоры,
        // иначе перепутать их местами было бы невозможно и тест пуст.
        XCTAssertNotEqual(batched[0], batched[1])
    }

    /// Подмножество документов в пакетной выдаче: порядок результата следует
    /// `docIds`, а не индексу.
    func testBatchedScoringHonoursDocIds() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        randomiseHead(model)
        let inf = RerankerInference(model: model, tokenizer: tok,
                                    config: serving(48, queryBatch: 3))
        let index = try inf.buildIndex(docs: docs)
        let queries = ["как пчёлы делают мёд", "кто изобрёл паровую машину"]
        let ids = [3, 0, 2]

        let batched = try inf.scoreIndexedBatch(queries: queries, index: index,
                                                docIds: ids)
        for (i, q) in queries.enumerated() {
            XCTAssertLessThan(maxDiff(batched[i],
                                      try inf.scoreIndexed(query: q, index: index,
                                                           docIds: ids)), 1e-5)
        }
        XCTAssertEqual(batched[0].count, ids.count)
        XCTAssertEqual(try inf.scoreIndexedBatch(queries: [], index: index).count, 0)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Края
    // ─────────────────────────────────────────────────────────────────

    func testEmptyAndOutOfRange() throws {
        let tok = try worldTokenizer()
        let (model, _) = try makeModel()
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        XCTAssertEqual(try inf.score(query: query, docs: []).count, 0)
        XCTAssertThrowsError(try inf.buildIndex(docs: []))

        let index = try inf.buildIndex(docs: docs)
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: index,
                                                  docIds: [docs.count]))
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: index,
                                                  docIds: [-1]))
        XCTAssertEqual(try inf.scoreIndexed(query: query, index: index,
                                            docIds: []).count, 0)
    }

    /// Токенизатор от другой модели ловится ДО прохода, а не после.
    ///
    /// Выборка за границей таблицы эмбеддингов в MLX не падает, а возвращает
    /// мусор. При выдаче это особенно неприятно: пользователь получил бы
    /// осмысленно выглядящий список.
    func testForeignTokenizerIsCaught() throws {
        let tok = try worldTokenizer()
        let (bb, _) = TinyBackbone.make(nLayer: 2, vocab: 64)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        XCTAssertThrowsError(try inf.score(query: query, docs: docs))
        XCTAssertThrowsError(try inf.buildIndex(docs: docs))
    }
}
