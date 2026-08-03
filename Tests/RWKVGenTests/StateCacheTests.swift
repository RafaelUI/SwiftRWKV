//
//  StateCacheTests.swift
//  Кэш состояний и кодирование пар.
//
//  Главный инвариант здесь — КЭШИРОВАННЫЙ путь равен СПЛОШНОМУ. Кэш
//  сворачивает «Instruct + Document» один раз и продолжает его хвостом
//  «Query: …», сплошной путь гоняет всё одним куском. Если они разойдутся,
//  обучение пойдёт на состояниях, которых при инференсе не бывает, — и
//  заметить это по лоссу нельзя, он будет исправно падать.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank

final class StateCacheTests: XCTestCase {

    /// Словарь ПОЛНОГО размера (65536), а не игрушечный.
    ///
    /// Это не перестраховка. Первая редакция этих тестов брала стандартный
    /// TinyBackbone со словарём на 64 строки, а токенизатор World выдаёт id
    /// до 65 тысяч: выборка уходила за границу таблицы эмбеддингов, MLX
    /// молча возвращал мусор, и мусор этот ЗАВИСЕЛ ОТ ФОРМЫ БАТЧА. Выглядело
    /// как расхождение кэшированного пути со сплошным на 3–5%, то есть как
    /// ошибка в самом кэше. Таблица 65536×128 в fp32 — 33 МБ, для теста
    /// приемлемо.
    func makeModel(nLayer: Int = 4, layerIdx: [Int] = [-1])
        throws -> (Reranker, X070Backbone, X070Config) {
        let (bb, cfg) = TinyBackbone.make(nLayer: nLayer, vocab: 65536)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: layerIdx))
        return (model, bb, cfg)
    }

    func makeSamples(_ n: Int = 6, nCand: Int = 4)
        throws -> (pool: [String], samples: [RerankSample]) {
        let rows = (0 ..< n).map { i in
            RerankRow(instruct: "инструкция", query: "вопрос номер \(i)",
                      positive: "правильный пассаж про тему \(i)",
                      negatives: ["похожий но неверный пассаж \(i)"])
        }
        return try RerankCandidates.build(rows, nCandidates: nCand, seed: 0)
    }

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        eval(ref, got)
        return MLX.abs(ref - got).max().item(Float.self)
             / (MLX.abs(ref).max().item(Float.self) + 1e-9)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Кэш как хранилище
    // ─────────────────────────────────────────────────────────────────

    /// Круг «записать → прочитать» и укладка строк.
    ///
    /// Строки пишутся ВРАЗБРОС (пары группируются по префиксу, а нумеруются
    /// по примерам), поэтому перепутанное смещение — реальный риск, и он
    /// молчаливый: формы сойдутся, состояния окажутся не от тех пар.
    func testWriterRoundTripWithScatteredRows() throws {
        let shape = [6, 1, 2, 4, 4]
        let writer = try StateCacheWriter(shape: shape, dtype: .float32)
        MLXRandom.seed(5)
        let all = MLXRandom.normal(shape) * 0.5
        eval(all)

        // Пишем в порядке 3,0,5,1,4,2 — как раз вразброс.
        for r in [3, 0, 5, 1, 4, 2] {
            try writer.write(rows: [r], all[r ..< (r + 1)])
        }
        let cache = try writer.finish(
            pairIndex: [[0, 1], [2, 3], [4, 5]], labels: [0, 1, 0],
            hardNegs: [[1], [0], [1]], contract: ["template": "doc_first"])

        XCTAssertEqual(cache.nPairs, 6)
        XCTAssertEqual(cache.nSamples, 3)
        XCTAssertEqual(cache.nCandidates, 2)
        XCTAssertEqual(maxAbsDiff(cache.gather([0, 1, 2, 3, 4, 5]), all), 0,
                       "строки легли не по своим смещениям")
        XCTAssertEqual(maxAbsDiff(cache.gather([5, 0]),
                                  concatenated([all[5 ..< 6], all[0 ..< 1]], axis: 0)), 0)
    }

    /// Батч по примерам собирает кандидатов подряд и отдаёт метки.
    func testBatchLaysCandidatesContiguously() throws {
        let shape = [6, 1, 2, 4, 4]
        let writer = try StateCacheWriter(shape: shape, dtype: .float32)
        MLXRandom.seed(6)
        let all = MLXRandom.normal(shape) * 0.5
        eval(all)
        try writer.write(rows: Array(0 ..< 6), all)
        let cache = try writer.finish(
            pairIndex: [[0, 1], [2, 3], [4, 5]], labels: [0, 1, 0],
            hardNegs: [[], [], []], contract: [:])

        let (states, labels) = cache.batch([2, 0])
        XCTAssertEqual(states.shape, [4, 1, 2, 4, 4])
        XCTAssertEqual(labels.asArray(Int32.self), [0, 0])
        let want = concatenated([all[4 ..< 6], all[0 ..< 2]], axis: 0)
        XCTAssertEqual(maxAbsDiff(states, want), 0)
    }

    /// Файл на диске: сохранить, перечитать через mmap, получить то же самое.
    func testDiskRoundTripMapped() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cache_\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: StateCache.statesURL(base))
            try? FileManager.default.removeItem(at: StateCache.indexURL(base))
        }

        let shape = [4, 1, 2, 4, 4]
        let writer = try StateCacheWriter(shape: shape, dtype: .float16, path: base)
        MLXRandom.seed(7)
        let all = MLXRandom.normal(shape) * 0.5
        eval(all)
        try writer.write(rows: Array(0 ..< 4), all)
        let built = try writer.finish(pairIndex: [[0, 1], [2, 3]], labels: [1, 0],
                                      hardNegs: [[0], [1]],
                                      contract: ["template": "doc_first"])

        let loaded = try StateCache.load(base)
        XCTAssertEqual(loaded.shape, shape)
        XCTAssertEqual(loaded.labels, [1, 0])
        XCTAssertEqual(loaded.hardNegs, [[0], [1]])
        XCTAssertEqual(loaded.contract["template"], "doc_first")
        XCTAssertEqual(maxAbsDiff(loaded.gather([0, 1, 2, 3]),
                                  built.gather([0, 1, 2, 3])), 0,
                       "перечитанный с диска кэш отличается от построенного")

        // fp16 округляет — но это округление ОДНОГО значения, а не потеря
        // строки: расхождение с оригиналом должно быть на уровне кванта fp16.
        XCTAssertLessThan(relDiff(all, loaded.gather([0, 1, 2, 3])), 1e-3,
                          "fp16 потерял больше, чем свой квант")
    }

    /// fp16 переполняется — внятная ошибка, а не тихие inf в кэше.
    func testFP16OverflowIsReported() throws {
        let writer = try StateCacheWriter(shape: [1, 1, 1, 4, 4], dtype: .float16)
        try writer.write(rows: [0], MLXArray.full([1, 1, 1, 4, 4],
                                                  values: MLXArray(Float(70000))))
        XCTAssertThrowsError(try writer.finish(pairIndex: [[0]], labels: [0],
                                               hardNegs: [[]], contract: [:])) { err in
            guard case StateCacheError.fp16Overflow = err else {
                return XCTFail("ожидалось fp16Overflow, получено \(err)")
            }
        }
    }

    /// Кэш от ДРУГОЙ головы отвергается. Формы у него правдоподобные, и без
    /// проверки обучение пошло бы как ни в чём не бывало.
    func testIncompatibleCacheRejected() throws {
        let (_, bb, _) = try makeModel(nLayer: 4)
        let oneLayer = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        let twoLayers = try RerankerHead(base: bb,
                                         cfg: RerankerConfig(layerIdx: [0, 2]))

        let writer = try StateCacheWriter(shape: [2, 1, bb.cfg.nHead, 64, 64],
                                          dtype: .float16)
        try writer.write(rows: [0, 1],
                         MLXArray.zeros([2, 1, bb.cfg.nHead, 64, 64]))
        let cache = try writer.finish(pairIndex: [[0, 1]], labels: [0],
                                      hardNegs: [[]],
                                      contract: ["template": "doc_first",
                                                 "max_doc_tokens": "384"])

        XCTAssertNoThrow(try cache.checkCompatible(
            head: oneLayer, contract: ["template": "doc_first"]))
        XCTAssertThrowsError(try cache.checkCompatible(
            head: twoLayers, contract: ["template": "doc_first"]),
            "кэш на один слой принят головой, читающей два")
        XCTAssertThrowsError(try cache.checkCompatible(
            head: oneLayer, contract: ["template": "query_first"]),
            "кэш принят при другом шаблоне подачи текста")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Кодирование
    // ─────────────────────────────────────────────────────────────────

    /// ГЛАВНОЕ: кэшированный путь == сплошной.
    ///
    /// Допуск, а не равенство: разрез последовательности меняет форму входа
    /// матмулов, и те раскладываются на GPU иначе. Причинность при этом
    /// точная — она проверена отдельно на уровне ядра (WKV7StateTests), где
    /// разрез даёт РОВНО ноль.
    ///
    /// Замерено 5.7e-7 относительного при границе 1e-4. Именно этот тест
    /// поймал промах со словарём: расхождение было 1.06, то есть на шесть
    /// порядков больше шума, — и слабая граница вроде 1e-2 всё равно бы его
    /// заметила, но списать на «bf16» уже не дала бы.
    func testCachedPathMatchesDirectPath() throws {
        let (model, _, _) = try makeModel()
        let tok = try worldTokenizer()
        let (pool, samples) = try makeSamples(4, nCand: 3)
        let cfg = RerankEncodeConfig(maxDocTokens: 64, maxQueryTokens: 32,
                                     docBatch: 3, queryBatch: 5, dtype: .float32)

        let cache = try RerankEncoder.encodePairs(model, tokenizer: tok,
                                                  pool: pool, samples: samples,
                                                  config: cfg)
        // Те же пары сплошным проходом, в порядке строк кэша.
        var pairs: [(instruct: String, document: String, query: String)] = []
        for s in samples {
            for did in s.docIds {
                pairs.append((s.instruct, pool[did], s.query))
            }
        }
        let direct = try RerankEncoder.encodePairsDirect(model, tokenizer: tok,
                                                         pairs: pairs, config: cfg)
        let cached = cache.gather(Array(0 ..< cache.nPairs))

        XCTAssertEqual(cached.shape, direct.shape)
        XCTAssertLessThan(relDiff(direct, cached), 1e-4,
                          "кэшированный путь разошёлся со сплошным")
    }

    /// Префиксы дедуплицируются: документ, встретившийся у нескольких
    /// запросов, кодируется ОДИН раз.
    ///
    /// Проверяется по числу вызовов кодирования префиксов (через прогресс),
    /// а не по времени: время шумит, а счётчик — нет. Ради этой экономии всё
    /// и построено: без неё «добавить кандидатов» стоило бы полного прохода
    /// на каждого.
    func testPrefixesAreDeduplicated() throws {
        let (model, _, _) = try makeModel()
        let tok = try worldTokenizer()
        // Один общий пул из шести документов, восемь запросов по три
        // кандидата — пар 24, уникальных префиксов не больше шести.
        let rows = (0 ..< 8).map { i in
            RerankRow(instruct: "и", query: "запрос \(i)",
                      positive: "документ \(i % 3)",
                      negatives: ["документ \((i + 1) % 3)"])
        }
        let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 3, seed: 0)

        var prefixTotal = 0, pairTotal = 0
        _ = try RerankEncoder.encodePairs(
            model, tokenizer: tok, pool: pool, samples: samples,
            config: RerankEncodeConfig(maxDocTokens: 32, maxQueryTokens: 16,
                                       dtype: .float32),
            progress: { _, pt, _, prt in prefixTotal = pt; pairTotal = prt })

        XCTAssertEqual(pairTotal, 24)
        XCTAssertEqual(prefixTotal, pool.count,
                       "префиксов \(prefixTotal) при \(pool.count) документах — "
                       + "дедупликация не сработала")
        XCTAssertLessThan(prefixTotal, pairTotal / 2,
                          "экономия на префиксах меньше двукратной")
    }

    /// Кэш даёт голове ровно те состояния, на которых она потом считает
    /// скоры, — и число кандидатов сохраняется.
    func testCacheFeedsHeadEndToEnd() throws {
        let (model, _, _) = try makeModel()
        let tok = try worldTokenizer()
        let (pool, samples) = try makeSamples(5, nCand: 4)
        let cache = try RerankEncoder.encodePairs(
            model, tokenizer: tok, pool: pool, samples: samples,
            config: RerankEncodeConfig(maxDocTokens: 48, maxQueryTokens: 24,
                                       dtype: .float32))

        XCTAssertEqual(cache.nSamples, 5)
        XCTAssertEqual(cache.nCandidates, 4)
        XCTAssertEqual(cache.labels, samples.map { $0.label })
        XCTAssertEqual(cache.hardNegs, samples.map { $0.hardNegs })
        try cache.checkCompatible(head: model.head,
                                  contract: ["template": "doc_first"])

        let (states, _) = cache.batch(Array(0 ..< 5))
        let scores = model.scoreStates(states).reshaped([5, 4])
        eval(scores)
        XCTAssertEqual(scores.shape, [5, 4])
        // Голова zero-init ⇒ ровно нули. Заодно проверяет, что кэш не
        // подсунул мусор, который дал бы NaN.
        XCTAssertEqual(MLX.abs(scores).max().item(Float.self), 0)
    }

    /// Округление длины батча НЕ меняет состояния.
    ///
    /// Округление введено ради памяти: без него почти каждая пачка имеет
    /// свою длину, формы буферов Metal не повторяются, и буферный кэш растёт
    /// линейно по числу пачек (замерено: 11.3 ГБ на прогоне 1000×8). Но
    /// вводить ради памяти то, что меняет числа, нельзя, — а пад-позиции
    /// нейтральны для рекуррентности, значит и не должно.
    func testLengthBucketingDoesNotChangeStates() throws {
        let (model, _, _) = try makeModel()
        let tok = try worldTokenizer()
        let (pool, samples) = try makeSamples(4, nCand: 3)

        // minT: 0 — округлять ВСЁ подряд. Обязательно явно: умолчание
        // округляет только батчи от 4×bucket токенов, а документы здесь
        // обрезаны до 64, так что при умолчании округления не случилось бы
        // вовсе и тест сравнивал бы путь сам с собой. Ровно так он и
        // выродился, когда умолчание сменили.
        func encode(_ bucket: Int) throws -> MLXArray {
            let cfg = RerankEncodeConfig(maxDocTokens: 64, maxQueryTokens: 32,
                                         dtype: .float32, lengthBucket: bucket,
                                         lengthBucketMinTokens: 0)
            let c = try RerankEncoder.encodePairs(model, tokenizer: tok,
                                                  pool: pool, samples: samples,
                                                  config: cfg)
            return c.gather(Array(0 ..< c.nPairs))
        }
        // Замерено 2.3e-7: то же, что между кэшированным и сплошным путём.
        // Ровного нуля нет и не будет — меняется форма входа матмулов.
        XCTAssertLessThan(relDiff(try encode(0), try encode(64)), 1e-5,
                          "округление длины изменило состояния")
        XCTAssertLessThan(relDiff(try encode(64), try encode(128)), 1e-5,
                          "разные размеры корзины дали разные состояния")
    }

    /// Округляются только ДЛИННЫЕ батчи, короткие остаются как есть.
    ///
    /// Это не микрооптимизация. Замерено на 0.1B (200 префиксов, три
    /// прогона): округление всего подряд до 64 стоит 38% полного времени
    /// кодирования, и почти вся эта цена — на хвостах. Хвост это около
    /// двадцати токенов, добивка до 64 утраивает работу: 14.2 мс против
    /// 6.4 мс на хвост. Порог 4×bucket держит накладные ниже четверти.
    func testOnlyLongBatchesAreBucketed() {
        let short = [[Int]](repeating: [Int](repeating: 1, count: 20), count: 4)
        let long = [[Int]](repeating: [Int](repeating: 1, count: 400), count: 4)

        // Короткий батч при пороге не трогается.
        XCTAssertEqual(RerankEncoder.batchIds(short, bucket: 64, minT: 256)
                        .idx.shape[1], 20)
        // Длинный — округляется вверх до кратной 64.
        XCTAssertEqual(RerankEncoder.batchIds(long, bucket: 64, minT: 256)
                        .idx.shape[1], 448)
        // Без порога округляется всё, в том числе короткий — то самое
        // поведение, которое стоило 38%.
        XCTAssertEqual(RerankEncoder.batchIds(short, bucket: 64, minT: 0)
                        .idx.shape[1], 64)
        // Ровно на пороге округление уже применяется.
        let atThreshold = [[Int]](repeating: [Int](repeating: 1, count: 256),
                                  count: 2)
        XCTAssertEqual(RerankEncoder.batchIds(atThreshold, bucket: 64, minT: 256)
                        .idx.shape[1], 256)
    }

    /// Умолчание порога выведено из размера корзины, а не задано отдельно.
    ///
    /// Иначе они разъезжаются: кто-нибудь поменяет `lengthBucket`, порог
    /// останется от прежнего, и правило «накладные ниже четверти» перестанет
    /// выполняться молча.
    func testBucketThresholdFollowsBucketSize() {
        XCTAssertEqual(RerankEncodeConfig().lengthBucket, 64)
        XCTAssertEqual(RerankEncodeConfig().lengthBucketMinTokens, 256)
        XCTAssertEqual(RerankEncodeConfig(lengthBucket: 32).lengthBucketMinTokens,
                       128)
        XCTAssertEqual(
            RerankEncodeConfig(lengthBucket: 64,
                               lengthBucketMinTokens: 9).lengthBucketMinTokens, 9,
            "явно заданный порог перебит выведенным")
    }

    /// Порог округления доезжает до батчера, а не остаётся в конфигурации.
    ///
    /// Проверяется СТРУКТУРНО — формами поданных батчей, а не числами.
    /// Иначе никак: выравнивание на состояния не влияет (это доказано
    /// отдельным тестом выше), поэтому кодирование с порогом и без него
    /// даёт побитово сравнимые кэши и отличается только объёмом лишней
    /// работы. Мутация «не передавать порог в batchIds» не ловилась ничем,
    /// пока этого теста не было.
    func testThresholdReachesTheBatcher() throws {
        let (model, _, _) = try makeModel()
        let tok = try worldTokenizer()
        let (pool, samples) = try makeSamples(4, nCand: 3)

        // Формы РАЗВЕДЕНЫ по видам батчей. Первая редакция теста складывала
        // их в один список — и мутация «не передавать порог префиксам»
        // проходила мимо: утверждение выполнялось за счёт хвостов, которые
        // порог получали исправно. Оба пути обязаны проверяться порознь.
        func shapes(_ minT: Int?) throws -> [String: [Int]] {
            let cfg = RerankEncodeConfig(maxDocTokens: 64, maxQueryTokens: 32,
                                         dtype: .float32, lengthBucket: 64,
                                         lengthBucketMinTokens: minT)
            var seen: [String: [Int]] = ["prefix": [], "tail": []]
            _ = try RerankEncoder.encodePairs(
                model, tokenizer: tok, pool: pool, samples: samples,
                config: cfg, onBatch: { kind, shape in
                    seen[kind, default: []].append(shape[1])
                })
            return seen
        }

        let all = try shapes(0)
        let withThreshold = try shapes(nil)

        for kind in ["prefix", "tail"] {
            let rounded = all[kind]!, kept = withThreshold[kind]!
            XCTAssertFalse(rounded.isEmpty, "\(kind): батчей не было вовсе")
            XCTAssertEqual(kept.count, rounded.count)
            // Порог 0 — округляется всё: каждая длина кратна 64.
            XCTAssertTrue(rounded.allSatisfy { $0 % 64 == 0 },
                          "\(kind): при пороге 0 округлены не все: \(rounded)")
            // Умолчание — тексты короткие, округляться нечему.
            XCTAssertTrue(kept.contains { $0 % 64 != 0 },
                          "\(kind): порог не доехал, все длины кратны 64: \(kept)")
            XCTAssertTrue(kept.allSatisfy { $0 < 256 },
                          "\(kind): тест вырожден — батчи длиннее порога")
        }
    }

    /// Инструкция попадает в контракт кэша, только если она в корпусе ОДНА.
    ///
    /// При обучении инструкция — поле примера, при выдаче — часть
    /// замороженного префикса. Когда она одна, кэш имеет право её
    /// зафиксировать, и голова унесёт её в свой чекпоинт. Когда их несколько,
    /// писать нечего, и молчание честнее умолчания: выдача, унаследовавшая
    /// «одну из» инструкций, давала бы правдоподобные числа не от того текста.
    func testInstructRecordedOnlyWhenUnique() throws {
        let tok = try worldTokenizer()
        let (model, _, _) = try makeModel()
        let cfg = RerankEncodeConfig(maxDocTokens: 32, maxQueryTokens: 16,
                                     dtype: .float32)

        func cache(_ instructs: [String]) throws -> StateCache {
            let rows = instructs.enumerated().map { i, ins in
                RerankRow(instruct: ins, query: "вопрос \(i)",
                          positive: "верный пассаж \(i)",
                          negatives: ["неверный пассаж \(i)"])
            }
            let (pool, samples) = try RerankCandidates.build(rows,
                                                             nCandidates: 2,
                                                             seed: 0)
            return try RerankEncoder.encodePairs(model, tokenizer: tok,
                                                 pool: pool, samples: samples,
                                                 config: cfg)
        }

        let one = try cache(["найди ответ", "найди ответ"])
        XCTAssertEqual(one.contract["instruct"], "найди ответ")

        let many = try cache(["найди ответ", "другая задача"])
        XCTAssertNil(many.contract["instruct"],
                     "инструкция записана при нескольких разных в корпусе")
        // Различающее утверждение: остальной контракт на месте в обоих —
        // иначе «нет ключа» означало бы просто пустой контракт.
        XCTAssertEqual(many.contract["max_doc_tokens"], "32")
        XCTAssertEqual(one.contract["max_doc_tokens"], "32")
    }

    /// Выравнивание НЕ входит в контракт подачи текста.
    ///
    /// Утверждение о границе контракта: пад-позиции для рекуррентности
    /// нейтральны, состояние от них не зависит, и требовать совпадения
    /// выравнивания значило бы запрещать переиспользовать кэш после смены
    /// параметра производительности. Проверяется вместе с тем, что всё
    /// ОСТАЛЬНОЕ в контракт входит, — иначе утверждение было бы про пустоту.
    func testAlignmentIsNotPartOfTheContract() {
        let a = RerankEncodeConfig(lengthBucket: 64, lengthBucketMinTokens: 256)
        let b = RerankEncodeConfig(lengthBucket: 0, lengthBucketMinTokens: 0)
        XCTAssertEqual(a.contract, b.contract)
        XCTAssertNotEqual(a.contract,
                          RerankEncodeConfig(maxDocTokens: 128).contract)
    }

    /// Токенизатор от ДРУГОЙ модели останавливает кодирование, а не портит
    /// кэш молча.
    ///
    /// Тест написан по факту: именно этот промах и дал первое расхождение
    /// кэшированного пути со сплошным. Выборка строки эмбеддинга за границей
    /// таблицы не падает — MLX возвращает мусор, причём зависящий от формы
    /// батча, — так что без явной проверки кодирование отработало бы минуты
    /// и выдало правдоподобный кэш, на котором голова обучилась бы шуму.
    func testForeignTokenizerIsRejected() throws {
        let tok = try worldTokenizer()
        // Словарь модели игрушечный, токенизатор — настоящий.
        let (bb, _) = TinyBackbone.make(nLayer: 2, vocab: 64)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        let (pool, samples) = try makeSamples(2, nCand: 2)

        XCTAssertThrowsError(try RerankEncoder.encodePairs(
            model, tokenizer: tok, pool: pool, samples: samples)) { err in
            guard case RerankEncodeError.vocabOverflow(let maxId, let vocab) = err else {
                return XCTFail("ожидалась vocabOverflow, получено \(err)")
            }
            XCTAssertGreaterThan(maxId, vocab)
            XCTAssertEqual(vocab, 64)
        }
    }

    /// Обрезка документа — по ТОКЕНАМ, а не по символам, и режется хвост.
    func testDocumentTruncationIsByTokens() throws {
        let tok = try worldTokenizer()
        let doc = String(repeating: "длинный документ ", count: 200)
        let short = RerankEncodeConfig(maxDocTokens: 16)
        let long = RerankEncodeConfig(maxDocTokens: 64)

        let a = RerankEncoder.prefixIds(tok, short, instruct: "и", document: doc)
        let b = RerankEncoder.prefixIds(tok, long, instruct: "и", document: doc)
        XCTAssertEqual(b.count - a.count, 48,
                       "обрезка не по токенам: разница \(b.count - a.count)")
        // Обрезается ХВОСТ: начало обеих последовательностей совпадает.
        XCTAssertEqual(Array(a.prefix(a.count - 1)),
                       Array(b.prefix(a.count - 1)),
                       "обрезано начало документа, а не хвост")
    }

    /// Настоящий токенизатор World. Синтетический тут не годится: обрезка
    /// по токенам и дедупликация префиксов зависят от того, как реальный
    /// словарь режет текст.
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
}
