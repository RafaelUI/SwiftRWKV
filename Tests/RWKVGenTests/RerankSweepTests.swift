//
//  RerankSweepTests.swift
//  Кэш НАДМНОЖЕСТВА слоёв и развёртка по сидам.
//
//  Ради чего это всё: раньше кэш был привязан к тому, что читает голова,
//  поэтому каждая конфигурация стоила своего кодирования (на реальной 0.1B —
//  12 минут). Теперь кэш держит надмножество, а голова берёт срез, и десять
//  конфигураций стоят одного кодирования.
//
//  Цена такой гибкости — новый молчаливый режим отказа: срез не той ширины
//  или не тех слоёв даёт ПРАВДОПОДОБНЫЕ числа. Состояние слоя 5 и состояние
//  слоя 11 неотличимы по форме, поэтому подстановка не падает нигде — она
//  просто обучает голову на чужом слое. Половина тестов здесь именно про это.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank

final class RerankSweepTests: XCTestCase {

    // ─────────────────────────────────────────────────────────────────
    //  Оснастка
    // ─────────────────────────────────────────────────────────────────

    /// Словарь ПОЛНОГО размера — по той же причине, что в StateCacheTests:
    /// выход за таблицу эмбеддингов в MLX не падает, а возвращает мусор,
    /// зависящий от формы батча.
    func makeBackbone(nLayer: Int = 4) -> (X070Backbone, X070Config) {
        TinyBackbone.make(nLayer: nLayer, vocab: 65536)
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

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    /// Кэш заданной формы, забитый известным шумом. Значение слота i несёт
    /// метку i, чтобы перепутанный слот было видно, а не только «не сошлось».
    ///
    /// Формы `[H, S, S]` — НАСТОЯЩИЕ формы состояния TinyBackbone, а не
    /// удобные маленькие: голова проверяет их preconditio'ом, и синтетика
    /// не той формы просто не дошла бы до обучения.
    func syntheticCache(nPairs: Int, sources: [Int]?, nSrc: Int,
                        seed: UInt64 = 11) throws -> (StateCache, MLXArray) {
        let shape = [nPairs, nSrc, 2, 64, 64]
        let writer = try StateCacheWriter(shape: shape, dtype: .float32)
        // Содержимое слоя определяется НОМЕРОМ СЛОЯ, а не позицией слота.
        // Иначе кэш на один слой и надмножество, его содержащее, содержали
        // бы разные числа в «одном и том же» слое, и сравнивать срез с
        // точным кэшем было бы нечего. Плюс к шуму добавлен номер слоя:
        // срез, взявший не тот слот, отличается на целую единицу, а не на
        // шум, и это видно прямо в сообщении об ошибке.
        let layers = sources ?? Array(0 ..< nSrc)
        precondition(layers.count == nSrc)
        var slabs: [MLXArray] = []
        for layer in layers {
            MLXRandom.seed(seed &+ UInt64(layer) &* 1_000)
            slabs.append(MLXRandom.normal([nPairs, 1, 2, 64, 64]) * 0.1
                         + Float(layer + 1))
        }
        let all = concatenated(slabs, axis: 1)
        eval(all)
        try writer.write(rows: Array(0 ..< nPairs), all)
        let nSamples = nPairs / 2
        let cache = try writer.finish(
            pairIndex: (0 ..< nSamples).map { [$0 * 2, $0 * 2 + 1] },
            labels: [Int](repeating: 0, count: nSamples),
            hardNegs: [[Int]](repeating: [1], count: nSamples),
            contract: ["template": "doc_first"],
            sources: sources)
        return (cache, all)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Срез кэша
    // ─────────────────────────────────────────────────────────────────

    /// Слоты разрешаются в порядке `uniqueSources`, а не в порядке `layerIdx`.
    ///
    /// Порядок здесь несущий: голова адресует слоты через `sourceSlot`,
    /// который построен по ВОЗРАСТАНИЮ. Перепутать порядок — значит подать
    /// каждому блоку состояние соседа, и формы при этом сойдутся.
    func testSlotsFollowUniqueSourcesOrder() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 4, sources: [0, 1, 2, 3],
                                            nSrc: 4)
        // layerIdx задан в порядке 2,0 — uniqueSources обязан быть [0,2].
        let head = try RerankerHead(base: bb,
                                    cfg: RerankerConfig(layerIdx: [2, 0]))
        XCTAssertEqual(head.uniqueSources, [0, 2])
        XCTAssertEqual(try cache.slots(for: head), [0, 2])

        let mid = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [3]))
        XCTAssertEqual(try cache.slots(for: mid), [3])
    }

    /// Срез надмножества побитово равен тому же слою, взятому целиком.
    ///
    /// Это и есть обещание фичи: «взять из кэша на четыре слоя один» обязано
    /// дать РОВНО то же, что кэш ровно на этот слой. Не «примерно» — здесь
    /// нет никакой арифметики, только копирование байтов, и любое
    /// расхождение означает промах по смещению.
    func testSlicedGatherIsBitIdenticalToWholeRow() throws {
        let nSrc = 4
        let (cache, all) = try syntheticCache(nPairs: 6, sources: [0, 1, 2, 3],
                                              nSrc: nSrc)
        let rows = [4, 1, 5, 0]
        for slot in 0 ..< nSrc {
            let sliced = cache.gather(rows, slots: [slot])
            let want = concatenated(rows.map { all[$0 ..< ($0 + 1)] }, axis: 0)[
                0..., slot ..< (slot + 1)]
            XCTAssertEqual(sliced.shape, [rows.count, 1, 2, 64, 64])
            XCTAssertEqual(maxAbsDiff(sliced, want), 0,
                           "слот \(slot): срез разошёлся со строкой целиком")
        }
        // Несколько слотов, и НЕ подряд.
        let two = cache.gather(rows, slots: [0, 2])
        let whole = concatenated(rows.map { all[$0 ..< ($0 + 1)] }, axis: 0)
        XCTAssertEqual(maxAbsDiff(two, concatenated(
            [whole[0..., 0 ..< 1], whole[0..., 2 ..< 3]], axis: 1)), 0)
    }

    /// Путь «вся строка» не изменился от появления срезов.
    ///
    /// В `gather` для непрерывного случая осталась одна memcpy на строку, и
    /// это ровно тот путь, которым кэш читался раньше. Тест сторожит его от
    /// подмены на посегментный: результат обязан совпасть побитово, а не
    /// «в пределах допуска», потому что арифметики тут нет вовсе.
    func testWholeRowPathUnchangedBySlicing() throws {
        let (cache, _) = try syntheticCache(nPairs: 6, sources: [0, 1, 2, 3],
                                            nSrc: 4)
        let rows = [3, 0, 5]
        XCTAssertEqual(maxAbsDiff(cache.gather(rows),
                                  cache.gather(rows, slots: [0, 1, 2, 3])), 0)
    }

    /// Голова, читающая слой, которого в кэше НЕТ, обрывается.
    ///
    /// Самое важное утверждение файла. Раньше проверялось только ЧИСЛО слоёв,
    /// поэтому кэш для слоя 0 и голова над слоем 3 (оба nSrc = 1) сходились
    /// молча и обучение шло на чужом слое до самого конца прогона.
    func testHeadOverAbsentLayerIsRejected() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 4, sources: [0], nSrc: 1)

        let onLayer0 = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [0]))
        XCTAssertNoThrow(try cache.slots(for: onLayer0))

        // Тот же ОДИН слот, но другой слой — то, что раньше проходило.
        let onLayer3 = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        XCTAssertEqual(onLayer3.uniqueSources.count, cache.nSources,
                       "тест вырожден: числа слоёв обязаны СОВПАДАТЬ, "
                       + "иначе он ловит старой проверкой по числу")
        XCTAssertThrowsError(try cache.slots(for: onLayer3),
                             "кэш слоя 0 принят головой над слоем 3")

        // И надмножество, которого не хватает лишь частично.
        let (superset, _) = try syntheticCache(nPairs: 4, sources: [0, 1],
                                               nSrc: 2)
        let onZeroAndThree = try RerankerHead(
            base: bb, cfg: RerankerConfig(layerIdx: [0, 3]))
        XCTAssertThrowsError(try superset.slots(for: onZeroAndThree))
    }

    /// Кэш без состава слоёв (собранный до появления поля) читается, но
    /// проверяется только по числу слотов — и об этом сказано вслух.
    func testLegacyCacheWithoutSourcesFallsBackToCount() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 4, sources: nil, nSrc: 1)
        XCTAssertNil(cache.sources)

        let one = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        XCTAssertEqual(try cache.slots(for: one), [0],
                       "старый кэш перестал читаться")

        let two = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [0, 2]))
        XCTAssertThrowsError(try cache.slots(for: two),
                             "число слотов не сошлось, а кэш принят")
    }

    /// Состав слоёв переживает круг через диск.
    func testSourcesSurviveDiskRoundTrip() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("sweep_sources_\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: base.appendingPathExtension("states"))
            try? FileManager.default.removeItem(at: base.appendingPathExtension("idx.json"))
        }
        let (cache, _) = try syntheticCache(nPairs: 4, sources: [1, 3], nSrc: 2)
        try cache.save(to: base)
        let loaded = try StateCache.load(base)
        XCTAssertEqual(loaded.sources, [1, 3])
    }

    // ─────────────────────────────────────────────────────────────────
    //  Кодирование надмножества
    // ─────────────────────────────────────────────────────────────────

    /// Кодирование надмножества и кодирование одного слоя дают ОДНО И ТО ЖЕ
    /// состояние для общего слоя — побитово.
    ///
    /// Ради этого равенства всё и затевалось: если оно есть, десять
    /// конфигураций честно стоят одного кодирования. Равенство строгое, а не
    /// с допуском: проход по базе один и тот же, отличается только то, какие
    /// слои с него снимают, — арифметике меняться негде.
    func testSupersetEncodingSlicesToExactSingleLayerCache() throws {
        let tok = try worldTokenizer()
        let (bb, _) = makeBackbone(nLayer: 4)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        let (pool, samples) = try makeSamples(4, nCand: 3)
        let cfg = RerankEncodeConfig(maxDocTokens: 32, maxQueryTokens: 16,
                                     docBatch: 2, queryBatch: 3,
                                     dtype: .float32)

        let exact = try RerankEncoder.encodePairs(
            model, tokenizer: tok, pool: pool, samples: samples, config: cfg)
        let superset = try RerankEncoder.encodePairs(
            model, tokenizer: tok, pool: pool, samples: samples, config: cfg,
            sources: [0, 1, 2, 3])

        XCTAssertEqual(exact.sources, [3])
        XCTAssertEqual(superset.sources, [0, 1, 2, 3])
        XCTAssertEqual(superset.nSources, 4)

        let rows = Array(0 ..< exact.nPairs)
        let slots = try superset.slots(for: model.head)
        XCTAssertEqual(slots, [3])
        XCTAssertEqual(maxAbsDiff(superset.gather(rows, slots: slots),
                                  exact.gather(rows)), 0,
                       "срез надмножества разошёлся с кэшем ровно этого слоя")
    }

    /// Отрицательные индексы слоёв нормализуются, повторы схлопываются,
    /// порядок — по возрастанию.
    func testSourceListIsNormalised() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let head = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        XCTAssertEqual(try RerankEncoder.resolveSources([3, -1, 0, 0],
                                                        head: head, nLayer: 4),
                       [0, 3])
        XCTAssertEqual(try RerankEncoder.resolveSources(nil, head: head,
                                                        nLayer: 4), [3])
        XCTAssertThrowsError(try RerankEncoder.resolveSources([4], head: head,
                                                              nLayer: 4))
        XCTAssertThrowsError(try RerankEncoder.resolveSources([-5], head: head,
                                                              nLayer: 4))
    }

    // ─────────────────────────────────────────────────────────────────
    //  Обучение на срезе
    // ─────────────────────────────────────────────────────────────────

    /// Обучение на срезе надмножества == обучение на кэше ровно этих слоёв.
    ///
    /// Побитовое равенство состояний (выше) ещё не означает равенства
    /// обучения: батч мог бы прийти в другой форме и развалить матмулы иначе.
    /// Здесь формы совпадают по построению, поэтому и результат обязан
    /// совпасть точно — что и проверяется, а не предполагается.
    func testTrainingOnSliceMatchesTrainingOnExactCache() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let cfg = RerankerConfig(layerIdx: [-1])

        let (exact, _) = try syntheticCache(nPairs: 8, sources: [3], nSrc: 1,
                                            seed: 21)
        // Тот же шум в слоте 3 надмножества — синтетика строится по слоту,
        // поэтому надмножество отдельно, а сверяется срез с точным кэшем.
        let (superset, _) = try syntheticCache(nPairs: 8, sources: [0, 1, 2, 3],
                                               nSrc: 4, seed: 21)
        // Проверяем предпосылку: слот 3 надмножества обязан совпасть с
        // единственным слотом точного кэша, иначе тест сравнивает разное.
        let rows = Array(0 ..< 8)
        try XCTSkipIf(maxAbsDiff(superset.gather(rows, slots: [3]),
                                 exact.gather(rows)) != 0,
                      "синтетика собралась несравнимо — тест не о том")

        let tcfg = RerankTrainConfig(batchSize: 2, epochs: 2, seed: 7)
        func run(_ cache: StateCache) throws -> RankingMetrics {
            let m = try Reranker(base: bb, cfg: cfg, seed: 3)
            let r = try RerankTraining.train(m, trainCache: cache,
                                             evalCache: cache, config: tcfg)
            return r.after!
        }
        XCTAssertEqual(try run(superset), try run(exact),
                       "обучение на срезе разошлось с обучением на точном кэше")
    }

    /// Оценка читает СВОЙ срез, а не срез обучающего кэша.
    ///
    /// Кэши train и eval — разные файлы, и собраны они могут быть в разное
    /// время с разным составом слоёв: обучающий впрок надмножеством,
    /// отложенный — ровно под одну конфигурацию. Слот слоя 3 в первом это
    /// номер 3, во втором — номер 0. Пока оба кэша в тестах были одним и тем
    /// же объектом, подмена одного среза другим не ловилась ничем: мутация
    /// `evalSlots = trainSlots` проходила молча.
    func testEvalCacheUsesItsOwnSlots() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (train, _) = try syntheticCache(nPairs: 12, sources: [0, 1, 2, 3],
                                            nSrc: 4, seed: 71)
        let (heldOut, _) = try syntheticCache(nPairs: 8, sources: [3], nSrc: 1,
                                              seed: 72)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]))
        XCTAssertEqual(try train.slots(for: model.head), [3])
        XCTAssertEqual(try heldOut.slots(for: model.head), [0],
                       "тест вырожден: срезы обязаны РАЗЛИЧАТЬСЯ")

        let r = try RerankTraining.train(
            model, trainCache: train, evalCache: heldOut,
            config: RerankTrainConfig(batchSize: 2, epochs: 1))
        XCTAssertNotNil(r.after)
        XCTAssertEqual(r.firstLoss, r.expectedFirstLoss, accuracy: 1e-5)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Отбор слоёв
    // ─────────────────────────────────────────────────────────────────

    /// `select(_:sources:)` берёт ИМЕННО заказанные слои состояния.
    ///
    /// Сверка идёт с `state.layerWKV(k)` напрямую, а не через второй путь
    /// отбора. Это принципиально: и кэш надмножества, и точный кэш, и
    /// сплошной путь пользуются ОДНОЙ этой функцией, поэтому «слой 5 читает
    /// слой 0» ломает их все ОДИНАКОВО, и любое сравнение пути с путём
    /// остаётся зелёным. Мутация ровно с таким смыслом и не ловилась, пока
    /// этого теста не было.
    func testSelectTakesRequestedLayers() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        let ids = MLXArray((0 ..< 12).map { Int32(($0 * 977) % 65536) }, [2, 6])
        let state = model.encode(ids)

        for k in 0 ..< 4 {
            let got = RerankerHead.select(state, sources: [k])
            XCTAssertEqual(maxAbsDiff(got, state.layerWKV(k)
                                              .expandedDimensions(axis: 1)), 0,
                           "sources: [\(k)] отдал не слой \(k)")
        }
        // Различающее утверждение: слои обязаны БЫТЬ разными, иначе
        // предыдущее выполняется при любой реализации.
        XCTAssertNotEqual(maxAbsDiff(state.layerWKV(0), state.layerWKV(3)), 0,
                          "слои состояния совпали — тест вырожден")

        // Порядок и состав нескольких слоёв.
        let pair = RerankerHead.select(state, sources: [1, 3])
        XCTAssertEqual(pair.shape[1], 2)
        XCTAssertEqual(maxAbsDiff(pair[0..., 0 ..< 1],
                                  state.layerWKV(1).expandedDimensions(axis: 1)), 0)
        XCTAssertEqual(maxAbsDiff(pair[0..., 1 ..< 2],
                                  state.layerWKV(3).expandedDimensions(axis: 1)), 0)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Контракт подачи текста при обучении
    // ─────────────────────────────────────────────────────────────────

    /// Отложенный кэш с ДРУГИМ контрактом обрывает обучение.
    ///
    /// train и eval — разные файлы, собираемые разными вызовами, и разъехаться
    /// им ничего не мешает. Раньше сюда передавался пустой словарь, то есть
    /// сверялась только форма состояния: обучение на паре «обрезка 128» +
    /// «обрезка 384» шло как ни в чём не бывало, и метрики на held-out мерили
    /// голову по тексту, которого она при обучении не видела.
    func testEvalCacheWithOtherContractIsRejected() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]))

        func cache(_ maxDoc: String, pairs: Int) throws -> StateCache {
            let (c, _) = try syntheticCache(nPairs: pairs, sources: [3],
                                            nSrc: 1, seed: 81)
            // Контракт синтетики задаётся напрямую: важен факт расхождения,
            // а не то, каким кодированием он получен.
            var idx = c.index
            idx.contract = ["template": "doc_first", "max_doc_tokens": maxDoc]
            return StateCache(storage: c.storage, index: idx)
        }

        let train = try cache("384", pairs: 12)
        let same = try cache("384", pairs: 8)
        let other = try cache("128", pairs: 8)
        let cfg = RerankTrainConfig(batchSize: 2, epochs: 1)

        XCTAssertNoThrow(try RerankTraining.train(model, trainCache: train,
                                                  evalCache: same, config: cfg))
        XCTAssertThrowsError(try RerankTraining.train(model, trainCache: train,
                                                      evalCache: other,
                                                      config: cfg),
                             "кэши с разной обрезкой документа приняты")
    }

    /// Явное ожидание вызывающего сильнее того, что лежит в кэше.
    ///
    /// Умолчание («взять из обучающего кэша») закрывает расхождение train/eval,
    /// но не случай «переиспользовали чужой готовый кэш». Прогонщик знает,
    /// каким текстом он хотел кормить модель, и обязан узнать о подмене.
    func testExplicitContractOverridesTheCache() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]))
        let (c, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                        seed: 82)
        var idx = c.index
        idx.contract = ["template": "doc_first", "max_doc_tokens": "128"]
        let cache = StateCache(storage: c.storage, index: idx)
        let cfg = RerankTrainConfig(batchSize: 2, epochs: 1)

        // Без ожидания — обучение идёт: кэш сам себе не противоречит.
        XCTAssertNoThrow(try RerankTraining.train(model, trainCache: cache,
                                                  evalCache: cache, config: cfg))
        // С ожиданием 384 — обрывается.
        XCTAssertThrowsError(try RerankTraining.train(
            model, trainCache: cache, evalCache: cache, config: cfg,
            contract: ["max_doc_tokens": "384"]),
            "чужая обрезка принята при явно заданном ожидании")
    }

    /// Контракт, на котором голова обучилась, отдаётся наружу.
    ///
    /// Ради этого он и нужен: чекпоинт сохраняется с ним, а не с тем, что
    /// вызывающий держит у себя. Совпадают они только когда кэш собран этим
    /// же запуском — то есть НЕ в самом частом случае.
    func testResultCarriesTheContractItTrainedWith() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]))
        let (c, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                        seed: 83)
        var idx = c.index
        idx.contract = ["template": "doc_first", "max_doc_tokens": "128",
                        "instruct": "инструкция из кэша"]
        let cache = StateCache(storage: c.storage, index: idx)

        let r = try RerankTraining.train(model, trainCache: cache,
                                         evalCache: cache,
                                         config: RerankTrainConfig(batchSize: 2,
                                                                   epochs: 1))
        XCTAssertEqual(r.contract["max_doc_tokens"], "128")
        XCTAssertEqual(r.contract["instruct"], "инструкция из кэша")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Оценка без обучения
    // ─────────────────────────────────────────────────────────────────

    /// Оценка без обучения даёт ровно то же, что оценка после обучения.
    ///
    /// Различающее утверждение обязательно: у НЕОБУЧЕННОЙ головы все скоры
    /// равны, MRR вырождается в пол случайного угадывания, и совпадение
    /// «до» с «до» ничего не значило бы. Поэтому сверяется обученная голова.
    func testEvaluateWithoutTrainingMatchesTrainedResult() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                            seed: 91)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]),
                                 seed: 5)
        let r = try RerankTraining.train(
            model, trainCache: cache, evalCache: cache,
            config: RerankTrainConfig(batchSize: 2, epochs: 2))

        let again = try RerankTraining.evaluate(model, cache: cache)
        XCTAssertEqual(again, r.after)

        // Тест не вырожден: обучение реально сдвинуло метрики.
        XCTAssertNotEqual(r.after, r.before)
    }

    /// Оценка сверяет контракт так же, как обучение.
    func testEvaluateChecksContract() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]))
        let (c, _) = try syntheticCache(nPairs: 8, sources: [3], nSrc: 1,
                                        seed: 92)
        var idx = c.index
        idx.contract = ["max_doc_tokens": "128"]
        let cache = StateCache(storage: c.storage, index: idx)

        XCTAssertNoThrow(try RerankTraining.evaluate(model, cache: cache))
        XCTAssertThrowsError(try RerankTraining.evaluate(
            model, cache: cache, contract: ["max_doc_tokens": "384"]),
            "чужая обрезка принята при оценке")
        // И чужой слой — тоже.
        let other = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [0]))
        XCTAssertThrowsError(try RerankTraining.evaluate(other, cache: cache))
    }

    /// Оценка по чекпоинту берёт контракт ИЗ ФАЙЛА.
    func testEvaluateFromCheckpointUsesFileContract() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (c, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                        seed: 93)
        var idx = c.index
        idx.contract = ["template": "doc_first", "max_doc_tokens": "128"]
        let cache = StateCache(storage: c.storage, index: idx)

        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [3]),
                                 seed: 5)
        _ = try RerankTraining.train(model, trainCache: cache, evalCache: cache,
                                     config: RerankTrainConfig(batchSize: 2,
                                                               epochs: 2))
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("head_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.saveHead(to: url, extra: ["template": "doc_first",
                                            "max_doc_tokens": "128"])

        let (m, contract) = try RerankTraining.evaluate(base: bb, head: url,
                                                        cache: cache)
        XCTAssertEqual(contract["max_doc_tokens"], "128")
        XCTAssertEqual(m, try RerankTraining.evaluate(model, cache: cache))

        // Чекпоинт с ЧУЖИМ контрактом на этом кэше обрывается.
        let bad = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("head_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: bad) }
        try model.saveHead(to: bad, extra: ["template": "doc_first",
                                            "max_doc_tokens": "384"])
        XCTAssertThrowsError(try RerankTraining.evaluate(base: bb, head: bad,
                                                          cache: cache))
    }

    // ─────────────────────────────────────────────────────────────────
    //  Добор слоёв в кэш
    // ─────────────────────────────────────────────────────────────────

    /// Слияние даёт ровно тот же кэш, что кодирование обоих слоёв сразу.
    ///
    /// Это и есть обещание: слой, которого не хватило, кодируется отдельно и
    /// приклеивается, а не гоняется вся база заново. Равенство побитовое —
    /// арифметики в слиянии нет, только перекладывание байтов.
    func testMergeEqualsEncodingBothLayersAtOnce() throws {
        let (a, _) = try syntheticCache(nPairs: 8, sources: [1], nSrc: 1,
                                        seed: 101)
        let (b, _) = try syntheticCache(nPairs: 8, sources: [3], nSrc: 1,
                                        seed: 101)
        let (both, _) = try syntheticCache(nPairs: 8, sources: [1, 3], nSrc: 2,
                                           seed: 101)

        let merged = try a.merged(with: b)
        XCTAssertEqual(merged.sources, [1, 3])
        XCTAssertEqual(merged.nPairs, 8)
        let rows = Array(0 ..< 8)
        XCTAssertEqual(maxAbsDiff(merged.gather(rows), both.gather(rows)), 0,
                       "слияние разошлось с кодированием обоих слоёв сразу")
        // Порядок аргументов не важен: слои сортируются, а не приписываются.
        XCTAssertEqual(maxAbsDiff(try b.merged(with: a).gather(rows),
                                  both.gather(rows)), 0)
    }

    /// Слияние кэшей от РАЗНЫХ баз обрывается.
    ///
    /// Единственная проверка, которая это ловит: модель в контракте не
    /// записана, и состояния двух моделей неотличимы ни по форме, ни по
    /// контракту. Ловится численным расхождением ОБЩИХ слоёв.
    func testMergeRejectsCachesFromDifferentBases() throws {
        let (a, _) = try syntheticCache(nPairs: 8, sources: [1, 3], nSrc: 2,
                                        seed: 111)
        let (b, _) = try syntheticCache(nPairs: 8, sources: [3, 5], nSrc: 2,
                                        seed: 222)   // другой шум = другая база
        XCTAssertThrowsError(try a.merged(with: b),
                             "кэши с расходящимся общим слоем слиты")

        // Без пересечения проверить нечем — и слияние проходит. Утверждение
        // о ГРАНИЦЕ проверки, а не о её отсутствии.
        let (c, _) = try syntheticCache(nPairs: 8, sources: [0], nSrc: 1,
                                        seed: 222)
        XCTAssertNoThrow(try a.merged(with: c))
    }

    /// Слияние отвергает кэши по разным парам и с разным контрактом.
    func testMergeRejectsMismatchedPairsAndContract() throws {
        let (a, _) = try syntheticCache(nPairs: 8, sources: [1], nSrc: 1,
                                        seed: 121)
        let (short, _) = try syntheticCache(nPairs: 6, sources: [3], nSrc: 1,
                                            seed: 121)
        XCTAssertThrowsError(try a.merged(with: short), "разное число пар слито")

        let (b, _) = try syntheticCache(nPairs: 8, sources: [3], nSrc: 1,
                                        seed: 121)
        var idx = b.index
        idx.contract = ["template": "query_first"]
        XCTAssertThrowsError(
            try a.merged(with: StateCache(storage: b.storage, index: idx)),
            "кэши с разным шаблоном слиты")

        // Расходится ТОЛЬКО число строк: pairIndex, labels и hardNegs
        // совпадают. Без этого случая проверка на nPairs замаскирована
        // соседними — в первой редакции теста мутация «не проверять число
        // пар» проходила именно так.
        let (big, _) = try syntheticCache(nPairs: 10, sources: [3], nSrc: 1,
                                          seed: 121)
        var wide = big.index
        wide.pairIndex = a.index.pairIndex
        wide.labels = a.index.labels
        wide.hardNegs = a.index.hardNegs
        wide.contract = a.index.contract
        let sameButLonger = StateCache(storage: big.storage, index: wide)
        XCTAssertEqual(sameButLonger.index.pairIndex, a.index.pairIndex,
                       "тест вырожден: расходится не только число строк")
        XCTAssertNotEqual(sameButLonger.nPairs, a.nPairs)
        XCTAssertThrowsError(try a.merged(with: sameButLonger),
                             "кэши с разным числом строк слиты")
    }

    /// Голова, читавшая один слой, читает слитый кэш и получает то же самое.
    func testHeadReadsMergedCacheIdentically() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (a, _) = try syntheticCache(nPairs: 12, sources: [1], nSrc: 1,
                                        seed: 131)
        let (b, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                        seed: 131)
        let merged = try a.merged(with: b)

        for layer in [1, 3] {
            let model = try Reranker(base: bb,
                                     cfg: RerankerConfig(layerIdx: [layer]),
                                     seed: 7)
            let exact = layer == 1 ? a : b
            XCTAssertEqual(try RerankTraining.evaluate(model, cache: merged),
                           try RerankTraining.evaluate(model, cache: exact),
                           "слой \(layer): слитый кэш прочитан иначе")
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Разброс
    // ─────────────────────────────────────────────────────────────────

    /// Разброс по одному прогону — НЕ ноль, а «не измерен».
    ///
    /// Ноль здесь был бы худшим из возможных значений: он читается как
    /// «разброса нет», то есть превращает одиночный шумный прогон в
    /// установленный факт. Ровно от этого весь механизм и заведён.
    func testSpreadOfSingleRunIsNotZero() {
        let one = Spread([0.9769])
        XCTAssertEqual(one.mean, 0.9769, accuracy: 1e-12)
        XCTAssertTrue(one.std.isNaN, "разброс по одному прогону объявлен нулём")
        XCTAssertTrue(one.description.contains("не измерен"))
    }

    /// Выборочное отклонение считается с делителем n−1.
    func testSpreadUsesSampleStandardDeviation() {
        let s = Spread([2, 4, 4, 4, 5, 5, 7, 9])
        XCTAssertEqual(s.mean, 5.0, accuracy: 1e-12)
        // Популяционное дало бы ровно 2.0 — именно это и различается.
        XCTAssertEqual(s.std, 2.13808993529939, accuracy: 1e-9)
        XCTAssertEqual(s.min, 2)
        XCTAssertEqual(s.max, 9)
        XCTAssertEqual(s.n, 8)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Развёртка
    // ─────────────────────────────────────────────────────────────────

    /// Каждый сид получает СВЕЖУЮ голову.
    ///
    /// Детектор ровно тот же, что и во всём остальном реранкере: у zero-init
    /// головы стартовый лосс обязан быть РОВНО ln(C). Если развёртка
    /// переиспользует одну модель, второй прогон стартует с обученных весов,
    /// и его первый лосс уже не ln(C). Проверка стоит одно сравнение и ловит
    /// самую вероятную ошибку в этом коде.
    func testEachSeedStartsFromAFreshHead() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                            seed: 31)
        let point = try RerankSweep.seeds(
            base: bb, cfg: RerankerConfig(layerIdx: [-1]),
            trainCache: cache, evalCache: cache,
            config: RerankTrainConfig(batchSize: 2, epochs: 2),
            seeds: [0, 1, 2])

        XCTAssertEqual(point.seeds.count, 3)
        XCTAssertEqual(point.after.count, 3)
        let want = point.expectedFirstLoss
        XCTAssertEqual(want, log(Float(cache.nCandidates)), accuracy: 1e-6)
        for (i, l) in point.firstLosses.enumerated() {
            XCTAssertEqual(l, want, accuracy: 1e-5,
                           "сид \(i): стартовый лосс не ln(C) — голова не свежая")
        }
    }

    /// Сид действительно доходит до обучения, и развёртка воспроизводима.
    ///
    /// Сверяются ЛОССЫ, а не метрики, и это не мелочь. Метрики на маленьком
    /// наборе упираются в потолок: голова запоминает шесть примеров за две
    /// эпохи, MRR = 1.0 при любом сиде, и тест, построенный на них, был бы
    /// зелёным даже при полностью оборванной проводке сида. Лосс —
    /// непрерывная величина и в потолок не упирается.
    ///
    /// Утверждения парные и оба нужны. Совпадение с ручным прогоном ловит
    /// «сид не передали»; РАЗЛИЧИЕ ручных прогонов между собой сторожит сам
    /// тест от вырождения. Первая редакция этого теста как раз и упала на
    /// втором утверждении — вырожденной оказалась синтетика, а не код.
    func testSweepPassesSeedThroughToTraining() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 12, sources: [3], nSrc: 1,
                                            seed: 41)
        let cfg = RerankerConfig(layerIdx: [-1])
        let tcfg = RerankTrainConfig(batchSize: 2, epochs: 2)

        func manual(_ seed: UInt64) throws -> [Float] {
            let m = try Reranker(base: bb, cfg: cfg, seed: seed)
            var c = tcfg
            c.seed = seed
            let r = try RerankTraining.train(m, trainCache: cache,
                                             evalCache: cache, config: c)
            return r.epochs.map { $0.loss }
        }
        let a = try manual(0), b = try manual(1)
        XCTAssertNotEqual(a, b, "два сида дали одни и те же лоссы — тест "
                          + "вырожден, проводку сида он не проверяет")

        func sweepLosses(_ seeds: [UInt64]) throws -> [[Float]] {
            var got: [[Float]] = []
            _ = try RerankSweep.seeds(base: bb, cfg: cfg, trainCache: cache,
                                      evalCache: cache, config: tcfg,
                                      seeds: seeds,
                                      onRun: { _, _, r in
                                          got.append(r.epochs.map { $0.loss })
                                      })
            return got
        }
        XCTAssertEqual(try sweepLosses([0, 1]), [a, b])
        // Воспроизводимость: та же развёртка — те же числа.
        XCTAssertEqual(try sweepLosses([0, 1]), [a, b])
    }

    /// Несколько конфигураций головы обучаются на ОДНОМ кэше надмножества.
    func testSeveralConfigsShareOneSupersetCache() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 12, sources: [0, 1, 2, 3],
                                            nSrc: 4, seed: 51)
        let configs = [RerankerConfig(layerIdx: [1]),
                       RerankerConfig(layerIdx: [3]),
                       RerankerConfig(layerIdx: [1, 3])]
        let res = try RerankSweep.run(base: bb, configs: configs,
                                      trainCache: cache, evalCache: cache,
                                      config: RerankTrainConfig(batchSize: 2,
                                                                epochs: 1),
                                      seeds: [0, 1])
        XCTAssertEqual(res.points.count, 3)
        XCTAssertEqual(res.points.map { $0.layers }, [[1], [3], [1, 3]])
        for p in res.points {
            XCTAssertEqual(p.mrr.n, 2)
            XCTAssertFalse(p.mrr.std.isNaN, "разброс по двум сидам не посчитан")
        }
        XCTAssertTrue(res.summary.contains("разрывы по MRR"))
    }

    /// Конфигурация, требующая слоя вне кэша, обрывается — не обучается на
    /// чужом.
    func testConfigOutsideCacheAborts() throws {
        let (bb, _) = makeBackbone(nLayer: 4)
        let (cache, _) = try syntheticCache(nPairs: 8, sources: [0, 1], nSrc: 2,
                                            seed: 61)
        XCTAssertThrowsError(try RerankSweep.run(
            base: bb, configs: [RerankerConfig(layerIdx: [3])],
            trainCache: cache, evalCache: cache,
            config: RerankTrainConfig(batchSize: 2, epochs: 1), seeds: [0]))
    }
}
