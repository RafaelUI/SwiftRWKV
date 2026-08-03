import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Дообучение эмбеддингов: связка «база + голова», задачные objective,
//  GradCache внутри общего тренера, метрики.
//
//  Порядок надёжности здесь тот же, что в остальном наборе: структурные
//  тождества → побитовые равенства → паритет с Python (в
//  EmbeddingParityTests). Структурные тесты проверяют то, что обязано
//  выполняться при любых весах; побитовые — что переписанный путь не изменил
//  арифметику; паритет — что арифметика вообще та, что задумана.
// ───────────────────────────────────────────────────────────────────────

final class EmbeddingFinetuneTests: XCTestCase {

    // ── Оснастка ─────────────────────────────────────────────────────

    /// - vocab: 64 хватает для тестов на синтетических id. Для тестов на
    ///   НАСТОЯЩЕМ тексте нужен словарь World целиком: иначе выборка из
    ///   таблицы эмбеддингов уходит за границу и лосс становится NaN — что
    ///   и случилось при первом прогоне.
    func makeModel(seed: UInt64 = 3, nLayer: Int = 2, vocab: Int = 64) -> EmbeddingModel {
        let cfg = TinyBackbone.config(nLayer: nLayer, vocab: vocab)
        let bb = X070Backbone(weights: TinyBackbone.weights(cfg, seed: seed), cfg: cfg)
        let head = EmbeddingHead(dim: cfg.nEmbd, seed: seed)
        // fc2 штатно нулевой (голова = тождество). Для тестов обучения это
        // плохая стартовая точка: нулевой fc2 даёт нулевой градиент по fc1
        // (df/dfc1 ∝ fc2), и «fc1 не изменился» означало бы не дефект, а
        // арифметику. Разбавляем, чтобы обе матрицы были живыми.
        MLXRandom.seed(seed &+ 100)
        head.fc2 = MLXRandom.normal([cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        return EmbeddingModel(backbone: bb, head: head)
    }

    /// Батч триплетов из случайных id. Длина кратна WKV7_CHUNK, т.к. обучаемые
    /// слои идут через дифференцируемое ядро.
    func makeTripletBatch(B: Int = 4, T: Int = 16, vocab: Int = 64,
                          seed: UInt64 = 5) -> TripletBatch {
        MLXRandom.seed(seed)
        func ids() -> MLXArray {
            MLXRandom.randInt(0 ..< vocab, [B, T]).asType(.int32)
        }
        let pool = MLXArray(Array(repeating: Int32(T - 1), count: B))
        let b = TripletBatch(anchorIdx: ids(), anchorPool: pool,
                             positiveIdx: ids(), positivePool: pool,
                             negativeIdx: ids(), negativePool: pool)
        eval(b.anchorIdx, b.positiveIdx, b.negativeIdx, pool)
        return b
    }

    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // ── Композиция обучаемых множеств ────────────────────────────────

    /// Заглушка: множество из N независимых тензоров, помнящее последнюю
    /// подстановку. Нужна, чтобы проверять саму композицию, не поднимая модель.
    final class FakeSet: TrainableSet {
        let names: [String]
        var initial: [MLXArray]
        var injected: [MLXArray] = []
        var committed: [MLXArray] = []
        init(prefix: String, count: Int) {
            names = (0 ..< count).map { "\(prefix).p\($0)" }
            initial = (0 ..< count).map { MLXArray(Float($0)) }
        }
        var parameterNames: [String] { names }
        func initialParameters() -> [MLXArray] { initial }
        func inject(_ ps: [MLXArray]) { injected = ps }
        func commit(_ ps: [MLXArray]) { committed = ps; injected = ps }
    }

    func testCompositeConcatenatesInDeclaredOrder() {
        let a = FakeSet(prefix: "base", count: 3)
        let b = FakeSet(prefix: "head", count: 2)
        let c = CompositeTrainableSet([a, b])

        XCTAssertEqual(c.parameterNames,
                       ["base.p0", "base.p1", "base.p2", "head.p0", "head.p1"],
                       "порядок частей обязан быть порядком объявления, не сортировкой")
        XCTAssertEqual(c.initialParameters().count, 5)
    }

    func testCompositeDistributesSlicesBackToParts() {
        let a = FakeSet(prefix: "base", count: 3)
        let b = FakeSet(prefix: "head", count: 2)
        let c = CompositeTrainableSet([a, b])
        _ = c.initialParameters()

        let ps = (0 ..< 5).map { MLXArray(Float(100 + $0)) }
        c.inject(ps)

        XCTAssertEqual(a.injected.count, 3)
        XCTAssertEqual(b.injected.count, 2)
        XCTAssertEqual(a.injected.map { $0.item(Float.self) }, [100, 101, 102])
        XCTAssertEqual(b.injected.map { $0.item(Float.self) }, [103, 104],
                       "второй части обязан достаться ХВОСТ, а не начало")
    }

    /// Границы срезов должны считаться и без предварительного вызова
    /// initialParameters — тренер после loadCheckpoint зовёт сразу inject.
    func testCompositeInjectsWithoutInitialParametersFirst() {
        let a = FakeSet(prefix: "base", count: 3)
        let b = FakeSet(prefix: "head", count: 2)
        let c = CompositeTrainableSet([a, b])

        c.inject((0 ..< 5).map { MLXArray(Float($0)) })
        XCTAssertEqual(a.injected.count, 3)
        XCTAssertEqual(b.injected.count, 2)
    }

    func testCompositeCommitReachesEveryPart() {
        let a = FakeSet(prefix: "base", count: 2)
        let b = FakeSet(prefix: "head", count: 2)
        let c = CompositeTrainableSet([a, b])
        _ = c.initialParameters()
        c.commit((0 ..< 4).map { MLXArray(Float($0)) })
        XCTAssertEqual(a.committed.count, 2)
        XCTAssertEqual(b.committed.count, 2, "commit не должен теряться по дороге")
    }

    // ── Голова обязана оставаться обучаемой ──────────────────────────

    func testFrozenBaseStillTrainsHead() {
        let model = makeModel()
        let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)

        XCTAssertEqual(trainable.parameterNames, EmbeddingHead.parameterNames,
                       "при замороженной базе обучается ровно голова")
        XCTAssertTrue(model.backbone.trainLayers.isEmpty,
                      "заморожённой базе дифференцируемое ядро не нужно")

        let fc1Before = model.head.fc1
        let wBefore = model.backbone.w["blocks.0.tmix.k_proj.weight"]!
        eval(fc1Before, wBefore)
        let fc1Copy = fc1Before.asType(.float32) + 0   // отвязать от объекта

        var batch = makeTripletBatch()
        let trainer = Trainer<TripletBatch>(
            trainable: trainable,
            objective: { EmbeddingObjective.retrievalLoss(model, $0) },
            nextBatch: { batch },
            config: TrainingConfig(lr: 1e-2, maxSteps: 2, logEvery: 0))
        _ = trainer.run()

        XCTAssertGreaterThan(maxAbs(model.head.fc1, fc1Copy), 0,
                             "голова не сдвинулась — обучать было нечего")
        XCTAssertEqual(maxAbs(model.backbone.w["blocks.0.tmix.k_proj.weight"]!, wBefore), 0,
                       "заморожённая база обязана остаться нетронутой")
    }

    /// Обратная сторона: при обучении базы голова НЕ должна тихо выпадать из
    /// множества. Именно этот дефект и мотивирует композицию.
    func testTopLayersModeTrainsBothBaseAndHead() {
        let model = makeModel()
        let trainable = EmbeddingTrainable.make(model: model, mode: .topLayers(1))

        XCTAssertTrue(trainable.parameterNames.contains("head.fc1"),
                      "голова обязана быть в множестве при любой судьбе базы")
        XCTAssertEqual(model.backbone.trainLayers, [1],
                       "верхний слой обязан идти через дифференцируемое ядро")

        let fc1 = model.head.fc1.asType(.float32) + 0
        let w = model.backbone.w["blocks.1.tmix.k_proj.weight"]!.asType(.float32) + 0
        let frozen = model.backbone.w["blocks.0.tmix.k_proj.weight"]!.asType(.float32) + 0
        eval(fc1, w, frozen)

        var batch = makeTripletBatch()
        let trainer = Trainer<TripletBatch>(
            trainable: trainable,
            objective: { EmbeddingObjective.retrievalLoss(model, $0) },
            nextBatch: { batch },
            config: TrainingConfig(lr: 1e-2, maxSteps: 2, logEvery: 0))
        _ = trainer.run()

        XCTAssertGreaterThan(maxAbs(model.head.fc1, fc1), 0, "голова не обучилась")
        XCTAssertGreaterThan(maxAbs(model.backbone.w["blocks.1.tmix.k_proj.weight"]!, w), 0,
                             "верхний слой базы не обучился")
        XCTAssertEqual(maxAbs(model.backbone.w["blocks.0.tmix.k_proj.weight"]!, frozen), 0,
                       "нижний слой не входит в topLayers(1) и меняться не должен")
    }

    /// Режим .full не должен тянуть в обучение LM-голову и таблицу
    /// эмбеддингов: логиты в этой задаче не считаются вовсе.
    func testFullModeExcludesEmbeddingTableAndLMHead() {
        let model = makeModel()
        let trainable = EmbeddingTrainable.make(model: model, mode: .full)
        let names = Set(trainable.parameterNames)

        XCTAssertFalse(names.contains("emb.weight"))
        XCTAssertFalse(names.contains("head.weight"),
                       "LM-голова в эмбеддинг-задаче не участвует")
        XCTAssertTrue(names.contains("ln_out.weight"))
        XCTAssertTrue(names.contains("head.fc2"), "голова эмбеддинга — участвует")
        XCTAssertEqual(model.backbone.trainLayers, Set(0 ..< model.backbone.cfg.nLayer))
    }

    // ── GradCache внутри общего тренера ──────────────────────────────

    /// Решающее тождество для интеграции: GradCache с ОДНИМ чанком обязан
    /// дать те же параметры, что обычный путь. Один чанк — это отсутствие
    /// разрезания, то есть остаётся только сама интеграция (провайдер
    /// градиента, подстановка внутри фаз, форма возврата). Расхождение здесь
    /// означало бы дефект проводки, а не численности.
    func testGradCacheProviderOneChunkMatchesEagerTrainer() {
        let batch = makeTripletBatch(B: 4)

        func runOnce(chunk: Int) -> [MLXArray] {
            let model = makeModel()
            let trainable = EmbeddingTrainable.make(model: model, mode: .topLayers(1))
            let provider: GradientProvider<TripletBatch>? = chunk > 0
                ? EmbeddingObjective.gradCacheProvider(
                    model: model, trainable: trainable, chunkSize: chunk,
                    temperature: 0.05, symmetric: false)
                : nil
            let trainer = Trainer<TripletBatch>(
                trainable: trainable,
                objective: { EmbeddingObjective.retrievalLoss(model, $0) },
                nextBatch: { batch },
                config: TrainingConfig(lr: 1e-3, maxSteps: 2, logEvery: 0),
                gradient: provider)
            _ = trainer.run()
            let ps = trainer.currentParameters
            eval(ps)
            return ps
        }

        let eager = runOnce(chunk: 0)
        let cached = runOnce(chunk: 4)      // один чанк на весь батч

        XCTAssertEqual(eager.count, cached.count)
        var worst: Float = 0
        for (a, b) in zip(eager, cached) {
            worst = Swift.max(worst, maxAbs(a, b) / (MLX.abs(a).max().item(Float.self) + 1e-9))
        }
        XCTAssertLessThan(worst, 1e-5,
                          "GradCache при одном чанке разошёлся с обычным путём на \(worst) "
                          + "— это дефект проводки, а не порядка суммирования")
    }

    /// Несколько чанков: лосс виден целиком, поэтому обязан совпасть; разница
    /// в параметрах остаётся на уровне порядка суммирования.
    func testGradCacheMultiChunkKeepsLoss() {
        let batch = makeTripletBatch(B: 4)
        let model = makeModel()
        let trainable = EmbeddingTrainable.make(model: model, mode: .topLayers(1))
        let params = trainable.initialParameters()

        let eagerLoss = EmbeddingObjective.retrievalLoss(model, batch)
        eval(eagerLoss)

        let provider = EmbeddingObjective.gradCacheProvider(
            model: model, trainable: trainable, chunkSize: 2,
            temperature: 0.05, symmetric: false)
        let (loss, grads) = provider(params, batch)
        eval([loss] + grads)

        XCTAssertEqual(loss.item(Float.self), eagerLoss.item(Float.self), accuracy: 1e-4,
                       "лосс считается на ПОЛНОМ батче векторов и обязан совпасть")
        XCTAssertEqual(grads.count, params.count)
    }

    /// Провайдер задан ⇒ objective не должен вызываться вовсе. Иначе лосс
    /// считался бы дважды, и вдобавок на eager-пути — то есть с той самой
    /// памятью, ради ухода от которой GradCache и введён.
    func testTrainerIgnoresObjectiveWhenProviderGiven() {
        let batch = makeTripletBatch(B: 2)
        let model = makeModel()
        let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)
        var objectiveCalls = 0

        let trainer = Trainer<TripletBatch>(
            trainable: trainable,
            objective: { b in
                objectiveCalls += 1
                return EmbeddingObjective.retrievalLoss(model, b)
            },
            nextBatch: { batch },
            config: TrainingConfig(lr: 1e-3, maxSteps: 2, logEvery: 0),
            gradient: EmbeddingObjective.gradCacheProvider(
                model: model, trainable: trainable, chunkSize: 2,
                temperature: 0.05, symmetric: false))
        _ = trainer.run()

        XCTAssertEqual(objectiveCalls, 0,
                       "при заданном провайдере objective вызван \(objectiveCalls) раз(а)")
    }

    // ── Выравнивание длины ───────────────────────────────────────────

    /// Причинность — БИТ-В-БИТ. При одинаковой длине содержимое правее
    /// poolIndex не может влиять на вектор вообще никак: позиция t зависит
    /// только от позиций ≤ t. Это равенство, а не приближение, и проверяется
    /// как равенство.
    ///
    /// Именно это утверждение и оправдывает добивку батча: если бы оно не
    /// выполнялось, любая правая добивка портила бы вектора коротких строк.
    func testContentAfterPoolIndexIsExactlyIrrelevant() {
        let model = makeModel()
        let vocab = model.backbone.cfg.vocab
        MLXRandom.seed(21)

        let T = 32, B = 3
        let ids = MLXRandom.randInt(0 ..< vocab, [B, T]).asType(.int32)
        let pool = MLXArray([Int32(20), Int32(16), Int32(5)])
        eval(ids, pool)

        // Тот же префикс, другой хвост: всё правее poolIndex переписано.
        var other = ids.asArray(Int32.self)
        MLXRandom.seed(22)
        let noise = MLXRandom.randInt(0 ..< vocab, [B, T]).asType(.int32).asArray(Int32.self)
        let poolVals = pool.asArray(Int32.self)
        for b in 0 ..< B {
            for t in (Int(poolVals[b]) + 1) ..< T { other[b * T + t] = noise[b * T + t] }
        }
        let mutated = MLXArray(other, [B, T])
        eval(mutated)

        let a = model.embed(ids, poolIndex: pool)
        let c = model.embed(mutated, poolIndex: pool)
        eval(a, c)

        XCTAssertEqual(maxAbs(a, c), 0,
                       "содержимое правее poolIndex повлияло на вектор — "
                       + "значит либо пулинг читает не poolIndex, либо путь не причинен")
    }

    /// Изменение самой ДЛИНЫ T — уже не побитовое тождество, и это стоит
    /// зафиксировать явно, а не выдавать желаемое за проверенное.
    ///
    /// Причинность при этом не нарушена: расхождение возникает не от добитых
    /// токенов, а от того, что матмулы и редукции при другой форме входа
    /// раскладываются на GPU иначе и складывают частичные суммы в другом
    /// порядке. Порядок величины (~1e-7 при вычислениях в bf16, то есть на
    /// пять порядков ниже шума самого bf16) показывает, что это округление,
    /// а не другая арифметика. Практический вывод: выравнивать длину батча
    /// безопасно, но воспроизводимость прогонов требует ОДИНАКОВОГО
    /// выравнивания, а не просто «достаточного».
    func testPadMultipleChangesResultOnlyByRounding() {
        let model = makeModel()
        let vocab = model.backbone.cfg.vocab
        MLXRandom.seed(21)

        let T = 20, B = 3
        let ids = MLXRandom.randInt(0 ..< vocab, [B, T]).asType(.int32)
        let pool = MLXArray([Int32(T - 1), Int32(T - 4), Int32(5)])
        eval(ids, pool)

        let padded = concatenated(
            [ids, MLXArray.zeros([B, 32 - T], dtype: .int32)], axis: 1)
        eval(padded)

        let a = model.embed(ids, poolIndex: pool)
        let b = model.embed(padded, poolIndex: pool)
        eval(a, b)

        let d = maxAbs(a, b)
        XCTAssertLessThan(d, 1e-5,
                          "выравнивание длины сдвинуло вектор на \(d) — это больше, "
                          + "чем объясняется порядком суммирования")
    }

    /// Та же нейтральность на уровне encodeBatch: выравнивание меняет ТОЛЬКО
    /// форму, poolIndex обязан остаться прежним.
    func testEncodeBatchPadMultipleChangesShapeNotPoolIndex() throws {
        let tok = try loadWorldTokenizerOrSkip()
        let texts = ["привет", "a much longer piece of text to shift the maximum"]

        let plain = encodeBatch(tokenizer: tok, texts: texts, terminator: 0,
                                maxTokens: nil, padMultiple: nil)
        let aligned = encodeBatch(tokenizer: tok, texts: texts, terminator: 0,
                                  maxTokens: nil, padMultiple: WKV7_CHUNK)
        eval(plain.idx, plain.poolIndex, aligned.idx, aligned.poolIndex)

        XCTAssertEqual(aligned.idx.shape[1] % WKV7_CHUNK, 0)
        XCTAssertGreaterThanOrEqual(aligned.idx.shape[1], plain.idx.shape[1])
        XCTAssertEqual(maxAbs(plain.poolIndex, aligned.poolIndex), 0,
                       "выравнивание сдвинуло позицию пулинга")
        // Сами токены до общей длины обязаны совпасть.
        let common = plain.idx.shape[1]
        XCTAssertEqual(maxAbs(plain.idx, aligned.idx[0..., 0 ..< common]), 0)
    }

    // ── Метрики ──────────────────────────────────────────────────────

    /// Метрики на матрице, посчитанной руками. Ранги: 1, 2, 3 ⇒
    /// MRR = (1 + 1/2 + 1/3)/3, recall@1 = 1/3, nDCG@10 = (1 + 1/log2 3 + 1/2)/3.
    func testRankingMetricsAgainstHandComputation() {
        let sims = MLXArray([
            0.9, 0.1, 0.2,     // верный 0 — ранг 1
            0.5, 0.4, 0.9,     // верный 1 — лучше только 0.9 и 0.5 ⇒ ранг 3
            0.3, 0.8, 0.7,     // верный 2 — лучше только 0.8 ⇒ ранг 2
        ].map { Float($0) }, [3, 3])
        eval(sims)

        let m = EmbeddingMetrics.rankingMetrics(similarities: sims,
                                                correctIndex: [0, 1, 2], ks: [1, 2, 3])
        XCTAssertEqual(m.mrr, (1.0 + 1.0 / 3.0 + 1.0 / 2.0) / 3.0, accuracy: 1e-9)
        XCTAssertEqual(m.recall[1]!, 1.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(m.recall[2]!, 2.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(m.recall[3]!, 1.0, accuracy: 1e-9)
        XCTAssertEqual(m.ndcg10,
                       (1.0 + 1.0 / log2(4.0) + 1.0 / log2(3.0)) / 3.0, accuracy: 1e-9)
        XCTAssertEqual(m.count, 3)
    }

    /// Ничья не должна считаться проигрышем: ранг определяется числом СТРОГО
    /// лучших. Иначе вырожденная модель, выдающая всем одинаковый балл,
    /// получала бы худшую оценку, чем модель, ставящая верный ответ последним.
    func testRankingMetricsTreatTiesAsRankOne() {
        let sims = MLXArray([Float](repeating: 0.5, count: 6), [2, 3])
        eval(sims)
        let m = EmbeddingMetrics.rankingMetrics(similarities: sims,
                                                correctIndex: [0, 2], ks: [1])
        XCTAssertEqual(m.mrr, 1.0, accuracy: 1e-9)
        XCTAssertEqual(m.recall[1]!, 1.0, accuracy: 1e-9)
    }

    /// Метрика ранжирования обязана реагировать на порядок: если верный ответ
    /// худший в строке, MRR = 1/C, а не что-то около единицы.
    func testRankingMetricsPunishWorstAnswer() {
        let sims = MLXArray([Float(0.0), 0.5, 0.9, 0.0, 0.5, 0.9], [2, 3])
        eval(sims)
        let m = EmbeddingMetrics.rankingMetrics(similarities: sims,
                                                correctIndex: [0, 0], ks: [1])
        XCTAssertEqual(m.mrr, 1.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(m.recall[1]!, 0.0, accuracy: 1e-9)
    }

    // ── Данные классификации ─────────────────────────────────────────

    /// Полный пул из 25 меток и семёрка из инструкции — разные задачи, и
    /// оценка обязана сообщать, какая из них считалась: 0.30 при семи
    /// кандидатах хуже случайного, а при 25 — вдвое лучше.
    func testFullPoolIsReportedAsHarderTask() throws {
        let tok = try loadWorldTokenizerOrSkip()
        let rows = try slice().filter { $0.task == .classification }
        try XCTSkipIf(rows.isEmpty, "в срезе нет строк классификации")
        let model = makeModel(vocab: 65536)

        let full = EmbeddingMetrics.evaluateClassification(
            model: model, tokenizer: tok, rows: Array(rows.prefix(4)),
            useFullPool: true, maxTokens: 64)
        let seven = EmbeddingMetrics.evaluateClassification(
            model: model, tokenizer: tok, rows: Array(rows.prefix(4)),
            useFullPool: false, maxTokens: 64)

        XCTAssertEqual(full.candidatesPerRow, 25.0, accuracy: 1e-9)
        XCTAssertEqual(seven.candidatesPerRow, 7.0, accuracy: 1e-9,
                       "в LitRetrieval у каждой строки ровно 7 кандидатов")
    }

    /// Выбор обязан лежать ВНУТРИ пула предъявленной строки.
    ///
    /// Добавлено по итогам мутационной проверки: мутация «искать максимум по
    /// всем известным меткам, а не по семи из инструкции» не ловилась ничем.
    /// Точность её не выдаёт — на этих данных она получается той же, — и
    /// candidatesPerRow тоже, потому что считается отдельно от самого выбора.
    /// Единственный признак — сам индекс: при поиске по 25 меткам он выходит
    /// за границу семёрки.
    func testClassificationChoosesWithinRowPool() throws {
        let tok = try loadWorldTokenizerOrSkip()
        let rows = try slice().filter { $0.task == .classification }
        try XCTSkipIf(rows.count < 8, "мало строк классификации")
        let model = makeModel(vocab: 65536)

        let m = EmbeddingMetrics.evaluateClassification(
            model: model, tokenizer: tok, rows: Array(rows.prefix(8)),
            useFullPool: false, maxTokens: 64)

        XCTAssertEqual(m.predictions.count, m.count)
        XCTAssertEqual(m.poolSizes.count, m.count)
        for (i, p) in m.predictions.enumerated() {
            XCTAssertLessThan(p, m.poolSizes[i],
                              "строка \(i): выбран индекс \(p) при пуле из "
                              + "\(m.poolSizes[i]) — максимум искался за пределами "
                              + "предъявленного набора")
            XCTAssertGreaterThanOrEqual(p, 0)
        }
        // И набор действительно узкий: если бы поиск шёл по всем 25, хотя бы
        // один индекс почти наверняка вышел бы за семёрку.
        XCTAssertTrue(m.poolSizes.allSatisfy { $0 == 7 })
    }

    /// Пул из данных обязан совпадать с константой в коде. Если корпус
    /// заменят, тест скажет об этом раньше, чем метрики поедут молча.
    func testLabelPoolMatchesData() throws {
        let rows = try slice().filter { $0.task == .classification }
        try XCTSkipIf(rows.isEmpty, "в срезе нет строк классификации")
        var seen = Set<String>()
        for r in rows {
            seen.insert(r.positive.trimmingCharacters(in: .whitespacesAndNewlines))
            for c in ClassificationLabels.parseCandidates(from: r.anchor) ?? [] {
                seen.insert(c)
            }
        }
        let pool = Set(ClassificationLabels.pool)
        XCTAssertTrue(seen.isSubset(of: pool),
                      "в данных есть метки вне пула: \(seen.subtracting(pool).sorted())")
    }

    // ── Стадия целиком ───────────────────────────────────────────────

    func testStageRejectsRowsOfAnotherTask() throws {
        let model = makeModel(vocab: 65536)
        let tok = try loadWorldTokenizerOrSkip()
        let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)
        let stage = EmbeddingStage(
            task: .retrieval,
            train: [EmbeddingSample(anchor: "a", positive: "b", negative: "c",
                                    task: .sts)],
            config: TrainingConfig(maxSteps: 1, logEvery: 0))

        XCTAssertThrowsError(try EmbeddingFinetune.runStage(
            model: model, tokenizer: tok, trainable: trainable, stage: stage),
            "стадия retrieval, набитая строками sts, обязана падать: "
            + "лоссы у них разные, и молча обучиться не тому — худший исход")
    }

    /// Токенизатор от другой модели обязан быть пойман, а не превращён в NaN.
    /// Это не гипотетический сценарий: словарь World содержит 65 536 id, и
    /// любая модель со своим BPE меньшего размера даст молчаливый выход за
    /// границу таблицы эмбеддингов.
    func testStageRejectsTokenizerFromAnotherModel() throws {
        let tok = try loadWorldTokenizerOrSkip()
        let rows = try slice().filter { $0.task == .retrieval }
        try XCTSkipIf(rows.count < 4, "мало строк в срезе")
        let model = makeModel(vocab: 64)          // словарь заведомо мал
        let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)
        let stage = EmbeddingStage(task: .retrieval, train: Array(rows.prefix(4)),
                                   config: TrainingConfig(maxSteps: 1, logEvery: 0),
                                   batchSize: 2, maxTokens: 32, maxChars: 200)

        XCTAssertThrowsError(try EmbeddingFinetune.runStage(
            model: model, tokenizer: tok, trainable: trainable, stage: stage)) { error in
            guard case EmbeddingFinetuneError.vocabOverflow = error else {
                return XCTFail("ожидалось vocabOverflow, получено \(error)")
            }
        }
    }

    func testStageRunsAndReportsBeforeAfter() throws {
        let tok = try loadWorldTokenizerOrSkip()
        let rows = try slice().filter { $0.task == .retrieval }
        try XCTSkipIf(rows.count < 8, "мало строк retrieval в срезе")
        let model = makeModel(vocab: 65536)
        let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)

        let stage = EmbeddingStage(
            task: .retrieval,
            train: Array(rows.prefix(4)),
            heldOut: Array(rows.suffix(4)),
            config: TrainingConfig(lr: 1e-3, maxSteps: 2, logEvery: 0),
            batchSize: 2, maxTokens: 48, maxChars: 200)

        let r = try EmbeddingFinetune.runStage(model: model, tokenizer: tok,
                                               trainable: trainable, stage: stage)
        XCTAssertEqual(r.steps, 2)
        XCTAssertNotNil(r.before)
        XCTAssertNotNil(r.after)
        XCTAssertFalse(r.finalLoss.isNaN, "лосс не посчитался")
        if case .ranking(let m) = r.after! {
            XCTAssertEqual(m.count, 4)
        } else {
            XCTFail("для retrieval ожидались метрики ранжирования")
        }
    }

    /// Curriculum над одной моделью: обучаемое множество общее, значит вторая
    /// стадия обязана стартовать с параметров, оставленных первой, а не с
    /// исходных весов модели.
    func testCurriculumCarriesParametersBetweenStages() throws {
        let tok = try loadWorldTokenizerOrSkip()
        let all = try slice()
        let ret = all.filter { $0.task == .retrieval }
        let sts = all.filter { $0.task == .sts }
        try XCTSkipIf(ret.count < 4 || sts.count < 4, "мало строк в срезе")

        let model = makeModel(vocab: 65536)
        let start = model.head.fc1.asType(.float32) + 0
        eval(start)

        func stage(_ t: EmbeddingTask, _ rows: [EmbeddingSample]) -> EmbeddingStage {
            EmbeddingStage(task: t, train: Array(rows.prefix(4)),
                           config: TrainingConfig(lr: 1e-2, maxSteps: 2, logEvery: 0),
                           batchSize: 2, maxTokens: 48, maxChars: 200)
        }

        let results = try EmbeddingFinetune.run(
            model: model, tokenizer: tok, mode: .frozen,
            stages: [stage(.retrieval, ret), stage(.sts, sts)])

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results.map(\.task), [.retrieval, .sts])
        XCTAssertGreaterThan(maxAbs(model.head.fc1, start), 0,
                             "после двух стадий голова обязана быть другой")
    }

    // ── Загрузка среза ───────────────────────────────────────────────

    func slicePath() -> String {
        ProcessInfo.processInfo.environment["RWKV_LITRETRIEVAL_SLICE"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl").path
    }

    func slice() throws -> [EmbeddingSample] {
        let p = slicePath()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p),
                          "нет среза LitRetrieval — тест пропущен (см. NEXT_SESSION.md)")
        return try EmbeddingDataset.loadJSONL(path: p)
    }

    func loadWorldTokenizerOrSkip() throws -> WorldTokenizer {
        let p = ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p),
                          "нет словаря World — тест пропущен")
        let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: p))
        return try XCTUnwrap(tok, "словарь World не прочитался: \(p)")
    }
}
