import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVQuant
@testable import RWKVRerank
@testable import RWKVEmbedding

// ───────────────────────────────────────────────────────────────────────
//  Реранкер и эмбеддер на базе из `.rwkvq` (пресеты REDUCTION и
//  COMPRESSION). До этого оба пути были проверены только на плотной базе.
//
//  Плечо сравнения — ТЕ ЖЕ квантованные веса, развёрнутые в плотную
//  bf16-модель (`dequantised`). Оно отделяет цену самого квантования (её
//  здесь нет: веса одни и те же) от арифметики квантованного пути: слитого
//  замороженного прохода, fp16-матмула, короткого прохода ядрами. Всё, что
//  расходится между плечами, — шум счёта порта.
//
//  Этот шум НЕ мал, и это свойство пути, а не дефект: в норме проекции
//  считаются «деквант + fp16-матмул» с fp16-накоплением суммы (так же идёт
//  путь эталона, rwkv-quant, на длинных последовательностях), и скрытое
//  состояние уходит от плотного счёта тех же весов примерно на процент.
//  Плотная модель, даже с весами в fp16, считает матмулы в fp32 (вход после
//  ядра WKV — fp32) и сходится с fp32-счётом до 1e-5. На логитах процент
//  состояния — это KL ~4e-6 (на четыре порядка ниже цены квантования); на
//  скорах реранкера со СЛУЧАЙНОЙ головой — 8-14 % размаха к плотному счёту
//  и около 1 % между разбиениями на пачки (у плотной базы 1e-4). На
//  обученной голове разницы в MRR между плечами нет (0.806 / 0.813 / 0.803,
//  четыре сида, 0.1B). Допуски ниже — по замеру, с запасом в 2-4 раза.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqDownstreamTests: XCTestCase {

    static func quantised(_ preset: String) throws -> X070Backbone {
        try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath(preset))
    }

    /// Те же квантованные веса плотной bf16-моделью.
    static func dequantised(_ bb: X070Backbone) -> X070Backbone {
        var w: [String: MLXArray] = [:]
        for key in bb.allWeightKeys { w[key] = bb.weight(key).asType(.bfloat16) }
        eval(Array(w.values))
        return X070Backbone(weights: w, cfg: bb.cfg)
    }

    func tokenizer() throws -> WorldTokenizer {
        let p = ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет словаря World")
        return try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: p)))
    }

    func slice() throws -> [EmbeddingSample] {
        let p = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет среза LitRetrieval")
        return try EmbeddingDataset.loadJSONL(path: p).filter { $0.task == .retrieval }
    }

    let docs = [
        "Пчёлы опыляют растения и производят мёд из нектара цветов.",
        "Паровая машина Уатта запустила промышленную революцию в Англии.",
        "Медоносная пчела танцем сообщает улью направление к взятку.",
        "Железные дороги девятнадцатого века связали города и порты.",
        "Мёд.",
    ]
    let query = "как пчёлы делают мёд"

    func serving(docBatch: Int = 2, queryBatch: Int = 3) -> RerankServingConfig {
        RerankServingConfig(
            encode: RerankEncodeConfig(maxDocTokens: 48, maxQueryTokens: 16,
                                       docBatch: docBatch, queryBatch: queryBatch),
            instruct: "найди документ, отвечающий на вопрос")
    }

    /// Голова с нулевым `fc2` даёт нулевой скор при любой базе.
    func reranker(_ bb: X070Backbone) throws -> Reranker {
        let m = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]), seed: 0)
        MLXRandom.seed(17)
        m.head.fc2Weight = MLXRandom.normal(m.head.fc2Weight.shape) * 0.3
        eval(m.head.fc2Weight)
        return m
    }

    func maxDiff(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).map { Swift.abs($0 - $1) }.max() ?? 0
    }
    func spread(_ a: [Float]) -> Float { (a.max() ?? 0) - (a.min() ?? 0) }

    static func kernelCalls(_ bb: X070Backbone) -> Int { bb.rwkvqSymGEMVCalls + bb.rwkvqSb6GEMVCalls }

    // ── Реранкер ─────────────────────────────────────────────────────

    /// Скоры на квантованной базе идут за скорами тех же весов в плотном
    /// счёте, и порядок документов тот же.
    func testRerankerScoresFollowDequantisedWeights() throws {
        let tok = try tokenizer()
        for preset in RwkvqFileTests.presets {
            let bb = try Self.quantised(preset)
            let q = try RerankerInference(model: try reranker(bb), tokenizer: tok, config: serving())
                .score(query: query, docs: docs)
            let d = try RerankerInference(model: try reranker(Self.dequantised(bb)), tokenizer: tok,
                                          config: serving()).score(query: query, docs: docs)
            let diff = maxDiff(q, d)
            print(String(format: "РЕРАНКЕР %@: размах скоров %.4f, |квант − плотный| %.3e",
                         preset, spread(d), diff))
            XCTAssertGreaterThan(spread(d), 1e-2, "\(preset): скоры вырождены")
            // замерено 0.155 при размахе 1.08 (REDUCTION), 0.109 при 1.37 (COMPRESSION)
            XCTAssertLessThan(diff, spread(d) * 0.3, "\(preset): квантованный путь ушёл от своих же весов")
            XCTAssertEqual(q.indices.sorted { q[$0] > q[$1] }, d.indices.sorted { d[$0] > d[$1] },
                           "\(preset): порядок документов")
            XCTAssertTrue(bb.trainLayers.isEmpty, "база реранкера заморожена")
            XCTAssertGreaterThan(bb.cmixFrozenCalls, 0, "\(preset): замороженный проход не включился")
        }
    }

    /// Индексный путь равен прямому и на квантованной базе — в том числе
    /// когда хвост запроса короче порога и идёт ядрами GEMV, а прямой путь
    /// считает всё матмулом.
    func testRerankerIndexedMatchesDirect() throws {
        let tok = try tokenizer()
        for preset in RwkvqFileTests.presets {
            let bb = try Self.quantised(preset)
            let limit = bb.shortPassKernelRows
            let model = try reranker(bb)
            // пачка запросов 1: хвост «Query: …» обязан попадать под порог
            bb.shortPassKernelRows = 64
            let inf = RerankerInference(model: model, tokenizer: tok,
                                        config: serving(docBatch: 2, queryBatch: 1))
            let index = try inf.buildIndex(docs: docs)
            let before = Self.kernelCalls(bb)
            let indexedKern = try inf.scoreIndexed(query: query, index: index)
            XCTAssertGreaterThan(Self.kernelCalls(bb), before, "\(preset): хвост запроса не дошёл до ядер")
            bb.shortPassKernelRows = 0
            let direct = try inf.score(query: query, docs: docs)
            let indexedMM = try inf.scoreIndexed(query: query, index: index)
            bb.shortPassKernelRows = limit
            let indexedDefault = try inf.scoreIndexed(query: query, index: index)
            print(String(format: "РЕРАНКЕР %@: размах %.4f; индекс−прямой %.3e (матмул), %.3e (ядра), %.3e (порог %d)",
                         preset, spread(direct), maxDiff(direct, indexedMM), maxDiff(direct, indexedKern),
                         maxDiff(direct, indexedDefault), limit))
            XCTAssertGreaterThan(spread(direct), 1e-2)
            for (name, got) in [("матмул", indexedMM), ("ядра", indexedKern), ("порог", indexedDefault)] {
                XCTAssertLessThan(maxDiff(direct, got), spread(direct) * 0.05, "\(preset): \(name)")
            }
        }
    }

    /// Размер пачки — параметр скорости, не чисел: на квантованной базе он
    /// ещё и переключает путь (строк B·T под порогом — ядра).
    func testRerankerScoresDoNotDependOnBatching() throws {
        let tok = try tokenizer()
        for preset in RwkvqFileTests.presets {
            let model = try reranker(try Self.quantised(preset))
            let small = RerankerInference(model: model, tokenizer: tok, config: serving(docBatch: 1, queryBatch: 1))
            let big = RerankerInference(model: model, tokenizer: tok, config: serving(docBatch: 8, queryBatch: 8))
            let a = try small.score(query: query, docs: docs), b = try big.score(query: query, docs: docs)
            let idx = try big.buildIndex(docs: docs)
            let c = try small.scoreIndexed(query: query, index: idx), d = try big.scoreIndexed(query: query, index: idx)
            print(String(format: "РЕРАНКЕР %@: пачка 1 против 8 — прямой %.3e, индексный %.3e (размах %.4f)",
                         preset, maxDiff(a, b), maxDiff(c, d), spread(a)))
            XCTAssertLessThan(maxDiff(a, b), spread(a) * 0.05, preset)
            XCTAssertLessThan(maxDiff(c, d), spread(a) * 0.05, preset)
        }
    }

    // ── Эмбеддер ─────────────────────────────────────────────────────

    /// Векторы на квантованной базе идут за векторами тех же весов в
    /// плотном счёте; короткий текст (под порогом, ядрами) — тоже.
    func testEmbeddingsFollowDequantisedWeights() throws {
        let tok = try tokenizer()
        // «Да» — два токена с терминатором: под порогом короткого прохода обоих пресетов
        let texts = docs + [query, "Да"]
        for preset in RwkvqFileTests.presets {
            let bb = try Self.quantised(preset)
            let before = Self.kernelCalls(bb)
            let q = Embedder(model: EmbeddingModel(backbone: bb), tokenizer: tok).embed(texts).asType(.float32)
            XCTAssertGreaterThan(Self.kernelCalls(bb), before, "\(preset): короткий текст не дошёл до ядер")
            let d = Embedder(model: EmbeddingModel(backbone: Self.dequantised(bb)), tokenizer: tok)
                .embed(texts).asType(.float32)
            let cos = (q * d).sum(axis: -1)
            let worst = cos.min().item(Float.self)
            // различающее: сами тексты между собой далеки от единицы
            let sim = cosineSimilarity(d).asArray(Float.self)
            let n = texts.count
            let offDiag = (0 ..< n * n).filter { $0 / n != $0 % n }.map { sim[$0] }.min() ?? 1
            print(String(format: "ЭМБЕДДЕР %@: худший косинус квант/плотный %.6f; разные тексты — до %.4f",
                         preset, worst, offDiag))
            XCTAssertLessThan(offDiag, 0.99, "\(preset): векторы вырождены")
            XCTAssertGreaterThan(worst, 0.999, "\(preset): квантованный путь ушёл от своих же весов")
        }
    }

    func stage(_ rows: [EmbeddingSample], steps: Int, lr: Float, heldOut: Int = 4) -> EmbeddingStage {
        EmbeddingStage(task: .retrieval, train: Array(rows.prefix(8)), heldOut: Array(rows.suffix(heldOut)),
                       config: TrainingConfig(lr: lr, maxSteps: steps, logEvery: 0),
                       batchSize: 2, maxTokens: 48, maxChars: 200)
    }

    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Замороженная квантованная база: обучается ровно голова, база не
    /// тронута, обратного прохода через базу нет.
    func testFrozenStageTrainsOnlyHead() throws {
        let tok = try tokenizer()
        let rows = try slice()
        try XCTSkipIf(rows.count < 12, "мало строк retrieval в срезе")
        for preset in RwkvqFileTests.presets {
            let bb = try Self.quantised(preset)
            let model = EmbeddingModel(backbone: bb)
            MLXRandom.seed(103)
            model.head.fc2 = MLXRandom.normal(model.head.fc2.shape) * 0.05
            eval(model.head.fc2)
            let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)
            XCTAssertEqual(trainable.parameterNames, EmbeddingHead.parameterNames)
            let snapshot = bb.w
            let backed = bb.rwkvqBackedKeys
            let fc1 = model.head.fc1.asType(.float32) + 0
            eval(fc1)
            let r = try EmbeddingFinetune.runStage(model: model, tokenizer: tok, trainable: trainable,
                                                   stage: stage(rows, steps: 3, lr: 1e-3))
            XCTAssertFalse(r.finalLoss.isNaN, preset)
            XCTAssertGreaterThan(maxAbs(model.head.fc1, fc1), 0, "\(preset): голова не сдвинулась")
            XCTAssertEqual(bb.w.count, snapshot.count)
            for (k, v) in snapshot { XCTAssertEqual(RwkvqFileTests.mismatches(bb.w[k]!, v), 0, "\(preset): \(k)") }
            XCTAssertEqual(bb.rwkvqBackedKeys, backed)
            XCTAssertFalse(backed.isEmpty, "\(preset): база не квантованная — тест пуст")
            XCTAssertGreaterThan(bb.cmixFrozenCalls, 0, "\(preset): замороженный проход не включился")
        }
    }

    /// QLoRA: обучаются адаптеры и голова, база остаётся сжатой и нетронутой,
    /// градиент до адаптеров доходит через квантованные проекции.
    func testQLoRAStageTrainsAdaptersAndKeepsBaseCompressed() throws {
        let tok = try tokenizer()
        let rows = try slice()
        try XCTSkipIf(rows.count < 12, "мало строк retrieval в срезе")
        for preset in RwkvqFileTests.presets {
            let bb = try Self.quantised(preset)
            LoRA.add(to: bb, spec: LoRASpec(rank: 4, alpha: 4))
            let model = EmbeddingModel(backbone: bb)
            MLXRandom.seed(103)
            model.head.fc2 = MLXRandom.normal(model.head.fc2.shape) * 0.05
            eval(model.head.fc2)
            let trainable = EmbeddingTrainable.make(model: model, mode: .lora)
            XCTAssertEqual(bb.trainLayers, Set(0 ..< bb.cfg.nLayer))
            let snapshot = bb.w
            let backed = bb.rwkvqBackedKeys
            let targets = bb.loraTargets
            XCTAssertFalse(targets.isEmpty)
            let b0 = targets.map { bb.loraB[$0]!.asType(.float32) + 0 }
            eval(b0)
            let probe = Embedder(model: model, tokenizer: tok)
            let v0 = probe.embed(docs[0]).asType(.float32) + 0
            eval(v0)
            let kern = Self.kernelCalls(bb), frozen = bb.cmixFrozenCalls
            // без отложенной выборки: оценка идёт прямым путём и счётчики сдвинула бы
            let r = try EmbeddingFinetune.runStage(model: model, tokenizer: tok, trainable: trainable,
                                                   stage: stage(rows, steps: 3, lr: 1e-3, heldOut: 0))
            XCTAssertFalse(r.finalLoss.isNaN, preset)
            // B стартует с нуля: сдвиг B означает, что градиент дошёл через базу
            let moved = zip(targets, b0).filter { maxAbs(bb.loraB[$0.0]!, $0.1) > 0 }.count
            // у слоя 0 адаптер на v_proj градиента не получает: в первом слое v — источник v_first
            XCTAssertGreaterThanOrEqual(moved, targets.count - 1, "\(preset): сдвинулось \(moved) из \(targets.count) адаптеров")
            for (k, v) in snapshot { XCTAssertEqual(RwkvqFileTests.mismatches(bb.w[k]!, v), 0, "\(preset): \(k)") }
            XCTAssertEqual(bb.rwkvqBackedKeys, backed, "\(preset): база развернулась")
            // обучаемые слои — не ядрами и не слитым замороженным проходом
            XCTAssertEqual(Self.kernelCalls(bb), kern, "\(preset): обучаемый слой ушёл в ядро без градиента")
            XCTAssertEqual(bb.cmixFrozenCalls, frozen, "\(preset): обучаемый слой ушёл в замороженный cmix")
            let v1 = probe.embed(docs[0]).asType(.float32)
            XCTAssertGreaterThan(maxAbs(v1, v0), 1e-4, "\(preset): вектор после обучения прежний")

            // Выдача с адаптерами идёт ПРЯМЫМ путём (замороженные ядра, короткий
            // проход), обучение — дифференцируемым. Адаптеры обязаны входить в
            // оба одинаково: иначе обучается одна модель, а выдаётся другая.
            let ids = MLXArray((0 ..< 32).map { Int32(($0 * 7919 + 13) % 60000 + 100) }, [2, 16])
            let pool = MLXArray([Int32(15), Int32(15)])
            let viaTrain = model.embed(ids, poolIndex: pool).asType(.float32)
            let kernBefore = Self.kernelCalls(bb) + bb.cmixFrozenCalls
            let viaForward = bb.withoutGradientPath { model.embed(ids, poolIndex: pool) }.asType(.float32)
            eval(viaTrain, viaForward)
            XCTAssertGreaterThan(Self.kernelCalls(bb) + bb.cmixFrozenCalls, kernBefore,
                                 "\(preset): прямой путь не включился")
            let cos = (viaTrain * viaForward).sum(axis: -1).min().item(Float.self)
            // без адаптеров тот же вход даёт заметно другой вектор — различающее
            let savedB = bb.loraB
            for t in targets { bb.loraB[t] = MLXArray.zeros(savedB[t]!.shape, dtype: savedB[t]!.dtype) }
            let bare = bb.withoutGradientPath { model.embed(ids, poolIndex: pool) }.asType(.float32)
            eval(bare)
            bb.loraB = savedB
            let cosBare = (bare * viaForward).sum(axis: -1).min().item(Float.self)
            print(String(format: "QLoRA %@: косинус прямой/обучаемый путь %.6f; с адаптерами/без %.6f",
                         preset, cos, cosBare))
            XCTAssertGreaterThan(cos, 0.9995, "\(preset): выдача считает не ту модель, что обучалась")
            XCTAssertLessThan(cosBare, cos, "\(preset): адаптеры на прямом пути не видны")
            XCTAssertEqual(bb.trainLayers, Set(0 ..< bb.cfg.nLayer), "\(preset): обёртка не вернула trainLayers")
        }
    }

    /// `.full` и `.topLayers` на квантованной базе — отказ с причиной, а не
    /// молчаливое обучение одних норм под именем «вся база».
    func testWeightModesAreRefusedOnQuantisedBase() throws {
        let bb = try Self.quantised("reduction")
        let model = EmbeddingModel(backbone: bb)
        XCTAssertNotNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .full))
        XCTAssertNotNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .topLayers(2)))
        XCTAssertNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .frozen))
        XCTAssertNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .lora))
        let dense = EmbeddingModel(backbone: Self.dequantised(bb))
        for mode in [BaseTrainingMode.full, .topLayers(2), .frozen, .lora] {
            XCTAssertNil(EmbeddingTrainable.unsupportedReason(model: dense, mode: mode), "плотная база: \(mode)")
        }
    }
}
