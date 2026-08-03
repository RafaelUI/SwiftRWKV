//
//  EmbeddingSmokeTests.swift
//  Публичный API RWKVEmbedding на РЕАЛЬНОЙ 0.1B.
//
//  Модульные тесты проверяют куски; здесь проверяется, что связка целиком
//  делает то, что обещает, на настоящей модели и настоящих данных. Разница
//  существенна: почти всё в этом модуле — числа без формы отказа, и «api
//  работает» нельзя вывести из того, что оно компилируется.
//
//  Пропускается без модели и среза LitRetrieval.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVEmbedding

final class EmbeddingSmokeTests: XCTestCase {

    func fixtures() throws -> (EmbeddingModel, WorldTokenizer, [EmbeddingSample]) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["RWKV_PARITY_MODEL"]
            ?? home.appendingPathComponent(
                "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let vocabPath = env["RWKV_WORLD_VOCAB"]
            ?? home.appendingPathComponent(
                "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        let slicePath = env["RWKV_LITRETRIEVAL_SLICE"]
            ?? home.appendingPathComponent(
                "Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl").path
        for p in [modelPath, vocabPath, slicePath] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p),
                              "нет фикстуры \(p) — тест пропущен")
        }
        let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        let bb = X070Backbone(weights: weights, cfg: cfg)
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: vocabPath))
        else { throw XCTSkip("словарь не разобрался") }
        let rows = try EmbeddingDataset.loadJSONL(path: slicePath, limit: 60)
        try XCTSkipUnless(!rows.isEmpty, "срез пуст")
        return (EmbeddingModel(backbone: bb), tok, rows)
    }

    /// Векторизация: формы, нормировка, батч == по одному.
    func testEmbedderProducesNormalisedVectors() throws {
        let (model, tok, _) = try fixtures()
        let e = Embedder(model: model, tokenizer: tok)

        let v = e.embed("медоносная пчела собирает нектар")
        XCTAssertEqual(v.shape, [model.backbone.cfg.nEmbd])
        eval(v)
        let norm = MLX.sqrt((v * v).sum()).item(Float.self)
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4, "вектор не L2-нормирован")

        let texts = ["пчёлы и мёд", "паровая машина Уатта", "короткий"]
        let m = e.embed(texts)
        XCTAssertEqual(m.shape, [texts.count, model.backbone.cfg.nEmbd])
        // Батчевый вызов обязан совпасть с поштучным: он и реализован
        // поштучно, но это свойство важнее реализации — если однажды путь
        // станет настоящим батчем, тест обязан заметить сдвиг.
        for (i, t) in texts.enumerated() {
            eval(m[i])
            XCTAssertEqual(MLX.abs(m[i] - e.embed(t)).max().item(Float.self), 0,
                           "текст \(i): батч разошёлся с поштучным")
        }
    }

    /// Косинус: сам с собой единица, симметричен.
    func testCosineSimilarityBasics() throws {
        let (model, tok, _) = try fixtures()
        let e = Embedder(model: model, tokenizer: tok)
        let a = e.embed("пчёлы делают мёд").reshaped([1, -1])
        let b = e.embed("паровые машины и железные дороги").reshaped([1, -1])
        let self_ = cosineSimilarity(a)
        eval(self_)
        XCTAssertEqual(self_[0, 0].item(Float.self), 1.0, accuracy: 1e-4)
        let ab = cosineSimilarity(a, b), ba = cosineSimilarity(b, a)
        eval(ab, ba)
        XCTAssertEqual(ab[0, 0].item(Float.self), ba[0, 0].item(Float.self),
                       accuracy: 1e-6)
    }

    /// Оба пулинга работают и дают РАЗНЫЕ векторы.
    ///
    /// Различие обязательно: совпади они, тест не отличал бы реализованный
    /// mean от копии last.
    func testBothPoolingsWork() throws {
        let (model, tok, _) = try fixtures()
        let mean = EmbeddingModel(backbone: model.backbone, pooling: .mean)
        let last = Embedder(model: model, tokenizer: tok)
        let meanE = Embedder(model: mean, tokenizer: tok)
        let text = "довольно длинная строка про пчёл, нектар и опыление растений"
        let a = last.embed(text), b = meanE.embed(text)
        eval(a, b)
        XCTAssertEqual(MLX.sqrt((b * b).sum()).item(Float.self), 1.0, accuracy: 1e-4)
        XCTAssertGreaterThan(MLX.abs(a - b).max().item(Float.self), 1e-3,
                             "mean и last дали один вектор")
    }

    /// Метрики считаются на сырой базе и дают осмысленный диапазон.
    func testMetricsRunOnRawBase() throws {
        let (model, tok, rows) = try fixtures()
        let retrieval = rows.filter { $0.task == .retrieval }
        try XCTSkipUnless(retrieval.count >= 8, "мало строк retrieval")

        let m = EmbeddingMetrics.evaluateRetrieval(
            model: model, tokenizer: tok, rows: Array(retrieval.prefix(24)))
        XCTAssertEqual(m.count, min(24, retrieval.count))
        XCTAssertGreaterThan(m.mrr, 0.0)
        XCTAssertLessThanOrEqual(m.mrr, 1.0)
        XCTAssertNotNil(m.recall[1])

        let sts = EmbeddingMetrics.evaluateSTS(
            model: model, tokenizer: tok, rows: Array(retrieval.prefix(24)))
        XCTAssertGreaterThanOrEqual(sts.accuracy, 0.0)
        XCTAssertLessThanOrEqual(sts.accuracy, 1.0)
    }

    /// Обучение головы на замороженной базе идёт и МЕНЯЕТ метрики.
    ///
    /// Проверяется не «лучше», а «изменилось»: на двадцати шагах направление
    /// не гарантировано, а вот полная неподвижность означала бы, что градиент
    /// до головы не дотекает — то есть ровно ту молчаливую поломку, которая
    /// в этом репозитории уже случалась дважды.
    func testFrozenBaseHeadTrainingMovesMetrics() throws {
        let (model, tok, rows) = try fixtures()
        let retrieval = rows.filter { $0.task == .retrieval }
        try XCTSkipUnless(retrieval.count >= 16, "мало строк retrieval")

        let stage = EmbeddingStage(
            task: .retrieval,
            train: Array(retrieval.prefix(16)),
            heldOut: Array(retrieval.suffix(8)),
            config: TrainingConfig(lr: 1e-3, schedule: .constant, maxSteps: 20),
            batchSize: 4, maxTokens: 128)
        let trainable = EmbeddingTrainable.make(model: model, mode: .frozen)
        let r = try EmbeddingFinetune.runStage(model: model, tokenizer: tok,
                                               trainable: trainable, stage: stage)
        XCTAssertEqual(r.steps, 20)
        XCTAssertFalse(r.finalLoss.isNaN, "лосс NaN — обучение вхолостую")
        XCTAssertNotNil(r.before)
        XCTAssertNotNil(r.after)
        XCTAssertNotEqual(r.before!.headline, r.after!.headline,
                          "метрики не сдвинулись — градиент до головы не дотёк")
    }

    /// Выдача обрезает текст ТАК ЖЕ, как оценка.
    ///
    /// Раньше расходилось: `EmbeddingMetrics.*` и `EmbeddingStage` работали с
    /// `maxTokens = 512`, а `Embedder.embed` не обрезал вовсе — значит на
    /// длинных текстах измеренное качество описывало не то, что делает
    /// выдача. Заметить это по формам невозможно: вектор нормирован в обоих
    /// случаях. Тест писался сперва как фиксация расхождения, потом стал
    /// проверкой того, что его нет.
    func testServingTruncatesLikeEvaluation() throws {
        let (model, tok, _) = try fixtures()
        let e = Embedder(model: model, tokenizer: tok)
        XCTAssertEqual(e.maxTokens, 512, "умолчание разъехалось с оценкой")

        let long = String(repeating: "пчёлы собирают нектар с цветов. ", count: 120)
        XCTAssertGreaterThan(tok.encode(long).count, 512,
                             "текст короче порога — тест вырожден")

        let served = e.embed(long)
        let (idx, pool) = encodeBatch(tokenizer: tok, texts: [long],
                                      maxTokens: 512)
        let evaluated = model.embed(idx, poolIndex: pool)[0]
        eval(served, evaluated)
        XCTAssertLessThan(MLX.abs(served - evaluated).max().item(Float.self),
                          1e-5, "выдача и оценка обрезают по-разному")

        // Различающее утверждение: БЕЗ обрезки вектор другой. Иначе
        // совпадение выше ничего не значило бы — оно выполнялось бы и при
        // полностью отключённой обрезке с обеих сторон.
        var unlimited = EmbeddingContract()
        unlimited.maxTokens = nil
        let whole = Embedder(model: model, tokenizer: tok,
                             contract: unlimited).embed(long)
        eval(whole)
        XCTAssertGreaterThan(MLX.abs(served - whole).max().item(Float.self),
                             1e-3, "обрезка ни на что не влияет — тест пуст")
    }

    /// Терминатор остаётся ПОСЛЕДНИМ и при обрезке.
    ///
    /// Обрезать после дописывания терминатора означало бы срезать его самого
    /// либо оставить не на конце — и тогда позиция пулинга указывает на
    /// обычный токен, а вектор снимается не с того места. Ошибка целиком
    /// молчаливая: длина сходится, вектор нормирован.
    func testTerminatorStaysLastAfterTruncation() throws {
        let (model, tok, _) = try fixtures()
        var c = EmbeddingContract()
        c.maxTokens = 16
        let e = Embedder(model: model, tokenizer: tok, contract: c)
        let ids = e.encode(String(repeating: "пчёлы и нектар ", count: 50))
        XCTAssertEqual(ids.count, 16, "обрезка не соблюдена")
        XCTAssertEqual(ids.last, 0, "терминатор не последний")

        var noTerm = EmbeddingContract()
        noTerm.maxTokens = 16
        noTerm.terminator = nil
        let e2 = Embedder(model: model, tokenizer: tok, contract: noTerm)
        XCTAssertEqual(e2.encode(String(repeating: "пчёлы и нектар ", count: 50)).count,
                       16, "без терминатора обрезка съела лишнее")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Чекпоинт
    // ─────────────────────────────────────────────────────────────────

    /// Круг «сохранить → загрузить» возвращает те же векторы и тот контракт.
    func testHeadCheckpointRoundTrip() throws {
        let (model, tok, rows) = try fixtures()
        // Голову надо СДВИНУТЬ с нуля: у свежей fc2 = 0, то есть она
        // тождество, и круг совпал бы при полностью сломанной записи весов.
        let retrieval = rows.filter { $0.task == .retrieval }
        try XCTSkipUnless(retrieval.count >= 12, "мало строк")
        let stage = EmbeddingStage(
            task: .retrieval, train: Array(retrieval.prefix(12)),
            config: TrainingConfig(lr: 1e-3, schedule: .constant, maxSteps: 10),
            batchSize: 4, maxTokens: 128)
        _ = try EmbeddingFinetune.runStage(
            model: model, tokenizer: tok,
            trainable: EmbeddingTrainable.make(model: model, mode: .frozen),
            stage: stage)

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("emb_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        var contract = EmbeddingContract()
        contract.maxTokens = 128
        try model.saveHead(to: url, contract: contract)

        let (loaded, back) = try EmbeddingModel.fromHead(
            backbone: model.backbone, url: url)
        XCTAssertEqual(back.maxTokens, 128)
        XCTAssertEqual(back.pooling, .last)

        let text = "пчёлы собирают нектар с цветов"
        let a = Embedder(model: model, tokenizer: tok, contract: contract).embed(text)
        let b = Embedder(model: loaded, tokenizer: tok, contract: back).embed(text)
        eval(a, b)
        XCTAssertEqual(MLX.abs(a - b).max().item(Float.self), 0,
                       "загруженная голова даёт другой вектор")

        // Различающее: НЕобученная голова даёт ДРУГОЙ вектор, значит круг
        // проверяет перенос весов, а не тождество.
        let fresh = EmbeddingModel(backbone: model.backbone)
        let c = Embedder(model: fresh, tokenizer: tok, contract: contract).embed(text)
        eval(c)
        XCTAssertGreaterThan(MLX.abs(a - c).max().item(Float.self), 1e-4,
                             "обучение не сдвинуло голову — круг вырожден")
    }

    /// Чекпоинт, обученный с другим пулингом, не грузится молча.
    func testCheckpointRejectsPoolingMismatch() throws {
        let (model, tok, _) = try fixtures()
        _ = tok
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("emb_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.saveHead(to: url)   // pooling = .last

        let meanModel = EmbeddingModel(backbone: model.backbone, pooling: .mean)
        XCTAssertThrowsError(try meanModel.loadHead(from: url),
                             "голова с .last загружена в модель с .mean")

        // fromHead собирает модель ПО ФАЙЛУ, поэтому расхождения не создаёт.
        let (rebuilt, contract) = try EmbeddingModel.fromHead(
            backbone: model.backbone, url: url)
        XCTAssertEqual(rebuilt.pooling, .last)
        XCTAssertEqual(contract.pooling, .last)
    }

    /// Чужой файл и голова другого размера отвергаются.
    func testCheckpointRejectsForeignAndMismatchedFiles() throws {
        let (model, _, _) = try fixtures()
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("emb_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: ["nonsense": MLXArray.zeros([2, 2])], url: url)
        XCTAssertThrowsError(try model.loadHead(from: url))
        XCTAssertThrowsError(try EmbeddingModel.fromHead(backbone: model.backbone,
                                                          url: url))

        let wide = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("emb_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: wide) }
        let big = EmbeddingModel(backbone: model.backbone,
                                 head: EmbeddingHead(dim: model.head.dim,
                                                     hidden: model.head.hidden * 2))
        try big.saveHead(to: wide)
        XCTAssertThrowsError(try model.loadHead(from: wide),
                             "голова другой ширины принята")
    }
}
