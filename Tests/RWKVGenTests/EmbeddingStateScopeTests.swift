//
//  EmbeddingStateScopeTests.swift
//  Состояние только для запросов (17.10): `X070Backbone.withoutInitialState`,
//  роль текста в эмбеддере, оценка, выдача и контракт чекпоинта.
//
//  Главное свойство: документ при `.queriesOnly` ПОБИТНО тот же, что без
//  подключённого состояния, а запрос — тот же, что при `.allTexts`.
//  Градиенты (обратный проход по блокам, куски, GradCache) — в
//  EmbeddingSequentialTests.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding

final class EmbeddingStateScopeTests: XCTestCase {

    func randomState(_ bb: X070Backbone, seed: UInt64 = 21) -> RWKVBatchState {
        MLXRandom.seed(seed)
        let z = bb.zeroInitialState()
        let s = RWKVBatchState(wkv: MLXRandom.normal(z.wkv.shape) * 0.05,
                               tmixShift: MLXRandom.normal(z.tmixShift.shape) * 0.5,
                               cmixShift: MLXRandom.normal(z.cmixShift.shape) * 0.5)
        eval(s.wkv, s.tmixShift, s.cmixShift)
        return s
    }

    func makeModel(vocab: Int = 64, scope: InitialStateScope = .queriesOnly) -> (EmbeddingModel, RWKVBatchState) {
        let (bb, _) = TinyBackbone.make(nLayer: 3, vocab: vocab)
        let head = EmbeddingHead(dim: bb.cfg.nEmbd, seed: 4)
        MLXRandom.seed(104)
        head.fc2 = MLXRandom.normal([bb.cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        let s = randomState(bb)
        bb.initialState = s
        return (EmbeddingModel(backbone: bb, head: head, stateScope: scope), s)
    }

    func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        eval(a, b)
        return a.shape == b.shape && MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) == 0
    }

    // ── RWKVGen: выключение подключённого состояния ──

    /// Под `withoutInitialState` проход побитно тот же, что без состояния;
    /// само состояние остаётся подключённым и видимым; выход из обёртки (и
    /// вложенной) возвращает прежнее.
    func testWithoutInitialStateIsBitwiseNoState() {
        let (bb, _) = TinyBackbone.make(nLayer: 3)
        let ids = TinyBackbone.ids(2, 20)
        let s = randomState(bb)
        let none = bb.body(ids), noneStates = bb.states(ids)
        bb.initialState = s
        let with = bb.body(ids)
        let fp = bb.initialStateFingerprint
        let suspended = bb.withoutInitialState { () -> (MLXArray, RWKVBatchState) in
            XCTAssertNotNil(bb.initialState, "состояние отключено, а не выключено")
            XCTAssertEqual(bb.initialStateFingerprint, fp)
            bb.withoutInitialState { }
            XCTAssertTrue(bb.initialStateSuspended, "вложенная обёртка вернула не то")
            return (bb.body(ids), bb.states(ids))
        }
        XCTAssertFalse(bb.initialStateSuspended)
        XCTAssertTrue(same(suspended.0, none), "под обёрткой проход не тот, что без состояния")
        XCTAssertTrue(same(suspended.1.wkv, noneStates.wkv))
        XCTAssertTrue(same(suspended.1.tmixShift, noneStates.tmixShift))
        XCTAssertFalse(same(with, none), "состояние ничего не меняет — проверять нечего")
        XCTAssertTrue(same(bb.body(ids), with), "после обёртки состояние не вернулось")
    }

    // ── Эмбеддер ──

    /// Документ при `.queriesOnly` — как без состояния, запрос — как при
    /// `.allTexts`; при `.allTexts` роль не значит ничего.
    func testRolesUnderQueriesOnly() {
        let (model, s) = makeModel()
        let bb = model.backbone
        let ids = TinyBackbone.ids(2, 20)
        let pool = MLXArray([Int32(19), 17])
        let q = model.embed(ids, poolIndex: pool, role: .query)
        let d = model.embed(ids, poolIndex: pool, role: .document)
        let pq = model.pooledOnly(ids, poolIndex: pool, role: .query)
        let pd = model.pooledOnly(ids, poolIndex: pool, role: .document)
        model.stateScope = .allTexts
        let allDoc = model.embed(ids, poolIndex: pool, role: .document)
        let allPooled = model.pooledOnly(ids, poolIndex: pool)
        bb.initialState = nil
        let none = model.embed(ids, poolIndex: pool), nonePooled = model.pooledOnly(ids, poolIndex: pool)
        bb.initialState = s
        XCTAssertTrue(same(d, none), "документ при .queriesOnly — не как без состояния")
        XCTAssertTrue(same(pd, nonePooled))
        XCTAssertTrue(same(q, allDoc), "запрос при .queriesOnly — не как при .allTexts")
        XCTAssertTrue(same(pq, allPooled))
        XCTAssertFalse(same(q, d), "роль ничего не меняет — проверять нечего")
    }

    /// Оценка: якоря — запросы, кандидаты — документы; sts — все запросы;
    /// классификация — метки как документы. Эталон собран вручную: документы
    /// при снятом состоянии.
    func testEvaluationUsesRoles() throws {
        let tok = try worldTokenizer()
        let (model, s) = makeModel(vocab: 65536)
        let bb = model.backbone
        let rows = (0 ..< 4).map { i in
            EmbeddingSample(anchor: "вопрос номер \(i) про пчёл", positive: "ответ \(i): пчёлы делают мёд",
                            negative: "паровая машина \(i)", task: .retrieval)
        }
        let anchors = rows.map { $0.anchor }, cands = rows.map { $0.positive } + rows.map { $0.negative }
        let aQ = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: anchors, role: .query)
        let cD = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: cands, role: .document)
        var gotSims: MLXArray? = nil
        EmbeddingMetrics.similarityObserver = { gotSims = $0 }
        defer { EmbeddingMetrics.similarityObserver = nil }
        let got = EmbeddingMetrics.evaluateRetrieval(model: model, tokenizer: tok, rows: rows)
        let retrievalSims = gotSims!
        let sts = EmbeddingMetrics.evaluateSTS(model: model, tokenizer: tok, rows: rows)
        model.stateScope = .allTexts
        let aAll = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: anchors)
        let stsAll = EmbeddingMetrics.evaluateSTS(model: model, tokenizer: tok, rows: rows)
        bb.initialState = nil
        let cNone = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: cands)
        bb.initialState = s
        XCTAssertTrue(same(aQ, aAll))
        XCTAssertTrue(same(cD, cNone), "кандидаты оценки при .queriesOnly — с состоянием")
        let ref = EmbeddingMetrics.rankingMetrics(
            similarities: matmul(aAll.asType(.float32), cNone.asType(.float32).transposed()),
            correctIndex: Array(0 ..< rows.count))
        XCTAssertTrue(same(retrievalSims, matmul(aAll.asType(.float32), cNone.asType(.float32).transposed())),
                      "оценка retrieval: кандидаты с состоянием")
        XCTAssertEqual(got.mrr, ref.mrr)
        XCTAssertEqual(got.description, ref.description)
        XCTAssertEqual(sts.description, stsAll.description, "sts: стороны пары равноправны")
        // классификация: якоря — запросы, метки — документы
        model.stateScope = .queriesOnly
        let cls = (0 ..< 3).map { i in
            EmbeddingSample(anchor: "текст номер \(i) о радости", positive: ClassificationLabels.pool[i],
                            negative: "", task: .classification)
        }
        gotSims = nil
        _ = EmbeddingMetrics.evaluateClassification(model: model, tokenizer: tok, rows: cls)
        let clsSims = gotSims!
        model.stateScope = .allTexts
        let clsA = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: cls.map { $0.anchor })
        bb.initialState = nil
        let labelsNone = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: ClassificationLabels.pool)
        bb.initialState = s
        XCTAssertTrue(same(clsSims, matmul(clsA.asType(.float32), labelsNone.asType(.float32).transposed())),
                      "оценка классификации: метки с состоянием")
    }

    /// Выдача: роль в `Embedder.embed`, область — из модели в контракт.
    func testEmbedderRoles() throws {
        let tok = try worldTokenizer()
        let (model, s) = makeModel(vocab: 65536)
        let e = Embedder(model: model, tokenizer: tok)
        XCTAssertEqual(e.contract.stateScope, .queriesOnly)
        let d = e.embed(["мёд и пчёлы", "паровая машина"], role: .document)
        let q = e.embed("мёд и пчёлы", role: .query)
        model.stateScope = .allTexts
        let qAll = e.embed("мёд и пчёлы")
        model.backbone.initialState = nil
        let dNone = e.embed(["мёд и пчёлы", "паровая машина"])
        model.backbone.initialState = s
        XCTAssertTrue(same(d, dNone))
        XCTAssertTrue(same(q, qAll))
    }

    // ── Контракт ──

    /// Область — в метаданных чекпоинта; нет ключа — `.allTexts`; `fromHead`
    /// собирает модель с областью из файла; голова с другой областью в модель
    /// не грузится.
    func testContractCarriesScope() throws {
        let (model, _) = makeModel()
        XCTAssertEqual(EmbeddingContract(stateScope: .queriesOnly).metadata["initial_state_scope"], "queries")
        XCTAssertEqual(EmbeddingContract().metadata["initial_state_scope"], "all")
        XCTAssertEqual(EmbeddingContract(metadata: ["pooling": "last"]).stateScope, .allTexts, "старый чекпоинт")
        XCTAssertEqual(EmbeddingContract(metadata: ["initial_state_scope": "queries"]).stateScope, .queriesOnly)

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("scope_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.saveHead(to: url)
        XCTAssertEqual(try EmbeddingModel.readContract(url).stateScope, .queriesOnly)
        let (loaded, contract) = try EmbeddingModel.fromHead(backbone: model.backbone, url: url)
        XCTAssertEqual(loaded.stateScope, .queriesOnly)
        XCTAssertEqual(contract.stateScope, .queriesOnly)
        let other = EmbeddingModel(backbone: model.backbone, head: EmbeddingHead(dim: model.head.dim, seed: 4))
        XCTAssertThrowsError(try other.loadHead(from: url), "голова .queriesOnly загружена в модель .allTexts")
    }

    private func worldTokenizer() throws -> WorldTokenizer {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет словаря World — тест пропущен")
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: path)) else {
            throw XCTSkip("словарь World не разобрался")
        }
        return tok
    }
}
