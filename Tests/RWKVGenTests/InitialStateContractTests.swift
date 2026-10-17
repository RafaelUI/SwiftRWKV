//
//  InitialStateContractTests.swift
//  Подключённое начальное состояние в контрактах кэшей реранкера (17.10).
//
//  Индекс префиксов и кэш состояний пар хранят РЕЗУЛЬТАТ счёта, а счёт
//  стартует с подключённого состояния. Кэш, снятый с одним состоянием, при
//  другом даёт правдоподобные и неверные числа — формы те же, скоры сырые.
//  Поэтому отпечаток состояния — часть контракта:
//    - другое состояние (и «без состояния» против «с состоянием») — отказ;
//    - то же состояние — принимается, числа побитно те же;
//    - кэш без ключа (записанный до его появления) — это «без состояния».
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank

final class InitialStateContractTests: XCTestCase {

    // ── Оснастка ──

    func makeModel(nLayer: Int = 4) throws -> (Reranker, X070Backbone) {
        let (bb, _) = TinyBackbone.make(nLayer: nLayer, vocab: 65536)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        MLXRandom.seed(17)
        model.head.fc2Weight = MLXRandom.normal(model.head.fc2Weight.shape) * 0.3
        eval(model.head.fc2Weight)
        return (model, bb)
    }

    /// Случайное состояние формы модели: заполнены все три части.
    func randomState(_ bb: X070Backbone, seed: UInt64, scale: Float = 0.3) -> RWKVBatchState {
        let c = bb.cfg
        MLXRandom.seed(seed)
        let s = RWKVBatchState(
            wkv: MLXRandom.normal([c.nLayer, 1, c.nHead, c.headSize, c.headSize]) * scale,
            tmixShift: MLXRandom.normal([c.nLayer, 1, 1, c.nEmbd]) * scale,
            cmixShift: MLXRandom.normal([c.nLayer, 1, 1, c.nEmbd]) * scale)
        eval(s.wkv, s.tmixShift, s.cmixShift)
        return s
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
    ]
    let query = "как пчёлы делают мёд"

    func serving() -> RerankServingConfig {
        RerankServingConfig(
            encode: RerankEncodeConfig(maxDocTokens: 48, maxQueryTokens: 16,
                                       docBatch: 2, queryBatch: 3),
            instruct: "найди документ, отвечающий на вопрос")
    }

    func tmpURL(_ ext: String = "safetensors") -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("isc_\(UUID().uuidString).\(ext)")
    }

    /// Детерминированное состояние для сверки с rwkv-metal: значения точны в fp16.
    func fixedState() -> RWKVBatchState {
        RWKVBatchState(
            wkv: ((MLXArray(Array(0 ..< 64).map { Float($0) }) - 20) * 0.125).reshaped([2, 1, 2, 4, 4]),
            tmixShift: (MLXArray(Array(0 ..< 16).map { Float($0) }) * 0.25 - 1).reshaped([2, 1, 1, 8]),
            cmixShift: (MLXArray(Array(0 ..< 16).map { Float($0) }) * -0.5 + 3).reshaped([2, 1, 1, 8]))
    }

    // ─────────────────────────────────────────────────────────────────
    //  Отпечаток
    // ─────────────────────────────────────────────────────────────────

    /// Тот же отпечаток бит в бит, что у rwkv_metal `state_fingerprint`
    /// (константа снята питоновской реализацией на том же состоянии).
    func testFingerprintMatchesRwkvMetal() {
        XCTAssertEqual(StateFile.stateFingerprint(fixedState()),
                       "sha256:b51da87efab20d186e0e30949566886fd0e293594a3ffe31e672b336e08b2f58")
    }

    /// Каждая из трёх частей, форма и место значения входят в отпечаток; тип
    /// хранения — нет (значения точны в fp16).
    func testFingerprintSeesEveryPart() {
        let s = fixedState()
        let f = StateFile.stateFingerprint(s)
        XCTAssertTrue(f.hasPrefix("sha256:"))
        var a = s; a.wkv = a.wkv + MLXArray(Float(0.125)) * (MLXArray(Array(0 ..< 64).map { Float($0) }).reshaped([2, 1, 2, 4, 4]) .== 63).asType(.float32)
        var b = s; b.tmixShift = b.tmixShift.asType(.float32) + (MLXArray(Array(0 ..< 16).map { Float($0) }).reshaped([2, 1, 1, 8]) .== 15).asType(.float32)
        var c = s; c.cmixShift = c.cmixShift.asType(.float32) + (MLXArray(Array(0 ..< 16).map { Float($0) }).reshaped([2, 1, 1, 8]) .== 15).asType(.float32)
        var swapped = s; swapped.tmixShift = s.cmixShift; swapped.cmixShift = s.tmixShift
        var reshaped = s; reshaped.wkv = s.wkv.reshaped([2, 1, 4, 2, 4])
        let all = [f, StateFile.stateFingerprint(a), StateFile.stateFingerprint(b),
                   StateFile.stateFingerprint(c), StateFile.stateFingerprint(swapped),
                   StateFile.stateFingerprint(reshaped)]
        XCTAssertEqual(Set(all).count, all.count, "разные состояния с одним отпечатком: \(all)")
        let half = RWKVBatchState(wkv: s.wkv.asType(.float16), tmixShift: s.tmixShift.asType(.float16),
                                  cmixShift: s.cmixShift.asType(.float16))
        XCTAssertEqual(StateFile.stateFingerprint(half), f, "отпечаток зависит от типа хранения")
        // fp32 — канон, а не fp16: различие ниже шага fp16 видно.
        var tiny = s; tiny.wkv = s.wkv + MLXArray(Float(1e-6)) * (MLXArray(Array(0 ..< 64).map { Float($0) }).reshaped([2, 1, 2, 4, 4]) .== 40).asType(.float32)
        XCTAssertNotEqual(StateFile.stateFingerprint(tiny), f, "отпечаток не видит различия ниже шага fp16")
    }

    /// Нет состояния и нулевое состояние — одно значение `none`: нули счёта не
    /// меняют. Ненулевое — не `none`, и подключённое к модели читается тем же.
    func testNoneAndZeroState() throws {
        let (_, bb) = try makeModel(nLayer: 2)
        XCTAssertEqual(StateFile.stateFingerprint(nil), StateFile.noInitialState)
        XCTAssertEqual(bb.initialStateFingerprint, "none")
        bb.initialState = bb.zeroInitialState()
        XCTAssertEqual(bb.initialStateFingerprint, "none", "нулевое состояние не равно отсутствию")
        let s = randomState(bb, seed: 1)
        bb.initialState = s
        XCTAssertNotEqual(bb.initialStateFingerprint, "none")
        XCTAssertEqual(bb.initialStateFingerprint, StateFile.stateFingerprint(s))
        bb.initialState = nil
        XCTAssertEqual(bb.initialStateFingerprint, "none")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Индекс выдачи
    // ─────────────────────────────────────────────────────────────────

    /// Индекс, снятый с состоянием A: при B и без состояния — отказ; при A —
    /// принят, скоры побитно те же (до и после переключений, из памяти и с
    /// диска, в том числе потоковый индекс).
    func testIndexContractCarriesInitialState() throws {
        let tok = try worldTokenizer()
        let (model, bb) = try makeModel()
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        let sA = randomState(bb, seed: 3), sB = randomState(bb, seed: 4)

        bb.initialState = sA
        let index = try inf.buildIndex(docs: docs)
        XCTAssertEqual(index.contract[StateFile.contractKey], StateFile.stateFingerprint(sA))
        let ref = try inf.scoreIndexed(query: query, index: index)
        let url = tmpURL(), streamURL = tmpURL()
        defer { try? FileManager.default.removeItem(at: url); try? FileManager.default.removeItem(at: streamURL) }
        try index.save(to: url)
        let streamed = try inf.buildIndexToDisk(docs: docs, at: streamURL)
        XCTAssertEqual(streamed.contract[StateFile.contractKey], StateFile.stateFingerprint(sA))

        bb.initialState = sB
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: index), "индекс с чужим состоянием принят")
        XCTAssertThrowsError(try inf.scoreIndexedBatch(queries: [query], index: index))
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: try DocIndex.load(url)))
        bb.initialState = nil
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: index), "индекс с состоянием принят без состояния")
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: streamed))

        bb.initialState = sA
        XCTAssertEqual(try inf.scoreIndexed(query: query, index: index), ref, "тот же индекс, то же состояние — другие скоры")
        XCTAssertEqual(try inf.scoreIndexed(query: query, index: try DocIndex.load(url)), ref)
        XCTAssertEqual(try inf.scoreIndexedBatch(queries: [query], index: index)[0], ref)
        XCTAssertNoThrow(try inf.scoreIndexed(query: query, index: streamed))

        // Индекс без состояния не принимается моделью с состоянием.
        bb.initialState = nil
        let plain = try inf.buildIndex(docs: docs)
        XCTAssertEqual(plain.contract[StateFile.contractKey], "none")
        bb.initialState = sA
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: plain))

        // `check` принимает и контракт без ключа: это «без состояния».
        var bare = inf.prefixContract()
        bare.removeValue(forKey: StateFile.contractKey)
        XCTAssertNoThrow(try plain.check(bare))
        XCTAssertThrowsError(try index.check(bare))
    }

    /// Состояние действительно меняет то, что кладётся в кэш: иначе отказ
    /// выше был бы придиркой, а не защитой. На крошечной модели вклад
    /// начального состояния к концу длинного текста затухает почти до нуля
    /// (скоры расходятся на 2e-7), поэтому — короткий префикс и конечное
    /// состояние, которое и хранит индекс.
    func testStateChangesCachedState() throws {
        let (model, bb) = try makeModel()
        let ids = MLXArray([Int32(5), 900, 31000], [1, 3])
        let plain = model.encode(ids).wkv
        bb.initialState = randomState(bb, seed: 3)
        let withState = model.encode(ids).wkv
        eval(plain, withState)
        let d = MLX.abs(plain - withState).max().item(Float.self)
        let scale = MLX.abs(plain).max().item(Float.self)
        XCTAssertGreaterThan(d, 0.1 * scale, "состояние не дошло до свёрнутого префикса: \(d) при \(scale)")
    }

    /// Индекс, записанный до появления ключа, — «без состояния»: читается и
    /// скорится моделью без состояния (побитно как новый), отвергается при
    /// подключённом.
    func testIndexWithoutStateKeyIsReadAsNone() throws {
        let tok = try worldTokenizer()
        let (model, bb) = try makeModel()
        let inf = RerankerInference(model: model, tokenizer: tok, config: serving())
        let fresh = try inf.buildIndex(docs: docs)
        var old = fresh.contract
        old.removeValue(forKey: StateFile.contractKey)
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try DocIndex(state: fresh.state, docs: fresh.docs, contract: old).save(to: url)
        let loaded = try DocIndex.load(url)
        XCTAssertNil(loaded.contract[StateFile.contractKey], "старый файл получил ключ")

        XCTAssertEqual(try inf.scoreIndexed(query: query, index: loaded),
                       try inf.scoreIndexed(query: query, index: fresh))
        bb.initialState = bb.zeroInitialState()
        XCTAssertNoThrow(try inf.scoreIndexed(query: query, index: loaded), "нулевое состояние — то же, что без него")
        bb.initialState = randomState(bb, seed: 5)
        XCTAssertThrowsError(try inf.scoreIndexed(query: query, index: loaded),
                             "старый индекс принят моделью с состоянием")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Кэш состояний пар
    // ─────────────────────────────────────────────────────────────────

    func pairs() throws -> (pool: [String], samples: [RerankSample]) {
        let rows = (0 ..< 4).map { i in
            RerankRow(instruct: "инструкция", query: "вопрос номер \(i)",
                      positive: "правильный пассаж про тему \(i)",
                      negatives: ["похожий но неверный пассаж \(i)"])
        }
        return try RerankCandidates.build(rows, nCandidates: 3, seed: 0)
    }

    func encCfg() -> RerankEncodeConfig {
        RerankEncodeConfig(maxDocTokens: 32, maxQueryTokens: 16, dtype: .float32)
    }

    /// Кэш пар пишет отпечаток; при другом состоянии (и без него) отказ, при
    /// том же — принят; голова с другим состоянием на нём не оценивается.
    func testPairCacheContractCarriesInitialState() throws {
        let tok = try worldTokenizer()
        let (model, bb) = try makeModel()
        let (pool, samples) = try pairs()
        let sA = randomState(bb, seed: 3), sB = randomState(bb, seed: 4)
        bb.initialState = sA
        let cache = try RerankEncoder.encodePairs(model, tokenizer: tok, pool: pool,
                                                  samples: samples, config: encCfg())
        let fpA = StateFile.stateFingerprint(sA)
        XCTAssertEqual(cache.contract[StateFile.contractKey], fpA)
        var want = encCfg().contract
        want[StateFile.contractKey] = fpA
        XCTAssertNoThrow(try cache.checkCompatible(head: model.head, contract: want))
        want[StateFile.contractKey] = StateFile.stateFingerprint(sB)
        XCTAssertThrowsError(try cache.checkCompatible(head: model.head, contract: want))
        want[StateFile.contractKey] = "none"
        XCTAssertThrowsError(try cache.checkCompatible(head: model.head, contract: want))

        // Без состояния — `none` в кэше.
        bb.initialState = nil
        let plain = try RerankEncoder.encodePairs(model, tokenizer: tok, pool: pool,
                                                  samples: samples, config: encCfg())
        XCTAssertEqual(plain.contract[StateFile.contractKey], "none")
        // Отложенный кэш с другим состоянием обучение не берёт.
        XCTAssertThrowsError(try RerankTraining.train(
            model, trainCache: plain, evalCache: cache,
            config: RerankTrainConfig(epochs: 1)), "отложенный кэш с другим состоянием принят")

        // Голова, записанная до появления ключа, — «без состояния».
        let head = tmpURL()
        defer { try? FileManager.default.removeItem(at: head) }
        try model.saveHead(to: head, extra: encCfg().contract.merging(["instruct": "инструкция"]) { a, _ in a })
        XCTAssertNoThrow(try RerankTraining.evaluate(base: bb, head: head, cache: plain))
        XCTAssertThrowsError(try RerankTraining.evaluate(base: bb, head: head, cache: cache),
                             "голова без состояния оценена на кэше с состоянием")

        // Голова, обученная на кэше с A (ключ в метаданных), — наоборот.
        let headA = tmpURL()
        defer { try? FileManager.default.removeItem(at: headA) }
        try model.saveHead(to: headA, extra: cache.contract.merging(["instruct": "инструкция"]) { a, _ in a })
        XCTAssertNoThrow(try RerankTraining.evaluate(base: bb, head: headA, cache: cache))
        XCTAssertThrowsError(try RerankTraining.evaluate(base: bb, head: headA, cache: plain),
                             "голова с состоянием оценена на кэше без него")
    }

    /// Кэш, собранный до ключа, — «без состояния»: принимается при `none`,
    /// отвергается при подключённом; слить его можно только с кэшем без
    /// состояния.
    func testPairCacheWithoutStateKey() throws {
        let (model, bb) = try makeModel()
        let H = bb.cfg.nHead, S = bb.cfg.headSize
        func cache(_ src: Int, _ contract: [String: String]) throws -> StateCache {
            let w = try StateCacheWriter(shape: [2, 1, H, S, S], dtype: .float16)
            try w.write(rows: [0, 1], MLXArray.zeros([2, 1, H, S, S]))
            return try w.finish(pairIndex: [[0, 1]], labels: [0], hardNegs: [[]],
                                contract: contract, sources: [src])
        }
        let base = ["template": "doc_first"]
        let old = try cache(bb.cfg.nLayer - 1, base)
        XCTAssertNoThrow(try old.checkCompatible(
            head: model.head, contract: base.merging([StateFile.contractKey: "none"]) { a, _ in a }))
        XCTAssertThrowsError(try old.checkCompatible(
            head: model.head, contract: base.merging([StateFile.contractKey: "sha256:ab"]) { a, _ in a }),
            "старый кэш принят моделью с состоянием")

        let noneNew = try cache(0, base.merging([StateFile.contractKey: "none"]) { a, _ in a })
        let withState = try cache(0, base.merging([StateFile.contractKey: "sha256:ab"]) { a, _ in a })
        let merged = try old.merged(with: noneNew)
        XCTAssertEqual(merged.contract[StateFile.contractKey], "none")
        XCTAssertThrowsError(try old.merged(with: withState), "слиты кэши с разным состоянием")
        XCTAssertThrowsError(try withState.merged(with: old))
    }
}
