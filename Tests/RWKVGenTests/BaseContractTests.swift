//
//  BaseContractTests.swift
//  Отпечаток БАЗЫ в контрактах кэшей и чекпоинтов (25.10).
//
//  Кэш пар и индекс реранкера хранят состояния — функцию весов базы; головы
//  реранкера и эмбеддера и LoRA-адаптеры — поправки к ЭТИМ весам. На другой
//  базе формы те же, ошибок нет, числа чужие. Поэтому:
//    - ключ `base_fingerprint` пишется везде, где пишется результат;
//    - другая база (ключ есть и не тот) — отказ; `allowOtherBase` — обход
//      там, где перенос осмыслен (головы, адаптеры);
//    - файл без ключа (до правки, rwkv-metal) по базе не сверяется.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank
@testable import RWKVEmbedding

final class BaseContractTests: XCTestCase {

    let key = StateFile.baseContractKey

    /// Две базы одной геометрии: разные веса, одинаковые формы.
    func bases(nLayer: Int = 4) -> (X070Backbone, X070Backbone) {
        let (a, _) = TinyBackbone.make(nLayer: nLayer, vocab: 65536, seed: 42)
        let (b, _) = TinyBackbone.make(nLayer: nLayer, vocab: 65536, seed: 43)
        return (a, b)
    }

    func reranker(_ bb: X070Backbone) throws -> Reranker {
        let m = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        MLXRandom.seed(17)
        m.head.fc2Weight = MLXRandom.normal(m.head.fc2Weight.shape) * 0.3
        eval(m.head.fc2Weight)
        return m
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

    func tmpURL() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bct_\(UUID().uuidString).safetensors")
    }

    /// Переписать файл без ключа базы — как записанный до правки.
    func stripBase(_ url: URL) throws {
        let (w, md) = try loadArraysAndMetadata(url: url)
        var m = md; m.removeValue(forKey: key)
        try save(arrays: w, metadata: m, url: url)
    }

    /// Отказ — именно по базе, а не по чему-то попутному.
    func namesBase(_ e: Error, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue("\(e)".contains(key), "отказ не по базе: \(e)", file: file, line: line)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Отпечаток модели
    // ─────────────────────────────────────────────────────────────────

    /// Свойство модели = `StateFile.baseFingerprint`; запоминается, но после
    /// записи в веса пересчитывается; разные базы — разные отпечатки.
    func testBackboneFingerprintFollowsWeights() {
        let (a, b) = bases(nLayer: 2)
        let fa = a.baseFingerprint
        XCTAssertEqual(fa, StateFile.baseFingerprint(a))
        XCTAssertNotEqual(fa, b.baseFingerprint, "две разные базы с одним отпечатком")
        XCTAssertEqual(a.baseFingerprint, fa, "повторный вызов дал другое")
        a.w["ln_out.weight"] = a.w["ln_out.weight"]! * 2
        XCTAssertNotEqual(a.baseFingerprint, fa, "отпечаток не увидел записи в веса (запомнен навсегда)")
        XCTAssertEqual(a.baseFingerprint, StateFile.baseFingerprint(a))
    }

    /// База QLoRA (без плотной копии проекций) — отпечаток считается (раньше
    /// `weight(_:)` падал) и отличается от плотной базы.
    func testQLoRABaseHasFingerprint() {
        let (a, _) = bases(nLayer: 2)
        let dense = a.baseFingerprint
        _ = LoRA.add(to: a, spec: LoRASpec(rank: 4, alpha: 8, quantizeBits: 4))
        XCTAssertNotNil(a.quant["blocks.0.tmix.r_proj.weight"], "r_proj не квантован — тест ничего не проверяет")
        XCTAssertNil(a.w["blocks.0.tmix.r_proj.weight"])
        let q = a.baseFingerprint
        XCTAssertNotEqual(q, dense, "квантованная база неотличима от плотной")
        XCTAssertEqual(q, StateFile.baseFingerprint(a))
    }

    // ─────────────────────────────────────────────────────────────────
    //  Кэш пар и обучение головы
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

    /// Кэш пишет отпечаток базы; голова на другой базе на нём не учится и не
    /// оценивается, на той же — учится; кэш без ключа по базе не сверяется.
    func testPairCacheCarriesBase() throws {
        let tok = try worldTokenizer()
        let (a, b) = bases()
        let mA = try reranker(a), mB = try reranker(b)
        let (pool, samples) = try pairs()
        let cache = try RerankEncoder.encodePairs(mA, tokenizer: tok, pool: pool,
                                                  samples: samples, config: encCfg())
        XCTAssertEqual(cache.contract[key], a.baseFingerprint)

        let cfg = RerankTrainConfig(batchSize: 2, epochs: 1)
        XCTAssertNoThrow(try RerankTraining.train(mA, trainCache: cache, evalCache: nil, config: cfg))
        XCTAssertThrowsError(try RerankTraining.train(mB, trainCache: cache, evalCache: nil, config: cfg),
                             "голова на базе B обучена на кэше базы A") { self.namesBase($0) }
        XCTAssertThrowsError(try RerankTraining.evaluate(mB, cache: cache),
                             "голова на базе B оценена на кэше базы A") { self.namesBase($0) }
        XCTAssertNoThrow(try RerankTraining.evaluate(mA, cache: cache))
        // Явный контракт без ключа базы — база модели подставляется сама.
        XCTAssertThrowsError(try RerankTraining.evaluate(mB, cache: cache, contract: encCfg().contract)) { self.namesBase($0) }

        // Отложенный кэш другой базы — отказ даже при верном обучающем.
        let cacheB = try RerankEncoder.encodePairs(mB, tokenizer: tok, pool: pool,
                                                   samples: samples, config: encCfg())
        XCTAssertThrowsError(try RerankTraining.train(mA, trainCache: cache, evalCache: cacheB, config: cfg),
                             "отложенный кэш другой базы принят") { self.namesBase($0) }
        XCTAssertThrowsError(try cache.merged(with: cacheB), "слиты кэши разных баз") { self.namesBase($0) }

        // Кэш без ключа (до правки): по базе не сверяется.
        let H = a.cfg.nHead, S = a.cfg.headSize
        let w = try StateCacheWriter(shape: [2, 1, H, S, S], dtype: .float16)
        try w.write(rows: [0, 1], MLXArray.zeros([2, 1, H, S, S]))
        let old = try w.finish(pairIndex: [[0, 1]], labels: [0], hardNegs: [[]],
                               contract: ["template": "doc_first"], sources: [a.cfg.nLayer - 1])
        XCTAssertNoThrow(try old.checkCompatible(head: mB.head, contract: RerankTraining.expected(nil, cache: old, mB)))
        XCTAssertNoThrow(try old.checkCompatible(head: mA.head, contract: RerankTraining.expected(nil, cache: old, mA)))
        // а тот же кэш с ключом базы A голову B не принимает
        let wA = try StateCacheWriter(shape: [2, 1, H, S, S], dtype: .float16)
        try wA.write(rows: [0, 1], MLXArray.zeros([2, 1, H, S, S]))
        let keyed = try wA.finish(pairIndex: [[0, 1]], labels: [0], hardNegs: [[]],
                                  contract: ["template": "doc_first", key: a.baseFingerprint],
                                  sources: [a.cfg.nLayer - 1])
        XCTAssertThrowsError(try keyed.checkCompatible(head: mB.head, contract: RerankTraining.expected(nil, cache: keyed, mB))) { self.namesBase($0) }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Индекс выдачи
    // ─────────────────────────────────────────────────────────────────

    let docs = [
        "Пчёлы опыляют растения и производят мёд из нектара цветов.",
        "Паровая машина Уатта запустила промышленную революцию в Англии.",
    ]

    func serving() -> RerankServingConfig {
        RerankServingConfig(
            encode: RerankEncodeConfig(maxDocTokens: 48, maxQueryTokens: 16,
                                       docBatch: 2, queryBatch: 3),
            instruct: "найди документ")
    }

    /// Индекс базы A моделью на базе B не скорится (из памяти, с диска,
    /// потоковый); той же базой — скоры побитно те же; индекс без ключа
    /// принимается.
    func testIndexCarriesBase() throws {
        let tok = try worldTokenizer()
        let (a, b) = bases()
        let infA = RerankerInference(model: try reranker(a), tokenizer: tok, config: serving())
        let infB = RerankerInference(model: try reranker(b), tokenizer: tok, config: serving())
        let index = try infA.buildIndex(docs: docs)
        XCTAssertEqual(index.contract[key], a.baseFingerprint)
        let url = tmpURL(), streamURL = tmpURL()
        defer { try? FileManager.default.removeItem(at: url); try? FileManager.default.removeItem(at: streamURL) }
        try index.save(to: url)
        let streamed = try infA.buildIndexToDisk(docs: docs, at: streamURL)
        XCTAssertEqual(streamed.contract[key], a.baseFingerprint)

        let ref = try infA.scoreIndexed(query: "мёд", index: index)
        XCTAssertEqual(try infA.scoreIndexed(query: "мёд", index: try DocIndex.load(url)), ref)
        XCTAssertThrowsError(try infB.scoreIndexed(query: "мёд", index: index), "индекс базы A принят базой B") { self.namesBase($0) }
        XCTAssertThrowsError(try infB.scoreIndexed(query: "мёд", index: try DocIndex.load(url)))
        XCTAssertThrowsError(try infB.scoreIndexed(query: "мёд", index: streamed))
        XCTAssertThrowsError(try infB.scoreIndexedBatch(queries: ["мёд"], index: index))

        var old = index.contract
        old.removeValue(forKey: key)
        let legacy = DocIndex(state: index.state, docs: index.docs, contract: old)
        XCTAssertNoThrow(try infB.scoreIndexed(query: "мёд", index: legacy))
        XCTAssertEqual(try infA.scoreIndexed(query: "мёд", index: legacy), ref)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Головы и адаптеры
    // ─────────────────────────────────────────────────────────────────

    /// Голова реранкера: отпечаток в файле; на другой базе — отказ (и в
    /// `fromHead`, и в оценке по чекпоинту), с `allowOtherBase` — загрузка с
    /// теми же весами; файл без ключа — без сверки.
    func testRerankerHeadCarriesBase() throws {
        let (a, b) = bases()
        let mA = try reranker(a)
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try mA.saveHead(to: url)
        XCTAssertEqual(try Reranker.readHeadMetadata(url)[key], a.baseFingerprint)

        XCTAssertNoThrow(try Reranker.fromHead(base: a, url: url))
        XCTAssertThrowsError(try Reranker.fromHead(base: b, url: url)) { e in
            guard case RerankerError.otherBase = e else { return XCTFail("не та ошибка: \(e)") }
        }
        let other = try Reranker.fromHead(base: b, url: url, allowOtherBase: true)
        XCTAssertEqual(other.head.fc2Weight.asArray(Float.self), mA.head.fc2Weight.asArray(Float.self))

        // `extra` (контракт обучающего кэша) перекрывает базу модели.
        let viaExtra = tmpURL()
        defer { try? FileManager.default.removeItem(at: viaExtra) }
        try mA.saveHead(to: viaExtra, extra: [key: b.baseFingerprint])
        XCTAssertNoThrow(try Reranker.fromHead(base: b, url: viaExtra))
        XCTAssertThrowsError(try Reranker.fromHead(base: a, url: viaExtra))

        // Оценка по чекпоинту: голова A, модель B, кэш B — отказ без обхода,
        // с обходом — кэш сверяется с базой модели.
        let H = a.cfg.nHead, S = a.cfg.headSize
        func cache(_ base: X070Backbone) throws -> StateCache {
            let w = try StateCacheWriter(shape: [2, 1, H, S, S], dtype: .float16)
            try w.write(rows: [0, 1], MLXArray.zeros([2, 1, H, S, S]))
            return try w.finish(pairIndex: [[0, 1]], labels: [0], hardNegs: [[]],
                                contract: [key: base.baseFingerprint], sources: [a.cfg.nLayer - 1])
        }
        let cA = try cache(a), cB = try cache(b)
        XCTAssertNoThrow(try RerankTraining.evaluate(base: a, head: url, cache: cA))
        XCTAssertThrowsError(try RerankTraining.evaluate(base: b, head: url, cache: cB))
        XCTAssertNoThrow(try RerankTraining.evaluate(base: b, head: url, cache: cB, allowOtherBase: true))
        XCTAssertThrowsError(try RerankTraining.evaluate(base: b, head: url, cache: cA, allowOtherBase: true),
                             "кэш с базы головы, а не модели, принят при allowOtherBase")

        try stripBase(url)
        XCTAssertNoThrow(try Reranker.fromHead(base: b, url: url), "голова без ключа сверена по базе")
    }

    /// Голова эмбеддера: то же.
    func testEmbeddingHeadCarriesBase() throws {
        let (a, b) = bases(nLayer: 2)
        let mA = EmbeddingModel(backbone: a, head: EmbeddingHead(dim: a.cfg.nEmbd, seed: 3))
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try mA.saveHead(to: url)
        XCTAssertEqual(try EmbeddingModel.readHeadMetadata(url)[key], a.baseFingerprint)
        XCTAssertNoThrow(try EmbeddingModel.fromHead(backbone: a, url: url))
        XCTAssertThrowsError(try EmbeddingModel.fromHead(backbone: b, url: url)) { e in
            guard case EmbeddingCheckpointError.otherBase = e else { return XCTFail("не та ошибка: \(e)") }
        }
        XCTAssertThrowsError(try EmbeddingModel(backbone: b).loadHead(from: url))
        let other = try EmbeddingModel.fromHead(backbone: b, url: url, allowOtherBase: true).model
        XCTAssertEqual(other.head.fc1.asArray(Float.self), mA.head.fc1.asArray(Float.self))
        try stripBase(url)
        XCTAssertNoThrow(try EmbeddingModel.fromHead(backbone: b, url: url))
    }

    /// Эмбеддер по чекпоинту (выдача) — та же сверка и тот же обход.
    func testEmbedderFromCheckpointChecksBase() throws {
        let tok = try worldTokenizer()
        let (a, b) = bases(nLayer: 2)
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try EmbeddingModel(backbone: a, head: EmbeddingHead(dim: a.cfg.nEmbd, seed: 3)).saveHead(to: url)
        XCTAssertNoThrow(try Embedder.fromCheckpoint(backbone: a, tokenizer: tok, head: url))
        XCTAssertThrowsError(try Embedder.fromCheckpoint(backbone: b, tokenizer: tok, head: url)) { self.namesBase($0) }
        XCTAssertNoThrow(try Embedder.fromCheckpoint(backbone: b, tokenizer: tok, head: url, allowOtherBase: true))
    }

    /// Адаптеры: отпечаток в файле; на другой базе — отказ, с обходом и на
    /// своей — те же числа; файл без ключа — без сверки; QLoRA-база —
    /// круг через диск.
    func testLoRACarriesBase() throws {
        let (a, b) = bases(nLayer: 2)
        let spec = LoRASpec(rank: 4, alpha: 8)
        _ = LoRA.add(to: a, spec: spec)
        _ = LoRA.add(to: b, spec: spec)
        MLXRandom.seed(5)
        for k in Array(a.loraB.keys) { a.loraB[k] = MLXRandom.normal(a.loraB[k]!.shape) * 0.1 }
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try LoRA.save(a, to: url)
        XCTAssertEqual(try loadArraysAndMetadata(url: url).1[key], a.baseFingerprint)

        XCTAssertThrowsError(try LoRA.load(b, from: url)) { e in
            guard case LoRA.LoadError.otherBase = e else { return XCTFail("не та ошибка: \(e)") }
        }
        try LoRA.load(b, from: url, allowOtherBase: true)
        let k = a.loraB.keys.sorted()[0]
        XCTAssertEqual(b.loraB[k]!.asArray(Float.self), a.loraB[k]!.asArray(Float.self))
        XCTAssertNoThrow(try LoRA.load(a, from: url))
        try stripBase(url)
        let (c, _) = TinyBackbone.make(nLayer: 2, vocab: 65536, seed: 44)
        _ = LoRA.add(to: c, spec: spec)
        XCTAssertNoThrow(try LoRA.load(c, from: url), "адаптеры без ключа сверены по базе")

        let (q, _) = TinyBackbone.make(nLayer: 2, vocab: 65536, seed: 45)
        _ = LoRA.add(to: q, spec: LoRASpec(rank: 4, alpha: 8, quantizeBits: 4))
        let qURL = tmpURL()
        defer { try? FileManager.default.removeItem(at: qURL) }
        try LoRA.save(q, to: qURL)
        XCTAssertNoThrow(try LoRA.load(q, from: qURL))
    }
}
