//
//  InstructionCacheTests.swift
//  Свёрнутые инструкции эмбеддера (`InstructionCache`, 19.10): инструкция
//  запроса сворачивается в состояние один раз и подключается к запросам этой
//  инструкции. Главное свойство: при той же токенизации кусков — тот же
//  вектор, что с инструкцией в строке (до порядка сложения на стыке).
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding

final class InstructionCacheTests: XCTestCase {

    func model(pooling: Pooling = .last, scope: InitialStateScope = .allTexts) -> EmbeddingModel {
        let (bb, _) = TinyBackbone.make(nLayer: 3, vocab: 65536)
        let head = EmbeddingHead(dim: bb.cfg.nEmbd, seed: 6)
        MLXRandom.seed(106)
        head.fc2 = MLXRandom.normal([bb.cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        return EmbeddingModel(backbone: bb, head: head, pooling: pooling, stateScope: scope)
    }

    func randomState(_ bb: X070Backbone, seed: UInt64 = 23) -> RWKVBatchState {
        MLXRandom.seed(seed)
        let z = bb.zeroInitialState()
        let s = RWKVBatchState(wkv: MLXRandom.normal(z.wkv.shape) * 0.05,
                               tmixShift: MLXRandom.normal(z.tmixShift.shape) * 0.5,
                               cmixShift: MLXRandom.normal(z.cmixShift.shape) * 0.5)
        eval(s.wkv, s.tmixShift, s.cmixShift)
        return s
    }

    private func tok() throws -> WorldTokenizer {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет словаря World")
        return try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: path)))
    }

    func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    static let instrA = "Instruct: You are an airline agent. Follow the policy: verify the user id before any change.\n"
    static let instrB = "Instruct: Выбери инструмент для запроса пользователя банка.\n"

    /// Вперемешку: две инструкции и строки без инструкции.
    static let anchors = [
        instrA + "Query: I want to change my flight to Monday",
        "просто запрос без инструкции",
        instrB + "Query: Покажи баланс карты",
        instrA + "Query: cancel reservation ABC123, user id mia_li_3668",
        instrB + "Query: переведи 500 рублей маме",
        "another plain query",
        instrA + "Query: how many bags can I take?",
    ]

    /// Свёрнутая инструкция = инструкция в строке, построчно и в исходном
    /// порядке; токенизация кусков совпадает со сплошной (условие равенства).
    func testFoldedMatchesInline() throws {
        let t = try tok(), m = model()
        for a in Self.anchors {
            guard let (p, q) = InstructionCache.split(a) else { continue }
            XCTAssertEqual(t.encode(p) + t.encode(q), t.encode(a), "токенизация кусков разошлась: \(a)")
        }
        let inline = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: Self.anchors, role: .query)
        let cache = InstructionCache(model: m, tokenizer: t)
        let folded = try cache.embedQueries(Self.anchors)
        XCTAssertEqual(folded.shape, inline.shape)
        XCTAssertLessThan(maxDiff(folded, inline), 1e-5)
        // строки без инструкции — побитно как есть на ТОМ ЖЕ составе батча
        // (длина добивки меняет последние биты, поэтому эталон — они одни)
        let plain = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: [Self.anchors[1], Self.anchors[5]],
                                              role: .query)
        XCTAssertEqual(maxDiff(folded[1], plain[0]), 0)
        XCTAssertEqual(maxDiff(folded[5], plain[1]), 0)
        // порядок: каждая строка ближе всего к своей
        let sims = matmul(folded, inline.transposed()).asArray(Float.self)
        let n = Self.anchors.count
        for i in 0 ..< n {
            let row = Array(sims[i * n ..< (i + 1) * n])
            XCTAssertEqual(row.firstIndex(of: row.max()!), i, "строка \(i) не на своём месте")
        }
    }

    /// Счётчики: одна свёртка на различную инструкцию; повторный вызов не
    /// сворачивает заново.
    func testCountersAndMemo() throws {
        let t = try tok(), m = model()
        let cache = InstructionCache(model: m, tokenizer: t)
        _ = try cache.embedQueries(Self.anchors)
        XCTAssertEqual(cache.foldedCount, 2)
        XCTAssertEqual(cache.foldedTokens, t.encode(Self.instrA).count + t.encode(Self.instrB).count)
        _ = try cache.embedQueries(Self.anchors)
        XCTAssertEqual(cache.foldedCount, 2, "свёрнуто заново без смены состояния")
    }

    /// Подключённое состояние после вызова — ровно прежнее (те же массивы);
    /// без состояния — по-прежнему без.
    func testAttachedStateRestored() throws {
        let t = try tok(), m = model()
        let bb = m.backbone
        _ = try InstructionCache(model: m, tokenizer: t).embedQueries(Self.anchors)
        XCTAssertNil(bb.initialState, "после вызова осталось подключённым свёрнутое")
        bb.initialState = randomState(bb)
        let before = bb.initialLayers!
        _ = try InstructionCache(model: m, tokenizer: t).embedQueries(Self.anchors)
        let after = bb.initialLayers!
        for (x, y) in zip(before, after) {
            XCTAssertTrue(x.wkv === y.wkv && x.tmixShift === y.tmixShift && x.cmixShift === y.cmixShift,
                          "подключённое состояние подменено копией или свёрнутым")
        }
    }

    /// `withInitialState` возвращает прежнее и при ошибке.
    func testWithInitialStateRestoresOnError() {
        let (bb, _) = TinyBackbone.make(nLayer: 2)
        let s0 = randomState(bb, seed: 1), s1 = randomState(bb, seed: 2)
        bb.initialState = s0
        let fp = bb.initialStateFingerprint
        struct Boom: Error {}
        XCTAssertThrowsError(try bb.withInitialState(s1) {
            XCTAssertNotEqual(bb.initialStateFingerprint, fp)
            throw Boom()
        })
        XCTAssertEqual(bb.initialStateFingerprint, fp)
    }

    /// Свёртка стартует с подключённого («обученное → инструкция → запрос»):
    /// с подключённым состоянием свёрнутая = в строке с ним же; смена
    /// подключённого — пересвёртка.
    func testFoldStartsFromAttachedAndRefolds() throws {
        let t = try tok(), m = model()
        let bb = m.backbone
        let cache = InstructionCache(model: m, tokenizer: t)
        _ = try cache.embedQueries(Self.anchors)
        XCTAssertEqual(cache.foldedCount, 2)
        bb.initialState = randomState(bb)
        let inline = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: Self.anchors, role: .query)
        let folded = try cache.embedQueries(Self.anchors)
        XCTAssertEqual(cache.foldedCount, 4, "подключённое сменилось, а свёрнутое старое")
        XCTAssertLessThan(maxDiff(folded, inline), 1e-5)
    }

    /// Состояние только для запросов: свёртка — запрос, она со состоянием.
    func testQueriesOnlyScope() throws {
        let t = try tok(), m = model(scope: .queriesOnly)
        m.backbone.initialState = randomState(m.backbone)
        let inline = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: Self.anchors, role: .query)
        let folded = try InstructionCache(model: m, tokenizer: t).embedQueries(Self.anchors)
        XCTAssertLessThan(maxDiff(folded, inline), 1e-5)
    }

    /// Среднее по хвосту не равно среднему по строке — отказ.
    func testMeanPoolingRefused() throws {
        let t = try tok(), m = model(pooling: .mean)
        XCTAssertThrowsError(try InstructionCache(model: m, tokenizer: t).embedQueries(Self.anchors))
    }

    /// Разрез — после «\n»: префикс кончается «\n», хвост начинается «Query: ».
    func testSplit() {
        let s = InstructionCache.split("Instruct: X\nY\nQuery: q\nQuery: r")
        XCTAssertEqual(s?.prefix, "Instruct: X\nY\n")
        XCTAssertEqual(s?.tail, "Query: q\nQuery: r")
        XCTAssertNil(InstructionCache.split("Query: q"))
        XCTAssertNil(InstructionCache.split("Instruct: без запроса"))
    }

    /// `maxTokens` режет хвост: длинная инструкция запрос не вытесняет.
    func testMaxTokensCutsTail() throws {
        let t = try tok(), m = model()
        let longInstr = "Instruct: " + String(repeating: "policy rule number seven applies here. ", count: 30) + "\n"
        let a = [longInstr + "Query: first query", longInstr + "Query: second, different query"]
        let folded = try InstructionCache(model: m, tokenizer: t).embedQueries(a, maxTokens: 32)
        XCTAssertGreaterThan(maxDiff(folded[0], folded[1]), 1e-3, "хвосты обрезаны, запросы неразличимы")
        let inline = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: a, maxTokens: 32, role: .query)
        XCTAssertEqual(maxDiff(inline[0], inline[1]), 0, "в строке запрос обязан быть вытеснен (проверка теста)")
    }

    /// Оценка и стадия: со свёрткой — те же метрики, кэш виден наблюдателю.
    func testEvaluateRetrievalFolded() throws {
        let t = try tok(), m = model()
        let docs = ["change flight date tool", "cancel reservation tool", "baggage policy tool",
                    "card balance tool", "money transfer tool"]
        let rows: [EmbeddingSample] = [
            .init(anchor: Self.anchors[0], positive: docs[0], negative: docs[1], task: .retrieval),
            .init(anchor: Self.anchors[2], positive: docs[3], negative: docs[4], task: .retrieval),
            .init(anchor: Self.anchors[3], positive: docs[1], negative: docs[2], task: .retrieval),
            .init(anchor: Self.anchors[4], positive: docs[4], negative: docs[3], task: .retrieval),
            .init(anchor: Self.anchors[6], positive: docs[2], negative: docs[0], task: .retrieval),
        ]
        var seen: InstructionCache? = nil
        EmbeddingMetrics.instructionCacheObserver = { seen = $0 }
        defer { EmbeddingMetrics.instructionCacheObserver = nil }
        var inlineRanks: [Double] = [], foldedRanks: [Double] = []
        EmbeddingMetrics.perRowObserver = { inlineRanks = $0 }
        _ = EmbeddingMetrics.evaluateRetrieval(model: m, tokenizer: t, rows: rows, dedupeCandidates: true)
        XCTAssertNil(seen, "без флага свёртки быть не должно")
        EmbeddingMetrics.perRowObserver = { foldedRanks = $0 }
        defer { EmbeddingMetrics.perRowObserver = nil }
        var stage = EmbeddingStage(task: .retrieval, train: rows, heldOut: rows,
                                   config: TrainingConfig(lr: 1e-3, maxSteps: 1), batchSize: 2)
        stage.dedupeEvaluationPool = true
        stage.foldInstructions = true
        _ = EmbeddingFinetune.evaluate(model: m, tokenizer: t, stage: stage)
        XCTAssertEqual(seen?.foldedCount, 2, "стадия не передала флаг")
        XCTAssertEqual(foldedRanks, inlineRanks)
    }

    /// Свёртка меняет числа оценки (обрезка хвоста) — входит в отпечаток
    /// плана curriculum; без неё отпечаток прежний.
    func testFoldInPlanFingerprint() {
        let rows = [EmbeddingSample(anchor: "a", positive: "b", negative: "c", task: .retrieval)]
        var st = EmbeddingStage(task: .retrieval, train: rows, heldOut: rows,
                                config: TrainingConfig(lr: 1e-3, maxSteps: 1), batchSize: 2)
        func fp() -> String {
            EmbeddingCurriculum.planFingerprint(mode: .frozen, stateScope: .allTexts, stages: [st], terminator: 0)
        }
        let plain = fp()
        st.foldInstructions = true
        XCTAssertNotEqual(fp(), plain)
    }

    /// Старт свёртки — подключённое, на КОРОТКОЙ инструкции: на длинной
    /// влияние начального состояния у крошечной случайной модели гаснет до
    /// 1e-5, и свёртка «с нуля» была неотличима (мутация не ловилась).
    func testFoldStartsFromAttachedShortInstruction() throws {
        let t = try tok(), m = model()
        let a = ["Instruct: A\nQuery: q one", "Instruct: A\nQuery: q two", "Instruct: B\nQuery: q three"]
        let none = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: a, role: .query)
        m.backbone.initialState = randomState(m.backbone)
        let inline = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: a, role: .query)
        // влияние ~1e-3 — на два порядка выше допуска; порог проверки — 10 допусков
        XCTAssertGreaterThan(maxDiff(inline, none), 1e-4, "состояние не влияет — проверка ничего не стоит")
        let folded = try InstructionCache(model: m, tokenizer: t).embedQueries(a)
        XCTAssertLessThan(maxDiff(folded, inline), 1e-5)
    }

    /// Свёртка кусками (сцепленными состоянием) = одним проходом, и с
    /// подключённым состоянием тоже; куски мельче инструкции.
    func testChunkedFoldMatchesWhole() throws {
        let t = try tok(), m = model()
        m.backbone.initialState = randomState(m.backbone)
        let whole = InstructionCache(model: m, tokenizer: t)
        whole.foldChunkTokens = 0
        let chunked = InstructionCache(model: m, tokenizer: t)
        chunked.foldChunkTokens = 5
        XCTAssertGreaterThan(t.encode(Self.instrA).count, 3 * chunked.foldChunkTokens, "кусков слишком мало")
        let a = whole.state(forPrefix: Self.instrA), b = chunked.state(forPrefix: Self.instrA)
        XCTAssertLessThan(maxDiff(a.wkv, b.wkv), 1e-5)
        XCTAssertLessThan(maxDiff(a.tmixShift, b.tmixShift), 1e-5)
        XCTAssertLessThan(maxDiff(a.cmixShift, b.cmixShift), 1e-5)
        let inline = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: Self.anchors, role: .query)
        XCTAssertLessThan(maxDiff(try chunked.embedQueries(Self.anchors), inline), 1e-5)
        XCTAssertEqual(chunked.foldedTokens, t.encode(Self.instrA).count + t.encode(Self.instrB).count,
                       "кусок посчитан не тот или дважды")
    }
}
