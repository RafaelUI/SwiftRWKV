//
//  EmbeddingStateBankTests.swift
//  Банк состояний в эмбеддере (20.10): своё начальное состояние у каждого
//  запроса (например, на его инструкцию), документы без состояния. Эталоны:
//  автоград eager-пути на ТОМ ЖЕ составе батча для обратного прохода по
//  блокам, кусков по строкам и GradCache; строка запроса = она же одна с
//  этим состоянием, подключённым как одно; банк свёрнутых инструкций =
//  инструкция в строке.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding

final class EmbeddingStateBankTests: XCTestCase {

    func makeModel(vocab: Int = 64) -> EmbeddingModel {
        let cfg = TinyBackbone.config(nLayer: 3, vocab: vocab)
        let bb = X070Backbone(weights: TinyBackbone.weights(cfg, seed: 3), cfg: cfg)
        let head = EmbeddingHead(dim: cfg.nEmbd, seed: 3)
        MLXRandom.seed(103)
        head.fc2 = MLXRandom.normal([cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        return EmbeddingModel(backbone: bb, head: head, stateScope: .queriesOnly)
    }

    func randomBank(_ bb: X070Backbone, K: Int, seed: UInt64 = 6) -> RWKVBatchState {
        MLXRandom.seed(seed)
        let z = RWKVBatchState(cfg: bb.cfg, batch: K, dtype: .float32)
        return RWKVBatchState(wkv: MLXRandom.normal(z.wkv.shape) * 0.3,
                              tmixShift: MLXRandom.normal(z.tmixShift.shape) * 0.5,
                              cmixShift: MLXRandom.normal(z.cmixShift.shape) * 0.5).evaluated()
    }

    static let rows: [Int32] = [2, 0, 2, 1]

    func triplets(vocab: Int = 64) -> TripletBatch {
        MLXRandom.seed(5)
        let B = Self.rows.count
        func ids(_ T: Int) -> MLXArray { MLXRandom.randInt(0 ..< vocab, [B, T]).asType(.int32) }
        func pool(_ T: Int) -> MLXArray { MLXArray((0 ..< B).map { Int32(T - 1 - $0) }) }
        let b = TripletBatch(anchorIdx: ids(16), anchorPool: pool(16),
                             positiveIdx: ids(32), positivePool: pool(32),
                             negativeIdx: ids(48), negativePool: pool(48),
                             anchorRows: MLXArray(Self.rows))
        eval(b.anchorIdx, b.positiveIdx, b.negativeIdx)
        return b
    }

    func diff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        diff(ref, got) / (MLX.abs(ref.asType(.float32)).max().item(Float.self) + 1e-12)
    }

    func check(_ what: String, _ set: TrainableSet, _ ps: [MLXArray], bound: Float = 1e-5,
               eager: @escaping () -> MLXArray,
               other: () -> (loss: MLXArray, gradients: [MLXArray]),
               file: StaticString = #filePath, line: UInt = #line) {
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            set.inject(p)
            return [eager().asType(.float32)]
        }, argumentNumbers: Array(ps.indices))
        let (v, gRef) = vg(ps)
        eval(v + gRef)
        set.inject(ps)
        let r = other()
        eval([r.loss] + r.gradients)
        var worst: Float = 0, live = 0
        for (i, (a, b)) in zip(gRef, r.gradients).enumerated() {
            if MLX.abs(a).max().item(Float.self) == 0 {
                XCTAssertEqual(MLX.abs(b).max().item(Float.self), 0, "\(what): \(set.parameterNames[i])", file: file, line: line)
                continue
            }
            live += 1
            worst = max(worst, rel(a, b))
        }
        print(String(format: "БАНК ЭМБЕДДЕРА, %@: лосс %.6f против %.6f; худший градиент %.1e (%d с градиентом)",
                     what, r.loss.item(Float.self), v[0].item(Float.self), worst, live))
        XCTAssertLessThan(worst, bound, what, file: file, line: line)
        XCTAssertEqual(r.loss.item(Float.self), v[0].item(Float.self), accuracy: abs(v[0].item(Float.self)) * 1e-5,
                       what, file: file, line: line)
        XCTAssertGreaterThan(live, 0, "\(what): сверять нечего", file: file, line: line)
    }

    /// По блокам, куски по строкам и GradCache = автоград на том же составе:
    /// обучается банк (WKV; всё), голова при банке, вся база при банке.
    func testGradientsMatchAutograd() {
        for what in ["банк WKV", "банк со сдвигами", "голова", "вся база"] {
            let model = makeModel()
            let bb = model.backbone
            bb.initialStateBank = randomBank(bb, K: 4)
            let set: TrainableSet
            switch what {
            case "банк WKV": set = StateBankTrainableSet(bb)
            case "банк со сдвигами": set = StateBankTrainableSet(bb, train: StateTrainableSet.Part.allCases)
            case "голова": set = EmbeddingHeadTrainableSet(model.head)
            default: set = EmbeddingTrainable.make(model: model, mode: .full)
            }
            let ps = set.initialParameters()
            let b = triplets()
            let eager = { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: false) }
            check("\(what), по блокам", set, ps, eager: eager, other: {
                EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                      symmetric: false)(ps, b)
            })
            check("\(what), куски по 16 токенов", set, ps, eager: eager, other: {
                EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                      symmetric: false, chunkTokens: 16)(ps, b)
            })
            check("\(what), GradCache по 3 строки", set, ps, eager: eager, other: {
                EmbeddingObjective.gradCacheProvider(model: model, trainable: set, chunkSize: 3,
                                                     temperature: 0.05, symmetric: false)(ps, b)
            })
            if set is StateBankTrainableSet {
                // строка банка 3 в батче не встречается
                let g = EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                              symmetric: false)(ps, b).gradients
                for x in g { XCTAssertEqual(MLX.abs(x[3]).max().item(Float.self), 0, "\(what): строка 3 получила градиент") }
            }
            XCTAssertNil(bb.initialRows, "строки остались после шага")
        }
    }

    /// Запрос со строкой банка = он же один с этим состоянием, подключённым как
    /// одно; документ не видит банка (побитно как без состояния).
    func testQueryRowEqualsSingleStateAndDocumentsIgnoreBank() {
        let model = makeModel()
        let bb = model.backbone
        let bank = randomBank(bb, K: 3)
        let b = triplets()
        let plainDocs = model.embed(b.positiveIdx, poolIndex: b.positivePool, role: .document)
        bb.initialStateBank = bank
        let q = model.embed(b.anchorIdx, poolIndex: b.anchorPool, role: .query, rows: b.anchorRows)
        let docs = model.embed(b.positiveIdx, poolIndex: b.positivePool, role: .document)
        eval(q, docs)
        XCTAssertEqual(diff(docs, plainDocs), 0, "документ увидел банк")
        bb.initialStateBank = nil
        for (i, k) in Self.rows.enumerated() {
            bb.initialState = RWKVBatchState(wkv: bank.wkv[0..., Int(k) ..< Int(k) + 1],
                                             tmixShift: bank.tmixShift[0..., Int(k) ..< Int(k) + 1],
                                             cmixShift: bank.cmixShift[0..., Int(k) ..< Int(k) + 1])
            let one = model.embed(b.anchorIdx[i ..< i + 1], poolIndex: b.anchorPool[i ..< i + 1], role: .query)
            XCTAssertLessThan(diff(q[i ..< i + 1], one), 1e-5, "строка \(i)")
        }
    }

    private func tok() throws -> WorldTokenizer {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет словаря World")
        return try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: path)))
    }

    static let instr = ["Instruct: Pick the airline tool for the request.\n",
                        "Instruct: Выбери инструмент банка для запроса.\n"]

    func samples() -> (inline: [EmbeddingSample], banked: [EmbeddingSample]) {
        // номера 0, 0, 1, 1, 0: не палиндром и не период пачки из двух —
        // иначе переворот и сдвиг строк банка неотличимы
        let q = [(0, "Query: change my flight"), (0, "Query: cancel the trip"), (1, "Query: покажи баланс"),
                 (1, "Query: переведи деньги"), (0, "Query: baggage allowance")]
        let docs = ["change flight tool", "card balance tool", "cancel reservation tool",
                    "money transfer tool", "baggage policy tool"]
        var inline: [EmbeddingSample] = [], banked: [EmbeddingSample] = []
        for (i, (k, tail)) in q.enumerated() {
            let neg = docs[(i + 1) % docs.count]
            inline.append(.init(anchor: Self.instr[k] + tail, positive: docs[i], negative: neg, task: .retrieval))
            banked.append(.init(anchor: tail, positive: docs[i], negative: neg, task: .retrieval, state: k))
        }
        return (inline, banked)
    }

    /// Банк свёрнутых инструкций: оценка запросов «Query: q» со строкой банка
    /// = оценка «Instruct: X\nQuery: q» в строке (те же ранги).
    func testFoldedBankEvaluationEqualsInline() throws {
        let t = try tok()
        let model = makeModel(vocab: 65536)
        let (inline, banked) = samples()
        var ranks: [[Double]] = []
        EmbeddingMetrics.perRowObserver = { ranks.append($0) }
        defer { EmbeddingMetrics.perRowObserver = nil }
        let ref = EmbeddingMetrics.evaluateRetrieval(model: model, tokenizer: t, rows: inline)
        let inlineVec = EmbeddingMetrics.embedAll(model: model, tokenizer: t, texts: inline.map(\.anchor), role: .query)
        let folded = InstructionCache(model: model, tokenizer: t).bank(Self.instr)   // до подключения банка
        model.backbone.initialStateBank = folded
        let got = EmbeddingMetrics.evaluateRetrieval(model: model, tokenizer: t, rows: banked)
        XCTAssertEqual(ranks[0], ranks[1])
        XCTAssertEqual(got.mrr, ref.mrr, accuracy: 1e-9)
        // векторы, а не только ранги: на пяти запросах ранги к перестановке
        // строк банка нечувствительны
        let bankVec = EmbeddingMetrics.embedAll(model: model, tokenizer: t, texts: banked.map(\.anchor), role: .query,
                                                states: banked.map { $0.state! })
        XCTAssertLessThan(diff(bankVec, inlineVec), 1e-5, "запрос со строкой банка ≠ инструкции в строке")
        // проверка теста: без банка (нули) ранги другие — инструкция влияет
        model.backbone.initialStateBank = RWKVBatchState(cfg: model.backbone.cfg, batch: 2, dtype: .float32)
        _ = EmbeddingMetrics.evaluateRetrieval(model: model, tokenizer: t, rows: banked)
        let a = EmbeddingMetrics.embedAll(model: model, tokenizer: t, texts: banked.map(\.anchor), role: .query,
                                          states: banked.map { $0.state! })
        model.backbone.initialStateBank = folded
        let b = EmbeddingMetrics.embedAll(model: model, tokenizer: t, texts: banked.map(\.anchor), role: .query,
                                          states: banked.map { $0.state! })
        XCTAssertGreaterThan(diff(a, b), 1e-3, "банк не влияет на векторы — проверка ничего не стоит")
        // по пачкам из двух строк — строки банка у каждой пачки свои
        let small = EmbeddingMetrics.embedAll(model: model, tokenizer: t, texts: banked.map(\.anchor), batchSize: 2,
                                              role: .query, states: banked.map { $0.state! })
        XCTAssertLessThan(diff(small, b), 1e-5, "пачки взяли не свои строки банка")
    }

    /// Батчер несёт номера состояний в порядке строк.
    func testBatcherCarriesStates() throws {
        let t = try tok()
        let (_, banked) = samples()
        var batcher = TripletBatcher(samples: banked, tokenizer: t, batchSize: 4, shuffle: false)
        let b = batcher.next()
        XCTAssertEqual(b.anchorRows?.asArray(Int32.self), [0, 0, 1, 1])
        var plain = TripletBatcher(samples: samples().inline, tokenizer: t, batchSize: 3, shuffle: false)
        XCTAssertNil(plain.next().anchorRows)
    }

    /// Стадия обучает банк: строки банка, встреченные в обучении, сдвигаются,
    /// не встреченная — нет (побитно); оценка до обучения — банк свёрнутых
    /// инструкций = инструкция в строке.
    func testStageTrainsOnlySeenRows() throws {
        let t = try tok()
        let model = makeModel(vocab: 65536)
        let bb = model.backbone
        let (_, banked) = samples()
        let prefixes = Self.instr + ["Instruct: an instruction no training row has.\n"]
        let start = InstructionCache(model: model, tokenizer: t).bank(prefixes)
        let set = StateBankTrainableSet(bb, initial: start)
        var stage = EmbeddingStage(task: .retrieval, train: banked, heldOut: banked,
                                   config: TrainingConfig(lr: 1e-2, warmupSteps: 0, maxSteps: 3), batchSize: 2)
        stage.maxTokens = 64
        _ = try EmbeddingFinetune.runStage(model: model, tokenizer: t, trainable: set, stage: stage)
        let after = bb.initialStateBank!
        for k in 0 ..< 2 {
            XCTAssertGreaterThan(diff(after.wkv[0..., k], start.wkv[0..., k]), 0, "строка \(k) не обучилась")
        }
        XCTAssertEqual(diff(after.wkv[0..., 2], start.wkv[0..., 2]), 0, "не встреченная строка сдвинулась")
        XCTAssertNil(bb.initialRows)
    }
}
