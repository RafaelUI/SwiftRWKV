import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Эмбеддер: градиент обратным проходом по блокам
//  (`EmbeddingObjective.sequentialValueAndGrad`) обязан быть тем же, что у
//  автограда по всему графу, — во всех режимах обучения базы, на обеих
//  формах задачи и по КАЖДОМУ параметру: блоков, основания (ln0), ln_out,
//  головы, адаптеров.
// ───────────────────────────────────────────────────────────────────────

final class EmbeddingSequentialTests: XCTestCase {

    func makeModel(seed: UInt64 = 3, nLayer: Int = 3, vocab: Int = 64) -> EmbeddingModel {
        let cfg = TinyBackbone.config(nLayer: nLayer, vocab: vocab)
        let bb = X070Backbone(weights: TinyBackbone.weights(cfg, seed: seed), cfg: cfg)
        let head = EmbeddingHead(dim: cfg.nEmbd, seed: seed)
        // штатный нулевой fc2 даёт нулевой градиент по fc1
        MLXRandom.seed(seed &+ 100)
        head.fc2 = MLXRandom.normal([cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        return EmbeddingModel(backbone: bb, head: head)
    }

    /// Длины пачек РАЗНЫЕ (как у настоящих триплетов: запрос короче документа).
    func triplets(B: Int = 3, vocab: Int = 64, seed: UInt64 = 5) -> TripletBatch {
        MLXRandom.seed(seed)
        func ids(_ T: Int) -> MLXArray { MLXRandom.randInt(0 ..< vocab, [B, T]).asType(.int32) }
        func pool(_ T: Int) -> MLXArray { MLXArray((0 ..< B).map { Int32(T - 1 - $0) }) }
        let b = TripletBatch(anchorIdx: ids(16), anchorPool: pool(16),
                             positiveIdx: ids(32), positivePool: pool(32),
                             negativeIdx: ids(48), negativePool: pool(48))
        eval(b.anchorIdx, b.positiveIdx, b.negativeIdx)
        return b
    }

    func classification(B: Int = 2, K: Int = 3, vocab: Int = 64) -> ClassificationBatch {
        MLXRandom.seed(11)
        let a = MLXRandom.randInt(0 ..< vocab, [B, 32]).asType(.int32)
        let c = MLXRandom.randInt(0 ..< vocab, [B, K, 16]).asType(.int32)
        eval(a, c)
        return ClassificationBatch(
            anchorIdx: a, anchorPool: MLXArray(Array(repeating: Int32(31), count: B)),
            candidateIdx: c, candidatePool: MLXArray(Array(repeating: Int32(15), count: B * K), [B, K]),
            mask: MLXArray(Array(repeating: Float(1), count: B * K), [B, K]),
            targetIndex: MLXArray((0 ..< B).map { Int32($0 % K) }))
    }

    func prepare(_ mode: BaseTrainingMode) -> (EmbeddingModel, TrainableSet, [MLXArray]) {
        let model = makeModel()
        if mode == .lora {
            LoRA.add(to: model.backbone, spec: LoRASpec(rank: 4, alpha: 8, cmixTargets: ["key"]))
            MLXRandom.seed(99)
            for t in model.backbone.loraTargets {
                let b = model.backbone.loraB[t]!
                model.backbone.loraB[t] = (MLXRandom.normal(b.shape) * 0.05).asType(b.dtype)
            }
        }
        let set = EmbeddingTrainable.make(model: model, mode: mode)
        return (model, set, set.initialParameters())
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        MLX.abs(ref.asType(.float32) - got.asType(.float32)).max().item(Float.self)
            / (MLX.abs(ref.asType(.float32)).max().item(Float.self) + 1e-12)
    }

    /// - bound: веса базы и голова сходятся до 1e-7 (тот же счёт). Адаптеры —
    ///   до решётки bf16 (2^-8): они входят в модель в bf16, и у автограда
    ///   вклады трёх пачек складываются в котангенте bf16, а здесь — уже в
    ///   fp32, после приведения каждого; замерено 3e-3…5e-3.
    func check(_ what: String, _ set: TrainableSet, _ ps: [MLXArray], bound: Float = 1e-5,
               eager: @escaping () -> MLXArray,
               sequential: () -> (loss: MLXArray, gradients: [MLXArray]),
               file: StaticString = #filePath, line: UInt = #line) {
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            set.inject(p)
            return [eager().asType(.float32)]
        }, argumentNumbers: Array(ps.indices))
        let (v, gRef) = vg(ps)
        eval(v + gRef)
        set.inject(ps)
        let r = sequential()
        eval([r.loss] + r.gradients)
        let names = set.parameterNames
        XCTAssertEqual(r.gradients.count, gRef.count, file: file, line: line)
        var worst: Float = 0, live = 0
        for (i, (a, b)) in zip(gRef, r.gradients).enumerated() {
            let scale = MLX.abs(a).max().item(Float.self)
            if scale == 0 {
                // градиента по этому параметру нет и у автограда — обязан быть нулём и здесь
                XCTAssertEqual(MLX.abs(b).max().item(Float.self), 0, "\(what): \(names[i])", file: file, line: line)
                continue
            }
            live += 1
            let e = rel(a, b)
            worst = Swift.max(worst, e)
            XCTAssertLessThan(e, bound, "\(what): \(names[i])", file: file, line: line)
        }
        let lossRef = v[0].item(Float.self), lossGot = r.loss.item(Float.self)
        print(String(format: "ЭМБЕДДЕР ПО БЛОКАМ, %@: лосс %.6f против %.6f; худший градиент %.1e (%d из %d параметров с градиентом)",
                     what, lossGot, lossRef, worst, live, gRef.count))
        XCTAssertEqual(lossGot, lossRef, accuracy: Swift.abs(lossRef) * 1e-5, what, file: file, line: line)
        XCTAssertGreaterThan(live, 0, "\(what): сверять нечего", file: file, line: line)
    }

    func testTripletGradientsMatchAutogradInEveryMode() {
        for (what, mode) in [("голова", BaseTrainingMode.frozen), ("верхний слой", .topLayers(1)),
                             ("два верхних слоя", .topLayers(2)), ("вся база", .full), ("адаптеры", .lora)] {
            for symmetric in [false, true] {
                let (model, set, ps) = prepare(mode)
                let b = triplets()
                check("\(what)\(symmetric ? ", sts" : "")", set, ps, bound: mode == .lora ? 2e-2 : 1e-5,
                      eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: symmetric) },
                      sequential: {
                          EmbeddingObjective.sequentialProvider(model: model, trainable: set,
                                                                temperature: 0.05, symmetric: symmetric)(ps, b)
                      })
            }
        }
    }

    func testClassificationGradientsMatchAutograd() {
        for (what, mode) in [("вся база", BaseTrainingMode.full), ("адаптеры", .lora)] {
            let (model, set, ps) = prepare(mode)
            let b = classification()
            check("классификация, \(what)", set, ps, bound: mode == .lora ? 2e-2 : 1e-5,
                  eager: { EmbeddingObjective.classificationLoss(model, b, temperature: 0.05) },
                  sequential: {
                      EmbeddingObjective.sequentialClassificationProvider(model: model, trainable: set,
                                                                          temperature: 0.05)(ps, b)
                  })
        }
    }

    /// Квантованная база с адаптерами (QLoRA).
    func testGradientsMatchAutogradOnQuantisedBase() throws {
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("compression"))
        LoRA.add(to: bb, spec: LoRASpec(rank: 4, alpha: 8, layers: 9 ..< 12))
        MLXRandom.seed(99)
        for t in bb.loraTargets {
            let b = bb.loraB[t]!
            bb.loraB[t] = (MLXRandom.normal(b.shape) * 0.05).asType(b.dtype)
        }
        let model = EmbeddingModel(backbone: bb)
        MLXRandom.seed(103)
        model.head.fc2 = MLXRandom.normal(model.head.fc2.shape) * 0.05
        let set = EmbeddingTrainable.make(model: model, mode: .lora)
        // режим .lora ставит дифференцируемое ядро на все слои; адаптеры — на трёх верхних
        let ps = set.initialParameters()
        let b = triplets(vocab: 60000)
        check(".rwkvq, адаптеры на верхних слоях", set, ps, bound: 2e-2,
              eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: false) },
              sequential: {
                  EmbeddingObjective.sequentialProvider(model: model, trainable: set,
                                                        temperature: 0.05, symmetric: false)(ps, b)
              })
    }

    /// Стадия идёт обратным проходом по блокам по умолчанию, не идёт под
    /// GradCache и при выключателе; обучение даёт те же параметры.
    func testStageUsesSequentialBackwardByDefault() throws {
        let tok = try EmbeddingFinetuneTests().loadWorldTokenizerOrSkip()
        let rows = try EmbeddingFinetuneTests().slice().filter { $0.task == .retrieval }
        try XCTSkipIf(rows.count < 8, "мало строк retrieval в срезе")
        func run(sequential: Bool, gradCache: Int = 0) throws -> ([MLXArray], Int) {
            let model = makeModel(vocab: 65536)
            let set = EmbeddingTrainable.make(model: model, mode: .topLayers(2))
            var stage = EmbeddingStage(task: .retrieval, train: Array(rows.prefix(4)),
                                       config: TrainingConfig(lr: 1e-3, maxSteps: 2, cacheLimitGB: 0, logEvery: 0),
                                       batchSize: 2, gradCacheChunk: gradCache, maxTokens: 48, maxChars: 200)
            stage.sequentialBackward = sequential
            let n = EmbeddingObjective.sequentialCalls
            _ = try EmbeddingFinetune.runStage(model: model, tokenizer: tok, trainable: set, stage: stage)
            let ps = set.initialParameters()
            return (ps, EmbeddingObjective.sequentialCalls - n)
        }
        XCTAssertTrue(EmbeddingStage(task: .retrieval, train: []).sequentialBackward, "умолчание")
        let (seq, nSeq) = try run(sequential: true)
        let (plain, nPlain) = try run(sequential: false)
        let (_, nCache) = try run(sequential: true, gradCache: 2)
        XCTAssertEqual(nSeq, 2); XCTAssertEqual(nPlain, 0); XCTAssertEqual(nCache, 0)
        var worst: Float = 0
        for (a, b) in zip(plain, seq) { worst = Swift.max(worst, rel(a, b)) }
        print(String(format: "ЭМБЕДДЕР ПО БЛОКАМ через стадию: параметры после 2 шагов %.1e", worst))
        XCTAssertLessThan(worst, 1e-2)      // веса после commit — в bf16 (замерено 1.1e-3)
    }
}
