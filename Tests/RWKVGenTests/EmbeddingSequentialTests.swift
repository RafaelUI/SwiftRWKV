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
        bb.castWKVOutputToComputeDType = false   // без приведения выхода WKV (явно): проверяется точное тождество путей / эталон из Python, снятый без него
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
        if case .state = mode {
            // ненулевое состояние: на нулях часть градиентов вырождена
            MLXRandom.seed(77)
            let z = model.backbone.zeroInitialState()
            model.backbone.initialState = RWKVBatchState(
                wkv: MLXRandom.normal(z.wkv.shape) * 0.05,
                tmixShift: MLXRandom.normal(z.tmixShift.shape) * 0.5,
                cmixShift: MLXRandom.normal(z.cmixShift.shape) * 0.5)
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
                             ("два верхних слоя", .topLayers(2)), ("вся база", .full), ("адаптеры", .lora),
                             ("состояние", .state()),
                             ("состояние со сдвигами", .state(parts: StateTrainableSet.Part.allCases))] {
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

    /// Режим состояния: в множестве состояние (с множителем шага) и голова,
    /// база в него не входит; все слои идут дифференцируемым ядром.
    func testStateModeTrainsStateAndHeadOnly() {
        let (model, set, _) = prepare(.state(learningRateScale: 50))
        let L = model.backbone.cfg.nLayer
        XCTAssertEqual(set.parameterNames, (0 ..< L).map { "blocks.\($0).state0.wkv" } + EmbeddingHead.parameterNames)
        XCTAssertEqual(set.learningRateScales, [Float](repeating: 50, count: L) + [1, 1, 1, 1])
        XCTAssertEqual(model.backbone.trainLayers, Set(0 ..< L))
        XCTAssertNil(prepare(.state()).1.learningRateScales, "множитель 1 — как без множителей")
        XCTAssertNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .state()))
        XCTAssertEqual(prepare(.state(parts: StateTrainableSet.Part.allCases)).1.parameterNames.count, 3 * L + 4,
                       "заказаны все части состояния")
    }

    func testClassificationGradientsMatchAutograd() {
        for (what, mode) in [("вся база", BaseTrainingMode.full), ("адаптеры", .lora), ("состояние", .state())] {
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

    // ── Состояние только для запросов (`InitialStateScope.queriesOnly`) ──

    /// Подключить ненулевое состояние, не делая его обучаемым (режимы базы).
    func attachState(_ model: EmbeddingModel) {
        MLXRandom.seed(78)
        let z = model.backbone.zeroInitialState()
        model.backbone.initialState = RWKVBatchState(
            wkv: MLXRandom.normal(z.wkv.shape) * 0.05,
            tmixShift: MLXRandom.normal(z.tmixShift.shape) * 0.5,
            cmixShift: MLXRandom.normal(z.cmixShift.shape) * 0.5)
    }

    /// Эталон — автоград на ТОМ ЖЕ составе пачек и кусков: обратный проход по
    /// блокам, GradCache и куски по строкам обязаны дать тот же градиент, что
    /// автоград eager-пути, когда документы идут без состояния. Пересчёт блока
    /// документа в обратном проходе без роли дал бы градиент состояния от
    /// документа, которого в прямом проходе не было.
    func testQueriesOnlyGradientsMatchAutograd() {
        for (what, mode) in [("состояние", BaseTrainingMode.state()),
                             ("состояние со сдвигами", .state(parts: StateTrainableSet.Part.allCases)),
                             ("вся база при подключённом состоянии", .full)] {
            let (model, set, ps) = prepare(mode)
            if mode == .full { attachState(model) }
            model.stateScope = .queriesOnly
            let b = triplets()
            check("только запросы, \(what)", set, ps,
                  eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: false) },
                  sequential: {
                      EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                            symmetric: false)(ps, b)
                  })
            check("только запросы, \(what), куски по 16 токенов", set, ps,
                  eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: false) },
                  sequential: {
                      EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                            symmetric: false, chunkTokens: 16)(ps, b)
                  })
            check("только запросы, \(what), GradCache по 2 строки", set, ps,
                  eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: false) },
                  sequential: {
                      EmbeddingObjective.gradCacheProvider(model: model, trainable: set, chunkSize: 2,
                                                           temperature: 0.05, symmetric: false)(ps, b)
                  })
            let c = classification()
            check("только запросы, классификация, \(what)", set, ps,
                  eager: { EmbeddingObjective.classificationLoss(model, c, temperature: 0.05) },
                  sequential: {
                      EmbeddingObjective.sequentialClassificationProvider(model: model, trainable: set,
                                                                          temperature: 0.05)(ps, c)
                  })
        }
    }

    /// Независимый эталон смысла: при `.queriesOnly` градиент тот же, что у
    /// лосса, где положительные и отрицательные посчитаны с ОТКЛЮЧЁННЫМ
    /// состоянием (подключённые слои сняты на время их прохода — другой
    /// механизм, не тот, что проверяется), а якорь — с ним. И он не равен
    /// градиенту при `.allTexts`: иначе роль ничего не меняла бы.
    func testQueriesOnlyLearnsStateFromQueriesOnly() {
        let (model, set, ps) = prepare(.state())
        let b = triplets()
        let bb = model.backbone
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            set.inject(p)
            let a = model.embed(b.anchorIdx, poolIndex: b.anchorPool)
            let layers = bb.initialLayers
            bb.initialLayers = nil
            let pos = model.embed(b.positiveIdx, poolIndex: b.positivePool)
            let neg = model.embed(b.negativeIdx, poolIndex: b.negativePool)
            bb.initialLayers = layers
            return [tripletPoolLoss(anchor: a, positive: pos, negative: neg, temperature: 0.05,
                                    symmetric: false).asType(.float32)]
        }, argumentNumbers: Array(ps.indices))
        let (vRef, gRef) = vg(ps)
        eval(vRef + gRef)
        set.inject(ps)
        model.stateScope = .queriesOnly
        let r = EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                      symmetric: false)(ps, b)
        eval([r.loss] + r.gradients)
        XCTAssertEqual(r.loss.item(Float.self), vRef[0].item(Float.self), accuracy: 1e-5)
        for (i, name) in set.parameterNames.enumerated() {
            XCTAssertLessThan(rel(gRef[i], r.gradients[i]), 1e-5, name)
        }
        model.stateScope = .allTexts
        let all = EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                        symmetric: false)(ps, b)
        eval(all.gradients)
        XCTAssertGreaterThan(rel(all.gradients[0], r.gradients[0]), 1e-2,
                             "градиент состояния не зависит от роли документов")
    }

    /// Тело по строкам (`bodyChunkTokens`): пачки режутся на куски строк, каждый
    /// идёт через тело порознь, лосс видит весь батч. Градиент тот же, что у
    /// автограда; кусков ровно столько, сколько следует из длин пачек (16 / 32 /
    /// 48 токенов при трёх строках), включая неполный последний.
    func testBodyChunksGiveTheSameGradient() {
        defer { SequentialBody.probe = nil }
        for (what, mode) in [("вся база", BaseTrainingMode.full), ("адаптеры", .lora),
                             ("два верхних слоя", .topLayers(2)), ("состояние", .state())] {
            for (chunk, expected) in [(40, 2 + 3 + 3), (16, 9), (1000, 3)] {
                let (model, set, ps) = prepare(mode)
                let b = triplets()
                var forwards = 0
                SequentialBody.probe = { if $0 == "forward" { forwards += 1 } }
                check("\(what), куски по \(chunk) токенов", set, ps, bound: mode == .lora ? 2e-2 : 1e-5,
                      eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: true) },
                      sequential: {
                          EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                                symmetric: true, chunkTokens: chunk)(ps, b)
                      })
                XCTAssertEqual(forwards, expected, "\(what): кусков при \(chunk) токенах")
            }
        }
        // классификация: пачка кандидатов [B·K, T] режется так же
        let (model, set, ps) = prepare(.full)
        let b = classification()
        var forwards = 0
        SequentialBody.probe = { if $0 == "forward" { forwards += 1 } }
        check("классификация, куски по 32 токена", set, ps,
              eager: { EmbeddingObjective.classificationLoss(model, b, temperature: 0.05) },
              sequential: {
                  EmbeddingObjective.sequentialClassificationProvider(model: model, trainable: set,
                                                                      temperature: 0.05, chunkTokens: 32)(ps, b)
              })
        XCTAssertEqual(forwards, 2 + 3, "якоря 2 × 32 — по строке, кандидаты 6 × 16 — по две")
    }

    /// Запись входов блоков с шагом (`traceStride`): недостающие входы
    /// пересчитываются в обратном проходе — тем же счётом на тех же числах,
    /// поэтому градиент ПОБИТНО тот же, что при записи всех входов; пересчётов
    /// ровно столько, сколько входов не записано.
    func testTraceStrideGivesBitIdenticalGradient() {
        defer { SequentialBody.probe = nil }
        for (what, mode, first) in [("вся база", BaseTrainingMode.full, 0), ("адаптеры", .lora, 0),
                                    ("два верхних слоя", .topLayers(2), 1)] {
            let (model, set, ps) = prepare(mode)
            let b = triplets()
            func run(_ stride: Int, chunk: Int = 0) -> ([MLXArray], Int) {
                var retraced = 0
                SequentialBody.probe = { if $0 == "retrace" { retraced += 1 } }
                set.inject(ps)
                let r = EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                              symmetric: false, chunkTokens: chunk,
                                                              traceStride: stride)(ps, b)
                eval([r.loss] + r.gradients)
                return ([r.loss] + r.gradients, retraced)
            }
            let (ref, none) = run(1)
            XCTAssertEqual(none, 0, "\(what): при записи всех входов пересчитывать нечего")
            let layers = 3 - first                     // блоков, через которые идёт градиент
            for stride in [2, 3, 100] {
                let (got, retraced) = run(stride)
                let stored = (layers + stride - 1) / stride
                XCTAssertEqual(retraced, 3 * (layers - stored), "\(what), шаг \(stride): пересчётов на три пачки")
                for (i, (x, y)) in zip(ref, got).enumerated() {
                    XCTAssertEqual(MLX.abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self), 0,
                                   "\(what), шаг \(stride): выход \(i) обязан быть побитно тем же")
                }
            }
            // вместе с кусками по строкам — тоже побитно против кусков без шага
            let (refChunk, _) = run(1, chunk: 16)
            let (gotChunk, _) = run(2, chunk: 16)
            for (x, y) in zip(refChunk, gotChunk) {
                XCTAssertEqual(MLX.abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self), 0, what)
            }
        }
        // пять блоков, шаг 2: записаны входы 0, 2, 4; вход блока 3 считается от
        // блока 2 (ближайшего записанного), а не от нижнего — по одному
        // пересчёту на каждый из двух незаписанных входов каждой из трёх пачек
        do {
            let model = makeModel(nLayer: 5)
            let set = EmbeddingTrainable.make(model: model, mode: .full)
            let ps = set.initialParameters()
            let b = triplets()
            func run(_ stride: Int) -> ([MLXArray], Int) {
                var retraced = 0
                SequentialBody.probe = { if $0 == "retrace" { retraced += 1 } }
                set.inject(ps)
                let r = EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                              symmetric: false, traceStride: stride)(ps, b)
                eval([r.loss] + r.gradients)
                return ([r.loss] + r.gradients, retraced)
            }
            let (ref, _) = run(1)
            let (got, retraced) = run(2)
            XCTAssertEqual(retraced, 3 * 2, "пять блоков, шаг 2")
            for (x, y) in zip(ref, got) {
                XCTAssertEqual(MLX.abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self), 0, "пять блоков")
            }
        }
        // запись действительно реже: при шаге 2 на трёх блоках хранятся входы блоков 0 и 2
        let (model, set, ps) = prepare(.full)
        set.inject(ps)
        let plan = SequentialBody.Plan(names: set.parameterNames, nLayer: 3)
        let ids = triplets().anchorIdx
        XCTAssertEqual(SequentialBody.forward(model.backbone, ids: ids, plan: plan).blockInput.keys.sorted(), [0, 1, 2])
        XCTAssertEqual(SequentialBody.forward(model.backbone, ids: ids, plan: plan, stride: 2).blockInput.keys.sorted(), [0, 2])
        XCTAssertEqual(SequentialBody.forward(model.backbone, ids: ids, plan: plan, stride: 9).blockInput.keys.sorted(), [0])
        XCTAssertEqual(EmbeddingStage(task: .retrieval, train: []).traceStride, 1, "умолчание")
    }

    /// Конвейер (`SequentialBody.pipelined`): после блока — asyncEval вместо
    /// eval. Граф тот же, меняется только, ждёт ли CPU, — выход побитно тот же,
    /// в том числе вместе с кусками по строкам и записью с шагом.
    func testPipelinedBackwardIsBitIdentical() {
        XCTAssertFalse(SequentialBody.pipelined, "умолчание")
        defer { SequentialBody.pipelined = false }
        for (what, mode) in [("вся база", BaseTrainingMode.full), ("адаптеры", .lora)] {
            for (chunk, stride) in [(0, 1), (16, 2)] {
                let (model, set, ps) = prepare(mode)
                let b = triplets()
                func run(_ pipelined: Bool) -> [MLXArray] {
                    SequentialBody.pipelined = pipelined
                    set.inject(ps)
                    let r = EmbeddingObjective.sequentialProvider(model: model, trainable: set, temperature: 0.05,
                                                                  symmetric: true, chunkTokens: chunk,
                                                                  traceStride: stride)(ps, b)
                    eval([r.loss] + r.gradients)
                    return [r.loss] + r.gradients
                }
                let plain = run(false), piped = run(true)
                XCTAssertGreaterThan(MLX.abs(plain[1]).max().item(Float.self), 0)
                for (i, (x, y)) in zip(plain, piped).enumerated() {
                    XCTAssertEqual(MLX.abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self), 0,
                                   "\(what), куски \(chunk), шаг \(stride): выход \(i)")
                }
            }
        }
    }

    /// Публичный выключатель конвейера — `TrainingConfig.pipelined`: по
    /// умолчанию выключен; включённый действует ТОЛЬКО на время своего запуска
    /// (счётчик `asyncEval` растёт, процессная настройка после — прежняя), и
    /// обучение с ним даёт побитно те же параметры.
    func testPipelineIsPerRunConfigAndOffByDefault() {
        XCTAssertFalse(TrainingConfig().pipelined)
        XCTAssertFalse(LoRAConfig().pipelined)
        XCTAssertFalse(StateTuneConfig().pipelined)
        XCTAssertFalse(EmbeddingStage(task: .retrieval, train: []).config.pipelined)
        XCTAssertTrue({ var c = LoRAConfig(); c.pipelined = true; return c.training.pipelined }())
        XCTAssertTrue({ var c = StateTuneConfig(); c.pipelined = true; return c.training.pipelined }())
        XCTAssertFalse(SequentialBody.pipelined)

        func train(_ pipelined: Bool) -> ([MLXArray], Int) {
            let (model, set, _) = prepare(.lora)
            let b = triplets()
            var cfg = TrainingConfig(lr: 1e-3, maxSteps: 3, cacheLimitGB: 0, logEvery: 0)
            cfg.pipelined = pipelined
            let calls = SequentialBody.asyncEvalCalls, outside = SequentialBody.pipelined
            let trainer = Trainer<TripletBatch>(
                trainable: set,
                objective: { EmbeddingObjective.tripletLoss(model, $0, temperature: 0.05, symmetric: false) },
                nextBatch: { b }, config: cfg,
                gradient: EmbeddingObjective.sequentialProvider(model: model, trainable: set,
                                                                temperature: 0.05, symmetric: false))
            trainer.run()
            XCTAssertEqual(SequentialBody.pipelined, outside, "запуск не вернул процессную настройку")
            return (trainer.currentParameters, SequentialBody.asyncEvalCalls - calls)
        }
        let (plain, plainCalls) = train(false)
        let (piped, pipedCalls) = train(true)
        XCTAssertEqual(plainCalls, 0, "выключенный конвейер всё равно звал asyncEval")
        XCTAssertGreaterThan(pipedCalls, 0, "включённый конвейер не дошёл до обратного прохода")
        for (i, (x, y)) in zip(plain, piped).enumerated() {
            XCTAssertEqual(MLX.abs(x - y).max().item(Float.self), 0, "параметр \(i)")
        }
        // запуск подчиняется своей конфигурации, а не процессной настройке
        SequentialBody.pipelined = true
        defer { SequentialBody.pipelined = false }
        XCTAssertEqual(train(false).1, 0, "процессная настройка перебила конфигурацию запуска")
    }

    /// Потолок кэша на время оценки (`EmbeddingStage.evaluationCacheLimitGB`):
    /// оценка идёт под ним, после неё — прежний; nil — оценка под потолком
    /// обучения; `.infinity` — до предела памяти MLX.
    func testEvaluationCacheLimit() throws {
        let tok = try EmbeddingFinetuneTests().loadWorldTokenizerOrSkip()
        let rows = try EmbeddingFinetuneTests().slice().filter { $0.task == .retrieval }
        try XCTSkipIf(rows.count < 12, "мало строк retrieval в срезе")
        let saved = MLX.Memory.cacheLimit
        defer { MLX.Memory.cacheLimit = saved; EmbeddingFinetune.evaluationProbe = nil }
        XCTAssertNil(EmbeddingStage(task: .retrieval, train: []).evaluationCacheLimitGB, "умолчание")
        for (limit, want) in [(nil, 0.25), (0.5, 0.5), (Double.infinity, -1.0)] as [(Double?, Double)] {
            let model = makeModel(vocab: 65536)
            var stage = EmbeddingStage(task: .retrieval, train: Array(rows.prefix(8)), heldOut: Array(rows.suffix(4)),
                                       config: TrainingConfig(lr: 1e-4, maxSteps: 1, cacheLimitGB: 0.25, logEvery: 0),
                                       batchSize: 2, maxTokens: 32)
            stage.evaluationCacheLimitGB = limit
            var seen: [Int] = []
            EmbeddingFinetune.evaluationProbe = { seen.append(MLX.Memory.cacheLimit) }
            _ = try EmbeddingFinetune.runStage(model: model, tokenizer: tok,
                                               trainable: EmbeddingTrainable.make(model: model, mode: .frozen),
                                               stage: stage)
            let expected = want < 0 ? MLX.Memory.memoryLimit : Int(want * 1e9)
            XCTAssertEqual(seen, [expected, expected], "оценки до и после, потолок \(String(describing: limit))")
            XCTAssertEqual(MLX.Memory.cacheLimit, Int(0.25e9), "после оценки не вернулся потолок обучения")
        }
    }

    /// Стадия передаёт `bodyChunkTokens` в обратный проход; умолчание — без кусков.
    func testStagePassesBodyChunkTokens() throws {
        let tok = try EmbeddingFinetuneTests().loadWorldTokenizerOrSkip()
        let rows = try EmbeddingFinetuneTests().slice().filter { $0.task == .retrieval }
        try XCTSkipIf(rows.count < 8, "мало строк retrieval в срезе")
        defer { SequentialBody.probe = nil }
        func forwards(chunk: Int?, stride: Int = 1) throws -> Int {
            let model = makeModel(vocab: 65536)
            let set = EmbeddingTrainable.make(model: model, mode: .topLayers(2))
            var stage = EmbeddingStage(task: .retrieval, train: Array(rows.prefix(4)),
                                       config: TrainingConfig(lr: 1e-3, maxSteps: 1, cacheLimitGB: 0, logEvery: 0),
                                       batchSize: 2, maxTokens: 48, maxChars: 200)
            if let chunk { stage.bodyChunkTokens = chunk }
            stage.traceStride = stride
            var n = 0, retraced = 0
            SequentialBody.probe = { if $0 == "forward" { n += 1 }; if $0 == "retrace" { retraced += 1 } }
            _ = try EmbeddingFinetune.runStage(model: model, tokenizer: tok, trainable: set, stage: stage)
            XCTAssertEqual(retraced > 0, stride > 1, "traceStride из стадии")
            return n
        }
        XCTAssertEqual(EmbeddingStage(task: .retrieval, train: []).bodyChunkTokens, 2048, "умолчание")
        XCTAssertEqual(try forwards(chunk: nil), 3, "три пачки целиком")
        XCTAssertEqual(try forwards(chunk: 1), 6, "по строке: две строки в каждой из трёх пачек")
        XCTAssertEqual(try forwards(chunk: nil, stride: 2), 3)
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

    /// Верхние слои базы из .rwkvq, развёрнутые в плотные (`unpackTopLayers`):
    /// низ остаётся сжатым, верх обучается как у плотной модели. Числа
    /// проекций те же деквантованные — выход тела не уходит дальше решётки
    /// типа; градиент по блокам сходится с автоградом по каждому весу.
    func testTopLayersTrainOverQuantisedBase() throws {
        let path = try RwkvqFileTests.sourcePath("reduction")
        let bb = try X070Backbone.load(fromRwkvq: path)
        let L = bb.cfg.nLayer
        let model = EmbeddingModel(backbone: bb)
        MLXRandom.seed(103)
        model.head.fc2 = MLXRandom.normal(model.head.fc2.shape) * 0.05
        let b = triplets(vocab: 60000)
        let before = bb.withoutGradientPath { model.embed(b.positiveIdx, poolIndex: b.positivePool) }
        eval(before)
        XCTAssertNotNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .topLayers(2)),
                        "сжатые верхние слои обучать нельзя")

        let packedBefore = bb.rwkvqBackedKeys.count
        let unpacked = bb.unpackTopLayers(2)
        XCTAssertFalse(unpacked.isEmpty)
        XCTAssertTrue(unpacked.allSatisfy { $0.hasPrefix("blocks.\(L - 1).") || $0.hasPrefix("blocks.\(L - 2).") },
                      "развёрнуты только два верхних слоя")
        XCTAssertTrue(unpacked.allSatisfy { bb.w[$0] != nil && !bb.isRwkvqBacked($0) }, "вес плотный, подложка снята")
        XCTAssertEqual(bb.rwkvqBackedKeys.count + unpacked.filter { !bb.rwkvqDenseTargets.contains($0) }.count,
                       packedBefore, "остальные слои остались сжатыми")
        XCTAssertFalse(bb.rwkvqBackedKeys.contains { $0.hasPrefix("blocks.\(L - 1).") || $0.hasPrefix("blocks.\(L - 2).") })
        XCTAssertTrue(bb.rwkvqBackedKeys.contains { $0.hasPrefix("blocks.\(L - 3).") })
        XCTAssertNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .topLayers(2)))
        XCTAssertNotNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .topLayers(3)), "третий сверху — сжат")
        XCTAssertNotNil(EmbeddingTrainable.unsupportedReason(model: model, mode: .full))

        let after = bb.withoutGradientPath { model.embed(b.positiveIdx, poolIndex: b.positivePool) }
        let drift = MLX.abs(before.asType(.float32) - after.asType(.float32)).max().item(Float.self)
        print(String(format: "ВЕРХ ИЗ .rwkvq ПЛОТНЫМ: развёрнуто %d весов (тип %@), сдвиг вектора %.1e",
                     unpacked.count, String(describing: bb.w[unpacked[0]]!.dtype), drift))
        XCTAssertLessThan(drift, 2e-3, "те же деквантованные числа: вектор (норма 1) сдвигается лишь на округления счёта (замерено 1.7e-4)")

        let set = EmbeddingTrainable.make(model: model, mode: .topLayers(2))
        XCTAssertTrue(unpacked.allSatisfy { set.parameterNames.contains($0) }, "развёрнутые веса — в обучаемом множестве")
        let ps = set.initialParameters()
        check(".rwkvq, два верхних слоя плотными", set, ps,
              eager: { EmbeddingObjective.tripletLoss(model, b, temperature: 0.05, symmetric: false) },
              sequential: {
                  EmbeddingObjective.sequentialProvider(model: model, trainable: set,
                                                        temperature: 0.05, symmetric: false)(ps, b)
              })
    }

    /// Оценка: батч ограничен токенами (строки × длина самой длинной), а не
    /// только числом строк; разбиение на батчи вектора не меняет.
    func testEvaluationBatchesAreBoundedByTokens() throws {
        let tok = try EmbeddingFinetuneTests().loadWorldTokenizerOrSkip()
        let model = makeModel(vocab: 65536)
        let texts = ["a", String(repeating: "слово ", count: 40), "bb", String(repeating: "x y ", count: 30),
                     "c", "d e f", "g", String(repeating: "долгий текст ", count: 20), "h"]
        let lens = texts.map { encodeRow(tokenizer: tok, text: $0, maxTokens: 64).count }
        let longest = lens.max()!
        XCTAssertGreaterThan(longest, 4 * lens.min()!, "нужны и длинные, и короткие строки")
        defer { SequentialBody.probe = nil }
        func run(batchSize: Int, tokens: Int) -> (MLXArray, Int) {
            var n = 0
            SequentialBody.probe = { if $0 == "embed batch" { n += 1 } }
            let v = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: texts, batchSize: batchSize,
                                              maxTokens: 64, batchTokens: tokens).asType(.float32)
            eval(v)
            return (v, n)
        }
        // правило, записанное заново: батч растёт, пока строк не больше batchSize и
        // строки × самая длинная не больше бюджета; одна строка проходит всегда
        func expected(batchSize: Int, tokens: Int) -> Int {
            var batches = 0, rows = 0, top = 0
            for l in lens {
                let fits = rows > 0 && rows < batchSize && (tokens == 0 || (rows + 1) * Swift.max(top, l) <= tokens)
                if fits { rows += 1; top = Swift.max(top, l) } else { batches += 1; rows = 1; top = l }
            }
            return batches
        }
        let (ref, one) = run(batchSize: 100, tokens: 0)
        XCTAssertEqual(one, 1, "без ограничений — один батч")
        XCTAssertEqual(ref.shape[0], texts.count)
        for (batchSize, tokens) in [(100, 2 * longest), (100, longest), (100, 1), (2, 0), (3, 3 * longest)] {
            let (v, n) = run(batchSize: batchSize, tokens: tokens)
            XCTAssertEqual(n, expected(batchSize: batchSize, tokens: tokens), "строк \(batchSize), токенов \(tokens)")
            XCTAssertEqual(v.shape, ref.shape)
            XCTAssertLessThan(MLX.abs(v - ref).max().item(Float.self), 1e-5,
                              "строк \(batchSize), токенов \(tokens): векторы те же, порядок строк тот же")
        }
        XCTAssertEqual(run(batchSize: 100, tokens: 1).1, texts.count, "бюджет меньше строки — по одной")
        XCTAssertGreaterThan(run(batchSize: 100, tokens: 2 * longest).1, 1, "бюджет действует")
    }

    /// Потолок кэша MLX действует уже на оценке «до обучения», а не только в
    /// цикле обучения: без него пик footprint всего прогона случался на оценке.
    func testCacheLimitIsAppliedBeforeFirstEvaluation() throws {
        let tok = try EmbeddingFinetuneTests().loadWorldTokenizerOrSkip()
        let rows = try EmbeddingFinetuneTests().slice().filter { $0.task == .retrieval }
        try XCTSkipIf(rows.count < 8, "мало строк retrieval в срезе")
        let saved = MLX.Memory.cacheLimit
        defer { MLX.Memory.cacheLimit = saved; EmbeddingFinetune.evaluationProbe = nil }
        func limitsSeen(cacheGB: Double) throws -> [Int] {
            MLX.Memory.cacheLimit = 7_000_000_000
            var seen: [Int] = []
            EmbeddingFinetune.evaluationProbe = { seen.append(MLX.Memory.cacheLimit) }
            let model = makeModel(vocab: 65536)
            let set = EmbeddingTrainable.make(model: model, mode: .frozen)
            let stage = EmbeddingStage(task: .retrieval, train: Array(rows.prefix(4)),
                                       heldOut: Array(rows.suffix(4)),
                                       config: TrainingConfig(lr: 1e-3, maxSteps: 1, cacheLimitGB: cacheGB, logEvery: 0),
                                       batchSize: 2, maxTokens: 48, maxChars: 200)
            _ = try EmbeddingFinetune.runStage(model: model, tokenizer: tok, trainable: set, stage: stage)
            return seen
        }
        XCTAssertEqual(try limitsSeen(cacheGB: 0.125), [125_000_000, 125_000_000], "оценка до и после обучения")
        XCTAssertEqual(try limitsSeen(cacheGB: 0), [7_000_000_000, 7_000_000_000], "0 — потолок не трогать")
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
