import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Общий тренер: то, чего в дорефакторных путях НЕ было и что понадобится
//  претрейну — спад LR и возобновление с чекпоинта.
//
//  Совпадение с прежним поведением проверено отдельно (характеризационные
//  тесты + побитовое сравнение траектории с HEAD), здесь — новое.
// ───────────────────────────────────────────────────────────────────────

final class TrainerTests: XCTestCase {

    static let T = 16

    func makeSetup(seed: UInt64 = 42)
        -> (X070Backbone, X070Config, LoRABatch) {
        let (bb, cfg) = TinyBackbone.make(seed: seed)
        LoRA.add(to: bb, spec: LoRASpec(rank: 8, alpha: 16))
        let ids = TinyBackbone.ids(1, Self.T + 1, vocab: cfg.vocab, seed: 7)
        return (bb, cfg, (x: ids[0..., 0 ..< Self.T], y: ids[0..., 1 ... Self.T]))
    }

    func trainer(_ bb: X070Backbone, _ batch: LoRABatch,
                 _ cfg: TrainingConfig) -> Trainer<LoRABatch> {
        Trainer<LoRABatch>(trainable: LoRATrainableSet(bb),
                           objective: { LoRAFinetune.languageModelLoss(bb, $0) },
                           nextBatch: { batch },
                           config: cfg)
    }

    // ── Расписание LR (чистая функция) ───────────────────────────────

    func testConstantScheduleIsFlat() {
        let c = TrainingConfig(lr: 1e-3, lrMin: 1e-5, schedule: .constant, maxSteps: 100)
        for s in [0, 1, 50, 99] {
            XCTAssertEqual(c.learningRate(at: s), 1e-3, accuracy: 0,
                           "constant обязан игнорировать lrMin и спад")
        }
    }

    /// Допуск 1e-9, а не 0: lr·(step+1)/warmup в Float не совпадает с литералом
    /// побитово (1e-3·1/10 = 1.000000047e-4 против 9.99999975e-5). При этом
    /// 1e-9 на пять порядков меньше любой осмысленной ошибки в дроби warmup.
    func testWarmupIsLinearThenHandsOff() {
        let c = TrainingConfig(lr: 1e-3, schedule: .constant, warmupSteps: 10, maxSteps: 100)
        XCTAssertEqual(c.learningRate(at: 0), 1e-4, accuracy: 1e-9)   // 1/10
        XCTAssertEqual(c.learningRate(at: 4), 5e-4, accuracy: 1e-9)   // 5/10
        XCTAssertEqual(c.learningRate(at: 9), 1e-3, accuracy: 1e-9)   // 10/10
        XCTAssertEqual(c.learningRate(at: 10), 1e-3, accuracy: 1e-9)  // дальше плоско
    }

    /// ВАЖНО про выбор точек. В прогрессе 0, 0.5 и 1 косинус и линейный спад
    /// СОВПАДАЮТ (при 0.5 оба дают decay = 0.5), поэтому проверка только в них
    /// не отличает одно от другого — мутация «косинус → линейный» такой тест
    /// проходит. Различающие точки — четверти: там decay = 0.8536 против 0.75
    /// и 0.1464 против 0.25.
    func testCosineDecaysFromLrToLrMin() {
        let c = TrainingConfig(lr: 1e-3, lrMin: 1e-4, schedule: .cosine,
                               warmupSteps: 0, maxSteps: 100)
        let span: Float = 9e-4      // lr − lrMin

        XCTAssertEqual(c.learningRate(at: 0), 1e-3, accuracy: 1e-9)
        XCTAssertEqual(c.learningRate(at: 50), 1e-4 + 0.5 * span, accuracy: 1e-9)
        XCTAssertEqual(c.learningRate(at: 100), 1e-4, accuracy: 1e-9)

        // четверти — здесь косинус отличим от прямой
        let cos25: Float = 0.5 * (1 + cos(Float.pi * 0.25))     // ≈ 0.853553
        let cos75: Float = 0.5 * (1 + cos(Float.pi * 0.75))     // ≈ 0.146447
        XCTAssertEqual(c.learningRate(at: 25), 1e-4 + cos25 * span, accuracy: 1e-9)
        XCTAssertEqual(c.learningRate(at: 75), 1e-4 + cos75 * span, accuracy: 1e-9)

        // и явно: косинус НЕ равен линейному в этих точках
        let lin = TrainingConfig(lr: 1e-3, lrMin: 1e-4, schedule: .linear,
                                 warmupSteps: 0, maxSteps: 100)
        XCTAssertGreaterThan(c.learningRate(at: 25) - lin.learningRate(at: 25), 1e-5,
                             "косинус неотличим от линейного при прогрессе 0.25")
        XCTAssertLessThan(c.learningRate(at: 75) - lin.learningRate(at: 75), -1e-5,
                          "косинус неотличим от линейного при прогрессе 0.75")

        // монотонность
        var prev = Float.infinity
        for s in stride(from: 0, through: 100, by: 5) {
            let lr = c.learningRate(at: s)
            XCTAssertLessThanOrEqual(lr, prev + 1e-12, "косинус не монотонен на шаге \(s)")
            prev = lr
        }
    }

    func testLinearDecayAndWarmupCompose() {
        let c = TrainingConfig(lr: 1e-3, lrMin: 0, schedule: .linear,
                               warmupSteps: 10, maxSteps: 110)
        XCTAssertEqual(c.learningRate(at: 9), 1e-3, accuracy: 1e-9)     // конец warmup
        XCTAssertEqual(c.learningRate(at: 10), 1e-3, accuracy: 1e-9)    // спад стартует отсюда
        XCTAssertEqual(c.learningRate(at: 35), 7.5e-4, accuracy: 1e-9)  // четверть спада
        XCTAssertEqual(c.learningRate(at: 60), 5e-4, accuracy: 1e-9)    // половина спада
        XCTAssertEqual(c.learningRate(at: 85), 2.5e-4, accuracy: 1e-9)  // три четверти
        XCTAssertEqual(c.learningRate(at: 110), 0, accuracy: 1e-9)
    }

    /// Спад действительно влияет на обучение, а не только на число:
    /// косинус даёт меньший суммарный сдвиг, чем плоский LR.
    func testScheduleAffectsTraining() {
        func finalLoss(_ sched: LRSchedule) -> Float {
            let (bb, _, batch) = makeSetup()
            return trainer(bb, batch, TrainingConfig(lr: 1e-2, schedule: sched,
                                                     maxSteps: 15, cacheLimitGB: 0,
                                                     logEvery: 1)).run().finalLoss
        }
        let flat = finalLoss(.constant)
        let cos = finalLoss(.cosine)
        XCTAssertNotEqual(flat, cos, "расписание не повлияло на обучение")
        XCTAssertLessThan(flat, cos, "плоский LR должен уйти дальше за 15 шагов")
    }

    // ── Чекпоинты ────────────────────────────────────────────────────

    /// 5 шагов → сохранить → загрузить в новый тренер → ещё 5 шагов
    /// обязано дать РОВНО то же, что 10 шагов подряд. Это проверяет, что в
    /// чекпоинт попали не только параметры, но и моменты Adam со счётчиком:
    /// без них возобновление даёт всплеск на первых шагах.
    func testResumeFromCheckpointMatchesUninterruptedRun() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("trainer_ckpt_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let cfgFull = TrainingConfig(lr: 1e-2, schedule: .cosine, maxSteps: 10,
                                     cacheLimitGB: 0, logEvery: 1)

        // эталон: 10 шагов без перерыва
        let (bbA, _, batchA) = makeSetup()
        let tA = trainer(bbA, batchA, cfgFull)
        tA.run()
        let refParams = tA.currentParameters
        eval(refParams)

        // С перерывом: те же 10 шагов, но прерванные отменой на пятом.
        //
        // Прерывать ИМЕННО отменой, а не maxSteps: 5 — косинусный спад
        // считается от maxSteps, поэтому тренер с maxSteps=5 прошёл бы первые
        // пять шагов по другому расписанию, и расхождение было бы следствием
        // теста, а не кода.
        let (bbB, _, batchB) = makeSetup()
        let tB = trainer(bbB, batchB, cfgFull)
        var done = 0
        tB.run(isCancelled: { done >= 5 }, onStep: { done = $0.step })
        XCTAssertEqual(tB.completedSteps, 5, "прерывание сработало не на пятом шаге")
        try tB.saveCheckpoint(to: url)

        let (bbC, _, batchC) = makeSetup()
        let tC = trainer(bbC, batchC, cfgFull)
        try tC.loadCheckpoint(from: url)
        XCTAssertEqual(tC.completedSteps, 5, "чекпоинт не восстановил счётчик шагов")
        let res = tC.run()
        XCTAssertEqual(res.steps, 10)

        let gotParams = tC.currentParameters
        eval(gotParams)
        XCTAssertEqual(gotParams.count, refParams.count)
        for (i, (r, g)) in zip(refParams, gotParams).enumerated() {
            let d = MLX.abs(r - g).max().item(Float.self)
            XCTAssertEqual(d, 0, "параметр \(i): возобновление разошлось (Δ=\(d))")
        }
    }

    /// Чекпоинт без нужного тензора обязан падать внятно, а не молча
    /// продолжать с нулями.
    func testLoadCheckpointRejectsMissingTensor() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bad_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(arrays: ["nonsense": MLXArray.zeros([2])], url: url)

        let (bb, _, batch) = makeSetup()
        let t = trainer(bb, batch, TrainingConfig(maxSteps: 1, cacheLimitGB: 0))
        XCTAssertThrowsError(try t.loadCheckpoint(from: url)) { err in
            guard case TrainerError.checkpointMissing = err else {
                return XCTFail("ожидалась checkpointMissing, получено \(err)")
            }
        }
    }

    // ── BackboneWeightsTrainableSet ──────────────────────────────────

    /// inject кладёт веса в wOverride (подмена на время forward), commit
    /// записывает их в саму модель и снимает подмену.
    func testBackboneWeightsSetInjectAndCommit() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let keys = BackboneWeightsTrainableSet.topLayerKeys(bb, from: 2)
        XCTAssertTrue(keys.contains("ln_out.weight"))
        XCTAssertTrue(keys.allSatisfy { !$0.hasPrefix("blocks.0.") && !$0.hasPrefix("blocks.1.") },
                      "в обучаемые попали замороженные слои")

        let head = ["head.weight": MLXArray.zeros([2, cfg.nEmbd])]
        let set = BackboneWeightsTrainableSet(bb, keys: keys, extra: head)
        XCTAssertEqual(set.parameterNames, keys + ["head.weight"])

        var ps = set.initialParameters()
        XCTAssertEqual(ps.count, keys.count + 1)
        XCTAssertTrue(ps.allSatisfy { $0.dtype == .float32 }, "мастер обязан быть fp32")

        // сдвигаем и подставляем
        ps = ps.map { $0 + 1.0 }
        set.inject(ps)
        XCTAssertNotNil(bb.wOverride)
        XCTAssertEqual(bb.wOverride?.count, keys.count)
        XCTAssertEqual(set.currentExtra["head.weight"]!.shape, [2, cfg.nEmbd])

        set.commit(ps)
        XCTAssertNil(bb.wOverride, "commit обязан снять подмену")
        // значение доехало до самой модели
        let after = bb.w[keys[0]]!.asType(.float32)
        let expected = ps[0]
        eval(after, expected)
        XCTAssertLessThan(MLX.abs(after - expected).max().item(Float.self), 1e-2,
                          "commit не записал вес в модель")
    }

    /// Обучение через BackboneWeightsTrainableSet идёт: лосс убывает, а
    /// замороженные слои остаются нетронутыми.
    func testBackboneWeightsTrainingLeavesFrozenLayersIntact() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let freeze = 2
        bb.trainLayers = Set(freeze ..< cfg.nLayer)
        let frozenBefore = bb.w.filter { $0.key.hasPrefix("blocks.0.") }

        let ids = TinyBackbone.ids(1, Self.T + 1, vocab: cfg.vocab, seed: 7)
        let batch: LoRABatch = (x: ids[0..., 0 ..< Self.T], y: ids[0..., 1 ... Self.T])
        let set = BackboneWeightsTrainableSet(
            bb, keys: BackboneWeightsTrainableSet.topLayerKeys(bb, from: freeze))

        var losses: [Float] = []
        let t = Trainer<LoRABatch>(
            trainable: set,
            objective: { LoRAFinetune.languageModelLoss(bb, $0) },
            nextBatch: { batch },
            config: TrainingConfig(lr: 1e-3, maxSteps: 12, cacheLimitGB: 0, logEvery: 1))
        t.run { losses.append($0.loss) }

        XCTAssertTrue(losses.allSatisfy { $0.isFinite })
        XCTAssertLessThan(losses.last!, losses.first!,
                          "лосс не убыл: \(losses.first!) → \(losses.last!)")
        for (k, v0) in frozenBefore {
            let v1 = bb.w[k]!
            eval(v0, v1)
            XCTAssertEqual(MLX.abs(v0.asType(.float32) - v1.asType(.float32))
                            .max().item(Float.self), 0,
                           "замороженный вес \(k) изменился")
        }
    }

    // ── Изоляция тренеров ────────────────────────────────────────────

    /// Два тренера в одном процессе не мешают друг другу. Раньше текущий батч
    /// жил в file-private глобали — два тренера затирали бы батчи друг друга.
    func testTwoTrainersDoNotInterfere() {
        let (bb1, _, batch1) = makeSetup(seed: 42)
        let (bb2, cfg2) = TinyBackbone.make(seed: 43)
        LoRA.add(to: bb2, spec: LoRASpec(rank: 8, alpha: 16))
        let ids2 = TinyBackbone.ids(1, Self.T + 1, vocab: cfg2.vocab, seed: 21)
        let batch2: LoRABatch = (x: ids2[0..., 0 ..< Self.T], y: ids2[0..., 1 ... Self.T])

        let cfg = TrainingConfig(lr: 1e-2, maxSteps: 4, cacheLimitGB: 0, logEvery: 1)
        let t1 = trainer(bb1, batch1, cfg)
        let t2 = Trainer<LoRABatch>(trainable: LoRATrainableSet(bb2),
                                    objective: { LoRAFinetune.languageModelLoss(bb2, $0) },
                                    nextBatch: { batch2 }, config: cfg)

        // чередуем прогоны — при общей глобали батчи бы перепутались
        let a1 = t1.run().finalLoss
        let a2 = t2.run().finalLoss

        // эталон: тот же t1 в одиночку
        let (bbSolo, _, batchSolo) = makeSetup(seed: 42)
        let solo = trainer(bbSolo, batchSolo, cfg).run().finalLoss

        XCTAssertEqual(a1, solo, accuracy: 0,
                       "первый тренер дал другой результат в присутствии второго")
        XCTAssertTrue(a2.isFinite)
    }
}
