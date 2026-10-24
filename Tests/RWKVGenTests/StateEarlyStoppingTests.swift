//
//  StateEarlyStoppingTests.swift
//  Ранняя остановка обучения состояния по отложенному лоссу (23.10):
//  отложенный лосс — тот же счёт, что лосс обучения; проверки не трогают
//  обучение; возвращается лучшее состояние; терпение; отмена посреди отрезка.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

final class StateEarlyStoppingTests: XCTestCase {

    func model() -> X070Backbone {
        let cfg = TinyBackbone.config()
        return X070Backbone(weights: TinyBackbone.weights(cfg), cfg: cfg, computeDType: .float32)
    }

    /// Пачка [B, T]; первые `mask` целей каждой строки вне лосса (как вопрос в дистилляции).
    func batch(_ seed: UInt64, B: Int = 2, T: Int = 32, mask: Int = 0) -> LoRABatch {
        let ids = TinyBackbone.ids(B, T + 1, vocab: 64, seed: seed)
        var y = ids[0..., 1 ... T]
        if mask > 0 {
            y = concatenated([MLXArray.full([B, mask], values: MLXArray(Int32(-1))), y[0..., mask...]], axis: 1)
        }
        return (x: ids[0..., 0 ..< T], y: y)
    }

    func start(_ bb: X070Backbone) -> RWKVBatchState {
        MLXRandom.seed(1)
        let c = bb.cfg
        return RWKVBatchState(wkv: MLXRandom.normal([c.nLayer, 1, c.nHead, c.headSize, c.headSize]) * 0.05,
                              tmixShift: MLXArray.zeros([c.nLayer, 1, 1, c.nEmbd]),
                              cmixShift: MLXArray.zeros([c.nLayer, 1, 1, c.nEmbd]))
    }

    func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
    }

    // ── Отложенный лосс ─────────────────────────────────────────────────

    /// На одной пачке при тех же параметрах — то же число, что лосс шага обучения
    /// (по блокам), бит в бит; и близко к лоссу одним графом.
    func testHeldOutLossIsTrainingLoss() {
        let bb = model()
        let set = StateTrainableSet(bb, initial: start(bb))
        let b = batch(5, mask: 7)
        let ps = set.initialParameters()
        let g = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 16)
        let train = g.loss.item(Float.self)
        set.inject(ps)
        let held = LoRAFinetune.heldOutLoss(bb, set: set, batches: [b], lossChunk: 16)
        XCTAssertEqual(held, train)
        let whole = LoRAFinetune.languageModelLoss(bb, b, chunk: 16).item(Float.self)
        XCTAssertEqual(held, whole, accuracy: 1e-5 * abs(whole))
    }

    /// Пачки взвешиваются числом токенов в лоссе, а не поровну.
    func testHeldOutLossWeightsByTokens() {
        let bb = model()
        let set = StateTrainableSet(bb, initial: start(bb))
        let b1 = batch(5, mask: 7), b2 = batch(6, mask: 20)          // 2·25 и 2·12 токенов
        let l1 = LoRAFinetune.heldOutLoss(bb, set: set, batches: [b1], lossChunk: 0)
        let l2 = LoRAFinetune.heldOutLoss(bb, set: set, batches: [b2], lossChunk: 0)
        let both = LoRAFinetune.heldOutLoss(bb, set: set, batches: [b1, b2], lossChunk: 0)
        XCTAssertEqual(both, (l1 * 50 + l2 * 24) / 74, accuracy: 1e-6 * abs(both))
        XCTAssertGreaterThan(abs(both - (l1 + l2) / 2), 1e-4 * abs(both), "пачки слишком похожи — проверка пуста")
    }

    // ── Ранняя остановка ────────────────────────────────────────────────

    func train(evalEvery: Int, patience: Int = 100, minDelta: Float = 0, maxSteps: Int = 8,
               cancelAfter: Int? = nil, restoreBest: Bool = true) -> (losses: [Float], out: (result: LoRATrainResult, state: RWKVBatchState),
                                            bb: X070Backbone, initial: RWKVBatchState) {
        let bb = model()
        let initial = start(bb)
        let data = (0 ..< 4).map { batch(UInt64(10 + $0), mask: 3) }
        var cfg = StateTuneConfig()
        cfg.maxSteps = maxSteps; cfg.warmupSteps = 2; cfg.logEvery = 1; cfg.lr = 1e-2; cfg.lrMin = 1e-3
        cfg.cacheLimitGB = 0; cfg.lossChunkTokens = 0
        cfg.evalEvery = evalEvery; cfg.patience = patience; cfg.minDelta = minDelta; cfg.restoreBest = restoreBest
        var i = 0
        var losses: [Float] = []
        let out = StateTuning.run(bb, nextBatch: { defer { i += 1 }; return data[i % data.count] }, config: cfg,
                                  initial: initial, heldOut: evalEvery > 0 ? [batch(99, mask: 3), batch(98)] : [],
                                  isCancelled: { cancelAfter.map { losses.count >= $0 } ?? false },
                                  onStep: { _, l, _, _ in losses.append(l) })
        return (losses, out, bb, initial)
    }

    /// Проверки не трогают обучение: лосс каждого шага тот же бит в бит; проверки на
    /// шагах 0, 3, 6 и на последнем.
    func testEvaluationDoesNotPerturbTraining() {
        let plain = train(evalEvery: 0)
        let checked = train(evalEvery: 3)
        XCTAssertEqual(plain.losses.count, 8)
        XCTAssertEqual(plain.losses, checked.losses)
        XCTAssertNil(plain.out.result.heldOut)
        let r = checked.out.result.heldOut!
        XCTAssertEqual(r.history.map(\.step), [0, 3, 6, 8])
        XCTAssertFalse(r.stoppedEarly)
        XCTAssertEqual(checked.out.result.steps, 8)
        XCTAssertEqual(r.bestLoss, r.history.map(\.loss).min()!)
        XCTAssertEqual(r.bestStep, r.history.first { $0.loss == r.bestLoss }!.step)
    }

    /// Возвращено и подключено состояние лучшей проверки: его отложенный лосс — лучший.
    func testReturnsBestState() {
        let t = train(evalEvery: 3)
        let r = t.out.result.heldOut!
        XCTAssertGreaterThan(r.bestStep, 0, "лучшее — исходное: проверка пуста")
        XCTAssertTrue(bitwise(t.bb.initialState!.wkv, t.out.state.wkv))
        let set = StateTrainableSet(t.bb, initial: t.out.state)
        t.bb.castDenseProjectionInput = true                         // как во время обучения (fp32: без разницы)
        let again = LoRAFinetune.heldOutLoss(t.bb, set: set, batches: [batch(99, mask: 3), batch(98)], lossChunk: 0)
        XCTAssertEqual(again, r.bestLoss)
    }

    /// `restoreBest = false`: проверки только наблюдают — возвращено последнее состояние, ход тот же.
    func testKeepLast() {
        // minDelta огромный: улучшений нет, лучшее — исходное (шаг 0), терпение не кончается
        let best = train(evalEvery: 3, minDelta: 1e9)
        let last = train(evalEvery: 3, minDelta: 1e9, restoreBest: false)
        let plain = train(evalEvery: 0)
        XCTAssertEqual(best.out.result.heldOut, last.out.result.heldOut)
        XCTAssertEqual(best.out.result.heldOut!.bestStep, 0)
        XCTAssertEqual(last.out.result.steps, 8)
        XCTAssertTrue(bitwise(best.out.state.wkv, best.initial.wkv))
        XCTAssertTrue(bitwise(last.out.state.wkv, plain.out.state.wkv), "возвращено не последнее состояние")
        XCTAssertTrue(bitwise(last.bb.initialState!.wkv, plain.out.state.wkv), "подключено не последнее состояние")
        XCTAssertFalse(bitwise(best.out.state.wkv, last.out.state.wkv))
    }

    /// Ни одного улучшения: стоп после `patience` проверок, возвращено исходное состояние.
    func testPatienceStopsAndKeepsInitial() {
        let t = train(evalEvery: 2, patience: 2, minDelta: 1e9, maxSteps: 20)
        let r = t.out.result.heldOut!
        XCTAssertEqual(r.history.map(\.step), [0, 2, 4])
        XCTAssertTrue(r.stoppedEarly)
        XCTAssertEqual(r.bestStep, 0)
        XCTAssertEqual(t.out.result.steps, 4)
        XCTAssertEqual(t.losses.count, 4)
        XCTAssertTrue(bitwise(t.out.state.wkv, t.initial.wkv))
    }

    /// Отмена посреди отрезка: сделанные шаги проверяются, дальше — ни шага.
    func testCancelMidSegmentEvaluatesPartial() {
        let t = train(evalEvery: 5, cancelAfter: 3)
        let r = t.out.result.heldOut!
        XCTAssertEqual(r.history.map(\.step), [0, 3])
        XCTAssertEqual(t.out.result.steps, 3)
        XCTAssertEqual(t.losses.count, 3)
        XCTAssertFalse(r.stoppedEarly)
    }
}
