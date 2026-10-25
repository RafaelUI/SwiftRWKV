//
//  RerankTrainingTests.swift
//  Цикл обучения головы на кэше состояний.
//
//  Задача здесь синтетическая, но НЕ вырожденная: состояние правильного
//  кандидата содержит различимый сигнал, остальные — шум. Голова обязана
//  этот сигнал найти. Тест «лосс убывает» без такой проверки ничего не
//  значит: лосс убывает и когда модель заучивает позицию, и когда она
//  скатывается в постоянный ответ.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank

final class RerankTrainingTests: XCTestCase {

    /// Кэш, в котором ПРАВИЛЬНЫЙ кандидат отличим: к его состоянию
    /// подмешан фиксированный образец. Метки расставлены по кругу, чтобы
    /// «всегда отвечать k» не работало.
    func makeLearnableCache(nSamples: Int, nCand: Int, nHead: Int,
                            seed: UInt64 = 1, signal: Float = 1.0)
        throws -> StateCache {
        let S = 64
        let shape = [nSamples * nCand, 1, nHead, S, S]
        MLXRandom.seed(seed)
        let noise = MLXRandom.normal(shape) * 0.2
        let pattern = MLXRandom.normal([1, 1, nHead, S, S]) * signal
        eval(noise, pattern)

        let labels = (0 ..< nSamples).map { $0 % nCand }
        var rows: [MLXArray] = []
        for s in 0 ..< nSamples {
            for c in 0 ..< nCand {
                let base = noise[(s * nCand + c) ..< (s * nCand + c + 1)]
                rows.append(c == labels[s] ? base + pattern : base)
            }
        }
        let all = concatenated(rows, axis: 0)
        eval(all)

        let writer = try StateCacheWriter(shape: shape, dtype: .float32)
        try writer.write(rows: Array(0 ..< nSamples * nCand), all)
        return try writer.finish(
            pairIndex: (0 ..< nSamples).map { s in
                (0 ..< nCand).map { s * nCand + $0 } },
            labels: labels,
            // Один «майненный» негатив на пример — чтобы колонка считалась.
            hardNegs: labels.map { [($0 + 1) % nCand] },
            contract: [:])
    }

    func makeModel(layers: [Int] = [-1], seed: UInt64 = 0)
        throws -> (Reranker, X070Backbone) {
        let (bb, _) = TinyBackbone.make(nLayer: 3, nEmbd: 128, vocab: 128)
        let m = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: layers),
                             seed: seed)
        return (m, bb)
    }

    // ─────────────────────────────────────────────────────────────────

    /// Голова УЧИТСЯ: метрики уходят выше пола случайного угадывания.
    ///
    /// Порог назначен по замеру, а не по вкусу: до обучения MRR равен ровно
    /// 2/(C+1) = 0.25 при C = 7... точнее, при C = 4 это 0.4. Требуем
    /// заметного отрыва, а не «хоть чуть-чуть больше», иначе тест прошёл бы
    /// и на случайном шевелении весов.
    func testHeadActuallyLearns() throws {
        let (model, _) = try makeModel()
        let cache = try makeLearnableCache(nSamples: 64, nCand: 4,
                                           nHead: model.base.cfg.nHead)
        let ev = try makeLearnableCache(nSamples: 32, nCand: 4,
                                        nHead: model.base.cfg.nHead, seed: 2)

        let r = try RerankTraining.train(
            model, trainCache: cache, evalCache: ev,
            config: RerankTrainConfig(lr: 3e-4, batchSize: 8, epochs: 6, seed: 3))

        let floor = RankingMetrics.randomFloor(nCandidates: 4)
        XCTAssertEqual(r.before!.mrr, floor, accuracy: 1e-9,
                       "до обучения голова обязана быть ровно на полу")
        XCTAssertGreaterThan(r.after!.mrr, floor + 0.2,
                             "после обучения MRR почти не сдвинулся: \(r.after!.mrr)")
        XCTAssertLessThan(r.epochs.last!.loss, r.firstLoss,
                          "лосс не убыл")
    }

    /// Кэш меньше пачки: обучение идёт пачками на весь кэш и даёт побитно то
    /// же, что `batchSize = n` (раньше — выход за массив и падение процесса).
    func testBatchLargerThanCache() throws {
        func run(_ bs: Int) throws -> ([Float], Int) {
            let (model, _) = try makeModel()
            let cache = try makeLearnableCache(nSamples: 3, nCand: 4, nHead: model.base.cfg.nHead)
            let r = try RerankTraining.train(model, trainCache: cache,
                                             config: RerankTrainConfig(batchSize: bs, epochs: 3, seed: 3))
            eval(model.head.fc2Weight)
            return (model.head.fc2Weight.asArray(Float.self), r.steps)
        }
        let (big, steps) = try run(8)
        let (exact, stepsExact) = try run(3)
        XCTAssertEqual(steps, 3, "по шагу на эпоху")
        XCTAssertEqual(steps, stepsExact)
        XCTAssertEqual(big, exact, "пачка больше кэша обучила не так, как пачка на весь кэш")
        let (one, _) = try run(1)
        XCTAssertNotEqual(one, exact, "сравнение вырождено: размер пачки ни на что не влияет")
    }

    /// Стартовый лосс — РОВНО ln(C), и это сообщается наружу.
    func testFirstLossIsLnC() throws {
        for C in [2, 4, 8] {
            let (model, _) = try makeModel()
            let cache = try makeLearnableCache(nSamples: 16, nCand: C,
                                               nHead: model.base.cfg.nHead)
            let r = try RerankTraining.train(
                model, trainCache: cache,
                config: RerankTrainConfig(batchSize: 4, epochs: 1))
            XCTAssertEqual(r.firstLoss, log(Float(C)), accuracy: 1e-6,
                           "C=\(C): стартовый лосс не ln(C)")
            XCTAssertEqual(r.expectedFirstLoss, log(Float(C)), accuracy: 1e-6)
        }
    }

    /// Один сид — одна траектория. Без этого сравнить две конфигурации
    /// нельзя: разница окажется шумом инициализации, а не конфигурации.
    func testDeterministicBySeed() throws {
        func run(_ seed: UInt64) throws -> [Float] {
            let (model, _) = try makeModel(seed: 7)
            let cache = try makeLearnableCache(nSamples: 32, nCand: 4,
                                               nHead: model.base.cfg.nHead)
            let r = try RerankTraining.train(
                model, trainCache: cache,
                config: RerankTrainConfig(batchSize: 8, epochs: 3, seed: seed))
            return r.epochs.map { $0.loss }
        }
        XCTAssertEqual(try run(11), try run(11), "один сид дал разные траектории")
        XCTAssertNotEqual(try run(11), try run(12),
                          "разные сиды дали одну траекторию")
    }

    /// keepBest возвращает веса ЛУЧШЕЙ эпохи, а не последней.
    ///
    /// Проверяется по результату: итоговые метрики обязаны совпасть с
    /// метриками эпохи `bestEpoch`. Если снимок весов не отвязан от объекта
    /// (частая ошибка — сохранить те же MLXArray), следующая эпоха перепишет
    /// «лучшие» веса, и совпадения не будет.
    func testKeepBestRestoresBestEpoch() throws {
        let (model, _) = try makeModel()
        let cache = try makeLearnableCache(nSamples: 48, nCand: 4,
                                           nHead: model.base.cfg.nHead)
        let ev = try makeLearnableCache(nSamples: 24, nCand: 4,
                                        nHead: model.base.cfg.nHead, seed: 5)
        let r = try RerankTraining.train(
            model, trainCache: cache, evalCache: ev,
            config: RerankTrainConfig(lr: 5e-4, batchSize: 8, epochs: 5,
                                      keepBest: true, seed: 3))

        XCTAssertGreaterThan(r.bestEpoch, 0)
        let best = r.epochs[r.bestEpoch - 1].metrics!
        XCTAssertEqual(r.after!.mrr, best.mrr, accuracy: 1e-9,
                       "восстановлены веса не лучшей эпохи")
        // И лучшая эпоха действительно лучшая.
        for e in r.epochs where e.metrics != nil {
            XCTAssertLessThanOrEqual(e.metrics!.mrr, best.mrr + 1e-9)
        }
    }

    /// База остаётся нетронутой: обучается только голова.
    func testBaseIsUntouched() throws {
        let (model, bb) = try makeModel()
        let ids = TinyBackbone.ids(2, 24, vocab: bb.cfg.vocab, seed: 3)
        let before = bb.body(ids)
        eval(before)
        let copy = before + 0

        let cache = try makeLearnableCache(nSamples: 32, nCand: 4,
                                           nHead: bb.cfg.nHead)
        _ = try RerankTraining.train(
            model, trainCache: cache,
            config: RerankTrainConfig(lr: 1e-3, batchSize: 8, epochs: 3))

        eval(copy)
        XCTAssertEqual(MLX.abs(copy - bb.body(ids)).max().item(Float.self), 0,
                       "обучение головы изменило базу")
        XCTAssertTrue(bb.trainLayers.isEmpty)
    }

    /// Веса головы после обучения ОТЛИЧАЮТСЯ от стартовых — во всех
    /// параметрах, а не только в последнем слое MLP.
    func testAllHeadParametersMove() throws {
        let (model, _) = try makeModel()
        let names = model.head.parameterNames
        let before = model.head.parameters.map { $0.asType(.float32) + 0 }
        eval(before)

        let cache = try makeLearnableCache(nSamples: 32, nCand: 4,
                                           nHead: model.base.cfg.nHead)
        _ = try RerankTraining.train(
            model, trainCache: cache,
            config: RerankTrainConfig(lr: 1e-3, batchSize: 8, epochs: 3))

        let after = model.head.parameters.map { $0.asType(.float32) }
        eval(after)
        for (i, name) in names.enumerated() {
            let d = MLX.abs(after[i] - before[i]).max().item(Float.self)
            XCTAssertGreaterThan(d, 0, "\(name) не сдвинулся ни на шаг")
        }
    }

    /// Кэш от другой головы отвергается ДО начала обучения, а не на первом
    /// шаге с невнятной ошибкой формы.
    func testIncompatibleCacheRejectedEarly() throws {
        let (model, _) = try makeModel(layers: [0, 2])       // читает два слоя
        let cache = try makeLearnableCache(nSamples: 16, nCand: 4,
                                           nHead: model.base.cfg.nHead)  // один
        XCTAssertThrowsError(try RerankTraining.train(
            model, trainCache: cache,
            config: RerankTrainConfig(batchSize: 4, epochs: 1))) { err in
            guard case StateCacheError.shapeMismatch = err else {
                return XCTFail("ожидалась shapeMismatch, получено \(err)")
            }
        }
    }

    /// Обучение без held-out работает (метрик просто нет).
    func testTrainsWithoutEvalCache() throws {
        let (model, _) = try makeModel()
        let cache = try makeLearnableCache(nSamples: 32, nCand: 4,
                                           nHead: model.base.cfg.nHead)
        let r = try RerankTraining.train(
            model, trainCache: cache, evalCache: nil,
            config: RerankTrainConfig(batchSize: 8, epochs: 2))
        XCTAssertNil(r.before)
        XCTAssertNil(r.after)
        XCTAssertEqual(r.epochs.count, 2)
        XCTAssertTrue(r.epochs.allSatisfy { $0.metrics == nil })
        XCTAssertTrue(r.epochs.allSatisfy { $0.loss.isFinite })
    }

    /// Число шагов ровно `epochs × (n / batchSize)` — границы эпох не
    /// съезжают. Съехавшая граница означала бы, что оценка снимается не
    /// после эпохи, а посреди неё.
    func testStepCountMatchesEpochs() throws {
        let (model, _) = try makeModel()
        let cache = try makeLearnableCache(nSamples: 50, nCand: 4,
                                           nHead: model.base.cfg.nHead)
        let r = try RerankTraining.train(
            model, trainCache: cache,
            config: RerankTrainConfig(batchSize: 10, epochs: 4))
        XCTAssertEqual(r.steps, 4 * (50 / 10))
        XCTAssertEqual(r.epochs.count, 4)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Возобновление (`checkpoint:`, `isCancelled:`)
    //
    //  Эталон — обучение без обрыва: возобновлённое обязано дать ПОБИТНО те
    //  же веса, отчёты эпох, лучшую эпоху и метрики. Модель продолжения —
    //  новая (как после настоящего обрыва процесса).
    // ─────────────────────────────────────────────────────────────────

    func resumeCaches(_ model: Reranker) throws -> (StateCache, StateCache) {
        (try makeLearnableCache(nSamples: 64, nCand: 4, nHead: model.base.cfg.nHead),
         try makeLearnableCache(nSamples: 32, nCand: 4, nHead: model.base.cfg.nHead, seed: 2))
    }

    func resumeConfig(keepBest: Bool = true, lr: Float = 1e-3) -> RerankTrainConfig {
        RerankTrainConfig(lr: lr, batchSize: 8, epochs: 4, keepBest: keepBest, seed: 3, logEvery: 0)
    }

    func headBits(_ m: Reranker) -> [MLXArray] {
        let ps = m.head.parameters.map { $0.asType(.float32) }
        eval(ps)
        return ps
    }

    func sameBits(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { x, y in
            x.shape == y.shape && MLX.abs(x - y).max().item(Float.self) == 0
        }
    }

    func tmpCheckpoint() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rerank_resume_\(UUID().uuidString).safetensors")
    }

    /// Чекпоинт чисел не меняет: то же обучение с ним и без него — побитно.
    func testCheckpointDoesNotChangeNumbers() throws {
        let url = tmpCheckpoint()
        defer { try? FileManager.default.removeItem(at: url) }
        let (a, _) = try makeModel()
        let (train, ev) = try resumeCaches(a)
        let ra = try RerankTraining.train(a, trainCache: train, evalCache: ev, config: resumeConfig())
        let (b, _) = try makeModel()
        let rb = try RerankTraining.train(b, trainCache: train, evalCache: ev, config: resumeConfig(),
                                          checkpoint: url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(sameBits(headBits(a), headBits(b)))
        XCTAssertEqual(ra.epochs, rb.epochs)
        XCTAssertEqual(ra.after, rb.after)
        XCTAssertTrue(rb.completed)
    }

    /// Обрыв после эпохи 1 и после эпохи 3, каждый раз продолжение новой
    /// моделью, — побитно то же, что без обрыва; с лучшей эпохой и без неё.
    func testResumeAfterCancelIsBitIdentical() throws {
        for keepBest in [true, false] {
            let (ref, _) = try makeModel()
            let (train, ev) = try resumeCaches(ref)
            let r0 = try RerankTraining.train(ref, trainCache: train, evalCache: ev,
                                              config: resumeConfig(keepBest: keepBest))
            let url = tmpCheckpoint()
            defer { try? FileManager.default.removeItem(at: url) }
            var seen = 0
            for stopAt in [1, 3] {
                let (m, _) = try makeModel()
                let r = try RerankTraining.train(m, trainCache: train, evalCache: ev,
                                                 config: resumeConfig(keepBest: keepBest), checkpoint: url,
                                                 isCancelled: { seen >= stopAt },
                                                 onEpoch: { _ in seen += 1 })
                XCTAssertFalse(r.completed)
                XCTAssertEqual(r.epochs.count, stopAt, "отмена не после эпохи \(stopAt)")
                XCTAssertNil(r.after)
                XCTAssertEqual(Array(r0.epochs.prefix(stopAt)), r.epochs)
            }
            let (m, _) = try makeModel(seed: 9)        // и голова с другим стартом: веса берутся из чекпоинта
            var resumedEpochs: [Int] = []
            let r = try RerankTraining.train(m, trainCache: train, evalCache: ev,
                                             config: resumeConfig(keepBest: keepBest), checkpoint: url,
                                             onEpoch: { resumedEpochs.append($0.epoch) })
            XCTAssertEqual(resumedEpochs, [4], "продолжение прогнало не только недостающую эпоху")
            XCTAssertTrue(r.completed)
            XCTAssertTrue(sameBits(headBits(ref), headBits(m)), "keepBest \(keepBest): веса не те")
            XCTAssertEqual(r.epochs, r0.epochs)
            XCTAssertEqual(r.after, r0.after)
            XCTAssertEqual(r.before, r0.before)
            XCTAssertEqual(r.bestEpoch, r0.bestEpoch)
            XCTAssertEqual(r.steps, r0.steps)
            XCTAssertEqual(r.firstLoss, r0.firstLoss)
        }
    }

    /// Чекпоинт другого обучения не продолжается.
    func testResumeOfOtherRunIsRefused() throws {
        let url = tmpCheckpoint()
        defer { try? FileManager.default.removeItem(at: url) }
        let (m, _) = try makeModel()
        let (train, ev) = try resumeCaches(m)
        _ = try RerankTraining.train(m, trainCache: train, evalCache: ev, config: resumeConfig(),
                                     checkpoint: url, isCancelled: { true })
        let (m2, _) = try makeModel()
        XCTAssertThrowsError(try RerankTraining.train(m2, trainCache: train, evalCache: ev,
                                                      config: resumeConfig(lr: 2e-3), checkpoint: url))
        XCTAssertThrowsError(try RerankTraining.train(m2, trainCache: train, evalCache: nil,
                                                      config: resumeConfig(), checkpoint: url))
        // Своя конфигурация — продолжается.
        XCTAssertNoThrow(try RerankTraining.train(m2, trainCache: train, evalCache: ev,
                                                  config: resumeConfig(), checkpoint: url))
    }

    /// Законченное обучение с тем же чекпоинтом не делает ни шага и даёт то же.
    func testFinishedCheckpointReturnsSameResult() throws {
        let url = tmpCheckpoint()
        defer { try? FileManager.default.removeItem(at: url) }
        let (a, _) = try makeModel()
        let (train, ev) = try resumeCaches(a)
        let ra = try RerankTraining.train(a, trainCache: train, evalCache: ev, config: resumeConfig(),
                                          checkpoint: url)
        let (b, _) = try makeModel(seed: 5)
        var epochs = 0
        let rb = try RerankTraining.train(b, trainCache: train, evalCache: ev, config: resumeConfig(),
                                          checkpoint: url, onEpoch: { _ in epochs += 1 })
        XCTAssertEqual(epochs, 0)
        XCTAssertEqual(ra.steps, rb.steps)
        XCTAssertEqual(ra.bestEpoch, rb.bestEpoch)
        XCTAssertEqual(ra.before, rb.before)
        XCTAssertTrue(sameBits(headBits(a), headBits(b)))
        XCTAssertEqual(ra.epochs, rb.epochs)
        XCTAssertEqual(ra.after, rb.after)
    }
}
