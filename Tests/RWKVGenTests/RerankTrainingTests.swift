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
}
