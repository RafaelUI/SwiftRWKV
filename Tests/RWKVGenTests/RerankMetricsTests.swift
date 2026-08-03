//
//  RerankMetricsTests.swift
//  Метрики ранжирования: ручной расчёт и — главное — разрешение ничьих.
//
//  Ничьи здесь не крайний случай, а ОСНОВНОЙ: голова zero-init выдаёт ровно
//  равные скоры, и именно на них снимается колонка «до обучения» в любой
//  таблице. Оптимистичное разрешение (взять лучший из равных) показало бы
//  MRR = 1.0 у модели, которая не знает ничего, — и вся таблица стала бы
//  ложью, причём убедительной.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVRerank

final class RerankMetricsTests: XCTestCase {

    // ─────────────────────────────────────────────────────────────────
    //  Ничьи
    // ─────────────────────────────────────────────────────────────────

    /// Полностью равные скоры ⇒ MRR ровно 2/(C+1), то есть случайное
    /// угадывание. Проверяется на нескольких C: одно совпадение могло бы
    /// оказаться случайностью выбранного числа кандидатов.
    func testAllTiedGivesRandomFloor() {
        for C in [2, 4, 8, 25] {
            let scores = [[Float]](repeating: [Float](repeating: 0, count: C),
                                   count: 40)
            let m = RerankMetrics.compute(scores: scores,
                                          labels: [Int](repeating: 0, count: 40),
                                          hardNegs: [])
            XCTAssertEqual(m.mrr, RankingMetrics.randomFloor(nCandidates: C),
                           accuracy: 1e-12,
                           "C=\(C): MRR ничьей не равен 2/(C+1)")
            // И recall@1 при полной ничьей — НЕ единица.
            XCTAssertEqual(m.recallAt1, 0,
                           "полная ничья засчитана как попадание в топ-1")
        }
    }

    /// Частичная ничья: gold делит второе место с одним кандидатом.
    /// scores = [3, 2, 2], label = 1 ⇒ строго больших 1, равных (кроме
    /// себя) 1 ⇒ ранг = 1 + 1 + 0.5 = 2.5, MRR = 0.4.
    func testPartialTie() {
        let m = RerankMetrics.compute(scores: [[3, 2, 2]], labels: [1], hardNegs: [])
        XCTAssertEqual(m.mrr, 0.4, accuracy: 1e-12)
        XCTAssertEqual(m.recallAt1, 0)
        XCTAssertEqual(m.recallAt3, 1)
    }

    /// Ранг 1.5 («поделил первое место») в recall@1 НЕ засчитывается.
    ///
    /// Отдельный тест, потому что здесь легко ошибиться в другую сторону:
    /// питоновский оригинал округляет ранг вверх, и это округление —
    /// тождественная операция (ceil(x) ≤ k ⟺ x ≤ k при целом k). Проверяется
    /// поведение, а не наличие округления.
    func testSharedFirstPlaceIsNotRecallAt1() {
        let m = RerankMetrics.compute(scores: [[5, 5, 1]], labels: [0], hardNegs: [])
        XCTAssertEqual(m.mrr, 1.0 / 1.5, accuracy: 1e-12)
        XCTAssertEqual(m.recallAt1, 0, "делёж первого места засчитан в recall@1")
        XCTAssertEqual(m.recallAt3, 1)
    }

    /// nDCG обрезается на десятке: кандидат с рангом 11 даёт РОВНО ноль.
    ///
    /// Нужен C > 10, иначе отсечка не срабатывает вовсе — и мутация «убрать
    /// отсечку» проходила мимо всех тестов, пока здесь было четыре кандидата.
    func testNDCGCutoffAtTen() {
        var row = [Float](repeating: 0, count: 12)
        for i in 0 ..< 12 { row[i] = Float(12 - i) }   // gold последний ⇒ ранг 12
        let m = RerankMetrics.compute(scores: [row], labels: [11], hardNegs: [])
        XCTAssertEqual(m.mrr, 1.0 / 12.0, accuracy: 1e-12)
        XCTAssertEqual(m.ndcgAt10, 0,
                       "кандидат за пределами десятки попал в nDCG@10")

        // А ранг 10 — ещё внутри: 1/log2(11).
        let m10 = RerankMetrics.compute(scores: [row], labels: [9], hardNegs: [])
        XCTAssertEqual(m10.ndcgAt10, 1.0 / log2(11.0), accuracy: 1e-12)
    }

    /// Идеальное ранжирование — MRR ровно 1, и это не должно зависеть от C.
    func testPerfectRanking() {
        let m = RerankMetrics.compute(scores: [[9, 1, 0, -3], [0, 5, 1, 2]],
                                      labels: [0, 1], hardNegs: [])
        XCTAssertEqual(m.mrr, 1.0, accuracy: 1e-12)
        XCTAssertEqual(m.recallAt1, 1.0)
        XCTAssertEqual(m.ndcgAt10, 1.0, accuracy: 1e-12)
    }

    /// Ручной расчёт на смешанном случае.
    ///   строка 1: [1,2,3], label 0 ⇒ ранг 3, 1/3
    ///   строка 2: [1,2,3], label 2 ⇒ ранг 1, 1
    ///   MRR = (1/3 + 1) / 2 = 0.6666667
    ///   recall@1 = 0.5, recall@3 = 1
    ///   nDCG@10 = (1/log2(4) + 1/log2(2)) / 2 = (0.5 + 1)/2 = 0.75
    func testAgainstHandComputation() {
        let m = RerankMetrics.compute(scores: [[1, 2, 3], [1, 2, 3]],
                                      labels: [0, 2], hardNegs: [])
        XCTAssertEqual(m.mrr, 2.0 / 3.0, accuracy: 1e-6)
        XCTAssertEqual(m.recallAt1, 0.5)
        XCTAssertEqual(m.recallAt3, 1.0)
        XCTAssertEqual(m.ndcgAt10, 0.75, accuracy: 1e-6)
        XCTAssertEqual(m.n, 2)
        XCTAssertEqual(m.nCandidates, 3)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Разложение на майненные и добранные
    // ─────────────────────────────────────────────────────────────────

    /// Колонка, ради которой реранкер и нужен. Здесь модель уверенно бьёт
    /// добранных из пула и проигрывает майненному — ровно то, что происходит
    /// с сырым эмбеддером, и общий MRR это различие размывает.
    ///
    ///   строка: [gold=5, hard=9, easy=0, easy=1], label 0, hardNegs [1]
    ///   против майненного: 5 > 9 — НЕТ ⇒ 0/1
    ///   против добранных:  5 > 0 и 5 > 1 ⇒ 2/2
    func testHardNegativeBreakdown() {
        let m = RerankMetrics.compute(scores: [[5, 9, 0, 1]], labels: [0],
                                      hardNegs: [[1]])
        XCTAssertEqual(m.pairwiseVsHardNegative, 0.0)
        XCTAssertEqual(m.nHardPairs, 1)
        XCTAssertEqual(m.pairwiseVsSampledNegative, 1.0)
        XCTAssertEqual(m.nSampledPairs, 2)
        // Общий MRR при этом выглядит прилично — в этом и дело.
        XCTAssertEqual(m.mrr, 0.5, accuracy: 1e-12)
    }

    /// Несколько майненных негативов на строку (reranker-triples-multi даёт
    /// до пяти) — все идут в знаменатель.
    func testMultipleHardNegatives() {
        let m = RerankMetrics.compute(scores: [[5, 1, 9, 0, 2]], labels: [0],
                                      hardNegs: [[1, 2, 3]])
        XCTAssertEqual(m.nHardPairs, 3)
        XCTAssertEqual(m.pairwiseVsHardNegative, 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(m.nSampledPairs, 1)
        XCTAssertEqual(m.pairwiseVsSampledNegative, 1.0)
    }

    /// Без майненных негативов колонка пустая, а не единица: делить не на что,
    /// и «1.0» здесь читалось бы как идеальный результат.
    func testNoHardNegatives() {
        let m = RerankMetrics.compute(scores: [[5, 1, 2]], labels: [0], hardNegs: [])
        XCTAssertEqual(m.nHardPairs, 0)
        XCTAssertEqual(m.pairwiseVsHardNegative, 0)
        XCTAssertEqual(m.nSampledPairs, 2)
    }

    // ─────────────────────────────────────────────────────────────────
    //  На настоящей голове и кэше
    // ─────────────────────────────────────────────────────────────────

    /// Сквозь кэш: у необученной головы метрики обязаны сесть ровно на пол
    /// случайного угадывания. Это тот же детектор, что и «стартовый лосс
    /// ln(C)», только со стороны метрик.
    func testUntrainedHeadSitsOnRandomFloor() throws {
        let (bb, _) = TinyBackbone.make(nLayer: 3, vocab: 65536)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))

        let nCand = 4, nSamples = 5
        let writer = try StateCacheWriter(
            shape: [nSamples * nCand, 1, bb.cfg.nHead, 64, 64], dtype: .float32)
        MLXRandom.seed(9)
        let states = MLXRandom.normal([nSamples * nCand, 1, bb.cfg.nHead, 64, 64]) * 0.3
        eval(states)
        try writer.write(rows: Array(0 ..< nSamples * nCand), states)
        let cache = try writer.finish(
            pairIndex: (0 ..< nSamples).map { s in
                (0 ..< nCand).map { s * nCand + $0 } },
            labels: [0, 1, 2, 3, 0], hardNegs: [[1], [0], [1], [0], [1]],
            contract: [:])

        let m = try RerankMetrics.evaluate(model.head, cache: cache)
        XCTAssertEqual(m.n, nSamples)
        XCTAssertEqual(m.nCandidates, nCand)
        XCTAssertEqual(m.mrr, RankingMetrics.randomFloor(nCandidates: nCand),
                       accuracy: 1e-6,
                       "необученная голова не села на пол случайного угадывания")
        XCTAssertEqual(m.pairwiseVsHardNegative, 0,
                       "при равных скорах gold не «побеждает» никого")
    }

    /// А обученная (здесь — подкрученная) голова обязана уйти ВЫШЕ пола.
    /// Без этой проверки предыдущая прошла бы и у метрики, которая всегда
    /// возвращает пол.
    func testNonDegenerateScoresBeatTheFloor() {
        // Скоры, где gold всегда первый.
        let scores = (0 ..< 20).map { _ in [Float(10), 1, 2, 3] }
        let m = RerankMetrics.compute(scores: scores,
                                      labels: [Int](repeating: 0, count: 20),
                                      hardNegs: [[Int]](repeating: [1], count: 20))
        XCTAssertEqual(m.mrr, 1.0, accuracy: 1e-12)
        XCTAssertGreaterThan(m.mrr, RankingMetrics.randomFloor(nCandidates: 4))
        XCTAssertEqual(m.pairwiseVsHardNegative, 1.0)
    }

    /// scoreAll не путает строки: у каждого примера свои скоры.
    func testScoreAllKeepsRowsAligned() throws {
        let (bb, _) = TinyBackbone.make(nLayer: 3, vocab: 65536)
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        MLXRandom.seed(21)
        model.head.fc2Weight = MLXRandom.normal([1, model.head.hidden]) * 0.05
        eval(model.head.fc2Weight)

        let nCand = 3, nSamples = 7
        let writer = try StateCacheWriter(
            shape: [nSamples * nCand, 1, bb.cfg.nHead, 64, 64], dtype: .float32)
        let states = MLXRandom.normal([nSamples * nCand, 1, bb.cfg.nHead, 64, 64]) * 0.3
        eval(states)
        try writer.write(rows: Array(0 ..< nSamples * nCand), states)
        let cache = try writer.finish(
            pairIndex: (0 ..< nSamples).map { s in
                (0 ..< nCand).map { s * nCand + $0 } },
            labels: [Int](repeating: 0, count: nSamples),
            hardNegs: [[Int]](repeating: [], count: nSamples), contract: [:])

        // Батчами по 2 и одним куском — результат обязан совпасть.
        let a = try RerankMetrics.scoreAll(model.head, cache: cache, batchSize: 2)
        let b = try RerankMetrics.scoreAll(model.head, cache: cache, batchSize: 64)
        XCTAssertEqual(a.count, nSamples)
        XCTAssertEqual(a[0].count, nCand)
        for i in 0 ..< nSamples {
            for j in 0 ..< nCand {
                XCTAssertEqual(a[i][j], b[i][j], accuracy: 1e-5,
                               "строка \(i), кандидат \(j): размер батча изменил скор")
            }
        }
        // И скоры не вырождены — иначе сравнение выше ничего не значит.
        let flat = a.flatMap { $0 }
        XCTAssertGreaterThan((flat.max() ?? 0) - (flat.min() ?? 0), 1e-4)
    }
}

import MLXRandom
