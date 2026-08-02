import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Метрики ранжирования. Порт evaluate/hard_negative_breakdown из
//  rwkv_metal/reranker/{train,../tools/run_reranker}.py.
//
//  Единственная неочевидная деталь здесь — РАЗРЕШЕНИЕ НИЧЬИХ, и она важнее,
//  чем кажется. Ранг считается со средним по связке:
//
//      rank = 1 + (строго больших) + (равных − 1) / 2
//
//  У необученной головы все скоры РАВНЫ (zero-init). При оптимистичном
//  разрешении ничьих (взять лучший из равных) такая голова получила бы
//  ранг 1 и MRR = 1.0 — то есть колонка «до обучения» в любой таблице стала
//  бы ложью, причём убедительной. Со средним она честно показывает
//  2/(C+1) — ровно случайное угадывание.
//
//  Это же число — пол для сравнения: 0.222 при восьми кандидатах.
// ───────────────────────────────────────────────────────────────────────

public struct RankingMetrics: Sendable, Equatable {
    public var mrr: Double
    public var recallAt1: Double
    public var recallAt3: Double
    public var recallAt5: Double
    public var ndcgAt10: Double
    public var n: Int
    public var nCandidates: Int

    /// Попарная точность против МАЙНЕННОГО негатива.
    ///
    /// Главная колонка. Общий MRR по восьми кандидатам, из которых шесть
    /// взяты из пула наугад, льстит всем: отличить пассаж про пчёл от
    /// пассажа про паровые машины несложно. Ради вот этого числа реранкер и
    /// нужен.
    public var pairwiseVsHardNegative: Double
    public var nHardPairs: Int
    /// Попарная точность против ДОБРАННЫХ из пула — «лёгкая» половина.
    public var pairwiseVsSampledNegative: Double
    public var nSampledPairs: Int

    /// Пол случайного угадывания при C кандидатах — то, с чем сравнивать.
    public static func randomFloor(nCandidates C: Int) -> Double {
        2.0 / Double(C + 1)
    }

    public var summary: String {
        String(format: """
            MRR %.4f | R@1 %.4f | R@3 %.4f | R@5 %.4f | nDCG@10 %.4f
            против майненного негатива %.4f (%d пар)
            против добранного из пула  %.4f (%d пар)
            примеров %d × %d кандидатов, пол случайного угадывания %.4f
            """, mrr, recallAt1, recallAt3, recallAt5, ndcgAt10,
            pairwiseVsHardNegative, nHardPairs,
            pairwiseVsSampledNegative, nSampledPairs,
            n, nCandidates, Self.randomFloor(nCandidates: nCandidates))
    }
}

public enum RerankMetrics {

    /// Метрики по готовой матрице скоров `[N, C]`.
    ///
    /// - scores: логиты кандидатов, строка на пример
    /// - labels: позиция правильного кандидата
    /// - hardNegs: позиции майненных негативов (пусто — их не было)
    public static func compute(scores: [[Float]], labels: [Int],
                               hardNegs: [[Int]]) -> RankingMetrics {
        precondition(scores.count == labels.count)
        precondition(hardNegs.isEmpty || hardNegs.count == labels.count)
        let n = scores.count
        guard n > 0 else {
            return RankingMetrics(mrr: 0, recallAt1: 0, recallAt3: 0,
                                  recallAt5: 0, ndcgAt10: 0, n: 0,
                                  nCandidates: 0, pairwiseVsHardNegative: 0,
                                  nHardPairs: 0, pairwiseVsSampledNegative: 0,
                                  nSampledPairs: 0)
        }
        let C = scores[0].count
        precondition(scores.allSatisfy { $0.count == C },
                     "у всех примеров должно быть одинаковое число кандидатов")

        var ranks = [Double](repeating: 0, count: n)
        var hardOK = 0, hardTotal = 0, easyOK = 0, easyTotal = 0

        for i in 0 ..< n {
            let row = scores[i]
            let gold = row[labels[i]]

            var greater = 0, ties = 0
            for v in row {
                if v > gold { greater += 1 }
                else if v == gold { ties += 1 }
            }
            // ties включает сам gold, отсюда −1.
            ranks[i] = 1.0 + Double(greater) + Double(ties - 1) / 2.0

            let hard = Set(hardNegs.isEmpty ? [] : hardNegs[i])
            for j in hard {
                hardTotal += 1
                if gold > row[j] { hardOK += 1 }
            }
            for j in 0 ..< C where j != labels[i] && !hard.contains(j) {
                easyTotal += 1
                if gold > row[j] { easyOK += 1 }
            }
        }

        // recall@k по ДРОБНОМУ рангу, без округления вверх.
        //
        // Питоновский оригинал здесь берёт np.ceil, и это округление —
        // тождественная операция: для целого k условия ceil(x) ≤ k и x ≤ k
        // совпадают при любом положительном x. Найдено мутацией (замена
        // ceil на тождество не поймалась ни одним тестом), после чего
        // проверено алгебраически, а не подобрано контрпримером.
        //
        // Смысл, который округление должно было нести, при этом сохраняется
        // сам собой: ранг 1.5 («поделил первое место») в recall@1 не
        // засчитывается, потому что 1.5 > 1.
        func recall(_ k: Int) -> Double {
            Double(ranks.filter { $0 <= Double(k) }.count) / Double(n)
        }

        let ndcg = ranks.reduce(0.0) {
            $0 + ($1 <= 10 ? 1.0 / log2($1 + 1.0) : 0.0)
        } / Double(n)

        return RankingMetrics(
            mrr: ranks.reduce(0.0) { $0 + 1.0 / $1 } / Double(n),
            recallAt1: recall(1),
            recallAt3: recall(3),
            recallAt5: recall(Swift.min(5, C)),
            ndcgAt10: ndcg,
            n: n, nCandidates: C,
            pairwiseVsHardNegative: hardTotal > 0
                ? Double(hardOK) / Double(hardTotal) : 0,
            nHardPairs: hardTotal,
            pairwiseVsSampledNegative: easyTotal > 0
                ? Double(easyOK) / Double(easyTotal) : 0,
            nSampledPairs: easyTotal)
    }

    /// Скоры головы по всему кэшу, `[nSamples, nCand]`.
    ///
    /// `slots` — срез кэша под эту голову; nil ⇒ вычислить самому. Передавать
    /// готовый стоит там, где он уже посчитан (цикл обучения зовёт оценку
    /// после каждой эпохи), но НЕ ради скорости: разрешение слотов — это и
    /// есть проверка «кэш от этой ли головы», и подсунуть сюда чужой срез
    /// молча нельзя ровно потому, что параметр явный.
    public static func scoreAll(_ head: RerankerHead, cache: StateCache,
                                batchSize: Int = 64,
                                slots: [Int]? = nil) throws -> [[Float]] {
        let sl = try slots ?? cache.slots(for: head)
        var out: [[Float]] = []
        out.reserveCapacity(cache.nSamples)
        var start = 0
        while start < cache.nSamples {
            let end = Swift.min(start + batchSize, cache.nSamples)
            let rows = Array(start ..< end)
            let (states, _) = cache.batch(rows, slots: sl)
            let s = head(states).reshaped([rows.count, cache.nCandidates])
                                .asType(.float32)
            eval(s)
            let flat = s.asArray(Float.self)
            for i in 0 ..< rows.count {
                out.append(Array(flat[i * cache.nCandidates
                                      ..< (i + 1) * cache.nCandidates]))
            }
            start = end
        }
        return out
    }

    /// Метрики головы на кэше.
    public static func evaluate(_ head: RerankerHead, cache: StateCache,
                                batchSize: Int = 64,
                                slots: [Int]? = nil) throws -> RankingMetrics {
        compute(scores: try scoreAll(head, cache: cache, batchSize: batchSize,
                                     slots: slots),
                labels: cache.labels, hardNegs: cache.hardNegs)
    }
}
