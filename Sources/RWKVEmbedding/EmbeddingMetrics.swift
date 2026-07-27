import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Оценка на отложенных данных.
//
//  Порт rwkv_metal/embedding/eval.py. Не MTEB — ровно столько, чтобы
//  отличить «стадия помогла» от «лосс убывал на данных, которые модель уже
//  видела». Лосс на обучающей выборке этого не показывает в принципе.
// ───────────────────────────────────────────────────────────────────────

/// Метрики ранжирования.
public struct RankingMetrics: Sendable, Equatable {
    public let mrr: Double
    public let recall: [Int: Double]
    public let ndcg10: Double
    public let count: Int

    public var description: String {
        let r = recall.keys.sorted().map { "recall@\($0) \(fmt(recall[$0]!))" }
            .joined(separator: ", ")
        return "MRR \(fmt(mrr)), \(r), nDCG@10 \(fmt(ndcg10)) (n=\(count))"
    }
}

/// Попарная проверка (STS).
public struct PairwiseMetrics: Sendable, Equatable {
    public let accuracy: Double
    public let meanSimilarityPositive: Double
    public let meanSimilarityNegative: Double
    public let count: Int

    public var description: String {
        "accuracy \(fmt(accuracy)), cos+ \(fmt(meanSimilarityPositive)), "
        + "cos− \(fmt(meanSimilarityNegative)) (n=\(count))"
    }
}

/// Точность классификации.
public struct AccuracyMetrics: Sendable, Equatable {
    public let accuracy: Double
    public let count: Int
    /// Сколько кандидатов предъявлялось (для полного пула — 25, для набора
    /// из инструкции — обычно 7). Без этого числа точность несравнима:
    /// случайное угадывание даёт 1/K, и 0.30 при K=7 хуже, чем 0.20 при K=25.
    public let candidatesPerRow: Double
    /// Выбор модели для каждой строки — индекс В ПУЛЕ ЭТОЙ СТРОКИ.
    ///
    /// Отдаётся наружу не для отладки. Точность — одно число, и на
    /// невырожденных данных она совпадает у правильной реализации и у той,
    /// что ищет максимум по всем известным меткам, а не по семи предъявленным
    /// этой строке. Проверить ограничение можно только по самим выборам:
    /// у правильной реализации индекс всегда меньше размера пула строки.
    /// Мутация «argmax по всем меткам» не ловилась ничем, пока этого поля
    /// не было.
    public let predictions: [Int]
    /// Размер пула у каждой строки — вместе с `predictions` даёт проверяемое
    /// утверждение «выбор лежит внутри предъявленного набора».
    public let poolSizes: [Int]

    public var description: String {
        "accuracy \(fmt(accuracy)) при \(fmt(candidatesPerRow)) кандидатах "
        + "(случайно \(fmt(1.0 / Swift.max(candidatesPerRow, 1)))), n=\(count)"
    }
}

private func fmt(_ x: Double) -> String { String(format: "%.4f", x) }

// ───────────────────────────────────────────────────────────────────────

public enum EmbeddingMetrics {

    /// Векторы плоского списка текстов, микробатчами.
    ///
    /// Выравнивание T здесь не нужно: оценка идёт БЕЗ градиента, а
    /// forward-ядро работает при любой длине. Именно поэтому параметра
    /// padMultiple тут нет — добавлять его значило бы платить лишними
    /// токенами за требование, которого на этом пути не существует.
    public static func embedAll(model: EmbeddingModel, tokenizer: WorldTokenizer,
                                texts: [String], batchSize: Int = 16,
                                terminator: Int = 0,
                                maxTokens: Int? = 512) -> MLXArray {
        var parts: [MLXArray] = []
        var i = 0
        while i < texts.count {
            let end = Swift.min(i + batchSize, texts.count)
            let (idx, pool) = encodeBatch(tokenizer: tokenizer,
                                          texts: Array(texts[i ..< end]),
                                          terminator: terminator,
                                          maxTokens: maxTokens)
            let v = model.embed(idx, poolIndex: pool)
            eval(v)                       // иначе ленивый граф удержит активации
            parts.append(v)
            i = end
        }
        let out = concatenated(parts, axis: 0)
        eval(out)
        return out
    }

    // ── Retrieval ────────────────────────────────────────────────────

    /// Ранжирование истинного положительного среди пула из ВСЕХ положительных
    /// и ВСЕХ отрицательных отложенной выборки: 2N кандидатов на запрос.
    ///
    /// Пул строится из самой выборки, а не из фиксированного индекса, поэтому
    /// числа зависят от её размера: чем больше N, тем труднее задача. Сравнивать
    /// прогоны между собой можно только при одинаковом N — это и есть причина,
    /// по которой `count` возвращается наружу, а не остаётся отладочным.
    public static func evaluateRetrieval(model: EmbeddingModel, tokenizer: WorldTokenizer,
                                         rows: [EmbeddingSample],
                                         maxChars: Int = 800,
                                         ks: [Int] = [1, 5, 10],
                                         terminator: Int = 0,
                                         maxTokens: Int? = 512) -> RankingMetrics {
        precondition(!rows.isEmpty, "оценка на пустой выборке бессмысленна")
        let anchors = rows.map { String($0.anchor.prefix(maxChars)) }
        let positives = rows.map { String($0.positive.prefix(maxChars)) }
        let negatives = rows.map { String($0.negative.prefix(maxChars)) }

        let a = embedAll(model: model, tokenizer: tokenizer, texts: anchors,
                         terminator: terminator, maxTokens: maxTokens)
        let c = embedAll(model: model, tokenizer: tokenizer, texts: positives + negatives,
                         terminator: terminator, maxTokens: maxTokens)

        let sims = matmul(a.asType(.float32), c.asType(.float32).transposed())
        eval(sims)
        return rankingMetrics(similarities: sims,
                              correctIndex: Array(0 ..< rows.count), ks: ks)
    }

    /// Метрики по готовой матрице похожести [N, C] и индексу верного
    /// кандидата в каждой строке. Вынесено отдельно, чтобы это можно было
    /// проверить на числах, не поднимая модель.
    public static func rankingMetrics(similarities: MLXArray,
                                      correctIndex: [Int],
                                      ks: [Int] = [1, 5, 10]) -> RankingMetrics {
        let n = similarities.shape[0]
        precondition(correctIndex.count == n,
                     "строк \(n), а верных индексов \(correctIndex.count)")
        let rowsData = similarities.asType(.float32).asArray(Float.self)
        let width = similarities.shape[1]

        var mrr = 0.0
        var hits = [Int: Int](uniqueKeysWithValues: ks.map { ($0, 0) })
        var ndcg = 0.0

        for i in 0 ..< n {
            let base = i * width
            let target = rowsData[base + correctIndex[i]]
            // 1-based ранг: сколько кандидатов СТРОГО лучше, плюс один.
            // Строго — потому что при равенстве баллов засчитывать проигрыш
            // значило бы штрафовать за ничью, а модель на ничью не влияет.
            var better = 0
            for j in 0 ..< width where rowsData[base + j] > target { better += 1 }
            let rank = better + 1
            mrr += 1.0 / Double(rank)
            for k in ks where rank <= k { hits[k]! += 1 }
            if rank <= 10 { ndcg += 1.0 / log2(Double(rank) + 1.0) }
        }

        return RankingMetrics(
            mrr: mrr / Double(n),
            recall: hits.mapValues { Double($0) / Double(n) },
            ndcg10: ndcg / Double(n),
            count: n)
    }

    // ── STS ──────────────────────────────────────────────────────────

    /// Попарное ранжирование: бьёт ли cos(якорь, положительный) косинус с
    /// отрицательным.
    ///
    /// Это НЕ классический STS по корреляции Спирмена: тот требует градуированных
    /// человеческих оценок похожести, а LitRetrieval размечен бинарно —
    /// положительный/отрицательный. Считать по бинарной разметке корреляцию
    /// можно, но она мерила бы не то, чем её принято называть.
    public static func evaluateSTS(model: EmbeddingModel, tokenizer: WorldTokenizer,
                                   rows: [EmbeddingSample],
                                   maxChars: Int = 800,
                                   terminator: Int = 0,
                                   maxTokens: Int? = 512) -> PairwiseMetrics {
        precondition(!rows.isEmpty, "оценка на пустой выборке бессмысленна")
        let a = embedAll(model: model, tokenizer: tokenizer,
                         texts: rows.map { String($0.anchor.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens).asType(.float32)
        let p = embedAll(model: model, tokenizer: tokenizer,
                         texts: rows.map { String($0.positive.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens).asType(.float32)
        let n = embedAll(model: model, tokenizer: tokenizer,
                         texts: rows.map { String($0.negative.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens).asType(.float32)

        let simPos = (a * p).sum(axis: -1)
        let simNeg = (a * n).sum(axis: -1)
        let correct = (simPos .> simNeg).asType(.float32)
        eval(simPos, simNeg, correct)

        return PairwiseMetrics(
            accuracy: Double(correct.mean().item(Float.self)),
            meanSimilarityPositive: Double(simPos.mean().item(Float.self)),
            meanSimilarityNegative: Double(simNeg.mean().item(Float.self)),
            count: rows.count)
    }

    // ── Классификация ────────────────────────────────────────────────

    /// Zero-shot top-1 по пулу меток.
    ///
    /// - useFullPool: предъявлять все 25 меток корпуса вместо семи из
    ///   инструкции строки. Честнее как оценка: семёрка своя у каждой строки,
    ///   и точность по ней меряет в том числе то, насколько лёгкой оказалась
    ///   конкретная выпавшая семёрка. Полный пул одинаков для всех строк,
    ///   поэтому числа сравнимы между прогонами и между моделями.
    ///
    /// Метки эмбеддятся ОДИН раз на весь вызов, а не построчно: при полном
    /// пуле их 25 на всю выборку, и повторять это на каждую строку — ровно
    /// та же арифметика, умноженная на N.
    public static func evaluateClassification(model: EmbeddingModel,
                                              tokenizer: WorldTokenizer,
                                              rows: [EmbeddingSample],
                                              maxChars: Int = 800,
                                              useFullPool: Bool = true,
                                              terminator: Int = 0,
                                              maxTokens: Int? = 512) -> AccuracyMetrics {
        // Строки, где верный ответ вне предъявленного пула, отбрасываются:
        // они не про качество модели, а про разметку.
        var anchors: [String] = []
        var pools: [[String]] = []
        var targets: [Int] = []
        for r in rows where r.task == .classification {
            let pool = useFullPool
                ? ClassificationLabels.pool
                : (ClassificationLabels.parseCandidates(from: r.anchor) ?? [])
            let label = r.positive.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let t = pool.firstIndex(of: label) else { continue }
            anchors.append(String(r.anchor.prefix(maxChars)))
            pools.append(pool)
            targets.append(t)
        }
        guard !anchors.isEmpty else {
            return AccuracyMetrics(accuracy: .nan, count: 0, candidatesPerRow: 0,
                                   predictions: [], poolSizes: [])
        }

        let a = embedAll(model: model, tokenizer: tokenizer, texts: anchors,
                         terminator: terminator, maxTokens: maxTokens).asType(.float32)

        // Уникальные метки по всей выборке — один прогон на каждую.
        var labelIndex: [String: Int] = [:]
        var labelTexts: [String] = []
        for pool in pools {
            for l in pool where labelIndex[l] == nil {
                labelIndex[l] = labelTexts.count
                labelTexts.append(l)
            }
        }
        let labelVecs = embedAll(model: model, tokenizer: tokenizer, texts: labelTexts,
                                 terminator: terminator, maxTokens: maxTokens)
            .asType(.float32)

        let sims = matmul(a, labelVecs.transposed())      // [N, уникальных меток]
        eval(sims)
        let flat = sims.asArray(Float.self)
        let width = labelTexts.count

        var correct = 0
        var totalCandidates = 0
        var predictions: [Int] = []
        var poolSizes: [Int] = []
        predictions.reserveCapacity(anchors.count)
        poolSizes.reserveCapacity(anchors.count)

        for i in anchors.indices {
            let pool = pools[i]
            totalCandidates += pool.count
            var best = -Float.infinity
            var bestJ = 0
            // Перебор строго по пулу ЭТОЙ строки: набор кандидатов у каждой
            // свой, и максимум по всем известным меткам был бы ответом на
            // другую задачу — более трудную, чем та, что предъявлена.
            for (j, label) in pool.enumerated() {
                let s = flat[i * width + labelIndex[label]!]
                if s > best { best = s; bestJ = j }
            }
            predictions.append(bestJ)
            poolSizes.append(pool.count)
            if bestJ == targets[i] { correct += 1 }
        }

        return AccuracyMetrics(
            accuracy: Double(correct) / Double(anchors.count),
            count: anchors.count,
            candidatesPerRow: Double(totalCandidates) / Double(anchors.count),
            predictions: predictions,
            poolSizes: poolSizes)
    }
}
