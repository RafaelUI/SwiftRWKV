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

    /// Бюджет токенов батча оценки по умолчанию — один на процесс: оценку
    /// зовёт стадия (`EmbeddingFinetune.evaluate`), и приложению с жёстким
    /// пределом памяти нужна одна ручка, а не параметр в каждом вызове.
    nonisolated(unsafe) public static var evaluationBatchTokens = 4096

    /// Поштучный результат оценки — для парного бутстрэпа по запросам: средние
    /// (MRR, точность) двух плеч сравнимы только по ОДНИМ И ТЕМ ЖЕ строкам, а
    /// наружу метрики отдают одно число. Вызывается один раз на оценку, в
    /// порядке строк выборки: ранжирование — 1-based ранг верного кандидата,
    /// sts и классификация — 1 за верный ответ, 0 за неверный. По умолчанию nil.
    nonisolated(unsafe) public static var perRowObserver: (([Double]) -> Void)? = nil

    /// Матрица похожести оценки retrieval / классификации [запросов, кандидатов]
    /// — только для тестов: ранги и точность к малым сдвигам векторов глухи, а
    /// то, какими векторами кандидатов (с состоянием или без) считалась
    /// оценка, видно лишь по самой матрице.
    nonisolated(unsafe) static var similarityObserver: ((MLXArray) -> Void)? = nil

    /// Кэш свёрнутых инструкций последней оценки с `foldInstructions` — для
    /// отчёта о цене (сколько инструкций свёрнуто, сколько их токенов).
    nonisolated(unsafe) public static var instructionCacheObserver: ((InstructionCache) -> Void)? = nil

    /// Векторы плоского списка текстов, микробатчами.
    ///
    /// Выравнивание T здесь не нужно: оценка идёт БЕЗ градиента, а
    /// forward-ядро работает при любой длине. Именно поэтому параметра
    /// padMultiple тут нет — добавлять его значило бы платить лишними
    /// токенами за требование, которого на этом пути не существует.
    ///
    /// Прямой путь здесь ВЫБИРАЕТСЯ ЯВНО (`withoutGradientPath`), а не
    /// предполагается: оценка идёт и между шагами обучения, когда у базы
    /// выставлен `trainLayers` (верхние слои, вся база, адаптеры). Без этого
    /// она уходила в дифференцируемое ядро и падала на длине, не кратной 16.
    ///
    /// Батч ограничен и строками (`batchSize`), и ТОКЕНАМИ (`batchTokens`,
    /// строки × длина самой длинной): память прохода идёт за токенами батча —
    /// ~0.14 ГБ на 1024 токена сверх модели (0.1B), и 16 текстов по 1024 токена
    /// одним батчем стоили больше шага обучения. Строки в теле независимы,
    /// разбиение на батчи вектора не меняет. 0 — без ограничения по токенам.
    ///
    /// `role` — роль ВСЕХ текстов списка (`EmbeddingModel.stateScope`): при
    /// `.queriesOnly` документы идут без подключённого состояния.
    public static func embedAll(model: EmbeddingModel, tokenizer: WorldTokenizer,
                                texts: [String], batchSize: Int = 16,
                                terminator: Int = 0,
                                maxTokens: Int? = 512,
                                batchTokens: Int = EmbeddingMetrics.evaluationBatchTokens,
                                role: TextRole? = nil) -> MLXArray {
        let rows = texts.map { encodeRow(tokenizer: tokenizer, text: $0, terminator: terminator, maxTokens: maxTokens) }
        let frozen = SequentialBody.Plan(names: [], nLayer: model.backbone.cfg.nLayer)
        var parts: [MLXArray] = []
        var i = 0
        while i < rows.count {
            var end = i + 1, longest = rows[i].count
            while end < rows.count, end - i < batchSize {
                let grown = Swift.max(longest, rows[end].count)
                if batchTokens > 0, (end - i + 1) * grown > batchTokens { break }
                longest = grown
                end += 1
            }
            let (idx, pool) = padBatch(Array(rows[i ..< end]), terminator: terminator)
            // Тело — по блокам, с вычислением после каждого (`SequentialBody.forward`
            // без обучаемых слоёв входов не хранит): те же операции, что у
            // `model.embed`, но одним ленивым графом всё тело держало ~1 ГБ
            // активаций уже на 2048 токенах батча (0.1B), по блокам — треть.
            let v = model.backbone.withoutGradientPath { () -> MLXArray in
                let top = model.building(role) { SequentialBody.forward(model.backbone, ids: idx, plan: frozen).top }
                return model.embedFromTop(top, poolIndex: pool)
            }
            eval(v)                       // иначе ленивый граф удержит активации
            SequentialBody.probe?("embed batch")
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
    ///
    /// - dedupeCandidates: одинаковые тексты кандидатов (после обрезки)
    ///   сливаются в одного. Нужно там, где документы повторяются — один
    ///   инструмент верен для многих запросов, документ позитивен у одного
    ///   запроса и негативен у другого: копии одного текста в разных пачках
    ///   оценки получают векторы, отличные в последних битах, и копия верного
    ///   документа может встать выше него самого. Ранг тогда — среди
    ///   РАЗЛИЧНЫХ документов. Умолчание false — прежний пул 2N (с ним сняты
    ///   все прежние серии).
    /// - foldInstructions: инструкции запросов («Instruct: X\nQuery: q»)
    ///   сворачиваются в состояние один раз на инструкцию (`InstructionCache`),
    ///   запрос платит только за хвост. Тот же счёт, что с инструкцией в
    ///   строке, кроме обрезки: `maxTokens` действует на хвост. Только пулинг
    ///   `.last`. Документы — как без свёртки.
    public static func evaluateRetrieval(model: EmbeddingModel, tokenizer: WorldTokenizer,
                                         rows: [EmbeddingSample],
                                         maxChars: Int = 800,
                                         ks: [Int] = [1, 5, 10],
                                         terminator: Int = 0,
                                         maxTokens: Int? = 512,
                                         dedupeCandidates: Bool = false,
                                         foldInstructions: Bool = false) -> RankingMetrics {
        precondition(!rows.isEmpty, "оценка на пустой выборке бессмысленна")
        let anchors = rows.map { String($0.anchor.prefix(maxChars)) }
        let positives = rows.map { String($0.positive.prefix(maxChars)) }
        let negatives = rows.map { String($0.negative.prefix(maxChars)) }

        var candidates = positives + negatives
        var correct = Array(0 ..< rows.count)
        if dedupeCandidates {
            var index: [String: Int] = [:]
            var unique: [String] = []
            for t in candidates where index[t] == nil {
                index[t] = unique.count
                unique.append(t)
            }
            correct = positives.map { index[$0]! }
            candidates = unique
        }

        let a: MLXArray
        if foldInstructions {
            precondition(model.pooling == .last, InstructionCacheError.meanPooling.description)
            let cache = InstructionCache(model: model, tokenizer: tokenizer)
            a = try! cache.embedQueries(anchors, terminator: terminator, maxTokens: maxTokens)
            instructionCacheObserver?(cache)
        } else {
            a = embedAll(model: model, tokenizer: tokenizer, texts: anchors,
                         terminator: terminator, maxTokens: maxTokens, role: .query)
        }
        let c = embedAll(model: model, tokenizer: tokenizer, texts: candidates,
                         terminator: terminator, maxTokens: maxTokens, role: .document)

        let sims = matmul(a.asType(.float32), c.asType(.float32).transposed())
        similarityObserver?(sims)
        eval(sims)
        return rankingMetrics(similarities: sims, correctIndex: correct, ks: ks)
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
        var ranks: [Double] = []
        ranks.reserveCapacity(n)

        for i in 0 ..< n {
            let base = i * width
            let target = rowsData[base + correctIndex[i]]
            // 1-based ранг: сколько кандидатов СТРОГО лучше, плюс один.
            // Строго — потому что при равенстве баллов засчитывать проигрыш
            // значило бы штрафовать за ничью, а модель на ничью не влияет.
            var better = 0
            for j in 0 ..< width where rowsData[base + j] > target { better += 1 }
            let rank = better + 1
            ranks.append(Double(rank))
            mrr += 1.0 / Double(rank)
            for k in ks where rank <= k { hits[k]! += 1 }
            if rank <= 10 { ndcg += 1.0 / log2(Double(rank) + 1.0) }
        }

        perRowObserver?(ranks)
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
        // Стороны пары равноправны: роль «запрос» у всех трёх (при
        // `.queriesOnly` состояние у всех — как при `.allTexts`).
        let a = embedAll(model: model, tokenizer: tokenizer,
                         texts: rows.map { String($0.anchor.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens, role: .query).asType(.float32)
        let p = embedAll(model: model, tokenizer: tokenizer,
                         texts: rows.map { String($0.positive.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens, role: .query).asType(.float32)
        let n = embedAll(model: model, tokenizer: tokenizer,
                         texts: rows.map { String($0.negative.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens, role: .query).asType(.float32)

        let simPos = (a * p).sum(axis: -1)
        let simNeg = (a * n).sum(axis: -1)
        let correct = (simPos .> simNeg).asType(.float32)
        eval(simPos, simNeg, correct)
        perRowObserver?(correct.asArray(Float.self).map(Double.init))

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
                         terminator: terminator, maxTokens: maxTokens, role: .query).asType(.float32)

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
                                 terminator: terminator, maxTokens: maxTokens, role: .document)
            .asType(.float32)

        let sims = matmul(a, labelVecs.transposed())      // [N, уникальных меток]
        similarityObserver?(sims)
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
        perRowObserver?(predictions.indices.map { predictions[$0] == targets[$0] ? 1 : 0 })

        return AccuracyMetrics(
            accuracy: Double(correct) / Double(anchors.count),
            count: anchors.count,
            candidatesPerRow: Double(totalCandidates) / Double(anchors.count),
            predictions: predictions,
            poolSizes: poolSizes)
    }
}
