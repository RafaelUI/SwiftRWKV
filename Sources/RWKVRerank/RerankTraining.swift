import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Обучение головы на кэше состояний.
//
//  Порт train_reranker из rwkv_metal/reranker/train.py, но поверх ОБЩЕГО
//  Trainer: AdamW, расписание, клип и чекпоинты там уже есть и проверены
//  характеризационными тестами. Здесь остаётся ровно специфика реранкера —
//  что считать батчем, чем меряться и когда останавливаться.
//
//  Шаг трогает ТОЛЬКО голову: один-два RWKV-блока на одном токене поверх
//  готового состояния. Отсюда и вся экономия — эпоха по десяткам тысяч пар
//  занимает секунды, а не десятки минут, и подбор гиперпараметров перестаёт
//  быть проектом.
// ───────────────────────────────────────────────────────────────────────

public struct RerankTrainConfig: Sendable {

    /// Голова маленькая и учится с нуля поверх замороженной карты признаков,
    /// поэтому терпит куда больше, чем файнтюн базы. Питоновское умолчание
    /// 3e-5 там названо консервативным; 2e-4 — то, на чём сняты замеры в
    /// docs/reranker.md.
    public var lr: Float
    public var weightDecay: Float
    public var gradClip: Float
    /// ЗАПРОСОВ на шаг. Пар на шаг — это `batchSize × nCandidates`.
    /// Ограничение здесь по памяти состояний, а не по вычислению.
    public var batchSize: Int
    public var epochs: Int
    /// Доля общего числа шагов на разогрев.
    public var warmupFrac: Float
    public var schedule: LRSchedule
    public var loss: RerankLoss
    public var temperature: Float
    /// Восстановить веса ЛУЧШЕЙ эпохи по held-out, а не последней.
    public var keepBest: Bool
    public var seed: UInt64
    public var logEvery: Int

    public init(lr: Float = 2e-4, weightDecay: Float = 0.01,
                gradClip: Float = 1.0, batchSize: Int = 32, epochs: Int = 8,
                warmupFrac: Float = 0.05, schedule: LRSchedule = .cosine,
                loss: RerankLoss = .listwise, temperature: Float = 1.0,
                keepBest: Bool = true, seed: UInt64 = 0, logEvery: Int = 10) {
        self.lr = lr; self.weightDecay = weightDecay; self.gradClip = gradClip
        self.batchSize = batchSize; self.epochs = epochs
        self.warmupFrac = warmupFrac; self.schedule = schedule
        self.loss = loss; self.temperature = temperature
        self.keepBest = keepBest; self.seed = seed; self.logEvery = logEvery
    }
}

public struct RerankEpochReport: Sendable {
    public var epoch: Int
    public var loss: Float
    public var metrics: RankingMetrics?
}

public struct RerankTrainResult: Sendable {
    public var before: RankingMetrics?
    public var after: RankingMetrics?
    public var epochs: [RerankEpochReport]
    public var bestEpoch: Int
    public var steps: Int
    public var firstLoss: Float
    /// Ожидаемый стартовый лосс `ln(C)` — рядом с `firstLoss` для сверки
    /// глазами. Расхождение означает сломанную проводку, а не плохой lr.
    public var expectedFirstLoss: Float
    /// Контракт подачи текста, на котором голова РЕАЛЬНО обучилась.
    ///
    /// Отдаётся наружу, чтобы чекпоинт головы сохранялся с ним, а не с тем,
    /// что вызывающий держит у себя в конфигурации. Это разные вещи ровно
    /// тогда, когда кэш переиспользован от прежнего прогона, — то есть в
    /// самом частом случае. Голова и условия её обучения обязаны ехать в
    /// одном файле.
    public var contract: [String: String]
}

public enum RerankTraining {

    /// Метрики готовой головы на кэше, БЕЗ обучения.
    ///
    /// Нужно ровно там, где обучение мешает: сравнить две сохранённые головы,
    /// переоценить старую на новом отложенном наборе, проверить чекпоинт
    /// после переноса. Раньше для этого приходилось прогонять обучение
    /// заново — то есть менять то, что собирались измерить.
    ///
    /// Контракт сверяется так же, как при обучении, и по той же причине:
    /// голова, обученная на одних обрезках, на кэше с другими даёт
    /// правдоподобные и неверные числа. Умолчание — контракт самого кэша;
    /// задать своё ожидание стоит, когда голова пришла из чекпоинта и её
    /// контракт известен.
    public static func evaluate(
        _ model: Reranker, cache: StateCache,
        contract: [String: String]? = nil, batchSize: Int = 64
    ) throws -> RankingMetrics {
        let want = contract ?? cache.contract
        try cache.checkCompatible(head: model.head, contract: want)
        return try RerankMetrics.evaluate(model.head, cache: cache,
                                          batchSize: batchSize,
                                          slots: try cache.slots(for: model.head))
    }

    /// То же по чекпоинту головы: контракт берётся ИЗ ФАЙЛА.
    ///
    /// Рекомендуемый вход. Голова и условия её обучения едут вместе, поэтому
    /// сверять кэш есть с чем — и несовпадение обрезок обрывает оценку, а не
    /// превращается в число, которое некому проверить.
    public static func evaluate(
        base: X070Backbone, head url: URL, cache: StateCache,
        batchSize: Int = 64, headDType: DType = .float32
    ) throws -> (metrics: RankingMetrics, contract: [String: String]) {
        let md = try Reranker.readHeadMetadata(url)
        let model = try Reranker.fromHead(base: base, url: url,
                                          headDType: headDType)
        // Из метаданных берутся только ключи контракта подачи текста:
        // остальное там — форма головы, и её уже сверил `fromHead`.
        let want = md.filter {
            ["template", "max_doc_tokens", "max_query_tokens",
             "terminator", "instruct"].contains($0.key)
        }
        return (try evaluate(model, cache: cache, contract: want,
                             batchSize: batchSize), want)
    }

    /// Обучить голову на кэше.
    ///
    /// - evalCache: held-out для оценки после каждой эпохи. nil ⇒ обучение
    ///   без оценки, и тогда `keepBest` неприменим (сравнивать не с чем).
    /// - contract: ожидаемый контракт подачи текста. nil ⇒ взять из
    ///   обучающего кэша. Задавать явно стоит там, где вызывающий знает,
    ///   чего хочет, и хочет узнать о расхождении, а не подчиниться ему.
    @discardableResult
    public static func train(
        _ model: Reranker, trainCache: StateCache, evalCache: StateCache? = nil,
        config: RerankTrainConfig = RerankTrainConfig(),
        contract: [String: String]? = nil,
        onEpoch: ((RerankEpochReport) -> Void)? = nil,
        onStep: ((TrainingStep) -> Void)? = nil
    ) throws -> RerankTrainResult {

        // Контракт подачи текста. Раньше сюда передавался пустой словарь,
        // то есть проверялась только ФОРМА состояния: кэш, собранный с
        // maxDocTokens = 128, молча обучал голову, которую потом применяли с
        // 384. Ошибок формы при этом не возникает нигде — только качество
        // хуже, и причину искать негде.
        //
        // Умолчание — контракт ОБУЧАЮЩЕГО кэша: он всегда есть и всегда
        // верен для самого обучения. Тогда проверка ловит главное, что здесь
        // может разъехаться, — отложенный кэш, собранный с другими
        // обрезками или другим шаблоном. Он отдельный файл и собирается
        // отдельным вызовом, так что это не гипотетический случай.
        let want = contract ?? trainCache.contract

        // Срез разрешается ОДИН раз на прогон и дальше передаётся явно.
        // Это же и проверка совместимости: слои, которых в кэше нет, здесь
        // и обрываются — до того, как голова обучится на чужом слое.
        try trainCache.checkCompatible(head: model.head, contract: want)
        let trainSlots = try trainCache.slots(for: model.head)
        var evalSlots: [Int]? = nil
        if let evalCache {
            try evalCache.checkCompatible(head: model.head, contract: want)
            evalSlots = try evalCache.slots(for: model.head)
        }

        let head = model.head
        let n = trainCache.nSamples
        let C = trainCache.nCandidates
        let stepsPerEpoch = Swift.max(1, n / config.batchSize)
        let totalSteps = stepsPerEpoch * config.epochs

        let trainable = RerankerHeadTrainableSet(head)
        let tcfg = TrainingConfig(
            lr: config.lr, lrMin: 0, schedule: config.schedule,
            warmupSteps: Int(Float(totalSteps) * config.warmupFrac),
            gradClip: config.gradClip, weightDecay: config.weightDecay,
            maxSteps: totalSteps, gradAccum: 1,
            // Кэш читается страницами с диска, а не держится в пуле Metal:
            // ограничивать буферный кэш незачем, и без ограничения шаг
            // заметно быстрее.
            cacheLimitGB: 0,
            // Считаем шаги ТОЧНО: границы эпох определяются по ним, и
            // пропуск логов сдвинул бы границу.
            logEvery: 1)

        // Порядок примеров перемешивается КАЖДУЮ эпоху. Без этого батчи из
        // эпохи в эпоху одни и те же, и градиентный шум перестаёт быть шумом.
        var rng = SplitMix64(seed: config.seed)
        var order = rng.shuffled(Array(0 ..< n))
        var cursor = 0

        func nextBatch() -> [Int] {
            if cursor + config.batchSize > order.count {
                order = rng.shuffled(order)
                cursor = 0
            }
            defer { cursor += config.batchSize }
            return Array(order[cursor ..< cursor + config.batchSize])
        }

        func objective(_ rows: [Int]) -> MLXArray {
            let (states, labels) = trainCache.batch(rows, slots: trainSlots)
            let scores = head(states).reshaped([rows.count, C])
            return config.loss(scores, labels, temperature: config.temperature)
        }

        let trainer = Trainer<[Int]>(trainable: trainable, objective: objective,
                                     nextBatch: nextBatch, config: tcfg)

        var before: RankingMetrics? = nil
        if let evalCache {
            before = try RerankMetrics.evaluate(head, cache: evalCache,
                                                slots: evalSlots)
        }

        // Стартовый лосс — на ПЕРВОМ батче, до единого шага. У zero-init
        // головы он обязан быть ровно ln(C); если нет, дело в данных или в
        // проводке, и дальше идти незачем.
        let firstLoss: Float = {
            let probe = Array(order.prefix(Swift.min(config.batchSize, n)))
            let l = objective(probe)
            eval(l)
            return l.item(Float.self)
        }()

        var reports: [RerankEpochReport] = []
        var bestScore = -Double.infinity
        var bestParams: [MLXArray]? = nil
        var bestEpoch = 0
        var stepsDone = 0
        var lastLoss: Float = .nan

        for epoch in 1 ... config.epochs {
            let epochEnd = Swift.min(epoch * stepsPerEpoch, totalSteps)
            let r = trainer.run(isCancelled: { stepsDone >= epochEnd },
                                onStep: { s in
                stepsDone = s.step
                lastLoss = s.loss
                if config.logEvery > 0 && s.step % config.logEvery == 0 {
                    onStep?(s)
                }
            })
            lastLoss = r.finalLoss

            var metrics: RankingMetrics? = nil
            if let evalCache {
                metrics = try RerankMetrics.evaluate(head, cache: evalCache,
                                                     slots: evalSlots)
                if config.keepBest, let m = metrics, m.mrr > bestScore {
                    bestScore = m.mrr
                    bestEpoch = epoch
                    // Снимок ОТВЯЗАННЫЙ: без «+ 0» это были бы те же объекты,
                    // и следующая эпоха переписала бы «лучшие» веса поверх.
                    bestParams = head.parameters.map { $0.asType(.float32) + 0 }
                    eval(bestParams!)
                }
            }
            let report = RerankEpochReport(epoch: epoch, loss: lastLoss,
                                           metrics: metrics)
            reports.append(report)
            onEpoch?(report)
        }

        if config.keepBest, let best = bestParams {
            head.setParameters(best)
            eval(head.parameters)
        }

        var after: RankingMetrics? = nil
        if let evalCache {
            after = try RerankMetrics.evaluate(head, cache: evalCache,
                                               slots: evalSlots)
        }

        return RerankTrainResult(
            before: before, after: after, epochs: reports,
            bestEpoch: bestEpoch, steps: stepsDone,
            firstLoss: firstLoss, expectedFirstLoss: log(Float(C)),
            contract: want)
    }
}
