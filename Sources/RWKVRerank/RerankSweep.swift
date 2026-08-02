import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Развёртка: несколько конфигураций головы и несколько сидов на ОДНОМ
//  кэше состояний.
//
//  Зачем это отдельная вещь, а не цикл в прогонщике. Одиночный прогон при
//  сотне-другой отложенных запросов шумит, и разницу между конфигурациями
//  по нему судить нельзя — но выглядит она при этом ровно так же
//  убедительно, как настоящая. Питоновский разброс по трём сидам был
//  ±0.004…0.014 по MRR: разрыв 0.024 между слоем 5 и слоем 11 больше этого,
//  но чтобы так СКАЗАТЬ, разброс надо измерить, а не вспомнить.
//
//  Стоило это раньше по 12 минут кодирования на конфигурацию, потому что
//  кэш был привязан к тому, что читает голова. С кэшем надмножества
//  (`RerankEncoder.encodePairs(sources:)`) кодирование одно на всю
//  развёртку, а каждое обучение — секунды. Отсюда и порядок работы здесь:
//  кэш дан снаружи и не пересчитывается ни разу.
// ───────────────────────────────────────────────────────────────────────

/// Среднее и разброс одного числа по прогонам.
///
/// `std` — ВЫБОРОЧНОЕ отклонение (делитель n−1) и при n < 2 равно NaN, а не
/// нулю. Ноль читался бы как «разброса нет», тогда как по одному прогону о
/// разбросе не известно ничего — а именно эта подмена и превращает шум в
/// вывод.
public struct Spread: Sendable, Equatable {
    public var values: [Double]

    public init(_ values: [Double]) { self.values = values }

    public var n: Int { values.count }
    public var mean: Double {
        values.isEmpty ? .nan : values.reduce(0, +) / Double(values.count)
    }
    public var std: Double {
        guard values.count > 1 else { return .nan }
        let m = mean
        let ss = values.reduce(0.0) { $0 + ($1 - m) * ($1 - m) }
        return (ss / Double(values.count - 1)).squareRoot()
    }
    public var min: Double { values.min() ?? .nan }
    public var max: Double { values.max() ?? .nan }

    public var description: String {
        guard n > 1 else {
            return String(format: "%.4f (один прогон, разброс не измерен)", mean)
        }
        return String(format: "%.4f ± %.4f (n=%d, %.4f…%.4f)",
                      mean, std, n, min, max)
    }
}

/// Итог одной конфигурации головы по всем сидам.
public struct RerankSweepPoint: Sendable {
    public var layers: [Int]
    public var nProbe: Int
    public var seeds: [UInt64]
    /// Метрики ПОСЛЕ обучения, по сидам, в порядке `seeds`.
    public var after: [RankingMetrics]
    public var firstLosses: [Float]
    public var expectedFirstLoss: Float
    public var seconds: Double

    public var mrr: Spread { Spread(after.map { $0.mrr }) }
    public var recallAt1: Spread { Spread(after.map { $0.recallAt1 }) }
    public var ndcgAt10: Spread { Spread(after.map { $0.ndcgAt10 }) }
    public var pairwiseVsHardNegative: Spread {
        Spread(after.map { $0.pairwiseVsHardNegative })
    }

    public var label: String {
        "слои \(layers.map(String.init).joined(separator: ","))"
            + (nProbe > 1 ? ", зондов \(nProbe)" : "")
    }
}

public struct RerankSweepResult: Sendable {
    public var points: [RerankSweepPoint]
    public var seconds: Double

    /// Таблица «конфигурация × метрика», среднее ± разброс.
    public var summary: String {
        var out = ""
        let w = points.map { $0.label.count }.max() ?? 0
        func pad(_ s: String) -> String {
            s + String(repeating: " ", count: Swift.max(0, w - s.count))
        }
        out += pad("") + "  MRR                        R@1"
             + "                        против майненного\n"
        for p in points {
            out += pad(p.label) + "  " + p.mrr.description
                 + "   " + p.recallAt1.description
                 + "   " + p.pairwiseVsHardNegative.description + "\n"
        }
        // Сравнение пар конфигураций имеет смысл только при измеренном
        // разбросе, поэтому при одном сиде его здесь просто нет — вместо
        // числа, которое нечем поверить.
        if points.count > 1, points.allSatisfy({ $0.mrr.n > 1 }) {
            out += "\nразрывы по MRR (разность средних против разброса):\n"
            for i in 0 ..< points.count {
                for j in (i + 1) ..< points.count {
                    let a = points[i].mrr, b = points[j].mrr
                    let d = a.mean - b.mean
                    let pooled = ((a.std * a.std + b.std * b.std) / 2).squareRoot()
                    out += String(
                        format: "  %@ − %@: %+.4f при разбросе %.4f — %@\n",
                        points[i].label, points[j].label, d, pooled,
                        abs(d) > 2 * pooled ? "больше двух разбросов"
                                            : "в пределах шума")
                }
            }
        }
        return out
    }
}

public enum RerankSweep {

    /// Обучить голову N раз с разными сидами на одном кэше.
    ///
    /// Каждый сид получает СВЕЖУЮ голову: `Reranker` строится заново, иначе
    /// второй прогон стартовал бы с весов первого и мерил бы дообучение, а
    /// не разброс. Тем же числом задаётся и перемешивание батчей — оба
    /// источника случайности двигаются вместе, потому что порознь они не
    /// разделимы никаким числом отложенных запросов.
    ///
    /// - evalCache: обязателен. Без held-out агрегировать нечего.
    public static func seeds(
        base: X070Backbone, cfg: RerankerConfig,
        trainCache: StateCache, evalCache: StateCache,
        config: RerankTrainConfig = RerankTrainConfig(),
        seeds: [UInt64] = [0], headDType: DType = .float32,
        contract: [String: String]? = nil,
        onRun: ((_ index: Int, _ seed: UInt64,
                 _ result: RerankTrainResult) -> Void)? = nil
    ) throws -> RerankSweepPoint {
        precondition(!seeds.isEmpty, "нужен хотя бы один сид")
        let t0 = Date()
        var after: [RankingMetrics] = []
        var firsts: [Float] = []
        var expected: Float = .nan
        var layers: [Int] = []

        for (i, seed) in seeds.enumerated() {
            let model = try Reranker(base: base, cfg: cfg,
                                     headDType: headDType, seed: seed)
            layers = model.head.layerIdx
            var c = config
            c.seed = seed
            let r = try RerankTraining.train(model, trainCache: trainCache,
                                             evalCache: evalCache, config: c,
                                             contract: contract)
            guard let m = r.after else {
                preconditionFailure("оценка не посчиталась при заданном evalCache")
            }
            after.append(m)
            firsts.append(r.firstLoss)
            expected = r.expectedFirstLoss
            onRun?(i, seed, r)
        }

        return RerankSweepPoint(
            layers: layers, nProbe: cfg.nProbe, seeds: seeds, after: after,
            firstLosses: firsts, expectedFirstLoss: expected,
            seconds: Date().timeIntervalSince(t0))
    }

    /// Несколько конфигураций головы × несколько сидов на одном кэше.
    ///
    /// Кэш обязан держать НАДМНОЖЕСТВО слоёв всех конфигураций; если какой-то
    /// не хватает, обучение обрывается на `StateCache.slots(for:)`, а не
    /// выдаёт числа по чужому слою.
    public static func run(
        base: X070Backbone, configs: [RerankerConfig],
        trainCache: StateCache, evalCache: StateCache,
        config: RerankTrainConfig = RerankTrainConfig(),
        seeds seedList: [UInt64] = [0], headDType: DType = .float32,
        contract: [String: String]? = nil,
        onPoint: ((RerankSweepPoint) -> Void)? = nil,
        onRun: ((_ config: Int, _ index: Int, _ seed: UInt64,
                 _ result: RerankTrainResult) -> Void)? = nil
    ) throws -> RerankSweepResult {
        let t0 = Date()
        var points: [RerankSweepPoint] = []
        for (ci, cfg) in configs.enumerated() {
            let p = try seeds(base: base, cfg: cfg, trainCache: trainCache,
                              evalCache: evalCache, config: config,
                              seeds: seedList, headDType: headDType,
                              contract: contract,
                              onRun: { i, s, r in onRun?(ci, i, s, r) })
            points.append(p)
            onPoint?(p)
        }
        return RerankSweepResult(points: points,
                                 seconds: Date().timeIntervalSince(t0))
    }
}
