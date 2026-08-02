import Foundation
import MLX
import MLXRandom
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Cross-encoder реранкер поверх RWKV-7.
//
//  Порт rwkv_metal/reranker/model.py.
//
//  Идея. Классический cross-encoder гоняет пару (запрос, документ) через
//  трансформер и снимает скор с [CLS]. У RWKV вся история пары уже свёрнута
//  в рекуррентное состояние фиксированного размера h [H,S,S] на слой —
//  значит, скор можно взять не из потокенных активаций, а прямо из
//  состояния: запустить поверх него несколько обучаемых токенов-зондов через
//  короткий стек RWKV-блоков и спроецировать выход в скаляр.
//
//  Что это даёт:
//   * голова читает МАТРИЦУ [S,S] на голову, а не пулинг-вектор [D];
//     y = h·r — это, по сути, один шаг внимания к содержимому состояния;
//   * стоимость головы не зависит от длины пары: один-два токена на блок;
//   * состояние префикса кэшируется, поэтому при шаблоне «документ → запрос»
//     документ сворачивается один раз, а запрос стоит O(своей длины).
//
//  База заморожена. Обучается только голова.
// ───────────────────────────────────────────────────────────────────────

public enum RerankerError: Error, CustomStringConvertible {
    case layerOutOfRange(Int, Int)
    case emptyLayerIdx
    case checkpointMismatch(String, String, String)
    case notARerankerCheckpoint(String)

    public var description: String {
        switch self {
        case .layerOutOfRange(let i, let n):
            return "слой \(i) вне диапазона для модели из \(n) слоёв"
        case .emptyLayerIdx:
            return "layerIdx пуст — голова обязана читать хотя бы один слой"
        case .checkpointMismatch(let key, let inFile, let here):
            return """
                чекпоинт не соответствует модели: \(key)='\(inFile)' в файле \
                против '\(here)' здесь. Собери Reranker с той же \
                конфигурацией — проще всего через Reranker.fromHead(base:url:).
                """
        case .notARerankerCheckpoint(let path):
            return """
                \(path): нет метаданных реранкера. Либо это чужой файл, либо \
                чекпоинт, сохранённый до появления метаданных — тогда собери \
                Reranker вручную и позови loadHead(..., strict: false).
                """
        }
    }
}

/// Нормализует индексы слоёв базы (поддерживает отрицательные).
public func resolveLayerIndices(_ layerIdx: [Int], nLayer: Int) throws -> [Int] {
    guard !layerIdx.isEmpty else { throw RerankerError.emptyLayerIdx }
    return try layerIdx.map { i in
        let j = i < 0 ? i + nLayer : i
        guard j >= 0 && j < nLayer else {
            throw RerankerError.layerOutOfRange(i, nLayer)
        }
        return j
    }
}

/// Конфигурация головы.
public struct RerankerConfig: Sendable, Equatable {

    /// Какие слои базы читает голова. Один блок на индекс; отрицательные
    /// считаются с конца.
    ///
    /// Умолчание `[-1]` — последний слой, самое дешёвое. По замерам Python
    /// это НЕ лучший выбор: середина стека даёт заметно больше (MRR 0.978
    /// против 0.922 у последнего слоя, три сида). Умолчание оставлено таким
    /// же, как в эталоне, чтобы числа сравнивались; в реальном прогоне стоит
    /// начинать с середины и сверять.
    public var layerIdx: [Int]

    /// Все блоки читают состояние ПОСЛЕДНЕГО слоя, а глубина стека остаётся
    /// равной `layerIdx.count`. Способ углубить голову, не трогая то, откуда
    /// она читает (и не увеличивая кэш состояний).
    public var sharedState: Bool

    /// Сколько обучаемых токенов проходит через голову. Каждый — ещё одно
    /// чтение состояния; скор снимается с последнего.
    public var nProbe: Int

    /// Ширина скрытого слоя MLP. nil ⇒ nEmbd.
    public var headHidden: Int?

    public init(layerIdx: [Int] = [-1], sharedState: Bool = false,
                nProbe: Int = 1, headHidden: Int? = nil) {
        self.layerIdx = layerIdx
        self.sharedState = sharedState
        self.nProbe = nProbe
        self.headHidden = headHidden
    }
}

/// Стек RWKV-блоков поверх состояния базы + проекция в скаляр.
///
/// СИБЛИНГ базы, а не её подмодуль: заморозка базы (и всё, что её зовёт —
/// навеска LoRA, квантование) не должна молча заморозить голову.
public final class RerankerHead {

    public let cfg: RerankerConfig
    /// Индексы слоёв базы после нормализации отрицательных.
    public let layerIdx: [Int]
    /// Какой слой базы читает блок i.
    public let sources: [Int]
    /// Слои, которые голове вообще нужны, по возрастанию.
    ///
    /// Решающе важно для кэша: хранить `[nUnique, H, S, S]` вместо
    /// `[L, H, S, S]` дешевле ровно в L/nUnique раз — для умолчания в 12.
    public let uniqueSources: [Int]
    /// Позиция состояния блока i внутри `uniqueSources`.
    public let sourceSlot: [Int]

    public let dim: Int
    public let hidden: Int

    /// Токены-зонды [nProbe, D]. Масштаб не важен: ln0 нормирует каждый токен.
    public var probe: MLXArray
    public var ln0Weight: MLXArray
    public var ln0Bias: MLXArray
    public var blocks: [RWKVBlock]
    public var lnOutWeight: MLXArray
    public var lnOutBias: MLXArray
    public var fc1Weight: MLXArray      // [hidden, D]
    public var fc1Bias: MLXArray        // [hidden]
    public var fc2Weight: MLXArray      // [1, hidden] — ZERO-INIT

    /// Собрать голову из базы: блоки инициализируются весами выбранных слоёв,
    /// ln0/ln_out — одноимёнными слоями базы.
    ///
    /// `dtype` по умолчанию fp32, и это не запас на всякий случай. Официальные
    /// веса лежат в bf16: 8 бит мантиссы против 24, и при lr порядка 1e-4 и
    /// весах порядка 0.05 часть каждого шага AdamW оказывается меньше кванта
    /// представления и теряется на округлении. Лосс при этом продолжает
    /// убывать за счёт последнего слоя MLP — то есть без явного каста
    /// проблема не видна. Голова 8–23 М параметров, fp32 ей ничего не стоит.
    public init(base: X070Backbone, cfg: RerankerConfig = RerankerConfig(),
                dtype: DType = .float32, seed: UInt64 = 0) throws {
        self.cfg = cfg
        self.layerIdx = try resolveLayerIndices(cfg.layerIdx, nLayer: base.cfg.nLayer)
        let resolved = self.layerIdx
        let last = resolved[resolved.count - 1]
        let srcs = resolved.map { cfg.sharedState ? last : $0 }
        let uniq = Array(Set(srcs)).sorted()
        self.sources = srcs
        self.uniqueSources = uniq
        self.sourceSlot = srcs.map { uniq.firstIndex(of: $0)! }

        let D = base.cfg.nEmbd
        self.dim = D
        self.hidden = cfg.headHidden ?? D

        MLXRandom.seed(seed)
        self.probe = MLXRandom.normal([cfg.nProbe, D]) * 0.02

        self.ln0Weight = base.weight("ln0.weight").asType(dtype)
        self.ln0Bias = base.weight("ln0.bias").asType(dtype)
        self.lnOutWeight = base.weight("ln_out.weight").asType(dtype)
        self.lnOutBias = base.weight("ln_out.bias").asType(dtype)

        self.blocks = try layerIdx.enumerated().map { i, src in
            try RWKVBlock.fromBase(base, layer: src, index: i, dtype: dtype)
        }

        // Как nn.Linear в MLX: uniform(-1/√in, 1/√in) и для веса, и для bias.
        let s1 = Float(1.0 / sqrt(Double(D)))
        self.fc1Weight = MLXRandom.uniform(low: -s1, high: s1, [hidden, D])
        self.fc1Bias = MLXRandom.uniform(low: -s1, high: s1, [hidden])
        // ZERO-INIT. До обучения все скоры РОВНО нули, значит listwise-лосс
        // стартует ровно с ln(C). Если первый залогированный лосс не ln(C) —
        // данные или голова собраны неверно; это самый дешёвый детектор
        // сломанной проводки во всём конвейере, и он существует только
        // благодаря этой строке.
        self.fc2Weight = MLXArray.zeros([1, hidden])

        self.probe = probe.asType(dtype)
        self.fc1Weight = fc1Weight.asType(dtype)
        self.fc1Bias = fc1Bias.asType(dtype)
        self.fc2Weight = fc2Weight.asType(dtype)
        eval(probe, ln0Weight, ln0Bias, lnOutWeight, lnOutBias,
             fc1Weight, fc1Bias, fc2Weight)
    }

    // ── Отбор состояния ──

    /// Состояние базы → `[B, nUnique, H, S, S]`: только читаемые головой слои.
    ///
    /// Это единица кэширования при обучении на замороженной базе: пара
    /// (документ, запрос) сворачивается в такой тензор один раз, дальше
    /// обучение головы состояние не пересчитывает.
    public func select(_ state: RWKVBatchState) -> MLXArray {
        Self.select(state, sources: uniqueSources)
    }

    /// Состояние базы → `[B, sources.count, H, S, S]` по ЯВНОМУ списку слоёв.
    ///
    /// Отдельно от `select(_:)` ради кэша НАДМНОЖЕСТВА: одно кодирование
    /// обслуживает несколько конфигураций головы, а какие слои в нём лежат,
    /// задаётся снаружи и головой не определяется.
    public static func select(_ state: RWKVBatchState,
                              sources: [Int]) -> MLXArray {
        stacked(sources.map { state.layerWKV($0) }, axis: 1)
    }

    // ── Проход ──

    /// Отобранное состояние `[B, nUnique, H, S, S]` → скоры `[B]`.
    public func callAsFunction(_ selected: MLXArray) -> MLXArray {
        apply(selected, parameters: parameters)
    }

    /// Состояние базы → скоры `[B]`.
    public func callAsFunction(_ state: RWKVBatchState) -> MLXArray {
        callAsFunction(select(state))
    }

    /// Вариант с явными параметрами — нужен, чтобы градиент тёк к ним внутри
    /// grad-замыкания (подстановка обязана происходить ВНУТРИ, иначе цепь к
    /// fp32-мастеру рвётся).
    public func apply(_ selected: MLXArray, parameters ps: [MLXArray]) -> MLXArray {
        let B = selected.shape[0]
        var i = 0
        func next() -> MLXArray { defer { i += 1 }; return ps[i] }

        let probe = next()
        let ln0W = next(), ln0B = next()

        var x = broadcast(probe.expandedDimensions(axis: 0),
                          to: [B, cfg.nProbe, dim])
        x = layerNormLast(x, ln0W, ln0B)

        var vFirst: MLXArray? = nil
        for (bi, block) in blocks.enumerated() {
            let count = block.weightKeys.count
            var over: [String: MLXArray] = [:]
            for key in block.weightKeys { over[key] = next() }
            precondition(over.count == count)
            block.wOverride = over
            defer { block.wOverride = nil }

            // Состояние берётся у БАЗЫ, а не у предыдущего блока: голова
            // читает разные слои, а не продолжает свою же рекуррентность.
            let hIn = selected[0..., sourceSlot[bi]].asType(.float32)
            let (xo, vf, _) = block(x, vFirst, hIn: hIn)
            x = xo
            vFirst = vf
        }

        let lnOutW = next(), lnOutB = next()
        let fc1W = next(), fc1B = next(), fc2W = next()

        // Скор снимается с ПОСЛЕДНЕГО зонда: каждый следующий читает
        // состояние, уже зная, что прочитали предыдущие.
        let lastProbe = x[0..., cfg.nProbe - 1]                      // [B, D]
        let h = layerNormLast(lastProbe, lnOutW, lnOutB)
        let mid = tanh(matmul(h, fc1W.transposed()) + fc1B)
        return matmul(mid, fc2W.transposed()).squeezed(axis: -1)     // [B]
    }

    private func layerNormLast(_ x: MLXArray, _ weight: MLXArray,
                               _ bias: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let varc = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) / sqrt(varc + eps) * weight + bias
    }

    // ── Параметры ──
    //
    // Порядок фиксирован и совпадает с порядком чтения в `apply`. Он же —
    // порядок в TrainableSet, в моментах Adam и в чекпоинте, поэтому меняться
    // он не должен, а если меняется — то во всех трёх местах сразу.

    public var parameterNames: [String] {
        var names = ["probe", "ln0.weight", "ln0.bias"]
        for (i, block) in blocks.enumerated() {
            names += block.weightKeys.map { "blocks.\(i).\($0)" }
        }
        names += ["ln_out.weight", "ln_out.bias",
                  "score_fc1.weight", "score_fc1.bias", "score_fc2.weight"]
        return names
    }

    public var parameters: [MLXArray] {
        var ps = [probe, ln0Weight, ln0Bias]
        for block in blocks {
            ps += block.weightKeys.map { block.param($0) }
        }
        ps += [lnOutWeight, lnOutBias, fc1Weight, fc1Bias, fc2Weight]
        return ps
    }

    public func setParameters(_ ps: [MLXArray]) {
        precondition(ps.count == parameterNames.count,
                     "ожидалось \(parameterNames.count) параметров, получено \(ps.count)")
        var i = 0
        func next() -> MLXArray { defer { i += 1 }; return ps[i] }
        probe = next(); ln0Weight = next(); ln0Bias = next()
        for block in blocks {
            var upd: [String: MLXArray] = [:]
            for key in block.weightKeys { upd[key] = next() }
            block.setWeights(upd)
        }
        lnOutWeight = next(); lnOutBias = next()
        fc1Weight = next(); fc1Bias = next(); fc2Weight = next()
    }

    /// Число обучаемых параметров.
    public var parameterCount: Int {
        parameters.reduce(0) { $0 + $1.size }
    }
}
