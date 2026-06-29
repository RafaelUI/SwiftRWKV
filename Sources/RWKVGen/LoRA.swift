import Foundation
import MLX
import MLXRandom

// ───────────────────────────────────────────────────────────────────────
//  LoRA / QLoRA для RWKV-7 x070 — порт rwkv_metal/lora/lora.py в
//  функциональном стиле поверх X070Backbone (без nn.Module-дерева).
//
//  Адаптеры ставятся на проекции tmix (r/k/v/o_proj), опционально на cmix
//  (key/value). Градиенты для r/k/v_proj текут через дифференцируемое WKV-ядро
//  (wkv7Train на trainLayers), для o_proj — напрямую после WKV.
//
//  y = W·x (frozen, возможно 4/8-бит) + scale·B(A(x)),   scale = alpha/rank.
//  B инициализируется нулём ⇒ в начале адаптер — no-op, выход равен базе.
//
//  Заморозка решается выбором diff-множества в LoRAFinetune (только loraA/loraB),
//  а не freeze()-деревом — функциональный аналог nn.value_and_grad.
// ───────────────────────────────────────────────────────────────────────

public enum LoRATargets {
    public static let tmix = ["r_proj", "k_proj", "v_proj", "o_proj"]
    public static let cmix = ["key", "value"]
}

/// Большие frozen-матрицы для QLoRA-квантизации (как BIG_QUANT_TARGETS в Python).
/// Низкоранговые w/a/g/v НЕ квантуем — их ранги не кратны group_size и квант
/// портит in-context динамику.
public let bigQuantTargets = ["cmix.key", "cmix.value", "head", "emb"]

public struct LoRASpec: Sendable {
    public var rank: Int
    public var alpha: Float
    public var tmixTargets: [String]
    public var cmixTargets: [String]
    public var layers: Range<Int>?     // nil = все блоки
    public var quantizeBits: Int       // 0 = bf16 база; 4/8 = QLoRA по таргетам
    public var quantGroupSize: Int

    public init(rank: Int = 16, alpha: Float = 16,
                tmixTargets: [String] = LoRATargets.tmix,
                cmixTargets: [String] = [],
                layers: Range<Int>? = nil,
                quantizeBits: Int = 0, quantGroupSize: Int = 64) {
        self.rank = rank; self.alpha = alpha
        self.tmixTargets = tmixTargets; self.cmixTargets = cmixTargets
        self.layers = layers
        self.quantizeBits = quantizeBits; self.quantGroupSize = quantGroupSize
    }
}

public struct LoRAInfo: Sendable {
    public let totalParams: Int
    public let trainableParams: Int
    public let trainablePct: Double
    public let numAdapters: Int
    public let wrappedPerBlock: [String]
}

public enum LoRA {

    // ── базовая геометрия веса [out, in] ──
    private static func outIn(_ w: MLXArray) -> (Int, Int) { (w.shape[0], w.shape[1]) }

    /// Квантовать одну frozen-матрицу в backbone.quant и выбросить bf16-копию.
    private static func quantizeWeight(_ bb: X070Backbone, _ wKey: String,
                                       bits: Int, groupSize: Int) {
        guard bb.quant[wKey] == nil, let W = bb.w[wKey] else { return }
        let (wq, sc, bi) = quantized(W, groupSize: groupSize, bits: bits)
        bb.quant[wKey] = QuantBase(wq: wq, scales: sc, biases: bi,
                                   groupSize: groupSize, bits: bits)
        bb.w[wKey] = nil                                   // освобождаем bf16-базу
    }

    /// Квантовать ТОЛЬКО большие frozen-матрицы (QLoRA), in-place.
    /// Вызывать ДО add(...) с тем же bits, чтобы add доквантовал r/k/v/o_proj.
    @discardableResult
    public static func quantizeBaseModel(_ bb: X070Backbone, bits: Int = 4,
                                         groupSize: Int = 64,
                                         targets: [String] = bigQuantTargets) -> Int {
        var n = 0
        for wKey in Array(bb.w.keys) {
            guard wKey.hasSuffix(".weight") else { continue }
            let stem = String(wKey.dropLast(".weight".count))   // "...cmix.key", "head", "emb"
            if targets.contains(where: { stem == $0 || stem.hasSuffix("." + $0) }) {
                quantizeWeight(bb, wKey, bits: bits, groupSize: groupSize)
                n += 1
            }
        }
        eval(bb.quant.values.flatMap { [$0.wq, $0.scales] + ($0.biases.map { [$0] } ?? []) })
        return n
    }

    /// Навесить LoRA-адаптеры на целевые проекции, пометить слои обучаемыми.
    @discardableResult
    public static func add(to bb: X070Backbone, spec: LoRASpec) -> LoRAInfo {
        let nLayer = bb.cfg.nLayer
        let sel = spec.layers ?? (0 ..< nLayer)
        let baseDType = bb.w["emb.weight"]?.dtype
            ?? bb.w.values.first?.dtype ?? .bfloat16
        let scale = spec.alpha / Float(spec.rank)

        // total params ДО возможного дропа bf16-базы при квантизации:
        var total = bb.w.values.reduce(0) { $0 + $1.size }
        total += bb.quant.values.reduce(0) { $0 + $1.scales.size * $1.groupSize }

        var wrapped = Set<String>()
        var nAdapters = 0
        var newAdapters: [MLXArray] = []

        func attach(_ wKey: String, _ target: String, _ shortName: String) {
            guard let W = bb.w[wKey] ?? dequantShape(bb, wKey) else { return }
            let (outF, inF) = outIn(W)
            let a = (MLXRandom.normal([spec.rank, inF]) * (1.0 / sqrt(Float(inF))))
                        .asType(baseDType)
            let b = MLXArray.zeros([outF, spec.rank], dtype: baseDType)
            bb.loraA[target] = a; bb.loraB[target] = b; bb.loraScale[target] = scale
            newAdapters.append(a); newAdapters.append(b)
            wrapped.insert(shortName); nAdapters += 1
            if spec.quantizeBits > 0 {
                quantizeWeight(bb, wKey, bits: spec.quantizeBits, groupSize: spec.quantGroupSize)
            }
        }

        for li in sel {
            for name in spec.tmixTargets {
                attach("blocks.\(li).tmix.\(name).weight", "blocks.\(li).tmix.\(name)", "tmix.\(name)")
            }
            for name in spec.cmixTargets {
                attach("blocks.\(li).cmix.\(name).weight", "blocks.\(li).cmix.\(name)", "cmix.\(name)")
            }
        }

        // WKV обучаемых слоёв — через дифференцируемое ядро.
        bb.trainLayers = Set(sel)
        eval(newAdapters)

        let trainable = bb.loraA.values.reduce(0) { $0 + $1.size }
                      + bb.loraB.values.reduce(0) { $0 + $1.size }
        return LoRAInfo(totalParams: total, trainableParams: trainable,
                        trainablePct: 100.0 * Double(trainable) / Double(max(1, total)),
                        numAdapters: nAdapters, wrappedPerBlock: wrapped.sorted())
    }

    // Форма веса даже если он уже квантован (out, in) — для повторного add поверх QLoRA.
    private static func dequantShape(_ bb: X070Backbone, _ wKey: String) -> MLXArray? {
        guard let q = bb.quant[wKey] else { return nil }
        // out = scales.shape[0]; in = scales.shape[1]*groupSize
        let outF = q.scales.shape[0], inF = q.scales.shape[1] * q.groupSize
        return MLXArray.zeros([outF, inF])               // только для чтения формы
    }

    // ── сохранение / загрузка / слияние адаптеров ──

    /// Только обучаемые тензоры: ключи "<target>.lora_a" / "<target>.lora_b".
    public static func adapterState(_ bb: X070Backbone) -> [String: MLXArray] {
        var d: [String: MLXArray] = [:]
        for (t, a) in bb.loraA { d[t + ".lora_a"] = a }
        for (t, b) in bb.loraB { d[t + ".lora_b"] = b }
        return d
    }

    public static func save(_ bb: X070Backbone, to url: URL) throws {
        let st = adapterState(bb); eval(Array(st.values))
        try MLX.save(arrays: st, url: url)
    }

    /// Загрузить тензоры адаптеров в УЖЕ навешенную через add(...) модель
    /// (scale/слои берутся из add). Перезаписывает loraA/loraB по имени.
    public static func load(_ bb: X070Backbone, from url: URL) throws {
        let w = try loadArrays(url: url)
        let dt = bb.loraA.values.first?.dtype ?? .bfloat16
        for (k, v) in w {
            if k.hasSuffix(".lora_a") { bb.loraA[String(k.dropLast(7))] = v.asType(dt) }
            else if k.hasSuffix(".lora_b") { bb.loraB[String(k.dropLast(7))] = v.asType(dt) }
        }
        eval(Array(bb.loraA.values) + Array(bb.loraB.values))
    }

    /// In-place слияние LoRA обратно в базу (для инференса/экспорта):
    /// W ← W + scale·(B·A); квант-база деквантуется в bf16. Сбрасывает адаптеры.
    public static func merge(_ bb: X070Backbone) {
        let bf: DType = .bfloat16
        for t in Array(bb.loraA.keys) {
            guard let a = bb.loraA[t], let b = bb.loraB[t] else { continue }
            let wKey = t + ".weight"
            let scale = bb.loraScale[t] ?? 1.0
            let baseW: MLXArray
            if let q = bb.quant[wKey] {
                baseW = dequantized(q.wq, scales: q.scales, biases: q.biases,
                                    groupSize: q.groupSize, bits: q.bits, dtype: bf)
                bb.quant[wKey] = nil
            } else if let w0 = bb.w[wKey] {
                baseW = w0
            } else { continue }
            let delta = scale * matmul(b, a)                  // [out,rank]·[rank,in]=[out,in]
            bb.w[wKey] = (baseW.asType(.float32) + delta.asType(.float32)).asType(bf)
            bb.loraA[t] = nil; bb.loraB[t] = nil; bb.loraScale[t] = nil
        }
        bb.trainLayers = []
        eval(bb.w.values.map { $0 })
    }
}
