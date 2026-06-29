import Foundation
import MLX
import MLXNN
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  RWKV-7 "Goose" x070 — точный порт rwkv_metal/model/rwkv7_x070.py для
//  загрузки официальных World-весов. Функциональный forward (без nn.Module-
//  дерева) ради прозрачного паритета logits против Python-эталона.
//
//  Отличия от RWKVBackbone (RWKVTrain, версия rwkv7 from-scratch):
//   1. decay w: sigmoid(w0 + B(tanh(A(xw))))     (w0 = bias w_lora_B)
//   2. iclr a:  sigmoid(a0 + B(A(xa)))           БЕЗ tanh  (a0 = bias a_lora_B)
//   3. gate g:  B(sigmoid(A(xg)))                sigmoid ВНУТРИ, линейно наружу
//   4. ln_x:    GroupNorm по головам (eps=64e-5, pytorch_compatible)
//   5. порядок: WKV → ln_x → +bonus → *g
//   6. token-shift: нулевой паддинг t=0 в каждом блоке, БЕЗ межблочного переноса
//   7. cmix FFN размер D*4
// ───────────────────────────────────────────────────────────────────────

/// Конфиг x070-модели.
public struct X070Config: Sendable {
    public var nLayer: Int
    public var nEmbd: Int
    public var headSize: Int
    public var vocab: Int
    public var nHead: Int { nEmbd / headSize }

    public init(nLayer: Int, nEmbd: Int, headSize: Int = 64, vocab: Int) {
        self.nLayer = nLayer
        self.nEmbd = nEmbd
        self.headSize = headSize
        self.vocab = vocab
    }
}

// ─────────────────── Утилиты ───────────────────

private func l2norm(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}

private func layerNorm(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
                       eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let varc = (x - mean).square().mean(axis: -1, keepDims: true)
    let normed = (x - mean) / sqrt(varc + eps)
    return normed * weight + bias
}

// Linear без bias: x[...,in] @ Wᵀ, W хранится [out, in].
private func linear(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
    matmul(x, w.transposed())
}
private func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    matmul(x, w.transposed()) + b
}

// ─────────────────── Backbone ───────────────────

/// Квантованная (frozen) матрица: упакованные веса + scale/bias по группам.
struct QuantBase {
    var wq: MLXArray
    var scales: MLXArray
    var biases: MLXArray?
    var groupSize: Int
    var bits: Int
}

/// x070-backbone поверх плоского словаря весов (MLX-имена тензоров).
public final class X070Backbone {
    public let cfg: X070Config
    var w: [String: MLXArray]

    // ── LoRA / QLoRA / training hooks ──
    // Инертны для обычного инференса (пустые словари + trainLayers пуст) ⇒
    // поведение идентично исходному, parity-тесты остаются зелёными.
    /// Слои, чьё WKV считается через дифференцируемое ядро (wkv7Train).
    public var trainLayers: Set<Int> = []
    /// Подмена весов на обучаемые (fp32) во время valueAndGrad для full-weight
    /// partial-finetune верхних слоёв. nil ⇒ читаются frozen-веса (инференс не меняется).
    public var wOverride: [String: MLXArray]? = nil
    /// Если true — каждый блок (ln1+tmix+resid+ln2+cmix+resid) считается через
    /// gradient checkpoint (recompute в backward). Дифф-входы (x, v_first, LoRA)
    /// протягиваются явными аргументами чекпоинта; frozen-веса захватываются.
    public var useBlockCheckpoint = false
    /// LoRA-адаптеры по таргету: "blocks.L.tmix.r_proj" и т.п.
    var loraA: [String: MLXArray] = [:]      // [rank, in]
    var loraB: [String: MLXArray] = [:]      // [out, rank]
    var loraScale: [String: Float] = [:]     // alpha / rank
    /// Квантованная замороженная база по имени веса (напр. "...r_proj.weight", "head.weight", "emb.weight").
    var quant: [String: QuantBase] = [:]

    // GroupNorm с pytorch_compatible-семантикой (eps=64e-5). Применяется
    // per-token к [N, D] (каждый токен нормализуется независимо по головам).
    private let groupNorm: GroupNorm

    public init(weights: [String: MLXArray], cfg: X070Config,
                computeDType: DType = .bfloat16) {
        self.cfg = cfg
        var conv: [String: MLXArray] = [:]
        for (k, v) in weights { conv[k] = v.asType(computeDType) }
        self.w = conv

        // affine=false: вес/смещение ln_x применяем вручную из весов модели,
        // т.к. они per-layer (blocks.N.tmix.ln_x.{weight,bias}).
        self.groupNorm = GroupNorm(groupCount: cfg.nHead, dimensions: cfg.nEmbd,
                                   eps: 64e-5, affine: false, pytorchCompatible: true)
    }

    // Чтение веса с учётом wOverride (обучаемая подмена) → frozen.
    private func wv(_ key: String) -> MLXArray { wOverride?[key] ?? w[key]! }
    private func g(_ key: String) -> MLXArray { wv(key) }

    // База проекции: если есть quant — x·Wᵀ с деквантизацией на лету
    // (веса [out,in] ⇒ transpose: true), иначе обычный linear.
    private func baseProj(_ x: MLXArray, _ wKey: String) -> MLXArray {
        if let q = quant[wKey] {
            return quantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                               transpose: true, groupSize: q.groupSize, bits: q.bits)
        }
        return matmul(x, wv(wKey).transposed())
    }

    // Проекция с опциональным LoRA: base + scale·(x·Aᵀ)·Bᵀ. target=nil ⇒ только база.
    private func proj(_ x: MLXArray, _ wKey: String, lora target: String?) -> MLXArray {
        let base = baseProj(x, wKey)
        guard let t = target, let a = loraA[t], let b = loraB[t] else { return base }
        let z = matmul(x, a.transposed())                       // [..., rank]
        return base + (loraScale[t] ?? 1.0) * matmul(z, b.transposed())
    }

    // Эмбеддинг: gather строк (с деквантизацией, если emb квантован).
    private func embed(_ ids: MLXArray) -> MLXArray {
        if let q = quant["emb.weight"] {
            let rows = q.wq.take(ids, axis: 0)
            let sc = q.scales.take(ids, axis: 0)
            let bi = q.biases?.take(ids, axis: 0)
            return dequantized(rows, scales: sc, biases: bi,
                               groupSize: q.groupSize, bits: q.bits,
                               dtype: w["emb.weight"]?.dtype ?? .bfloat16)
        }
        return wv("emb.weight").take(ids, axis: 0)
    }

    // token-shift: prev[t]=x[t-1], prev[0]=0 (нулевой паддинг). xx = prev - x.
    private func tokenShift(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0], T = x.shape[1], D = x.shape[2]
        let zero = MLXArray.zeros([B, 1, D], dtype: x.dtype)
        let shifted = concatenated([zero, x[0..., 0 ..< (T - 1)]], axis: 1)
        return shifted - x
    }

    // GroupNorm ln_x (per-token). Канон RWKV: ln_x(x.view(B*T, C)) — каждый
    // токен нормируется независимо по головам. КАУЗАЛЬНО: токен t не видит
    // t+1..T-1, поэтому параллельный путь совпадает с рекуррентным lnXStep.
    //
    // ВАЖНО: подаём [B*T, D], НЕ [B,T,D]. MLX GroupNorm на [B,T,D] трактует
    // batch=B и смешал бы все T внутри головы (cross-token, утечка будущего) —
    // это была причина старого расхождения уже на позиции 0. Регресс ловится
    // рекуррентным parity-тестом (recurrent vs parallel).
    private func lnX(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
        let B = x.shape[0], T = x.shape[1], D = x.shape[2]
        let normed = groupNorm(x.reshaped([B * T, D])).reshaped([B, T, D])
        return normed * weight + bias
    }

    // time-mix блока layer. Возвращает (выход, обновлённый v_first).
    private func tmix(_ x: MLXArray, _ vFirst: MLXArray?, _ layer: Int)
        -> (MLXArray, MLXArray) {
        let p = "blocks.\(layer).tmix."
        let B = x.shape[0], T = x.shape[1], D = cfg.nEmbd
        let H = cfg.nHead, S = cfg.headSize

        let xx = tokenShift(x)
        let xr = x + xx * g(p + "x_r")
        let xw = x + xx * g(p + "x_w")
        let xk = x + xx * g(p + "x_k")
        let xv = x + xx * g(p + "x_v")
        let xa = x + xx * g(p + "x_a")
        let xg = x + xx * g(p + "x_g")

        var r = proj(xr, p + "r_proj.weight", lora: p + "r_proj").reshaped([B, T, H, S])
        var k = proj(xk, p + "k_proj.weight", lora: p + "k_proj").reshaped([B, T, H, S])
        var v = proj(xv, p + "v_proj.weight", lora: p + "v_proj").reshaped([B, T, H, S])

        // gate: B(sigmoid(A(xg))) — sigmoid ВНУТРИ, линейно наружу, без bias.
        let gate = linear(sigmoid(linear(xg, g(p + "g_lora_A.weight"))),
                          g(p + "g_lora_B.weight"))

        // value-residual (слои > 0): v0 = bias v_lora_B
        var vFirstOut: MLXArray
        if layer == 0 {
            vFirstOut = v
        } else {
            let vv = sigmoid(linear(linear(xv, g(p + "v_lora_A.weight")),
                                    g(p + "v_lora_B.weight"), g(p + "v_lora_B.bias")))
                        .reshaped([B, T, H, S])
            v = v + (vFirst! - v) * vv
            vFirstOut = vFirst!
        }

        // iclr a: sigmoid(a0 + B(A(xa))) — БЕЗ tanh.
        let a = sigmoid(linear(linear(xa, g(p + "a_lora_A.weight")),
                               g(p + "a_lora_B.weight"), g(p + "a_lora_B.bias")))
                    .reshaped([B, T, H, S])

        // decay w: exp(-0.606531 * sigmoid(w0 + B(tanh(A(xw))))), reductions в fp32.
        var ww = linear(tanh(linear(xw, g(p + "w_lora_A.weight"))),
                        g(p + "w_lora_B.weight"), g(p + "w_lora_B.bias"))
        ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype)
        ww = ww.reshaped([B, T, H, S])

        // kk = l2norm(k * k_k);  k = k*(1+(a-1)*k_a)
        let kk = l2norm(k * g(p + "k_k"))
        k = k * (1.0 + (a - 1.0) * g(p + "k_a"))

        // WKV-7: a_kernel = -kk, b_kernel = kk * a
        var out = trainLayers.contains(layer)
            ? wkv7Train(r, ww, k, v, -kk, kk * a)
            : wkv7Forward(r, ww, k, v, -kk, kk * a)         // [B,T,H,S]

        // Порядок официала: ln_x (GroupNorm) ДО bonus.
        out = lnX(out.reshaped([B, T, D]), g(p + "ln_x.weight"), g(p + "ln_x.bias"))
              .reshaped([B, T, H, S])
        let bonus = (r * k * g(p + "r_k")).sum(axis: -1, keepDims: true) * v
        out = (out + bonus).reshaped([B, T, D])

        let res = proj(out * gate, p + "o_proj.weight", lora: p + "o_proj")
        return (res, vFirstOut)
    }

    // channel-mix: value(relu(key(xk))^2). token-shift свой, нулевой паддинг.
    private func cmix(_ x: MLXArray, _ layer: Int) -> MLXArray {
        let p = "blocks.\(layer).cmix."
        let xx = tokenShift(x)
        let xk = x + xx * g(p + "x_k")
        let h = relu(proj(xk, p + "key.weight", lora: p + "key"))
        return proj(h * h, p + "value.weight", lora: p + "value")
    }

    /// Один блок: ln1+tmix+resid+ln2+cmix+resid. (x0, vFirst?) -> (x', vFirstOut).
    func blockForward(_ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int) -> (MLXArray, MLXArray) {
        let (h, vf) = tmix(layerNorm(x0, g("blocks.\(layer).ln1.weight"),
                                     g("blocks.\(layer).ln1.bias")), vFirst, layer)
        var x = x0 + h
        x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                               g("blocks.\(layer).ln2.bias")), layer)
        return (x, vf)
    }

    /// Отсортированные ключи LoRA-таргетов слоя (детерминированный порядок упаковки).
    private func loraKeysForLayer(_ layer: Int) -> [String] {
        loraA.keys.filter { $0.hasPrefix("blocks.\(layer).") }.sorted()
    }

    /// blockForward через gradient checkpoint. LoRA-адаптеры слоя идут ЯВНЫМИ
    /// входами чекпоинта (иначе vjp не даст к ним градиент). v_first и x — тоже
    /// явные входы, чтобы grad тёк сквозь value-residual и остаточный поток.
    func blockCheckpointed(_ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int) -> (MLXArray, MLXArray) {
        let keys = loraKeysForLayer(layer)
        let hasV = vFirst != nil
        var inputs: [MLXArray] = [x0]
        if hasV { inputs.append(vFirst!) }
        for t in keys { inputs.append(loraA[t]!); inputs.append(loraB[t]!) }

        let f: ([MLXArray]) -> [MLXArray] = { ins in
            var i = 0
            let xin = ins[i]; i += 1
            var vin: MLXArray? = nil
            if hasV { vin = ins[i]; i += 1 }
            for t in keys {
                self.loraA[t] = ins[i]; i += 1
                self.loraB[t] = ins[i]; i += 1
            }
            let (xo, vo) = self.blockForward(xin, vin, layer)
            return [xo, vo]
        }
        let out = checkpointed(f)(inputs)
        return (out[0], out[1])
    }

    /// body: всё кроме головы. ids [B,T] → ln_out [B,T,D].
    public func body(_ ids: MLXArray) -> MLXArray {
        let emb = embed(ids)        // [B,T,D]
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil
        for layer in 0 ..< cfg.nLayer {
            let (xo, vf): (MLXArray, MLXArray) = useBlockCheckpoint
                ? blockCheckpointed(x, vFirst, layer)
                : blockForward(x, vFirst, layer)
            x = xo
            vFirst = vf
        }
        return layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
    }

    /// Полный forward в логиты: ids [B,T] → logits [B,T,vocab].
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        proj(body(ids), "head.weight", lora: nil)
    }

    // ─────────── Partial-finetune: разрез сети на слое f ───────────
    // token-shift внутриблочный ⇒ между блоками течёт только (x, vFirst),
    // межблочного xPrev НЕТ. Поэтому граница = (x после блока f-1, vFirst).

    /// Frozen-проход блоков [0..<f]. Возвращает (x, vFirst) на границе.
    /// Без gradient checkpoint — этот участок не обучается.
    public func boundaryState(_ ids: MLXArray, upTo f: Int) -> (MLXArray, MLXArray?) {
        let emb = embed(ids)
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil
        for layer in 0 ..< f {
            let (xo, vf) = blockForward(x, vFirst, layer)
            x = xo
            vFirst = vf
        }
        return (x, vFirst)
    }

    /// Обучаемый хвост: блоки [f..<nLayer] + ln_out. Граничные (x, vFirst)
    /// приходят из boundaryState (или из дискового кэша). Уважает useBlockCheckpoint.
    public func forwardFrom(_ x0: MLXArray, _ vFirst0: MLXArray?, from f: Int) -> MLXArray {
        var x = x0
        var vFirst = vFirst0
        for layer in f ..< cfg.nLayer {
            let (xo, vf): (MLXArray, MLXArray) = useBlockCheckpoint
                ? blockCheckpointed(x, vFirst, layer)
                : blockForward(x, vFirst, layer)
            x = xo
            vFirst = vf
        }
        return layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
    }

    /// vFirst (= v слоя 0) для пересчёта на train-шаге без хранения на диске.
    /// Слой 0 frozen ⇒ вызывать вне grad-тейпа.
    public func vFirstFrom(_ ids: MLXArray) -> MLXArray {
        let (_, vFirst) = boundaryState(ids, upTo: 1)
        return vFirst!
    }

    // ── Отладка паритета: промежуточные этапы (internal, для тестов) ──
    func debugPerLayer(_ ids: MLXArray) -> [String: MLXArray] {
        let emb = embed(ids)
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil
        var out: [String: MLXArray] = [:]
        for layer in 0 ..< cfg.nLayer {
            let (h, vf) = tmix(layerNorm(x, g("blocks.\(layer).ln1.weight"),
                                         g("blocks.\(layer).ln1.bias")), vFirst, layer)
            vFirst = vf
            x = x + h
            x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                                   g("blocks.\(layer).ln2.bias")), layer)
            out["after_blk\(layer)"] = x
        }
        return out
    }

    func debugTmix(_ ids: MLXArray) -> [String: MLXArray] {
        let emb = embed(ids)
        let afterLn0 = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        let x = layerNorm(afterLn0, g("blocks.0.ln1.weight"), g("blocks.0.ln1.bias"))
        let p = "blocks.0.tmix."
        let B = x.shape[0], T = x.shape[1], D = cfg.nEmbd, H = cfg.nHead, S = cfg.headSize
        let xx = tokenShift(x)
        let xr = x + xx * g(p+"x_r"), xw = x + xx * g(p+"x_w"), xk = x + xx * g(p+"x_k")
        let xv = x + xx * g(p+"x_v"), xa = x + xx * g(p+"x_a"), xg = x + xx * g(p+"x_g")
        let r = linear(xr, g(p+"r_proj.weight")).reshaped([B,T,H,S])
        let k = linear(xk, g(p+"k_proj.weight")).reshaped([B,T,H,S])
        let v = linear(xv, g(p+"v_proj.weight")).reshaped([B,T,H,S])
        let gate = linear(sigmoid(linear(xg, g(p+"g_lora_A.weight"))), g(p+"g_lora_B.weight"))
        let a = sigmoid(linear(linear(xa, g(p+"a_lora_A.weight")),
                               g(p+"a_lora_B.weight"), g(p+"a_lora_B.bias"))).reshaped([B,T,H,S])
        var ww = linear(tanh(linear(xw, g(p+"w_lora_A.weight"))),
                        g(p+"w_lora_B.weight"), g(p+"w_lora_B.bias"))
        ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype).reshaped([B,T,H,S])
        let kk = l2norm(k * g(p+"k_k"))
        let k2 = k * (1.0 + (a - 1.0) * g(p+"k_a"))
        let wkv = wkv7Forward(r, ww, k2, v, -kk, kk * a)
        let outLnx = lnX(wkv.reshaped([B,T,D]), g(p+"ln_x.weight"), g(p+"ln_x.bias")).reshaped([B,T,H,S])
        let bonus = (r * k2 * g(p+"r_k")).sum(axis: -1, keepDims: true) * v
        let outF = (outLnx + bonus).reshaped([B,T,D])
        let res = linear(outF * gate, g(p+"o_proj.weight"))
        return ["r":r,"k":k,"v":v,"g":gate,"a":a,"w":ww,"kk":kk,"k2":k2,
                "wkv":wkv,"out_lnx":outLnx,"res":res]
    }

    func debugStages(_ ids: MLXArray) -> [String: MLXArray] {
        let emb = embed(ids)
        let afterLn0 = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        let (h, _) = tmix(layerNorm(afterLn0, g("blocks.0.ln1.weight"),
                                    g("blocks.0.ln1.bias")), nil, 0)
        var x2 = afterLn0 + h
        x2 = x2 + cmix(layerNorm(x2, g("blocks.0.ln2.weight"),
                                 g("blocks.0.ln2.bias")), 0)
        return ["after_ln0": afterLn0, "blk0_tmix": h, "after_blk0": x2]
    }
}
