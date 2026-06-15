import Foundation
import MLX
import MLXNN
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Инкрементальный (рекуррентный) инференс x070 для генерации (B=1).
//
//  RWKV — RNN: состояние полностью описывает контекст, не нужен KV-кэш по
//  длине. Состояние на слой:
//    • wkv      [H,D,D]  — матрица состояния WKV-ядра,
//    • tmixPrev [1,D]    — предыдущий x для token-shift в tmix,
//    • cmixPrev [1,D]    — предыдущий x для token-shift в cmix.
//  Плюс глобальный v_first [1,H,D] (value первого слоя, фиксируется на слое 0).
//
//  Decode по одному токену: чистые MLX-операции (вариант A), без Metal-ядра —
//  на 1 токен матрица [H,D,D] мала и это быстро.
// ───────────────────────────────────────────────────────────────────────

/// Рекуррентное состояние модели (B=1).
public struct RWKVState {
    public var wkv: [MLXArray]        // [nLayer] × [H, D, D]  fp32
    public var tmixPrev: [MLXArray]   // [nLayer] × [1, D]
    public var cmixPrev: [MLXArray]   // [nLayer] × [1, D]
    public var vFirst: MLXArray?      // [1, H, D] (устанавливается на слое 0)

    /// Пустое состояние (нули) для модели заданной геометрии.
    public init(cfg: X070Config, dtype: DType = .bfloat16) {
        let H = cfg.nHead, D = cfg.headSize, E = cfg.nEmbd
        wkv = (0 ..< cfg.nLayer).map { _ in MLXArray.zeros([H, D, D], dtype: .float32) }
        tmixPrev = (0 ..< cfg.nLayer).map { _ in MLXArray.zeros([1, E], dtype: dtype) }
        cmixPrev = (0 ..< cfg.nLayer).map { _ in MLXArray.zeros([1, E], dtype: dtype) }
        vFirst = nil
    }

    /// Зафиксировать состояние (eval) — полезно между шагами генерации.
    public mutating func eval() {
        MLX.eval(wkv + tmixPrev + cmixPrev + (vFirst.map { [$0] } ?? []))
    }
}

extension X070Backbone {

    private func gg(_ key: String) -> MLXArray { w[key]! }

    // ─────────────── Один рекуррентный WKV-шаг (вариант A) ───────────────
    // Вход: r,w,k,v,a,b — [H,D];  h — [H,D,D] (h[head, dv, dk]).
    // Возврат: (out [H,D], h' [H,D,D]). Всё в fp32 (как ядро).
    private func wkvStep(_ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
                         _ a: MLXArray, _ b: MLXArray, _ h: MLXArray) -> (MLXArray, MLXArray) {
        let rf = r.asType(.float32), wf = w.asType(.float32), kf = k.asType(.float32)
        let vf = v.asType(.float32), af = a.asType(.float32), bf = b.asType(.float32)

        // sa[head,dv] = Σ_dk h[head,dv,dk] * a[head,dk]
        let sa = (h * af.expandedDimensions(axis: 1)).sum(axis: -1)          // [H,D]

        // h'[head,dv,dk] = w[dk]*h + v[dv]*k[dk] + sa[dv]*b[dk]
        let wTerm = h * wf.expandedDimensions(axis: 1)                       // [H,D,D] (по dk)
        let vk = vf.expandedDimensions(axis: 2) * kf.expandedDimensions(axis: 1)  // [H,D,D]
        let sab = sa.expandedDimensions(axis: 2) * bf.expandedDimensions(axis: 1) // [H,D,D]
        let hNew = wTerm + vk + sab

        // out[head,dv] = Σ_dk h'[head,dv,dk] * r[head,dk]
        let out = (hNew * rf.expandedDimensions(axis: 1)).sum(axis: -1)      // [H,D]
        return (out, hNew)
    }

    // ─────────────── tmix для одного токена ───────────────
    // x: [1, D] (выход ln1). Возвращает (res [1,D]) и мутирует state.
    private func tmixStep(_ x: MLXArray, _ layer: Int, _ state: inout RWKVState) -> MLXArray {
        let p = "blocks.\(layer).tmix."
        let D = cfg.nEmbd, H = cfg.nHead, S = cfg.headSize

        let prev = state.tmixPrev[layer]      // [1,D]
        let xx = prev - x                     // token-shift: prev - x
        state.tmixPrev[layer] = x             // обновляем shift-state

        let xr = x + xx * gg(p+"x_r"), xw = x + xx * gg(p+"x_w"), xk = x + xx * gg(p+"x_k")
        let xv = x + xx * gg(p+"x_v"), xa = x + xx * gg(p+"x_a"), xg = x + xx * gg(p+"x_g")

        let r = linear_(xr, gg(p+"r_proj.weight")).reshaped([H, S])
        let k0 = linear_(xk, gg(p+"k_proj.weight")).reshaped([H, S])
        var v = linear_(xv, gg(p+"v_proj.weight")).reshaped([H, S])

        let gate = linear_(sigmoid(linear_(xg, gg(p+"g_lora_A.weight"))), gg(p+"g_lora_B.weight"))

        if layer == 0 {
            state.vFirst = v.reshaped([1, H, S])
        } else {
            let vv = sigmoid(linear_(linear_(xv, gg(p+"v_lora_A.weight")),
                                     gg(p+"v_lora_B.weight"), gg(p+"v_lora_B.bias"))).reshaped([H, S])
            let vf = state.vFirst!.reshaped([H, S])
            v = v + (vf - v) * vv
        }

        let a = sigmoid(linear_(linear_(xa, gg(p+"a_lora_A.weight")),
                                gg(p+"a_lora_B.weight"), gg(p+"a_lora_B.bias"))).reshaped([H, S])

        var ww = linear_(tanh(linear_(xw, gg(p+"w_lora_A.weight"))),
                         gg(p+"w_lora_B.weight"), gg(p+"w_lora_B.bias"))
        ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype).reshaped([H, S])

        let kk = l2normLast(k0 * gg(p+"k_k"))
        let k = k0 * (1.0 + (a - 1.0) * gg(p+"k_a"))

        // WKV-шаг (fp32)
        let (outHD, hNew) = wkvStep(r, ww, k, v, -kk, kk * a, state.wkv[layer])
        state.wkv[layer] = hNew

        // ln_x (GroupNorm) для одного токена: [1, D]
        var out = lnXStep(outHD.reshaped([1, D]), gg(p+"ln_x.weight"), gg(p+"ln_x.bias"))
            .reshaped([H, S])
        // bonus = (r*k*r_k).sum(-1, keepdims) * v
        let bonus = (r * k * gg(p+"r_k")).sum(axis: -1, keepDims: true) * v   // [H,S]
        out = out + bonus

        return linear_(out.reshaped([1, D]) * gate, gg(p+"o_proj.weight"))
    }

    // cmix для одного токена.
    private func cmixStep(_ x: MLXArray, _ layer: Int, _ state: inout RWKVState) -> MLXArray {
        let p = "blocks.\(layer).cmix."
        let prev = state.cmixPrev[layer]
        let xx = prev - x
        state.cmixPrev[layer] = x
        let xk = x + xx * gg(p+"x_k")
        let h = relu(linear_(xk, gg(p+"key.weight")))
        return linear_(h * h, gg(p+"value.weight"))
    }

    // ─────────────── Публичные методы генерации ───────────────

    /// Обработать промпт (prefill). Прогоняет токены по одному, обновляя state.
    /// Возвращает logits последнего токена [vocab] (не evaluated).
    /// Для B=1; ids — массив id токенов промпта.
    public func prefill(_ ids: [Int], state: inout RWKVState) -> MLXArray {
        precondition(!ids.isEmpty, "prefill: пустой промпт")
        var logits = MLXArray.zeros([cfg.vocab])
        for id in ids {
            logits = step(id, state: &state)
        }
        return logits
    }

    /// Один шаг декодирования: id токена → logits [vocab] (не evaluated).
    public func step(_ id: Int, state: inout RWKVState) -> MLXArray {
        let emb = gg("emb.weight")[id].reshaped([1, cfg.nEmbd])    // [1,D]
        var x = layerNorm_(emb, gg("ln0.weight"), gg("ln0.bias"))
        for layer in 0 ..< cfg.nLayer {
            let h = tmixStep(layerNorm_(x, gg("blocks.\(layer).ln1.weight"),
                                        gg("blocks.\(layer).ln1.bias")), layer, &state)
            x = x + h
            x = x + cmixStep(layerNorm_(x, gg("blocks.\(layer).ln2.weight"),
                                        gg("blocks.\(layer).ln2.bias")), layer, &state)
        }
        let lnOut = layerNorm_(x, gg("ln_out.weight"), gg("ln_out.bias"))
        return linear_(lnOut, gg("head.weight")).reshaped([cfg.vocab])
    }
}

// ─────────────── Локальные утилиты (повтор для extension-доступа) ───────────────

private func l2normLast(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}
private func layerNorm_(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
                        eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let varc = (x - mean).square().mean(axis: -1, keepDims: true)
    return (x - mean) / sqrt(varc + eps) * weight + bias
}
private func linear_(_ x: MLXArray, _ w: MLXArray) -> MLXArray { matmul(x, w.transposed()) }
private func linear_(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    matmul(x, w.transposed()) + b
}
