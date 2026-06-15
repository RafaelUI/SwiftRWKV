import Foundation
import MLX
import MLXNN   // для gelu/relu/sigmoid/tanh при желании; используем mx-функции
import RWKVKernel

// ─────────────────────── Конфиг ru10m ───────────────────────
struct RWKVConfig {
    var nLayer = 6
    var nEmbd  = 256
    var vocab  = 16000
    var headSize = 64
    var nHead: Int { nEmbd / headSize }
}

// ─────────────────── Утилиты ───────────────────
private func l2norm(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}

// LayerNorm с весом и смещением по последней оси (eps как в nn.LayerNorm = 1e-5)
private func layerNorm(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
                       eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let varc = (x - mean).square().mean(axis: -1, keepDims: true)
    let normed = (x - mean) / sqrt(varc + eps)
    return normed * weight + bias
}

// Linear без bias: x[...,D] @ Wᵀ, где W хранится [out, in]
private func linear(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
    matmul(x, w.transposed())
}
private func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    matmul(x, w.transposed()) + b
}

// ─────────────────── Backbone ───────────────────
final class RWKVBackbone {
    let cfg: RWKVConfig
    var w: [String: MLXArray]

    // fp32 — основной режим: стабильнее и точнее, а по ПИКОВОЙ памяти не дороже bf16
    // (пик держат активации, не веса; bf16-веса экономят только диск/загрузку).
    // Параметр computeDType оставлен в сигнатуре для совместимости вызовов, но
    // паритет и фичи работают в fp32. WKV-ядро (WKV7.swift) всегда fp32.
    init(weights: [String: MLXArray], cfg: RWKVConfig = RWKVConfig(),
         computeDType: DType = .float32) {
        self.cfg = cfg
        var conv: [String: MLXArray] = [:]
        for (k, v) in weights { conv[k] = v.asType(computeDType) }
        self.w = conv
    }

    // частичный файнтюн: обучаемые веса и слои с дифференцируемым WKV7
    var wOverride: [String: MLXArray]? = nil
    var trainLayers: Set<Int> = []
    private func g(_ key: String) -> MLXArray { wOverride?[key] ?? w[key]! }

    // token-shift: xx = concat(x_prev, x[:,:-1]) - x
    private func tokenShift(_ x: MLXArray, _ xPrev: MLXArray) -> MLXArray {
        let T = x.shape[1]
        let shifted = concatenated([xPrev, x[0..., 0 ..< (T - 1)]], axis: 1)
        return shifted - x
    }

    // time-mix блока layer. Возвращает (выход, обновлённый v_first)
    private func tmix(_ x: MLXArray, _ xPrev: MLXArray, _ vFirst: MLXArray?,
                      _ layer: Int) -> (MLXArray, MLXArray) {
        let p = "blocks.\(layer).tmix."
        let B = x.shape[0], T = x.shape[1], D = cfg.nEmbd
        let H = cfg.nHead, S = cfg.headSize

        let xx = tokenShift(x, xPrev)
        let xr = x + xx * g(p + "x_r")
        let xw = x + xx * g(p + "x_w")
        let xk = x + xx * g(p + "x_k")
        let xv = x + xx * g(p + "x_v")
        let xa = x + xx * g(p + "x_a")
        let xg = x + xx * g(p + "x_g")

        var r = linear(xr, g(p + "r_proj.weight")).reshaped([B, T, H, S])
        var k = linear(xk, g(p + "k_proj.weight")).reshaped([B, T, H, S])
        var v = linear(xv, g(p + "v_proj.weight")).reshaped([B, T, H, S])

        // gate = sigmoid(g_B(gelu(g_A(xg))))
        let gate = sigmoid(linear(gelu(linear(xg, g(p + "g_lora_A.weight"))),
                                  g(p + "g_lora_B.weight")))

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

        // iclr = sigmoid(a_B(tanh(a_A(xa))))
        let iclr = sigmoid(linear(tanh(linear(xa, g(p + "a_lora_A.weight"))),
                                  g(p + "a_lora_B.weight"), g(p + "a_lora_B.bias")))
                    .reshaped([B, T, H, S])

        // w = exp(-0.606531 * sigmoid(w_B(tanh(w_A(xw)))))
        var ww = sigmoid(linear(tanh(linear(xw, g(p + "w_lora_A.weight"))),
                                g(p + "w_lora_B.weight"))).reshaped([B, T, H, S])
        ww = exp(-0.606531 * ww)

        // kk = l2norm(k * k_k);  k = k*(1+(iclr-1)*k_a);  a=-kk; b=kk*iclr
        let kk = l2norm(k * g(p + "k_k"))
        k = k * (1.0 + (iclr - 1.0) * g(p + "k_a"))
        let a = -kk
        let b = kk * iclr

        // WKV-7 forward (наше ядро)
        var out = trainLayers.contains(layer)
            ? wkv7Train(r, ww, k, v, a, b)
            : wkv7Forward(r, ww, k, v, a, b)        // [B,T,H,S]

        // bonus = (r*k*r_k).sum(-1, keepdims) * v
        let bonus = (r * k * g(p + "r_k")).sum(axis: -1, keepDims: true) * v
        out = (out + bonus).reshaped([B, T, D])

        out = layerNorm(out, g(p + "ln_x.weight"), g(p + "ln_x.bias"))
        let res = linear(out * gate, g(p + "o_proj.weight"))
        return (res, vFirstOut)
    }

    // channel-mix: value(relu(key(xk))^2)
    private func cmix(_ x: MLXArray, _ xPrev: MLXArray, _ layer: Int) -> MLXArray {
        let p = "blocks.\(layer).cmix."
        let xx = tokenShift(x, xPrev)
        let xk = x + xx * g(p + "x_k")
        let h = relu(linear(xk, g(p + "key.weight")))
        return linear(h * h, g(p + "value.weight"))
    }

    // Полный forward: ids [B,T] -> ln_out [B,T,D]
    func forward(_ ids: MLXArray) -> MLXArray {
        let B = ids.shape[0]
        let emb = g("emb.weight").take(ids, axis: 0)   // gather -> [B,T,D]
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var xPrev = MLXArray.zeros([B, 1, cfg.nEmbd], dtype: .float32)
        var vFirst: MLXArray? = nil
        for layer in 0 ..< cfg.nLayer {
            let (h, vf) = tmix(layerNorm(x, g("blocks.\(layer).ln1.weight"),
                                         g("blocks.\(layer).ln1.bias")),
                               xPrev, vFirst, layer)
            vFirst = vf
            x = x + h
            x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                                   g("blocks.\(layer).ln2.bias")),
                         xPrev, layer)
            xPrev = x[0..., (x.shape[1] - 1)...]
        }
        return layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
    }

    // ───── частичный файнтюн ─────
    // frozen forward слоёв [0, upTo): состояние на входе слоя upTo.
    func boundaryState(_ ids: MLXArray, upTo: Int)
        -> (x: MLXArray, xPrev: MLXArray, vFirst: MLXArray) {
        let B = ids.shape[0]
        let emb = g("emb.weight").take(ids, axis: 0)
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var xPrev = MLXArray.zeros([B, 1, cfg.nEmbd], dtype: .float32)
        var vFirst: MLXArray? = nil
        for layer in 0 ..< upTo {
            let (h, vf) = tmix(layerNorm(x, g("blocks.\(layer).ln1.weight"),
                                         g("blocks.\(layer).ln1.bias")),
                               xPrev, vFirst, layer)
            vFirst = vf
            x = x + h
            x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                                   g("blocks.\(layer).ln2.bias")),
                         xPrev, layer)
            xPrev = x[0..., (x.shape[1] - 1)...]
        }
        return (x, xPrev, vFirst!)
    }

    // forward слоёв [from, nLayer) от готового состояния + ln_out.
    // Обучаемые слои должны быть в trainLayers, веса — в wOverride.
    func forwardFrom(_ x0: MLXArray, _ xPrev0: MLXArray, _ vFirst0: MLXArray,
                     from: Int) -> MLXArray {
        var x = x0, xPrev = xPrev0
        var vFirst: MLXArray? = vFirst0
        for layer in from ..< cfg.nLayer {
            let (h, vf) = tmix(layerNorm(x, g("blocks.\(layer).ln1.weight"),
                                         g("blocks.\(layer).ln1.bias")),
                               xPrev, vFirst, layer)
            vFirst = vf
            x = x + h
            x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                                   g("blocks.\(layer).ln2.bias")),
                         xPrev, layer)
            xPrev = x[0..., (x.shape[1] - 1)...]
        }
        return layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
    }

    // vFirst (v-проекция слоя 0) из id — пересчёт вместо кэширования.
    func vFirstFrom(_ ids: MLXArray) -> MLXArray {
        let B = ids.shape[0]
        let emb = g("emb.weight").take(ids, axis: 0)
        let x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        let xPrev = MLXArray.zeros([B, 1, cfg.nEmbd], dtype: .float32)
        let (_, vf) = tmix(layerNorm(x, g("blocks.0.ln1.weight"),
                                     g("blocks.0.ln1.bias")), xPrev, nil, 0)
        return vf
    }
}

