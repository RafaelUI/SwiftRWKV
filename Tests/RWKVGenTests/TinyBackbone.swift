import Foundation
import MLX
import MLXRandom
@testable import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Крошечный X070Backbone со случайными весами — чтобы модельные тесты были
//  самодостаточны (как WKV7KernelParityTests: фиксированный seed, без файлов
//  с весами в репозитории).
//
//  Веса случайные и модель ничего осмысленного не предсказывает — и не должна.
//  Проверяемые свойства (продолжение == сплошной проход, независимость строки
//  от паддинга и соседей по батчу) — структурные: они обязаны выполняться при
//  ЛЮБЫХ весах. Обученные веса тут ничего не добавили бы, а зависимость от
//  внешнего файла сделала бы тест невоспроизводимым.
//
//  headSize фиксирован в 64: ядро WKV-7 скомпилировано под HEAD_SIZE=64.
// ───────────────────────────────────────────────────────────────────────

enum TinyBackbone {

    /// Официальная формула x070 для low-rank размерностей.
    static func loraRanks(_ D: Int) -> [String: Int] {
        func f(_ c: Double, _ p: Double) -> Int {
            max(32, Int((c * pow(Double(D), p) / 32).rounded()) * 32)
        }
        return ["w": f(1.8, 0.5), "a": f(1.8, 0.5),
                "v": f(1.3, 0.5), "g": f(0.6, 0.8)]
    }

    /// cfg по умолчанию: 2 слоя, D=128 (H=2), словарь 64.
    static func config(nLayer: Int = 2, nEmbd: Int = 128, vocab: Int = 64) -> X070Config {
        X070Config(nLayer: nLayer, nEmbd: nEmbd, headSize: 64, vocab: vocab)
    }

    /// Случайные веса под заданный cfg. Масштабы подобраны так, чтобы forward
    /// не уходил в насыщение и не вырождался в нули.
    static func weights(_ cfg: X070Config, seed: UInt64 = 42) -> [String: MLXArray] {
        MLXRandom.seed(seed)
        let D = cfg.nEmbd, H = cfg.nHead, S = cfg.headSize, V = cfg.vocab
        let r = loraRanks(D)

        func n(_ shape: [Int], _ scale: Float = 0.05) -> MLXArray {
            MLXRandom.normal(shape) * scale
        }

        var w: [String: MLXArray] = [
            "emb.weight":    n([V, D], 0.1),
            "ln0.weight":    MLXArray.ones([D]),
            "ln0.bias":      MLXArray.zeros([D]),
            "ln_out.weight": MLXArray.ones([D]),
            "ln_out.bias":   MLXArray.zeros([D]),
            "head.weight":   n([V, D], 0.1),
        ]

        for i in 0 ..< cfg.nLayer {
            let bp = "blocks.\(i)."
            w[bp + "ln1.weight"] = MLXArray.ones([D])
            w[bp + "ln1.bias"]   = MLXArray.zeros([D])
            w[bp + "ln2.weight"] = MLXArray.ones([D])
            w[bp + "ln2.bias"]   = MLXArray.zeros([D])

            let tp = bp + "tmix."
            // token-shift lerp: ненулевые, иначе сдвиг не влияет и тест на
            // перенос shift-состояния стал бы бессодержательным.
            for name in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                w[tp + name] = n([D], 0.3) + 0.5
            }
            w[tp + "k_k"] = MLXArray.ones([H, S]) + n([H, S], 0.1)
            w[tp + "k_a"] = n([H, S], 0.1)
            w[tp + "r_k"] = n([H, S], 0.1)

            for (name, rank) in [("w", r["w"]!), ("a", r["a"]!), ("g", r["g"]!)] {
                w[tp + "\(name)_lora_A.weight"] = n([rank, D])
                w[tp + "\(name)_lora_B.weight"] = n([D, rank])
            }
            w[tp + "w_lora_B.bias"] = n([D], 0.1)
            w[tp + "a_lora_B.bias"] = n([D], 0.1)
            // g_lora_B без bias — как в x070.

            if i > 0 {
                w[tp + "v_lora_A.weight"] = n([r["v"]!, D])
                w[tp + "v_lora_B.weight"] = n([D, r["v"]!])
                w[tp + "v_lora_B.bias"]   = n([D], 0.1)
            }

            for name in ["r_proj", "k_proj", "v_proj", "o_proj"] {
                w[tp + "\(name).weight"] = n([D, D])
            }
            w[tp + "ln_x.weight"] = MLXArray.ones([D])
            w[tp + "ln_x.bias"]   = MLXArray.zeros([D])

            let cp = bp + "cmix."
            w[cp + "x_k"]         = n([D], 0.3) + 0.5
            w[cp + "key.weight"]   = n([D * 4, D])
            w[cp + "value.weight"] = n([D, D * 4])
        }

        eval(Array(w.values))
        return w
    }

    /// Готовый бэкбон. computeDType fp32: тесты проверяют структурные
    /// тождества, и bf16 замусорил бы их собственным шумом округления.
    static func make(nLayer: Int = 2, nEmbd: Int = 128, vocab: Int = 64,
                     seed: UInt64 = 42) -> (X070Backbone, X070Config) {
        let cfg = config(nLayer: nLayer, nEmbd: nEmbd, vocab: vocab)
        let bb = X070Backbone(weights: weights(cfg, seed: seed), cfg: cfg,
                              computeDType: .float32)
        return (bb, cfg)
    }

    /// Детерминированный SplitMix64 — намеренно СВОЙ, а не Int32.random(in:).
    ///
    /// Int32.random(in:) берёт системный RNG и seed'ом не управляется: id
    /// получались бы разными при каждом запуске, и любой тест, сравнивающий
    /// ДВА прогона (детерминизм обучения, эквивалентность grad-accum), падал
    /// бы из-за разных данных, а не из-за кода. MLXRandom.seed() на него не
    /// влияет — он сеет только генератор MLX.
    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    /// Случайные, но воспроизводимые id формы [B, T].
    static func ids(_ B: Int, _ T: Int, vocab: Int = 64, seed: UInt64 = 1) -> MLXArray {
        var rng = SplitMix64(state: seed &* 0x2545F4914F6CDD1D &+ 0x9E3779B9)
        let flat = (0 ..< (B * T)).map { _ in Int32(rng.next() % UInt64(vocab)) }
        return MLXArray(flat, [B, T])
    }
}
