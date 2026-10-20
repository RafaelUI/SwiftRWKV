//
//  LossMaskTests.swift
//  Маска лосса языковой модели (20.10): цель < 0 — токен вне лосса, лосс —
//  среднее по вошедшим. Нужна обучению состояния на ответах (вопрос и
//  добивка не учатся). Без отрицательных целей — прежний путь побитно.
//
import XCTest
import MLX
import MLXRandom
import MLXNN
@testable import RWKVGen

final class LossMaskTests: XCTestCase {

    func tiny() -> (X070Backbone, X070Config) { TinyBackbone.make(nLayer: 3, seed: 42) }

    func batch(_ cfg: X070Config) -> LoRABatch {
        let ids = TinyBackbone.ids(2, 33, vocab: cfg.vocab, seed: 7)
        return (x: ids[0..., 0 ..< 32], y: ids[0..., 1 ... 32])
    }

    /// Цели: строка 0 — вопрос (первые 12) вне лосса; строка 1 — хвост
    /// (последние 8, добивка) вне лосса.
    func masked(_ b: LoRABatch) -> LoRABatch {
        var m = b.y.asArray(Int32.self)
        for t in 0 ..< 12 { m[t] = -1 }
        for t in 24 ..< 32 { m[32 + t] = -1 }
        return (x: b.x, y: MLXArray(m, [2, 32]))
    }

    func randomState(_ bb: X070Backbone) -> RWKVBatchState {
        MLXRandom.seed(3)
        let z = bb.zeroInitialState()
        return RWKVBatchState(wkv: MLXRandom.normal(z.wkv.shape) * 0.05, tmixShift: z.tmixShift,
                              cmixShift: z.cmixShift).evaluated()
    }

    func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self) / (MLX.abs(a).max().item(Float.self) + 1e-12)
    }

    /// Лосс с маской = среднее кросс-энтропии по вошедшим токенам, посчитанное
    /// вручную; кусками — то же.
    func testMaskedLossIsMeanOverKeptTokens() {
        let (bb, cfg) = tiny()
        let b = masked(batch(cfg))
        let logits = bb(b.x).asType(.float32)
        let lp = logits - logSumExp(logits, axis: -1, keepDims: true)
        let y = b.y.asArray(Int32.self)
        var sum: Float = 0, n = 0
        for r in 0 ..< 2 { for t in 0 ..< 32 where y[r * 32 + t] >= 0 {
            sum -= lp[r, t, Int(y[r * 32 + t])].item(Float.self); n += 1
        } }
        XCTAssertEqual(n, 32 + 32 - 12 - 8)
        let whole = LoRAFinetune.languageModelLoss(bb, b).item(Float.self)
        let chunked = LoRAFinetune.languageModelLoss(bb, b, chunk: 16).item(Float.self)
        XCTAssertEqual(whole, sum / Float(n), accuracy: 1e-5 * abs(whole))
        XCTAssertEqual(chunked, sum / Float(n), accuracy: 1e-5 * abs(whole))
    }

    /// Без отрицательных целей — прежняя формула побитно.
    func testUnmaskedIsOldFormula() {
        let (bb, cfg) = tiny()
        let b = batch(cfg)
        let logits = bb(b.x)
        let old = crossEntropy(logits: logits.reshaped([64, cfg.vocab]), targets: b.y.reshaped([64]).asType(.int32),
                               reduction: .mean)
        XCTAssertEqual(LoRAFinetune.languageModelLoss(bb, b).item(Float.self), old.item(Float.self))
    }

    /// Градиент состояния с маской: по блокам (целиком и кусками) = автоград;
    /// строка, целиком вне лосса, ничего не добавляет: батч = одна строка 0.
    func testMaskedGradients() {
        let (bb, cfg) = tiny()
        let set = StateTrainableSet(bb, initial: randomState(bb))
        let ps = set.initialParameters()
        let b = masked(batch(cfg))
        func autograd(_ bt: LoRABatch) -> [MLXArray] {
            let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
                set.inject(p); return [LoRAFinetune.languageModelLoss(bb, bt).asType(.float32)]
            }, argumentNumbers: Array(ps.indices))
            let g = vg(ps).1; eval(g); set.inject(ps); return g
        }
        let ref = autograd(b)
        for chunk in [0, 16] {
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: chunk)
            eval([r.loss] + r.gradients)
            for i in ps.indices { XCTAssertLessThan(rel(ref[i], r.gradients[i]), 1e-5, "кусок \(chunk), \(i)") }
        }
        // строка 1 целиком вне лосса
        var y1 = b.y.asArray(Int32.self)
        for t in 32 ..< 64 { y1[t] = -1 }
        let onlyRow0: LoRABatch = (x: b.x, y: MLXArray(y1, [2, 32]))
        let alone: LoRABatch = (x: b.x[0 ..< 1], y: b.y[0 ..< 1])
        let g2 = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: onlyRow0, lossChunk: 0).gradients
        let g1 = autograd(alone)
        for i in ps.indices { XCTAssertLessThan(rel(g1[i], g2[i]), 1e-5, "строка вне лосса повлияла: \(i)") }
    }
}
