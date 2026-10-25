//
//  StateTuningCastTests.swift
//  Типы в шаге state-tuning (23.10): приведение входа плотной проекции к типу веса
//  (`X070Backbone.castDenseProjectionInput`) и выход WKV в типе вычислений
//  (`castWKVOutputToComputeDType`); `StateTuneConfig.castProjectionInput` /
//  `castWKVOutput` выставляют их на время `StateTuning.run` и возвращают прежние.
//
//  Бэкбон — крошечный, но в bf16: в fp32 оба приведения — тождество, и тесты
//  ничего бы не видели.
//
import XCTest
import MLX
@testable import RWKVGen

final class StateTuningCastTests: XCTestCase {

    func bf16Tiny(seed: UInt64 = 42) -> (X070Backbone, X070Config) {
        let cfg = TinyBackbone.config(nLayer: 2, nEmbd: 128, vocab: 64)
        return (X070Backbone(weights: TinyBackbone.weights(cfg, seed: seed), cfg: cfg, computeDType: .bfloat16), cfg)
    }

    func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype
            && allClose(a.asType(.float32), b.asType(.float32), rtol: 0, atol: 0).item(Bool.self)
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        (MLX.abs(ref.asType(.float32) - got.asType(.float32)).max()
            / (MLX.abs(ref.asType(.float32)).max() + 1e-12)).item(Float.self)
    }

    func batch(_ cfg: X070Config, B: Int, T: Int, seed: UInt64) -> LoRABatch {
        let ids = TinyBackbone.ids(B, T + 1, vocab: cfg.vocab, seed: seed)
        return (x: ids[0..., 0 ..< T], y: ids[0..., 1 ... T])
    }

    /// Проекция с флагом — матмул в типе веса, результат в типе входа; без флага
    /// fp32-вход считает в fp32 (вес поднимается). На входе в типе веса флаг —
    /// тождество. Голова и проекции обоих миксов — одним путём.
    func testProjectionCastIsMatmulInWeightType() {
        let (bb, cfg) = bf16Tiny()
        MLXRandom.seed(7)
        let x = MLXRandom.normal([1, 16, cfg.nEmbd])
        for key in ["blocks.0.tmix.r_proj.weight", "blocks.1.tmix.o_proj.weight", "blocks.1.cmix.key.weight", "head.weight"] {
            let W = bb.weight(key)
            XCTAssertEqual(W.dtype, .bfloat16, key)
            let xin = x
            bb.castDenseProjectionInput = false
            let off = bb.projectForBlock(xin, key, lora: nil)
            bb.castDenseProjectionInput = true
            let on = bb.projectForBlock(xin, key, lora: nil)
            let onHalf = bb.projectForBlock(xin.asType(.bfloat16), key, lora: nil)
            bb.castDenseProjectionInput = false
            let offHalf = bb.projectForBlock(xin.asType(.bfloat16), key, lora: nil)
            eval(off, on, onHalf, offHalf)
            XCTAssertEqual(on.dtype, .float32, "\(key): выход проекции обязан остаться в типе входа")
            XCTAssertTrue(same(on, matmul(xin.asType(.bfloat16), W.transposed()).asType(.float32)),
                          "\(key): с флагом — не матмул в типе веса")
            XCTAssertTrue(same(off, matmul(xin, W.transposed())), "\(key): без флага путь изменился")
            XCTAssertFalse(same(on, off), "\(key): флаг ничего не меняет на fp32-входе")
            XCTAssertTrue(same(onHalf, offHalf), "\(key): на входе в типе веса флаг обязан быть тождеством")
        }
    }

    /// С приведением градиент состояния по блокам == автоград по всему графу
    /// (оба — с тем же флагом); оба приведения меняют градиент.
    func testSequentialGradientMatchesAutogradWithCasts() {
        for (what, proj, wkv) in [("вход проекций", true, false), ("выход WKV", false, true), ("оба", true, true)] {
            let (bb, cfg) = bf16Tiny()
            bb.castDenseProjectionInput = proj; bb.castWKVOutputToComputeDType = wkv
            let set = StateTrainableSet(bb, initial: nil)
            let ps = set.initialParameters()
            let b = batch(cfg, B: 2, T: 32, seed: 5)
            let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
                set.inject(p)
                return [LoRAFinetune.languageModelLoss(bb, b, chunk: 0).asType(.float32)]
            }, argumentNumbers: Array(ps.indices))
            let (v, gRef) = vg(ps); eval(v + gRef); set.inject(ps)
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 0)
            eval([r.loss] + r.gradients)
            var worst: Float = 0
            for i in ps.indices { worst = Swift.max(worst, rel(gRef[i], r.gradients[i])) }
            // без приведения — тот же счёт; градиент обязан отличаться
            bb.castDenseProjectionInput = false; bb.castWKVOutputToComputeDType = false
            let plain = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 0)
            eval(plain.gradients)
            print(String(format: "ПРИВЕДЕНИЕ %@: по блокам к автограду %.1e; к счёту без приведения %.1e",
                         what, worst, rel(plain.gradients[0], r.gradients[0])))
            XCTAssertLessThan(worst, 1e-2, what)    // bf16: та же решётка округлений, разный порядок сложения
            XCTAssertEqual(r.loss.item(Float.self), v[0].item(Float.self), accuracy: 1e-3 * Swift.abs(v[0].item(Float.self)), what)
            XCTAssertFalse(same(plain.gradients[0], r.gradients[0]), "\(what): приведение не дошло до градиента")
        }
    }

    func testDefaults() {
        let c = StateTuneConfig()
        XCTAssertTrue(c.castWKVOutput, "умолчание — приведение выхода WKV (25.10): быстрее, исход тот же")
        XCTAssertFalse(c.castProjectionInput, "приведение входа проекций — по выбору")
        let (bb, _) = bf16Tiny()
        XCTAssertFalse(bb.castDenseProjectionInput, "у самого бэкбона флаг выключен: инференс не меняется")
        XCTAssertFalse(bb.castWKVOutputToComputeDType)
    }

    /// Результат обучения зависит только от StateTuneConfig, а не от флагов, с
    /// которыми пришёл бэкбон; после обучения флаги — прежние.
    func testRunSetsFlagsFromConfigAndRestores() {
        func tune(_ c: StateTuneConfig, pre: (Bool, Bool)) -> (RWKVBatchState, (Bool, Bool)) {
            let (bb, cfg) = bf16Tiny()
            bb.castDenseProjectionInput = pre.0; bb.castWKVOutputToComputeDType = pre.1
            let b = batch(cfg, B: 1, T: 32, seed: 13)
            var during: [(Bool, Bool)] = []
            var cl = c; cl.logEvery = 1
            let out = StateTuning.run(bb, nextBatch: { b }, config: cl, onStep: { _, _, _, _ in
                during.append((bb.castDenseProjectionInput, bb.castWKVOutputToComputeDType))
            })
            eval(out.state.wkv)
            XCTAssertFalse(during.isEmpty)
            XCTAssertTrue(during.allSatisfy { $0 == (c.castProjectionInput, c.castWKVOutput) },
                          "во время обучения флаги \(during.first!), а в конфигурации \((c.castProjectionInput, c.castWKVOutput))")
            return (out.state, (bb.castDenseProjectionInput, bb.castWKVOutputToComputeDType))
        }
        var base = StateTuneConfig()
        base.lr = 0.05; base.lrMin = 0.01; base.warmupSteps = 0; base.maxSteps = 3; base.logEvery = 0
        base.cacheLimitGB = 0; base.lossChunkTokens = 0
        var noCast = base; noCast.castProjectionInput = false; noCast.castWKVOutput = false
        var lin = base; lin.castProjectionInput = true; lin.castWKVOutput = false

        let (a, after1) = tune(base, pre: (false, false))
        let (a2, after2) = tune(base, pre: (true, true))
        let (n, after3) = tune(noCast, pre: (true, false))
        let (n2, _) = tune(noCast, pre: (false, true))
        let (l, after4) = tune(lin, pre: (false, false))
        let (l2, _) = tune(lin, pre: (false, true))
        XCTAssertTrue(same(a.wkv, a2.wkv), "умолчание: исход зависит от флагов бэкбона")
        XCTAssertTrue(same(n.wkv, n2.wkv), "без приведения: исход зависит от флагов бэкбона")
        XCTAssertTrue(same(l.wkv, l2.wkv), "вход проекций: исход зависит от флагов бэкбона")
        XCTAssertFalse(same(a.wkv, n.wkv), "castWKVOutput (умолчание) не действует")
        XCTAssertFalse(same(l.wkv, n.wkv), "castProjectionInput не действует")
        XCTAssertTrue(after1 == (false, false) && after2 == (true, true) && after3 == (true, false)
                      && after4 == (false, false), "флаги бэкбона после обучения не возвращены")
    }
}
