import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Обратный проход по блокам (`LoRAConfig.sequentialBackward`) и лосс кусками
//  (`lossChunkTokens`): оба меняют только то, что живёт в памяти одновременно.
//  Градиент обязан остаться тем же, что у автограда по всему графу, — это
//  здесь и проверяется, по каждому параметру.
// ───────────────────────────────────────────────────────────────────────

final class SequentialBackwardTests: XCTestCase {

    func batch(_ cfg: X070Config, B: Int = 1, T: Int = 48, seed: UInt64 = 7) -> LoRABatch {
        let ids = TinyBackbone.ids(B, T + 1, vocab: cfg.vocab, seed: seed)
        return (x: ids[0..., 0 ..< T], y: ids[0..., 1 ... T])
    }

    /// Модель с адаптерами, у которых B НЕ нулевой: при нулевом B градиент по
    /// A тождественно нуль, и половина сверки была бы пустой.
    func model(nLayer: Int = 4, layers: Range<Int>? = nil, cmix: Bool = false,
               seed: UInt64 = 42) -> (X070Backbone, X070Config) {
        let (bb, cfg) = TinyBackbone.make(nLayer: nLayer, seed: seed)
        LoRA.add(to: bb, spec: LoRASpec(rank: 4, alpha: 8, cmixTargets: cmix ? ["key", "value"] : [],
                                        layers: layers))
        liven(bb)
        return (bb, cfg)
    }

    func liven(_ bb: X070Backbone) {
        MLXRandom.seed(99)
        for t in bb.loraTargets {
            let b = bb.loraB[t]!
            bb.loraB[t] = (MLXRandom.normal(b.shape) * 0.05).asType(b.dtype)
        }
        eval(Array(bb.loraB.values))
    }

    /// Автоград по всему графу — то, с чем сверяем.
    func whole(_ bb: X070Backbone, _ set: LoRATrainableSet, _ ps: [MLXArray],
               _ b: LoRABatch, chunk: Int = 0) -> (Float, [MLXArray]) {
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            set.inject(p)
            return [LoRAFinetune.languageModelLoss(bb, b, chunk: chunk).asType(.float32)]
        }, argumentNumbers: Array(ps.indices))
        let (v, g) = vg(ps)
        eval(v + g)
        set.inject(ps)
        return (v[0].item(Float.self), g)
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        MLX.abs(ref.asType(.float32) - got.asType(.float32)).max().item(Float.self)
            / (MLX.abs(ref.asType(.float32)).max().item(Float.self) + 1e-12)
    }

    /// Худшее относительное расхождение по параметрам; попутно — ни один
    /// градиент эталона не нулевой (иначе сверять нечего).
    func compare(_ ref: [MLXArray], _ got: [MLXArray], _ names: [String], _ what: String,
                 bound: Float, file: StaticString = #filePath, line: UInt = #line) -> Float {
        XCTAssertEqual(ref.count, got.count, file: file, line: line)
        var worst: Float = 0
        for (i, (r, g)) in zip(ref, got).enumerated() {
            XCTAssertGreaterThan(MLX.abs(r).max().item(Float.self), 0, "\(what): эталонный градиент \(names[i]) нулевой",
                                 file: file, line: line)
            let e = rel(r, g)
            worst = Swift.max(worst, e)
            XCTAssertLessThan(e, bound, "\(what): \(names[i])", file: file, line: line)
        }
        return worst
    }

    // ── обратный проход по блокам ────────────────────────────────────

    func testSequentialGradientsMatchAutograd() {
        for (what, layers, cmix, B, chunk) in [
            ("все слои", nil as Range<Int>?, false, 1, 0),
            ("верхние слои", 2 ..< 4, false, 1, 0),
            ("один верхний слой", 3 ..< 4, false, 1, 0),
            ("с адаптерами cmix", nil, true, 1, 0),
            ("батч 2", nil, false, 2, 0),
            ("лосс кусками", nil, false, 2, 16),
            ("лосс неровными кусками", 1 ..< 4, true, 1, 20),
        ] {
            let (bb, cfg) = model(layers: layers, cmix: cmix)
            let set = LoRATrainableSet(bb)
            let ps = set.initialParameters()
            let b = batch(cfg, B: B)
            let (lossRef, gRef) = whole(bb, set, ps, b)
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: chunk)
            eval([r.loss] + r.gradients)
            let lossGot = r.loss.item(Float.self)
            // Без кусков — ПОБИТНО автоград (замерено 0): те же операции в том же
            // порядке. С кусками меняется порядок сложения в голове, а котангенты
            // идут через адаптеры в bf16 (решётка 2^-8): замерено 1.3e-3…1.9e-3.
            let worst = compare(gRef, r.gradients, set.parameterNames, what, bound: chunk > 0 ? 1e-2 : 1e-6)
            print(String(format: "ПО БЛОКАМ, %@: лосс %.6f против %.6f; худший градиент %.1e (%d параметров)",
                         what, lossGot, lossRef, worst, gRef.count))
            XCTAssertEqual(lossGot, lossRef, accuracy: Swift.abs(lossRef) * 1e-5, what)
        }
    }

    /// После счёта в модели — значения адаптеров, а не трассеры градиента:
    /// модель считается дальше и даёт то же, что до вызова.
    func testModelIsLeftWithPlainAdapters() {
        let (bb, cfg) = model()
        let set = LoRATrainableSet(bb)
        let ps = set.initialParameters()
        let b = batch(cfg)
        set.inject(ps)
        let before = bb(b.x); eval(before)
        let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 16)
        eval(r.gradients)
        let after = bb(b.x); eval(after)
        XCTAssertEqual(MLX.abs(before - after).max().item(Float.self), 0)
        XCTAssertEqual(bb.trainLayers, Set(0 ..< 4))
    }

    /// Через тренер: тот же путь обучения, что без режима.
    func testTrainerWithSequentialBackwardFollowsThePlainRun() {
        func run(_ sequential: Bool) -> (Float, [String: MLXArray]) {
            let (bb, cfg) = model(seed: 5)
            var c = LoRAConfig(lr: 1e-2, maxSteps: 4, cacheLimitGB: 0, logEvery: 1)
            c.sequentialBackward = sequential
            c.lossChunkTokens = sequential ? 16 : 0
            let b = batch(cfg)
            let calls = LoRAFinetune.sequentialGradientCalls
            let res = LoRAFinetune.run(bb, nextBatch: { b }, config: c)
            // градиент побитно тот же — режим виден только по счётчику
            XCTAssertEqual(LoRAFinetune.sequentialGradientCalls - calls, sequential ? 4 : 0,
                           "режим \(sequential ? "не включился" : "включился без спроса")")
            return (res.finalLoss, bb.loraB)
        }
        let (lossPlain, bPlain) = run(false), (lossSeq, bSeq) = run(true)
        XCTAssertEqual(lossSeq, lossPlain, accuracy: Swift.abs(lossPlain) * 1e-4)
        var worst: Float = 0
        for (k, v) in bPlain { worst = Swift.max(worst, rel(v, bSeq[k]!)) }
        print(String(format: "ПО БЛОКАМ через тренер: лосс %.6f против %.6f, адаптеры после 4 шагов %.1e",
                     lossSeq, lossPlain, worst))
        XCTAssertLessThan(worst, 2e-2)      // замерено 0; допуск — решётка bf16 адаптеров
    }

    /// Квантованная база (0.1B REDUCTION): градиент идёт через свой VJP
    /// проекций и ядра WKV — режим обязан сойтись с автоградом и здесь.
    func testSequentialGradientsOnQuantisedBase() throws {
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("reduction"))
        LoRA.add(to: bb, spec: LoRASpec(rank: 4, alpha: 8, layers: 8 ..< 12))
        liven(bb)
        let set = LoRATrainableSet(bb)
        let ps = set.initialParameters()
        let ids = MLXArray((0 ..< 33).map { Int32(($0 * 7919 + 13) % 60000 + 100) }, [1, 33])
        let b: LoRABatch = (x: ids[0..., 0 ..< 32], y: ids[0..., 1 ... 32])
        let (lossRef, gRef) = whole(bb, set, ps, b)
        let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 16)
        eval([r.loss] + r.gradients)
        // замерено 0: и здесь тот же счёт; допуск — на порядок сложения в кусках
        let worst = compare(gRef, r.gradients, set.parameterNames, ".rwkvq", bound: 1e-2)
        print(String(format: "ПО БЛОКАМ на .rwkvq: лосс %.5f против %.5f; худший градиент %.1e",
                     r.loss.item(Float.self), lossRef, worst))
        XCTAssertEqual(r.loss.item(Float.self), lossRef, accuracy: Swift.abs(lossRef) * 1e-3)
    }

    // ── лосс кусками ─────────────────────────────────────────────────

    func testChunkedLossMatchesWholeLoss() {
        let (bb, cfg) = model()
        let set = LoRATrainableSet(bb)
        let ps = set.initialParameters()
        let b = batch(cfg, B: 2)
        let (lossRef, gRef) = whole(bb, set, ps, b)
        for chunk in [16, 20, 47] {
            let (loss, g) = whole(bb, set, ps, b, chunk: chunk)
            let worst = compare(gRef, g, set.parameterNames, "кусок \(chunk)", bound: 1e-2)   // замерено 1.3e-3
            print(String(format: "ЛОСС КУСКАМИ по %d: %.6f против %.6f; худший градиент %.1e", chunk, loss, lossRef, worst))
            XCTAssertEqual(loss, lossRef, accuracy: Swift.abs(lossRef) * 1e-5)
        }
        // T <= куска — прежний путь побитно
        let (same, gSame) = whole(bb, set, ps, b, chunk: 48)
        XCTAssertEqual(same, lossRef)
        for (r, g) in zip(gRef, gSame) { XCTAssertEqual(MLX.abs(r - g).max().item(Float.self), 0) }
    }

    /// Обучаемая голова (претрейн, частичное дообучение): чекпоинт куска до
    /// неё градиент не донёс бы — лосс обязан идти прежним путём.
    func testChunkedLossFallsBackWhenTheHeadIsTrained() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2)
        let b = batch(cfg)
        let head = bb.w["head.weight"]!.asType(.float32)
        func gradHead(_ chunk: Int) -> MLXArray {
            let g = grad({ (p: [MLXArray]) -> [MLXArray] in
                bb.wOverride = ["head.weight": p[0]]
                return [LoRAFinetune.languageModelLoss(bb, b, chunk: chunk).asType(.float32)]
            }, argumentNumbers: [0])([head])[0]
            eval(g); bb.wOverride = nil
            return g
        }
        let plain = gradHead(0), chunked = gradHead(16)
        XCTAssertGreaterThan(MLX.abs(plain).max().item(Float.self), 0)
        XCTAssertEqual(MLX.abs(plain - chunked).max().item(Float.self), 0,
                       "при обучаемой голове лосс пошёл кусками")
        XCTAssertTrue(bb.headIsConstant)
    }
}
