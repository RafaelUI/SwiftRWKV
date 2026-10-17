import XCTest
import MLX
import MLXRandom
import MLXNN
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Характеризационные тесты partial-finetune (обучение верхних N слоёв).
//
//  Как и для LoRA — фиксируем ПОВЕДЕНИЕ перед вынесением общего тренера.
//
//  Самое ценное здесь — тождество разреза:
//      forwardFrom(boundaryState(ids, upTo: f), from: f) == body(ids)
//  На нём стоит вся конструкция: если разрез сети на слое f не эквивалентен
//  сплошному проходу, то дисковый кэш границы кэширует не то, и обучение
//  идёт по слегка другой модели, чем инференс. Ядро такое поймать не может —
//  это свойство модели.
// ───────────────────────────────────────────────────────────────────────

final class PartialFinetuneCharacterizationTests: XCTestCase {

    static let ctxLen = 16      // обязана делиться на WKV7_CHUNK

    override func setUp() {
        super.setUp()
        X070PartialFinetune.clearCache()
    }

    override func tearDown() {
        X070PartialFinetune.clearCache()
        super.tearDown()
    }

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        eval(ref, got)
        return MLX.abs(ref - got).max().item(Float.self)
             / (MLX.abs(ref).max().item(Float.self) + 1e-9)
    }

    /// Разделимая игрушечная задача: класс определяется первым токеном.
    /// Модель со случайными весами не обязана её решить идеально, но сигнал
    /// в данных есть, значит accuracy обязана иметь возможность вырасти.
    func examples(_ n: Int, classes: Int = 2, vocab: Int = 64,
                  seed: UInt64 = 3) -> [X070Example] {
        (0 ..< n).map { i in
            let label = i % classes
            var ids = Array(TinyBackbone.ids(1, Self.ctxLen, vocab: vocab,
                                             seed: seed &+ UInt64(i))
                              .asArray(Int32.self)).map { Int($0) }
            ids[0] = label                       // метка закодирована в первом токене
            return X070Example(ids: ids, label: label)
        }
    }

    // ── Тождество разреза ────────────────────────────────────────────

    /// Разрез сети на слое f эквивалентен сплошному проходу — при любом f.
    func testSplitAtLayerEqualsFullBody() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 4)
        let ids = TinyBackbone.ids(2, Self.ctxLen, vocab: cfg.vocab)
        let full = bb.body(ids)

        for f in 0 ... cfg.nLayer {
            let (x, vFirst) = bb.boundaryState(ids, upTo: f)
            let tail = bb.forwardFrom(x, vFirst, from: f)
            XCTAssertLessThan(relDiff(full, tail), 1e-5,
                              "разрез на слое \(f) разошёлся со сплошным проходом")
        }
    }

    /// vFirstFrom пересчитывает ровно тот vFirst, что даёт граница на слое 1 —
    /// именно поэтому его не обязательно хранить на диске.
    func testVFirstFromMatchesBoundary() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let ids = TinyBackbone.ids(2, Self.ctxLen, vocab: cfg.vocab)
        let (_, vf) = bb.boundaryState(ids, upTo: 1)
        XCTAssertEqual(maxAbsDiff(vf!, bb.vFirstFrom(ids)), 0,
                       "vFirstFrom разошёлся с границей слоя 1")
    }

    // ── Дисковый кэш границы ─────────────────────────────────────────

    /// Кэш — это round-trip через диск в bf16. Проверяем и метаданные,
    /// и то, что прочитанное совпадает с посчитанным (с точностью bf16).
    func testBoundaryCacheRoundTrip() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 4)
        let ex = examples(6)
        let freeze = 2

        let cache = X070PartialFinetune.buildBoundaryCache(
            backbone: bb, examples: ex, freeze: freeze, name: "train",
            ctxLen: Self.ctxLen, batch: 4)

        XCTAssertEqual(cache.count, ex.count)
        XCTAssertEqual(cache.T, Self.ctxLen)
        XCTAssertEqual(cache.D, cfg.nEmbd)
        XCTAssertEqual(cache.label.map { Int($0) }, ex.map(\.label))
        XCTAssertEqual(cache.length, ex.map { min($0.ids.count, Self.ctxLen) })

        // прочитанное из кэша == посчитанное заново (bf16 ⇒ допуск ~1e-2)
        let idx = [0, 3, 5]
        let fromCache = cache.readX(idx)
        let idsBatch = cache.idsBatch(idx)
        let (fresh, _) = bb.boundaryState(idsBatch, upTo: freeze)
        XCTAssertLessThan(relDiff(fresh.asType(.float32), fromCache.asType(.float32)),
                          2e-2, "кэш границы вернул не то, что посчитал бэкбон")

        // порядок строк в кэше сохранён: строка 3 — это именно пример 3
        XCTAssertEqual(cache.label[3], Int32(ex[3].label))
    }

    /// clearCache() действительно убирает файлы: следующий build начинает с нуля,
    /// а не дописывает в хвост прошлого прогона.
    func testClearCacheRemovesFiles() {
        let (bb, _) = TinyBackbone.make(nLayer: 3)
        let c1 = X070PartialFinetune.buildBoundaryCache(
            backbone: bb, examples: examples(4), freeze: 1, name: "train",
            ctxLen: Self.ctxLen)
        XCTAssertEqual(c1.count, 4)

        X070PartialFinetune.clearCache()

        let c2 = X070PartialFinetune.buildBoundaryCache(
            backbone: bb, examples: examples(2), freeze: 1, name: "train",
            ctxLen: Self.ctxLen)
        XCTAssertEqual(c2.count, 2, "кэш не был очищен — счётчик поехал")
    }

    /// buildBoundaryCache обязан сбросить wOverride/trainLayers: кэш строится
    /// по FROZEN базе, и если в бэкбоне остались обучаемые подмены, кэш
    /// окажется собран по другой модели, чем та, что потом обучается.
    func testCacheBuildResetsTrainingHooks() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        bb.trainLayers = Set(0 ..< cfg.nLayer)
        bb.wOverride = ["ln_out.weight": MLXArray.zeros([cfg.nEmbd])]

        _ = X070PartialFinetune.buildBoundaryCache(
            backbone: bb, examples: examples(2), freeze: 1, name: "train",
            ctxLen: Self.ctxLen)

        XCTAssertNil(bb.wOverride, "wOverride не сброшен перед сборкой кэша")
        XCTAssertTrue(bb.trainLayers.isEmpty, "trainLayers не сброшены")
    }

    // ── Обучение ─────────────────────────────────────────────────────

    /// Обучаются ТОЛЬКО верхние слои + голова. Ключи результата обязаны
    /// содержать блоки [freeze..<nLayer], ln_out и голову — и ничего ниже.
    func testOnlyTopLayersAndHeadAreTrained() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 4)
        let freeze = 2
        let res = X070PartialFinetune.run(
            backbone: bb, cfg: cfg, numClasses: 2,
            trainSet: examples(8), valSet: examples(4, seed: 99),
            freeze: freeze, ctxLen: Self.ctxLen, epochs: 1, batchSize: 4, lr: 1e-3)

        let keys = Set(res.params.keys)
        XCTAssertTrue(keys.contains("head.weight") && keys.contains("head.bias"))
        XCTAssertTrue(keys.contains("ln_out.weight") && keys.contains("ln_out.bias"))
        for k in keys where k.hasPrefix("blocks.") {
            let layer = Int(k.split(separator: ".")[1])!
            XCTAssertGreaterThanOrEqual(layer, freeze,
                                        "в обучаемых оказался замороженный слой: \(k)")
        }
        for l in freeze ..< cfg.nLayer {
            XCTAssertTrue(keys.contains { $0.hasPrefix("blocks.\(l).") },
                          "слой \(l) должен обучаться, но его нет в результате")
        }
    }

    /// run() обязан оставить бэкбон чистым: hooks сброшены, иначе последующий
    /// инференс пойдёт через обучаемые подмены.
    func testRunLeavesBackboneClean() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        _ = X070PartialFinetune.run(
            backbone: bb, cfg: cfg, numClasses: 2,
            trainSet: examples(4), valSet: examples(4, seed: 99),
            freeze: 1, ctxLen: Self.ctxLen, epochs: 1, batchSize: 2, lr: 1e-3)
        XCTAssertNil(bb.wOverride)
        XCTAssertTrue(bb.trainLayers.isEmpty)
    }

    /// Лосс убывает на разделимой задаче — обучение действительно учит.
    func testLossDecreasesOnSeparableTask() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        var losses: [Float] = []
        _ = X070PartialFinetune.run(
            backbone: bb, cfg: cfg, numClasses: 2,
            trainSet: examples(16), valSet: examples(8, seed: 99),
            freeze: 2, ctxLen: Self.ctxLen, epochs: 4, batchSize: 4, lr: 1e-3,
            onStep: { _, loss, _ in losses.append(loss) })

        XCTAssertGreaterThan(losses.count, 4)
        XCTAssertTrue(losses.allSatisfy { $0.isFinite }, "NaN/inf в лоссе")
        let firstHalf = losses.prefix(losses.count / 2).reduce(0, +) / Float(losses.count / 2)
        let lastHalf = losses.suffix(losses.count / 2).reduce(0, +) / Float(losses.count / 2)
        XCTAssertLessThan(lastHalf, firstHalf,
                          "лосс не убыл: \(firstHalf) → \(lastHalf)")
    }

    /// Детерминизм при фиксированном seed — предпосылка любого регрессионного
    /// сравнения до/после рефакторинга.
    func testDeterministicGivenSeed() {
        func finalAcc() -> Float {
            let (bb, cfg) = TinyBackbone.make(nLayer: 3, seed: 42)
            X070PartialFinetune.clearCache()
            return X070PartialFinetune.run(
                backbone: bb, cfg: cfg, numClasses: 2,
                trainSet: examples(8), valSet: examples(4, seed: 99),
                freeze: 2, ctxLen: Self.ctxLen, epochs: 2, batchSize: 8, lr: 1e-3).valAcc
        }
        XCTAssertEqual(finalAcc(), finalAcc(), accuracy: 0,
                       "два одинаковых прогона дали разную accuracy")
    }

    /// Отмена прерывает обучение.
    func testCancellationStopsEarly() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        var steps = 0
        let tc = X070PartialFinetune.buildBoundaryCache(
            backbone: bb, examples: examples(32), freeze: 1, name: "train",
            ctxLen: Self.ctxLen)
        let vc = X070PartialFinetune.buildBoundaryCache(
            backbone: bb, examples: examples(4, seed: 99), freeze: 1, name: "val",
            ctxLen: Self.ctxLen)
        _ = X070PartialFinetune.train(
            backbone: bb, trainCache: tc, valCache: vc, cfg: cfg, numClasses: 2,
            freeze: 1, epochs: 10, batchSize: 4, lr: 1e-3,
            isCancelled: { steps >= 2 },
            onStep: { s, _, _ in steps = s })
        XCTAssertLessThan(steps, 80, "отмена не сработала: \(steps) шагов")
    }

    // ── Воспроизводимость по сиду ────────────────────────────────────

    /// `globalNoise` — разное состояние глобального MLXRandom перед обучением:
    /// с сидом результат от него зависеть не обязан (TinyBackbone сеет
    /// глобальный генератор сам, и без этого голова «от сида» и голова от
    /// глобального генератора были бы неразличимы).
    func seededRun(seed: UInt64?, globalNoise: UInt64 = 0, steps: inout [Float]) -> X070PartialFinetune.TrainResult {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3, seed: 42)
        MLXRandom.seed(1000 &+ globalNoise)
        X070PartialFinetune.clearCache()
        var losses: [Float] = []
        let r = X070PartialFinetune.run(
            backbone: bb, cfg: cfg, numClasses: 2,
            trainSet: examples(10), valSet: examples(4, seed: 99),
            freeze: 1, ctxLen: Self.ctxLen, epochs: 3, batchSize: 4, lr: 1e-3, seed: seed,
            onStep: { _, l, _ in losses.append(l) })
        steps = losses
        return r
    }

    /// С сидом два прогона ПОБИТНО одинаковы (веса, лоссы по шагам, точность);
    /// другой сид — другие веса. Предпосылка побитной сверки переноса на
    /// общий Trainer. (Неполный последний батч: 10 примеров по 4.)
    func testSeededRunsAreBitIdentical() {
        var la: [Float] = [], lb: [Float] = [], lc: [Float] = []
        let a = seededRun(seed: 7, steps: &la)
        let b = seededRun(seed: 7, globalNoise: 1, steps: &lb)
        let c = seededRun(seed: 8, steps: &lc)
        XCTAssertEqual(la.count, 9, "3 эпохи по 3 шага (4 + 4 + 2)")
        XCTAssertEqual(la, lb, "лоссы двух прогонов с одним сидом разошлись")
        XCTAssertEqual(a.valAcc, b.valAcc)
        XCTAssertEqual(a.params.keys.sorted(), b.params.keys.sorted())
        for k in a.params.keys {
            XCTAssertEqual(MLX.abs(a.params[k]! - b.params[k]!).max().item(Float.self), 0, k)
        }
        XCTAssertGreaterThan(MLX.abs(a.params["head.weight"]! - c.params["head.weight"]!).max().item(Float.self), 0,
                             "сид не влияет на начальные веса головы")
        XCTAssertNotEqual(la, lc, "сид не влияет на порядок примеров или голову")
    }

    // ── Эталон цикла обучения (перенос на общий Trainer, 17.10) ─────

    /// Прежний собственный цикл X070PartialFinetune — дословно (после
    /// выравнивания порядка операций в шаге Adam с Trainer): эпохи, по
    /// эпохе перестановка от SplitMix64(seed), батчи с неполным последним,
    /// Adam β₂ = 0.999 без отсечения и затухания, постоянный шаг.
    func referenceTrain(backbone: X070Backbone, trainCache tc: X070BoundaryCache, cfg: X070Config,
                        numClasses: Int, freeze: Int, epochs: Int, batchSize: Int, lr: Float,
                        seed: UInt64) -> (params: [String: MLXArray], losses: [Float]) {
        let dim = cfg.nEmbd
        let trainLayerSet = Set(freeze ..< cfg.nLayer)
        var keys = backbone.w.keys.filter { k in (freeze ..< cfg.nLayer).contains { k.hasPrefix("blocks.\($0).") } }
        keys.append("ln_out.weight"); keys.append("ln_out.bias"); keys.sort()
        let nBB = keys.count
        var params: [MLXArray] = keys.map { backbone.w[$0]!.asType(.float32) }
        params.append(MLXRandom.normal([numClasses, dim], key: MLXRandom.key(seed)) * 0.02)
        params.append(MLXArray.zeros([numClasses]))
        eval(params)
        func logits(_ ps: [MLXArray], _ x: MLXArray, _ vf: MLXArray, _ lens: [Int]) -> MLXArray {
            var ov: [String: MLXArray] = [:]
            for j in 0 ..< nBB { ov[keys[j]] = ps[j] }
            backbone.wOverride = ov; backbone.trainLayers = trainLayerSet
            let pooled = X070Pooling.pool(backbone.forwardFrom(x, vf, from: freeze), lengths: lens, kind: .mean)
            return matmul(pooled, ps[nBB].transposed()) + ps[nBB + 1]
        }
        var m = params.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        var v = params.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        var t = 0
        var rng = SplitMix64(seed: seed)
        var losses: [Float] = []
        for _ in 0 ..< epochs {
            var perm = Array(0 ..< tc.count); perm.shuffle(using: &rng)
            var s0 = 0
            while s0 < tc.count {
                let e = min(s0 + batchSize, tc.count), idx = Array(perm[s0 ..< e])
                backbone.wOverride = nil; backbone.trainLayers = []
                let x = tc.readX(idx).asType(.float32)
                let vf = backbone.vFirstFrom(tc.idsBatch(idx)).asType(.float32)
                eval(x, vf)
                let y = MLXArray(idx.map { tc.label[$0] }), lens = idx.map { tc.length[$0] }
                let (vals, grads) = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
                    [crossEntropy(logits: logits(ps, x, vf, lens), targets: y, reduction: .mean)]
                }, argumentNumbers: Array(params.indices))(params)
                t += 1
                let c1 = 1 - Float(pow(Double(Float(0.9)), Double(t)))
                let c2 = 1 - Float(pow(Double(Float(0.999)), Double(t)))
                for i in params.indices {
                    m[i] = Float(0.9) * m[i] + (1 - Float(0.9)) * grads[i]
                    v[i] = Float(0.999) * v[i] + (1 - Float(0.999)) * (grads[i] * grads[i])
                    let mhat = m[i] / c1, vhat = v[i] / c2
                    params[i] = params[i] - lr * (mhat / (sqrt(vhat) + Float(1e-8)))
                }
                eval(params + m + v)
                losses.append(vals[0].item(Float.self))
                s0 = e
            }
        }
        backbone.wOverride = nil; backbone.trainLayers = []
        var out: [String: MLXArray] = [:]
        for j in 0 ..< nBB { out[keys[j]] = params[j] }
        out["head.weight"] = params[nBB]; out["head.bias"] = params[nBB + 1]
        return (out, losses)
    }

    /// Обучение на общем Trainer ПОБИТНО равно прежнему собственному циклу:
    /// веса, лоссы по шагам; верх из одного и из двух слоёв, неполный батч.
    func testTrainerPathEqualsReferenceLoop() {
        for (freeze, n) in [(2, 10), (1, 14)] {
            let (bb, cfg) = TinyBackbone.make(nLayer: 3, seed: 42)
            let tc = X070PartialFinetune.buildBoundaryCache(backbone: bb, examples: examples(n, classes: 3),
                                                            freeze: freeze, name: "train", ctxLen: Self.ctxLen)
            let vc = X070PartialFinetune.buildBoundaryCache(backbone: bb, examples: examples(4, classes: 3, seed: 99),
                                                            freeze: freeze, name: "val", ctxLen: Self.ctxLen)
            var losses: [Float] = [], atEpoch: [Int] = []
            let got = X070PartialFinetune.train(backbone: bb, trainCache: tc, valCache: vc, cfg: cfg, numClasses: 3,
                                                freeze: freeze, epochs: 3, batchSize: 4, lr: 1e-3, seed: 11,
                                                onStep: { _, l, _ in losses.append(l) },
                                                onEpoch: { _, _ in atEpoch.append(losses.count) })
            let ref = referenceTrain(backbone: bb, trainCache: tc, cfg: cfg, numClasses: 3, freeze: freeze,
                                     epochs: 3, batchSize: 4, lr: 1e-3, seed: 11)
            XCTAssertEqual(losses, ref.losses, "freeze \(freeze): лоссы по шагам")
            let perEpoch = (n + 3) / 4
            XCTAssertEqual(losses.count, 3 * perEpoch)
            XCTAssertEqual(atEpoch, [perEpoch, 2 * perEpoch, 3 * perEpoch], "точность не на границах эпох")
            XCTAssertEqual(got.params.keys.sorted(), ref.params.keys.sorted())
            for (k, r) in ref.params {
                XCTAssertEqual(MLX.abs(got.params[k]! - r).max().item(Float.self), 0, "freeze \(freeze): \(k)")
            }
            XCTAssertNil(bb.wOverride, "подмена весов осталась после обучения")
            XCTAssertTrue(bb.trainLayers.isEmpty)
        }
    }
}
