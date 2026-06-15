import Foundation
import MLX
import MLXNN          // crossEntropy
import MLXRandom

// ───────────────────────────────────────────────────────────────────────
//  Обучение верхушки на диск-кэше фич.
//  x читается с mmap; vFirst пересчитывается из id (frozen, слой 0);
//  xPrev = x[:, -1]. Frozen-константы считаются ВНЕ valueAndGrad.
// ───────────────────────────────────────────────────────────────────────

extension PartialFinetune {

    struct TrainResult {
        var valAcc: Float
        var params: [String: MLXArray]
    }

    static func train(
        backbone: RWKVBackbone,
        trainCache: BoundaryCache,
        valCache: BoundaryCache,
        cfg: RWKVConfig,
        numClasses: Int,
        freeze: Int,
        epochs: Int = 5,
        batchSize: Int = 8,
        lr: Float = 1e-4,
        isCancelled: @escaping () -> Bool = { false },
        onStep: @escaping (_ step: Int, _ loss: Float, _ peakMB: Double) -> Void = { _, _, _ in },
        onEpoch: @escaping (_ epoch: Int, _ valAcc: Float) -> Void = { _, _ in }
    ) -> TrainResult {

        let dim = cfg.nEmbd
        let trainLayerSet = Set(freeze ..< cfg.nLayer)

        var keys = backbone.w.keys.filter { k in
            (freeze ..< cfg.nLayer).contains { k.hasPrefix("blocks.\($0).") }
        }
        keys.append("ln_out.weight"); keys.append("ln_out.bias")
        keys.sort()
        let nBB = keys.count

        // fp32 обучаемые веса (bf16-апдейты Adam теряются на округлении)
        var params: [MLXArray] = keys.map { backbone.w[$0]!.asType(.float32) }
        params.append(MLXRandom.normal([numClasses, dim]) * 0.02)   // head.weight
        params.append(MLXArray.zeros([numClasses]))                 // head.bias
        eval(params)

        // frozen-константы батча: x (с диска), xPrev (= x[:,-1]), vFirst (из id)
        func inputs(_ cache: BoundaryCache, _ idx: [Int]) -> (MLXArray, MLXArray, MLXArray) {
            backbone.wOverride = nil; backbone.trainLayers = []
            let x = cache.readX(idx).asType(.float32)              // [B,T,D]
            let xPrev = x[0..., (cache.T - 1)...]                  // [B,1,D]
            let vFirst = backbone.vFirstFrom(cache.idsBatch(idx)).asType(.float32)
            eval(x, vFirst)
            return (x, xPrev, vFirst)
        }

        // обучаемая часть: forwardFrom(14..) + голова → logits [B,C]
        func logits(_ ps: [MLXArray], _ x: MLXArray, _ xPrev: MLXArray,
                    _ vFirst: MLXArray, _ lengths: [Int]) -> MLXArray {
            var ov: [String: MLXArray] = [:]
            for j in 0 ..< nBB { ov[keys[j]] = ps[j] }
            backbone.wOverride = ov
            backbone.trainLayers = trainLayerSet
            let lnOut = backbone.forwardFrom(x, xPrev, vFirst, from: freeze)
            let pooled = Pooling.pool(lnOut, lengths: lengths, kind: .mean)
            return matmul(pooled, ps[nBB].transposed()) + ps[nBB + 1]
        }

        func accuracy(_ cache: BoundaryCache) -> Float {
            var correct = 0, total = 0, s = 0
            let N = cache.count
            while s < N {
                let e = min(s + batchSize, N); let idx = Array(s ..< e)
                let (x, xp, vf) = inputs(cache, idx)
                let lg = logits(params, x, xp, vf, idx.map { cache.length[$0] })
                let pred = lg.argMax(axis: 1).asType(.int32)
                let y = MLXArray(idx.map { cache.label[$0] })
                correct += Int((pred .== y).asType(.int32).sum().item(Int32.self))
                total += idx.count; s = e
            }
            return Float(correct) / Float(max(total, 1))
        }

        // ── ручной Adam ──
        var m = params.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        var v = params.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        let b1: Float = 0.9, b2: Float = 0.999, aeps: Float = 1e-8
        var t = 0

        let N = trainCache.count
        var step = 0
        for epoch in 0 ..< epochs {
            if isCancelled() { break }
            var perm = Array(0 ..< N); perm.shuffle()
            var s = 0
            while s < N {
                if isCancelled() { break }
                let e = min(s + batchSize, N)
                let idx = Array(perm[s ..< e])
                let y = MLXArray(idx.map { trainCache.label[$0] })
                let lens = idx.map { trainCache.length[$0] }

                // frozen-константы — ВНЕ grad-тейпа
                let (x, xp, vf) = inputs(trainCache, idx)

                let vg = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
                    [crossEntropy(logits: logits(ps, x, xp, vf, lens),
                                  targets: y, reduction: .mean)]
                }, argumentNumbers: Array(params.indices))
                let (vals, grads) = vg(params)

                t += 1
                let c1 = 1 - Float(pow(Double(b1), Double(t)))
                let c2 = 1 - Float(pow(Double(b2), Double(t)))
                for i in params.indices {
                    m[i] = b1 * m[i] + (1 - b1) * grads[i]
                    v[i] = b2 * v[i] + (1 - b2) * (grads[i] * grads[i])
                    params[i] = params[i] - lr * (m[i] / c1) / (sqrt(v[i] / c2) + aeps)
                }
                eval(params + m + v)

                step += 1
                onStep(step, vals[0].item(Float.self), residentMemoryMB())
                s = e
            }
            if isCancelled() { break }
            onEpoch(epoch + 1, accuracy(valCache))
        }

        let finalAcc = accuracy(valCache)
        var trained: [String: MLXArray] = [:]
        for j in 0 ..< nBB { trained[keys[j]] = params[j] }
        trained["head.weight"] = params[nBB]
        trained["head.bias"]   = params[nBB + 1]
        backbone.wOverride = nil; backbone.trainLayers = []
        return TrainResult(valAcc: finalAcc, params: trained)
    }

    static func runDemo(
        backbone: RWKVBackbone, cfg: RWKVConfig, numClasses: Int,
        trainSet: [Example], valSet: [Example], freeze: Int,
        ctxLen: Int = 128, epochs: Int = 5, batchSize: Int = 8, lr: Float = 1e-4
    ) {
        clearCache()                       // снести возможный стейл
        defer { clearCache() }             // авто-удаление: успех/отмена/возврат
        print("[PF] кэш границы train (\(trainSet.count)) → диск…")
        let tc = buildBoundaryCache(backbone: backbone, examples: trainSet,
                                    freeze: freeze, name: "train", ctxLen: ctxLen) { p, mb in
            print(String(format: "  train cache %3d%%  peak %d MB", Int(p * 100), Int(mb)))
        }
        print("[PF] кэш границы val (\(valSet.count)) → диск…")
        let vc = buildBoundaryCache(backbone: backbone, examples: valSet,
                                    freeze: freeze, name: "val", ctxLen: ctxLen)
        print("[PF] обучение слоёв \(freeze)..<\(cfg.nLayer) + голова…")
        var best: Float = 0
        let res = train(backbone: backbone, trainCache: tc, valCache: vc, cfg: cfg,
                        numClasses: numClasses, freeze: freeze,
                        epochs: epochs, batchSize: batchSize, lr: lr,
                        onStep: { s, l, mb in
                            if s % 20 == 0 { print(String(format: "  step %d  loss %.4f  peak %d MB", s, l, Int(mb))) }
                        },
                        onEpoch: { e, acc in
                            best = max(best, acc)
                            print(String(format: "  epoch %d  valAcc %.4f  (best %.4f)", e, acc, best))
                        })
        print(String(format: "[PF] ГОТОВО. финальная %.4f, лучшая %.4f", res.valAcc, best))
    }
}
