import Foundation
import MLX
import MLXNN          // crossEntropy
import MLXRandom
import RWKVKernel     // WKV7_CHUNK

// ───────────────────────────────────────────────────────────────────────
//  Partial-finetune (обучение верхних N слоёв) на КАНОНИЧЕСКОМ X070Backbone.
//  Перенос из RWKVTrain. Отличия от старого пути:
//   • token-shift внутриблочный ⇒ межблочного xPrev НЕТ. Граница = (x, vFirst).
//   • обучаемые fp32-веса инжектятся через backbone.wOverride.
//   • vFirst пересчитывается из id (слой 0 frozen), на диск идёт только x.
// ───────────────────────────────────────────────────────────────────────

public struct X070Example {
    public var ids: [Int]
    public var label: Int
    public init(ids: [Int], label: Int) { self.ids = ids; self.label = label }
}

public enum X070PoolKind { case mean, last }

enum X070Pooling {
    /// ln_out [B,T,D] + lengths [B] → [B,D]; паддинг (>= length) не учитывается.
    static func pool(_ lnOut: MLXArray, lengths: [Int], kind: X070PoolKind) -> MLXArray {
        let B = lnOut.shape[0], T = lnOut.shape[1], D = lnOut.shape[2]
        switch kind {
        case .mean:
            var maskRows: [MLXArray] = []
            for b in 0 ..< B {
                let len = min(lengths[b], T)
                var m = [Float](repeating: 0, count: T)
                for t in 0 ..< len { m[t] = 1 }
                maskRows.append(MLXArray(m).reshaped([1, T, 1]))
            }
            let mask = concatenated(maskRows, axis: 0)
            let summed = (lnOut * mask).sum(axis: 1)
            let counts = mask.sum(axis: 1)
            return summed / maximum(counts, MLXArray(1.0))
        case .last:
            var rows: [MLXArray] = []
            for b in 0 ..< B {
                let idx = max(0, min(lengths[b], T) - 1)
                rows.append(lnOut[b, idx].reshaped([1, D]))
            }
            return concatenated(rows, axis: 0)
        }
    }
}

// Диск-кэш граничного x [N,T,D] bf16 (mmap). vFirst и (отсутствующий) xPrev не храним.
public struct X070BoundaryCache {
    let mapped: Data
    let T: Int
    let D: Int
    var ids: [[Int32]]
    var length: [Int]
    var label: [Int32]
    var count: Int { label.count }
    private var rowBytes: Int { T * D * DType.bfloat16.size }

    func readX(_ idx: [Int]) -> MLXArray {
        var arrs: [MLXArray] = []; arrs.reserveCapacity(idx.count)
        for i in idx {
            let off = i * rowBytes
            let chunk = mapped.subdata(in: off ..< off + rowBytes)
            arrs.append(MLXArray(chunk, [1, T, D], dtype: .bfloat16))
        }
        return concatenated(arrs, axis: 0)
    }
    func idsBatch(_ idx: [Int]) -> MLXArray {
        var flat = [Int32](); flat.reserveCapacity(idx.count * T)
        for i in idx { flat.append(contentsOf: ids[i]) }
        return MLXArray(flat, [idx.count, T])
    }
}

public enum X070PartialFinetune {

    public struct TrainResult {
        public var valAcc: Float
        public var params: [String: MLXArray]
    }

    public static func clearCache() {
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        for n in ["train", "val"] {
            try? FileManager.default.removeItem(at: dir.appendingPathComponent("pf_x070_\(n).bin"))
        }
    }

    public static func buildBoundaryCache(
        backbone: X070Backbone,
        examples: [X070Example],
        freeze: Int,
        name: String,
        ctxLen: Int = 128,
        batch: Int = 5,
        isCancelled: @escaping () -> Bool = { false },
        progress: @escaping (_ pct: Double, _ peakMB: Double) -> Void = { _, _ in }
    ) -> X070BoundaryCache {
        precondition(ctxLen % WKV7_CHUNK == 0,
                     "ctxLen (\(ctxLen)) должен делиться на CHUNK \(WKV7_CHUNK)")
        backbone.wOverride = nil
        backbone.trainLayers = []

        let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pf_x070_\(name).bin")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try! FileHandle(forWritingTo: url)

        var ids: [[Int32]] = []; ids.reserveCapacity(examples.count)
        var length: [Int] = []
        var label: [Int32] = []
        var D = -1

        var start = 0
        while start < examples.count {
            if isCancelled() { break }
            let end = min(start + batch, examples.count)
            let bs = Array(examples[start ..< end]); let B = bs.count

            var idsFlat = [Int32](repeating: 0, count: B * ctxLen)
            for (bi, ex) in bs.enumerated() {
                let L = min(ex.ids.count, ctxLen)
                for t in 0 ..< L { idsFlat[bi * ctxLen + t] = Int32(ex.ids[t]) }
                length.append(L); label.append(Int32(ex.label))
                ids.append(Array(idsFlat[bi * ctxLen ..< bi * ctxLen + ctxLen]))
            }
            let idsArr = MLXArray(idsFlat, [B, ctxLen])
            let (x, _) = backbone.boundaryState(idsArr, upTo: freeze)   // только x
            let xb = x.asType(.bfloat16)
            eval(xb)
            if D < 0 { D = xb.shape[2] }
            try! fh.write(contentsOf: xb.asData(access: .copy).data)

            progress(Double(end) / Double(examples.count), residentMemoryMB())
            start = end
        }
        try! fh.close()
        let mapped = try! Data(contentsOf: url, options: .alwaysMapped)
        return X070BoundaryCache(mapped: mapped, T: ctxLen, D: max(D, 1),
                                 ids: ids, length: length, label: label)
    }

    public static func train(
        backbone: X070Backbone,
        trainCache: X070BoundaryCache,
        valCache: X070BoundaryCache,
        cfg: X070Config,
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

        var params: [MLXArray] = keys.map { backbone.w[$0]!.asType(.float32) }
        params.append(MLXRandom.normal([numClasses, dim]) * 0.02)   // head.weight
        params.append(MLXArray.zeros([numClasses]))                 // head.bias
        eval(params)

        // frozen-константы: x (диск), vFirst (из id, слой 0) — ВНЕ grad-тейпа
        func inputs(_ cache: X070BoundaryCache, _ idx: [Int]) -> (MLXArray, MLXArray) {
            backbone.wOverride = nil; backbone.trainLayers = []
            let x = cache.readX(idx).asType(.float32)
            let vFirst = backbone.vFirstFrom(cache.idsBatch(idx)).asType(.float32)
            eval(x, vFirst)
            return (x, vFirst)
        }

        func logits(_ ps: [MLXArray], _ x: MLXArray, _ vFirst: MLXArray,
                    _ lengths: [Int]) -> MLXArray {
            var ov: [String: MLXArray] = [:]
            for j in 0 ..< nBB { ov[keys[j]] = ps[j] }
            backbone.wOverride = ov
            backbone.trainLayers = trainLayerSet
            let lnOut = backbone.forwardFrom(x, vFirst, from: freeze)
            let pooled = X070Pooling.pool(lnOut, lengths: lengths, kind: .mean)
            return matmul(pooled, ps[nBB].transposed()) + ps[nBB + 1]
        }

        func accuracy(_ cache: X070BoundaryCache) -> Float {
            var correct = 0, total = 0, s = 0
            let N = cache.count
            while s < N {
                let e = min(s + batchSize, N); let idx = Array(s ..< e)
                let (x, vf) = inputs(cache, idx)
                let lg = logits(params, x, vf, idx.map { cache.length[$0] })
                let pred = lg.argMax(axis: 1).asType(.int32)
                let y = MLXArray(idx.map { cache.label[$0] })
                correct += Int((pred .== y).asType(.int32).sum().item(Int32.self))
                total += idx.count; s = e
            }
            return Float(correct) / Float(max(total, 1))
        }

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

                let (x, vf) = inputs(trainCache, idx)

                let vg = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
                    [crossEntropy(logits: logits(ps, x, vf, lens),
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
    /// Публичный end-to-end вход: строит кэши границы, обучает верхушку, чистит кэш.
    public static func run(
        backbone: X070Backbone, cfg: X070Config, numClasses: Int,
        trainSet: [X070Example], valSet: [X070Example], freeze: Int,
        ctxLen: Int = 128, epochs: Int = 5, batchSize: Int = 8, lr: Float = 1e-4,
        onStep: @escaping (_ step: Int, _ loss: Float, _ peakMB: Double) -> Void = { _, _, _ in },
        onEpoch: @escaping (_ epoch: Int, _ valAcc: Float) -> Void = { _, _ in }
    ) -> TrainResult {
        clearCache(); defer { clearCache() }
        let tc = buildBoundaryCache(backbone: backbone, examples: trainSet,
                                    freeze: freeze, name: "train", ctxLen: ctxLen)
        let vc = buildBoundaryCache(backbone: backbone, examples: valSet,
                                    freeze: freeze, name: "val", ctxLen: ctxLen)
        return train(backbone: backbone, trainCache: tc, valCache: vc, cfg: cfg,
                     numClasses: numClasses, freeze: freeze, epochs: epochs,
                     batchSize: batchSize, lr: lr, onStep: onStep, onEpoch: onEpoch)
    }
}
