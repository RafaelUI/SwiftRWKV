import Foundation
import MLX

enum PoolKind {
    case mean      // masked mean по реальным токенам
    case last      // вектор последнего реального токена (классика RWKV)
}

enum Pooling {
    /// ln_out [B,T,D] + lengths [B] (число реальных токенов) -> [B,D]
    /// mask: паддинг-позиции (>= length) не учитываются.
    static func pool(_ lnOut: MLXArray, lengths: [Int], kind: PoolKind) -> MLXArray {
        let B = lnOut.shape[0], T = lnOut.shape[1], D = lnOut.shape[2]

        switch kind {
        case .mean:
            // строим маску [B,T,1]
            var maskRows: [MLXArray] = []
            for b in 0 ..< B {
                let len = min(lengths[b], T)
                var m = [Float](repeating: 0, count: T)
                for t in 0 ..< len { m[t] = 1 }
                maskRows.append(MLXArray(m).reshaped([1, T, 1]))
            }
            let mask = concatenated(maskRows, axis: 0)          // [B,T,1]
            let summed = (lnOut * mask).sum(axis: 1)            // [B,D]
            let counts = mask.sum(axis: 1)                      // [B,1]
            return summed / maximum(counts, MLXArray(1.0))

        case .last:
            var rows: [MLXArray] = []
            for b in 0 ..< B {
                let idx = max(0, min(lengths[b], T) - 1)
                rows.append(lnOut[b, idx].reshaped([1, D]))
            }
            return concatenated(rows, axis: 0)                  // [B,D]
        }
    }
}
