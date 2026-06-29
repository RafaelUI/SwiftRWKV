import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Эталонное ("оригинальное") WKV-7 — наивная рекуррентность DPLR на чистом
//  MLX, БЕЗ кастомного Metal-backward. Градиенты даёт автоград MLX, поэтому
//  служит двумя целями:
//
//   1. GROUND-TRUTH для проверки кастомного wkv7Train (его ручной reverse-
//      recurrence с делением на w нигде иначе численно не валидируется).
//   2. Опциональный путь обучения N слоёв "на оригинальном ядре": когда
//      корректность важнее скорости (малый N, короткий ctx), train-слой
//      может идти через wkv7Reference вместо wkv7Train. Drop-in: та же
//      сигнатура [B,T,H,D] -> [B,T,H,D].
//
//  Рекуррентность (для каждой головы, состояние S[dv,dk], h_in = 0):
//      sa[dv]   = Σ_dk S[dv,dk]·a[dk]                 = S·a
//      S[dv,dk] = w[dk]·S[dv,dk] + v[dv]·k[dk] + sa[dv]·b[dk]
//      y[dv]    = Σ_dk S[dv,dk]·r[dk]                 = S·r
//  Это в точности тело forward-кернела в WKV7.swift и каноничная DPLR-форма
//  RWKV-7: диагональ w плюс низкоранговая поправка (S·a)bᵀ.
//
//  Цена: граф разворачивается на T шагов. Для тестов/малых ctx это приемлемо;
//  для продакшен-обучения — wkv7Train.
// ───────────────────────────────────────────────────────────────────────

/// Наивная рекуррентная WKV-7, дифференцируемая автоградом MLX.
/// Входы [B,T,H,D]. h_in = 0. Возвращает out [B,T,H,D].
public func wkv7Reference(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray
) -> MLXArray {
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]

    // S[b,h,dv,dk]; старт с нуля (h_in = 0, как в wkv7Train).
    var s = MLXArray.zeros([B, H, D, D], dtype: .float32)
    var outs: [MLXArray] = []
    outs.reserveCapacity(T)

    // срез шага t -> [B,H,D] (range + squeeze, без индексации голым Int)
    func step(_ x: MLXArray, _ t: Int) -> MLXArray {
        x[0..., t ..< (t + 1)].squeezed(axis: 1).asType(.float32)
    }

    for t in 0 ..< T {
        let rt = step(r, t)   // индекс dk (для y = S·r)
        let wt = step(w, t)   // индекс dk (диагональ по столбцам)
        let kt = step(k, t)   // индекс dk
        let vt = step(v, t)   // индекс dv (строки)
        let at = step(a, t)   // индекс dk
        let bt = step(b, t)   // индекс dk

        // dv -> ось -2 (строка), dk -> ось -1 (столбец)
        let kCol = kt.expandedDimensions(axis: -2)   // [B,H,1,D] по dk
        let bCol = bt.expandedDimensions(axis: -2)
        let aCol = at.expandedDimensions(axis: -2)
        let wCol = wt.expandedDimensions(axis: -2)
        let rCol = rt.expandedDimensions(axis: -2)

        // sa[dv] = Σ_dk S[dv,dk]·a[dk]
        let sa = (s * aCol).sum(axis: -1)            // [B,H,D] индекс dv

        // S[dv,dk] = w[dk]·S + v[dv]·k[dk] + sa[dv]·b[dk]
        let decay = s * wCol
        let vk    = vt.expandedDimensions(axis: -1) * kCol   // outer(v,k)
        let sab   = sa.expandedDimensions(axis: -1) * bCol   // outer(sa,b)
        s = decay + vk + sab

        // y[dv] = Σ_dk S[dv,dk]·r[dk]
        let y = (s * rCol).sum(axis: -1)             // [B,H,D] индекс dv
        outs.append(y.expandedDimensions(axis: 1))   // [B,1,H,D]
    }
    return concatenated(outs, axis: 1)               // [B,T,H,D]
}
