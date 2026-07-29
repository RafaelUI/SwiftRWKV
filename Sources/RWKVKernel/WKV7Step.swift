import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  WKV-7: ОДИН шаг рекуррентности на чистых MLX-операциях.
//
//  Зачем отдельная точка входа, если есть wkv7ForwardWithState / wkv7Train.
//
//  Оба существующих пути идут через Metal-ядро, которому нужен T, кратный
//  CHUNK (=16): длина добивается no-op шагами (w=1, остальное 0). Для T=1
//  это в 16 раз больше работы, чем нужно, и — что важнее — обучаемый путь
//  тянет за собой checkpoint-ядро с ручным backward там, где хватает
//  автоградиента MLX: один шаг разворачивается в десяток обычных операций.
//
//  Это путь головы реранкера: один-два обучаемых токена-зонда поверх
//  состояния базы, по блоку на слой. Он же — путь пошагового декода, если
//  когда-нибудь потребуется дифференцировать по нему.
//
//  Математика ТА ЖЕ, что в ядре (kernel/wkv7.py → wkv7_step):
//      sa[dv] = Σ_dk h[dv,dk]·a[dk]
//      h[dv,dk] ← w[dk]·h[dv,dk] + v[dv]·k[dk] + sa[dv]·b[dk]
//      out[dv]  = Σ_dk h[dv,dk]·r[dk]
//  Первый индекс h — ось value, второй — ось key; порядок тот же, что в
//  раскладке буфера ядра (h_base = ((b·H+h)·S + dv)·S).
//
//  Побитового равенства с ядром НЕТ и быть не может: ядро складывает sa и y
//  последовательным циклом по dk, а MLX-редукция — деревом. Расхождение
//  порядка 1e-7 при fp32, что и проверяется тестом.
// ───────────────────────────────────────────────────────────────────────

/// Один шаг WKV-7 без Metal-ядра и без добивки до CHUNK.
///
/// - Parameters:
///   - r, w, k, v, a, b: `[B, 1, H, D]` либо `[B, H, D]` — ранг входа
///     определяет ранг выхода.
///   - hIn: начальное состояние `[B, H, D, D]`; nil ⇒ нули.
/// - Returns: `(out, hOut)`, где `out` повторяет ранг входа, а
///   `hOut` — `[B, H, D, D]`.
///
/// Дифференцируемо по всем аргументам, включая `hIn`: градиент до состояния
/// нужен, если обучается то, что это состояние породило (state tuning), и
/// безвреден, если нет.
///
/// Для нескольких токенов — `wkv7Steps`.
public func wkv7Step(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    precondition(r.ndim == 3 || r.ndim == 4,
                 "wkv7Step ждёт [B,H,D] или [B,1,H,D], получено \(r.shape)")
    if r.ndim == 4 {
        precondition(r.shape[1] == 1,
                     "wkv7Step — ровно один токен, для нескольких wkv7Steps "
                     + "(получено T=\(r.shape[1]))")
        return wkv7Steps(r, w, k, v, a, b, hIn)
    }

    let B = r.shape[0], H = r.shape[1], D = r.shape[2]
    var h = hIn?.asType(.float32) ?? MLXArray.zeros([B, H, D, D], dtype: .float32)
    precondition(h.shape == [B, H, D, D],
                 "hIn \(h.shape) должен быть [B,H,D,D] = [\(B),\(H),\(D),\(D)]")
    let out: MLXArray
    (out, h) = wkv7StepCore(r.asType(.float32), w.asType(.float32),
                            k.asType(.float32), v.asType(.float32),
                            a.asType(.float32), b.asType(.float32), h)
    return (out, h)
}

/// Несколько шагов подряд, развёрнутых в граф: `[B, T, H, D]` → `[B, T, H, D]`.
///
/// Стоимость линейна по T и без единого запуска ядра, поэтому это путь для
/// НЕСКОЛЬКИХ токенов, а не для последовательности. Ровно такой случай —
/// голова реранкера с `nProbe > 1`: ядру там нужен T, кратный 16, то есть на
/// два токена-зонда пришлось бы четырнадцать пустых.
///
/// Обучаемое ядро на такой длине не работает вовсе: `wkv7TrainWithState`
/// требует кратности CHUNK и на T=2 падает по precondition, а не добивает
/// длину само.
public func wkv7Steps(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    precondition(r.ndim == 4, "wkv7Steps ждёт [B,T,H,D], получено \(r.shape)")
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
    var h = hIn?.asType(.float32) ?? MLXArray.zeros([B, H, D, D], dtype: .float32)
    precondition(h.shape == [B, H, D, D],
                 "hIn \(h.shape) должен быть [B,H,D,D] = [\(B),\(H),\(D),\(D)]")

    var outs: [MLXArray] = []
    outs.reserveCapacity(T)
    for t in 0 ..< T {
        func at(_ x: MLXArray) -> MLXArray {
            x[0..., t ..< (t + 1)].squeezed(axis: 1).asType(.float32)
        }
        let o: MLXArray
        (o, h) = wkv7StepCore(at(r), at(w), at(k), at(v), at(a), at(b), h)
        outs.append(o.expandedDimensions(axis: 1))
    }
    return (concatenated(outs, axis: 1), h)
}

/// Тело шага. Все входы уже [B,H,D] в fp32, состояние [B,H,D,D] в fp32.
///
/// Всё в fp32: состояние — накапливающая величина, и bf16 здесь съедает
/// младшие разряды на каждом шаге (то же решение, что в ядре).
@inline(__always)
private func wkv7StepCore(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray
) -> (MLXArray, MLXArray) {
    // [B,H,1,D] — вдоль оси key (последней у h).
    let aK = a.expandedDimensions(axis: 2)
    let wK = w.expandedDimensions(axis: 2)
    let kK = k.expandedDimensions(axis: 2)
    let bK = b.expandedDimensions(axis: 2)
    let rK = r.expandedDimensions(axis: 2)
    // [B,H,D,1] — вдоль оси value.
    let vV = v.expandedDimensions(axis: 3)

    let sa = (hIn * aK).sum(axis: -1)                     // [B,H,D] по value
    let h = hIn * wK + vV * kK + sa.expandedDimensions(axis: 3) * bK
    return ((h * rK).sum(axis: -1), h)                    // [B,H,D]
}
