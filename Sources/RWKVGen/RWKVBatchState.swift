import Foundation
import MLX
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Батчевое рекуррентное состояние RWKV-7 — всё, что нужно, чтобы продолжить
//  последовательность с места обрыва, для B строк сразу.
//
//  Порт rwkv_metal/model/state.py.
//
//  Зачем отдельный тип, если в X070Generation уже есть RWKVState
//  ─────────────────────────────────────────────────────────────
//  Тот — для генерации: B=1, слои лежат отдельными массивами, и он хранит
//  vFirst, потому что при пошаговом декодировании одной непрерывной
//  последовательности vFirst обязан пережить границу вызова step().
//
//  Здесь семантика ДРУГАЯ: это граница между независимыми проходами (документ
//  → запрос), а не середина одного. vFirst в x070 не бегущая величина — слой 0
//  считает её заново на каждой позиции, а слои выше потребляют на той же
//  позиции. Продолжение считает свой vFirst из своих же токенов, переносить
//  нечего. Хранить его здесь означало бы протащить в продолжение чужую
//  величину — тихая ошибка, дающая правдоподобный, но неверный результат.
//
//  Поэтому это самостоятельный тип, а не обобщение существующего. Сливать их
//  без разбора нельзя.
//
//  Раскладка (как в Python — стопкой, а не массивом на слой): срез/repeat/
//  concat по батчу становятся одной операцией, а это основной сценарий
//  (один документ → много запросов).
//
//      wkv        [L, B, H, S, S]  матрица WKV-рекуррентности, ВСЕГДА fp32
//      tmixShift  [L, B, 1, D]     вход tmix (ln1(x)) на последней позиции
//      cmixShift  [L, B, 1, D]     вход cmix (ln2(x)) на последней позиции
//
//  Про сдвиги: их часто забывают, а без них продолжение расходится со сплошным
//  проходом на ПЕРВОМ токене каждого слоя — там, где token-shift тянется за
//  предыдущим токеном и находит ноль. Расхождение маленькое и правдоподобное,
//  то есть худшего сорта.
// ───────────────────────────────────────────────────────────────────────

public struct RWKVBatchState {

    public var wkv: MLXArray          // [L, B, H, S, S] fp32
    public var tmixShift: MLXArray    // [L, B, 1, D]
    public var cmixShift: MLXArray    // [L, B, 1, D]

    public init(wkv: MLXArray, tmixShift: MLXArray, cmixShift: MLXArray) {
        self.wkv = wkv
        self.tmixShift = tmixShift
        self.cmixShift = cmixShift
    }

    // ── Конструкторы ────────────────────────────────────────────────────

    /// Нулевое состояние — начало последовательности.
    public init(cfg: X070Config, batch: Int = 1, dtype: DType = .bfloat16) {
        let L = cfg.nLayer, H = cfg.nHead, S = cfg.headSize, D = cfg.nEmbd
        self.wkv = MLXArray.zeros([L, batch, H, S, S], dtype: .float32)
        self.tmixShift = MLXArray.zeros([L, batch, 1, D], dtype: dtype)
        self.cmixShift = MLXArray.zeros([L, batch, 1, D], dtype: dtype)
    }

    /// Собрать из послойных кусков (порядок — снизу вверх).
    public static func stacked(wkv: [MLXArray], tmix: [MLXArray],
                               cmix: [MLXArray]) -> RWKVBatchState {
        precondition(wkv.count == tmix.count && tmix.count == cmix.count,
                     "разное число слоёв: wkv=\(wkv.count) tmix=\(tmix.count) cmix=\(cmix.count)")
        return RWKVBatchState(wkv: stackedArrays(wkv, axis: 0),
                              tmixShift: stackedArrays(tmix, axis: 0),
                              cmixShift: stackedArrays(cmix, axis: 0))
    }

    // ── Свойства ────────────────────────────────────────────────────────

    public var nLayer: Int { wkv.shape[0] }
    public var batch: Int  { wkv.shape[1] }

    /// Размер в байтах (для решений про кэш: влезет ли в память).
    public var nbytes: Int {
        [wkv, tmixShift, cmixShift].reduce(0) { $0 + $1.size * $1.dtype.size }
    }

    /// Состояние WKV одного слоя: [B, H, S, S].
    public func layerWKV(_ i: Int) -> MLXArray { wkv[i] }
    public func layerTmixShift(_ i: Int) -> MLXArray { tmixShift[i] }
    public func layerCmixShift(_ i: Int) -> MLXArray { cmixShift[i] }

    // ── Манипуляции по оси батча (ось 1 у всех трёх тензоров) ───────────

    /// Диапазон строк.
    public subscript(range: Range<Int>) -> RWKVBatchState {
        RWKVBatchState(wkv: wkv[0..., range],
                       tmixShift: tmixShift[0..., range],
                       cmixShift: cmixShift[0..., range])
    }

    /// Одна строка (остаётся batch=1, а не схлопывается).
    public subscript(row: Int) -> RWKVBatchState { self[row ..< (row + 1)] }

    /// Произвольная выборка строк по индексам.
    public func gather(_ idx: MLXArray) -> RWKVBatchState {
        RWKVBatchState(wkv: takeAlong(wkv, idx),
                       tmixShift: takeAlong(tmixShift, idx),
                       cmixShift: takeAlong(cmixShift, idx))
    }

    public func gather(_ idx: [Int]) -> RWKVBatchState {
        gather(MLXArray(idx.map { Int32($0) }))
    }

    /// Размножить состояние одной строки на n — один документ, много запросов.
    public func repeated(_ n: Int) -> RWKVBatchState {
        precondition(batch == 1, "repeated() ждёт batch=1, получил \(batch)")
        return RWKVBatchState(wkv: repeatedArray(wkv, count: n, axis: 1),
                              tmixShift: repeatedArray(tmixShift, count: n, axis: 1),
                              cmixShift: repeatedArray(cmixShift, count: n, axis: 1))
    }

    /// Склеить состояния по батчу.
    public static func concatenated(_ states: [RWKVBatchState]) -> RWKVBatchState {
        precondition(!states.isEmpty, "concatenated: пустой список")
        return RWKVBatchState(
            wkv: concatenatedArrays(states.map(\.wkv), axis: 1),
            tmixShift: concatenatedArrays(states.map(\.tmixShift), axis: 1),
            cmixShift: concatenatedArrays(states.map(\.cmixShift), axis: 1))
    }

    // ── Прочее ──────────────────────────────────────────────────────────

    /// Приводит ТОЛЬКО сдвиги. wkv остаётся fp32 всегда: ядро считает
    /// рекуррентность в fp32, и именно её точность решает, совпадёт ли
    /// продолжение со сплошным проходом.
    public func asType(_ dtype: DType) -> RWKVBatchState {
        RWKVBatchState(wkv: wkv,
                       tmixShift: tmixShift.asType(dtype),
                       cmixShift: cmixShift.asType(dtype))
    }

    public func stopGradient() -> RWKVBatchState {
        RWKVBatchState(wkv: MLX.stopGradient(wkv),
                       tmixShift: MLX.stopGradient(tmixShift),
                       cmixShift: MLX.stopGradient(cmixShift))
    }

    @discardableResult
    public func evaluated() -> RWKVBatchState {
        MLX.eval(wkv, tmixShift, cmixShift)
        return self
    }
}

// ── Вспомогательное для right-padding ───────────────────────────────────

/// Маска реальных токенов [B, T] для right-padded батча: 1 — токен, 0 — паддинг.
///
/// Пад-позиции делаются НЕЙТРАЛЬНЫМИ для рекуррентности (см. tmix: w←1, k←0,
/// b←0), поэтому финальное состояние строки замирает на её последнем реальном
/// токене и не зависит ни от числа пад-токенов, ни от соседей по батчу.
public func buildMask(lengths: [Int], total: Int, dtype: DType = .float32) -> MLXArray {
    let lens = MLXArray(lengths.map { Int32($0) }).reshaped([-1, 1])
    let pos  = MLXArray(Array(0 ..< total).map { Int32($0) }).reshaped([1, -1])
    return (pos .< lens).asType(dtype)
}

/// [B, T, D] → [B, 1, D] на позиции endIdx (или на последней, если nil).
public func gatherLast(_ x: MLXArray, _ endIdx: MLXArray? = nil) -> MLXArray {
    guard let endIdx else { return x[0..., (x.shape[1] - 1) ..< x.shape[1]] }
    return takeAlong(x, endIdx.reshaped([-1, 1, 1]), axis: 1)
}

/// Индексы последнего РЕАЛЬНОГО токена каждой строки — для gatherLast.
/// Пустая строка (length 0) даёт 0: брать нечего, но и падать не за что.
public func lastRealIndex(lengths: [Int]) -> MLXArray {
    MLXArray(lengths.map { Int32(max(0, $0 - 1)) })
}

// ── Локальные обёртки над MLX (имена конфликтуют с методами структуры) ──

private func stackedArrays(_ xs: [MLXArray], axis: Int) -> MLXArray {
    MLX.stacked(xs, axis: axis)
}
private func concatenatedArrays(_ xs: [MLXArray], axis: Int) -> MLXArray {
    MLX.concatenated(xs, axis: axis)
}
private func repeatedArray(_ x: MLXArray, count: Int, axis: Int) -> MLXArray {
    MLX.repeated(x, count: count, axis: axis)
}
private func takeAlong(_ x: MLXArray, _ idx: MLXArray) -> MLXArray {
    MLX.take(x, idx, axis: 1)
}
