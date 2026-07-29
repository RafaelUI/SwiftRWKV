import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Лоссы реранкера. Порт rwkv_metal/reranker/loss.py.
//
//  Оригинальный EmbeddingRWKV учит голову поточечно: BCE на логите позитива
//  (цель 1) и негативов (цель 0). Это работает, но оптимизирует не то, что
//  меряется: реранкеру не нужно попасть в абсолютную «релевантность», ему
//  нужно поставить правильный документ выше остальных. Listwise softmax по
//  кандидатам оптимизирует ровно порядок и стоит здесь умолчанием.
//
//  ВАЖНО про замеры: на бенчмарке из docs/reranker.md listwise (0.9781),
//  mixed (0.9763) и чистый BCE (0.9733) НЕразличимы — разброс по сидам той
//  же величины. Умолчание выбрано по принципу «оптимизируем то, что меряем»,
//  а НЕ потому, что оно измерилось лучше.
//
//  Единственная конкретная причина держать поточечный член — калибровка
//  абсолютного уровня скоров. Если кандидаты только сортируются внутри
//  одного запроса, она не нужна.
// ───────────────────────────────────────────────────────────────────────

/// Listwise softmax-кросс-энтропия по списку кандидатов.
///
/// - scores: `[B, C]` логиты кандидатов
/// - labels: `[B]` индекс правильного кандидата
///
/// При zero-init голове все скоры РОВНО нули, и лосс в точности `ln(C)`.
/// Это самый дешёвый детектор сломанной проводки во всём конвейере: если
/// первый залогированный лосс не `ln(C)`, дело в данных или в голове, и
/// искать надо там, а не в расписании lr.
public func listwiseLoss(_ scores: MLXArray, _ labels: MLXArray,
                         temperature: Float = 1.0) -> MLXArray {
    let s = scores.asType(.float32) / temperature
    // logsumexp со сдвигом на максимум: без него exp переполняется на
    // разъехавшихся скорах, а они разъезжаются ровно тогда, когда обучение
    // идёт хорошо.
    let mx = s.max(axis: -1, keepDims: true)
    let logZ = mx.squeezed(axis: -1) + log(exp(s - mx).sum(axis: -1))
    let picked = takeAlong(s, labels.reshaped([-1, 1]), axis: -1).squeezed(axis: -1)
    return (logZ - picked).mean()
}

/// Поточечный BCE: позитив → 1, остальные → 0 (рецепт оригинала).
///
/// Устойчивая форма: `max(x,0) − x·t + log1p(exp(−|x|))`. Наивная
/// `−t·log σ(x) − (1−t)·log(1−σ(x))` даёт inf, как только |x| подрастает.
public func bceLoss(_ scores: MLXArray, _ labels: MLXArray) -> MLXArray {
    let s = scores.asType(.float32)
    let B = s.shape[0], C = s.shape[1]
    let cols = MLXArray(Array(0 ..< C).map { Int32($0) }).reshaped([1, C])
    let targets = (cols .== labels.reshaped([B, 1])).asType(.float32)
    let loss = maximum(s, MLXArray(Float(0))) - s * targets
              + log1p(exp(-MLX.abs(s)))
    return loss.mean()
}

/// `alpha · listwise + (1 − alpha) · BCE`.
///
/// Крайние значения замыкаются накоротко, а не считаются с нулевым весом:
/// иначе неиспользуемое слагаемое всё равно попадало бы в граф и в
/// backward — работа ради множителя 0.
public func mixedLoss(_ scores: MLXArray, _ labels: MLXArray,
                      alpha: Float = 0.9, temperature: Float = 1.0) -> MLXArray {
    if alpha >= 1.0 { return listwiseLoss(scores, labels, temperature: temperature) }
    if alpha <= 0.0 { return bceLoss(scores, labels) }
    return alpha * listwiseLoss(scores, labels, temperature: temperature)
         + (1.0 - alpha) * bceLoss(scores, labels)
}

/// Какой лосс использовать.
public enum RerankLoss: Sendable, Equatable {
    case listwise
    case bce
    /// `alpha` — доля listwise.
    case mixed(Float)

    public func callAsFunction(_ scores: MLXArray, _ labels: MLXArray,
                               temperature: Float = 1.0) -> MLXArray {
        switch self {
        case .listwise: return listwiseLoss(scores, labels, temperature: temperature)
        case .bce:      return bceLoss(scores, labels)
        case .mixed(let a): return mixedLoss(scores, labels, alpha: a,
                                             temperature: temperature)
        }
    }
}
