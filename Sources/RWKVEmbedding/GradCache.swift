import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  GradCache (Gao et al. 2021) — точный лосс и градиенты при памяти
//  активаций, ограниченной размером ЧАНКА, а не батча.
//
//  Порт rwkv_metal/embedding/gradcache.py.
//
//  Зачем. В tripletPoolLoss пул отрицательных — это и есть батч, поэтому
//  качество retrieval растёт с размером батча. Но память активаций растёт
//  вместе с ним, а активации RWKV-7 на длинных пассажах съедают unified
//  memory быстро. GradCache разрывает эту связь: лосс по-прежнему видит весь
//  батч, а активации — только чанк.
//
//  Три фазы (математически ТОЧНО, не приближение):
//    1. прогнать все чанки БЕЗ градиента, запомнить векторы и сразу eval —
//       так граф не удерживает активации;
//    2. посчитать лосс и dL/dE на ПОЛНОЙ матрице векторов: здесь весь батч
//       и взаимодействует, это [N,N]-матмул плюс софтмакс, без активаций
//       модели;
//    3. прогнать каждый чанк заново С градиентом, засеяв backward его срезом
//       dL/dE, и накопить градиенты параметров.
//
//  Фаза 3 опирается на тождество
//        d/dθ  Σ (embed(chunk) · stop_grad(dL/dE_chunk))  ==  VJP,
//  то есть градиент параметров от этого скаляра равен вектор-якобиан
//  произведению, засеянному нужным котангенсом.
//
//  Почему это НЕ то же, что grad-accumulation: накопление разбивает и сам
//  ЛОСС, поэтому каждый микробатч видит только свои отрицательные — это
//  другая математика (в Python замерено расхождение градиента ~400%).
//  GradCache разбивает только вычисление, но не контрастную задачу.
//
//  Требование: embedChunk обязан быть ДЕТЕРМИНИРОВАННЫМ — фазы 1 и 3 должны
//  давать одинаковые векторы. Dropout'а в модели нет, так что это выполняется.
// ───────────────────────────────────────────────────────────────────────

/// Результат: лосс и градиенты по обучаемым параметрам, в том же порядке,
/// в каком их вернул `parameters`.
public struct GradCacheResult {
    public let loss: MLXArray
    public let gradients: [MLXArray]
}

/// Точный лосс и градиенты с памятью, ограниченной чанком.
///
/// - parameters: обучаемые тензоры (fp32-мастер).
/// - chunks: входы по чанкам; каждый уходит в `embedChunk`.
/// - embedChunk: (параметры, чанк) → «поля» векторов, [b_i, D] каждое.
///   Полей может быть несколько — например (anchor, positive, negative);
///   их число обязано совпадать у всех чанков, а строки поля f склеиваются
///   по оси 0.
/// - lossFromEmbeddings: полные матрицы [N,D] по полям → скаляр. Именно
///   здесь виден ВЕСЬ батч, ради чего всё и затевалось.
public func gradCacheValueAndGrad(
    parameters: [MLXArray],
    chunks: [Int],
    embedChunk: @escaping ([MLXArray], Int) -> [MLXArray],
    lossFromEmbeddings: @escaping ([MLXArray]) -> MLXArray
) -> GradCacheResult {

    precondition(!chunks.isEmpty, "gradCache: пустой список чанков")

    // ── Фаза 1: forward без градиента, кэшируем векторы ──────────────
    //
    // stopGradient и eval здесь — ПОДСТРАХОВКА, а не несущая часть, и это
    // проверено: удаление любого из них не меняет ни результат, ни пик
    // памяти (см. EmbeddingTests, раздел про мутации). Причина в том, что
    // forward идёт вне grad-трансформации, поэтому графа для автодиффа не
    // строится, а память MLX освобождает и без явного eval.
    //
    // Питоновская версия описывает eval как необходимый для освобождения
    // активаций; в mlx-swift на проверенных масштабах это не подтвердилось.
    // Оставлено намеренно: цена нулевая, а гарантия «векторы материализованы
    // до следующего чанка» перестаёт зависеть от деталей планировщика.
    var cached: [[MLXArray]] = []
    cached.reserveCapacity(chunks.count)
    for chunk in chunks {
        let embs = embedChunk(parameters, chunk).map { MLX.stopGradient($0) }
        eval(embs)
        cached.append(embs)
    }
    let nFields = cached[0].count
    for c in cached {
        precondition(c.count == nFields,
                     "embedChunk вернул разное число полей: \(c.count) против \(nFields)")
    }

    let full = (0 ..< nFields).map { f in
        concatenated(cached.map { $0[f] }, axis: 0)
    }
    eval(full)

    // ── Фаза 2: лосс и dL/dE на полном батче ─────────────────────────
    let vg = valueAndGrad({ (fields: [MLXArray]) -> [MLXArray] in
        [lossFromEmbeddings(fields)]
    }, argumentNumbers: Array(0 ..< nFields))
    let (lossVals, dFull) = vg(full)
    eval(lossVals + dFull)
    let loss = lossVals[0]

    // Нарезаем dL/dE обратно по чанкам — отдельное смещение на каждое поле,
    // т.к. поля могут иметь разное число строк в чанке.
    var offsets = [Int](repeating: 0, count: nFields)
    var cotangents: [[MLXArray]] = []
    for c in cached {
        var cots: [MLXArray] = []
        for f in 0 ..< nFields {
            let rows = c[f].shape[0]
            cots.append(dFull[f][offsets[f] ..< (offsets[f] + rows)])
            offsets[f] += rows
        }
        cotangents.append(cots)
    }

    // ── Фаза 3: суррогатный backward по чанкам ───────────────────────
    var total: [MLXArray]? = nil
    for (chunk, cots) in zip(chunks, cotangents) {
        let surrogate = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
            let embs = embedChunk(ps, chunk)
            var acc: MLXArray? = nil
            for (e, c) in zip(embs, cots) {
                let term = (e * MLX.stopGradient(c)).sum()
                acc = acc == nil ? term : acc! + term
            }
            return [acc!]
        }, argumentNumbers: Array(parameters.indices))

        let (_, g) = surrogate(parameters)
        eval(g)
        if var t = total {
            for i in t.indices { t[i] = t[i] + g[i] }
            eval(t)
            total = t
        } else {
            total = g
        }
    }

    return GradCacheResult(loss: loss, gradients: total!)
}

/// Разбить B строк на чанки по `chunkSize`; возвращает стартовые индексы.
/// Сам чанк описывается парой (start, size) — размер выводится из следующего
/// старта, поэтому наружу отдаются только старты.
public func chunkStarts(batch B: Int, chunkSize: Int) -> [Int] {
    precondition(chunkSize > 0, "chunkSize должен быть положительным")
    return Array(stride(from: 0, to: B, by: chunkSize))
}
