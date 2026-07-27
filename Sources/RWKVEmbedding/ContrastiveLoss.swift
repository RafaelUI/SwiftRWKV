import Foundation
import MLX
import MLXNN

// ───────────────────────────────────────────────────────────────────────
//  Контрастные лоссы. Чистая тензорная математика — ни про модель, ни про
//  токенизатор здесь ничего не знают.
//
//  Порт rwkv_metal/embedding/loss.py.
//
//  Все входы предполагаются L2-НОРМИРОВАННЫМИ: тогда скалярное произведение
//  и есть косинус, а температура делит именно косинус, а не произведение
//  неизвестных норм.
// ───────────────────────────────────────────────────────────────────────

/// InfoNCE: q[i] ↔ d[i] — положительные пары, всё остальное в батче —
/// отрицательные, в обе стороны.
///
/// Оставлено для случая, когда явных hard-negative нет. Когда они есть,
/// `tripletPoolLoss` строго лучше: он добавляет их в тот же пул.
public func infoNCELoss(query q: MLXArray, document d: MLXArray,
                        temperature: Float = 0.05) -> MLXArray {
    let logits = matmul(q.asType(.float32), d.asType(.float32).transposed()) / temperature
    let labels = MLXArray(Array(0 ..< q.shape[0]).map { Int32($0) })
    let q2d = crossEntropy(logits: logits, targets: labels, reduction: .mean)
    let d2q = crossEntropy(logits: logits.transposed(), targets: labels, reduction: .mean)
    return (q2d + d2q) / 2.0
}

/// Триплет с ПУЛОМ отрицательных.
///
/// Пул — это не только свой hard-negative, но и ВСЕ положительные и все
/// отрицательные остальных строк батча: 2B кандидатов на якорь. Отсюда и
/// зависимость качества от размера батча — и, как следствие, необходимость
/// GradCache, если батч не влезает в память.
///
/// symmetric=false (retrieval): только якорь→кандидат. Запрос и отвечающий
/// на него документ не симметричны, поэтому обратное направление не учим.
/// symmetric=true (STS): оба направления — там обе стороны «документы» в
/// действительно симметричном отношении похожести.
public func tripletPoolLoss(anchor: MLXArray, positive: MLXArray,
                            negative: MLXArray, temperature: Float = 0.05,
                            symmetric: Bool = false) -> MLXArray {
    let a = anchor.asType(.float32)
    let p = positive.asType(.float32)
    let n = negative.asType(.float32)

    let candidates = concatenated([p, n], axis: 0)              // [2B, D]
    let logits = matmul(a, candidates.transposed()) / temperature
    let labels = MLXArray(Array(0 ..< a.shape[0]).map { Int32($0) })
    let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
    guard symmetric else { return loss }

    // Обратное направление берёт только блок [B,B] по положительным:
    // у отрицательных нет «своего» якоря, на который они должны указывать.
    let back = matmul(p, a.transposed()) / temperature
    let lossBack = crossEntropy(logits: back, targets: labels, reduction: .mean)
    return (loss + lossBack) / 2.0
}

/// Zero-shot классификация: метки эмбеддятся как обычный текст и сравниваются
/// косинусом — никакой обученной классификационной головы.
///
/// candidates [B,K,D] — пул кандидатов СВОЙ у каждой строки (набор меток
/// различается), поэтому он добит до максимального K, а mask [B,K] отмечает
/// реальные позиции (1) против добивки (0). Пад-позиции получают −1e9 до
/// софтмакса, иначе добивка конкурировала бы с настоящими метками.
public func zeroShotClassificationLoss(anchor: MLXArray, candidates: MLXArray,
                                       mask: MLXArray, targetIndex: MLXArray,
                                       temperature: Float = 0.05) -> MLXArray {
    let a = anchor.asType(.float32).expandedDimensions(axis: 1)   // [B,1,D]
    let c = candidates.asType(.float32)                            // [B,K,D]
    var logits = (a * c).sum(axis: -1) / temperature               // [B,K]
    logits = MLX.where(mask .> 0, logits,
                       MLXArray.full(logits.shape, values: MLXArray(Float(-1e9))))
    return crossEntropy(logits: logits, targets: targetIndex.asType(.int32),
                        reduction: .mean)
}
