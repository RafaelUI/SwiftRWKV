//
//  PartialFinetuneDump.swift
//  Отпечаток обучения X070PartialFinetune с сидом: веса после обучения, лоссы
//  по шагам, точность. Для правок, которые не должны менять числа (перенос на
//  общий Trainer): снимок до и после обязан совпасть побитно.
//
//      RWKV_PARTIAL_DUMP=…/after.safetensors  swift test --filter PartialFinetuneDump
//      (код до правки) RWKV_PARTIAL_DUMP=…/before.safetensors swift test --filter PartialFinetuneDump
//
//  Без RWKV_PARTIAL_DUMP тест пропускается.
//
import XCTest
import MLX
@testable import RWKVGen

final class PartialFinetuneDump: XCTestCase {
    func testDump() throws {
        guard let path = ProcessInfo.processInfo.environment["RWKV_PARTIAL_DUMP"] else {
            throw XCTSkip("RWKV_PARTIAL_DUMP не задан — дамп снимается вручную")
        }
        var out: [String: MLXArray] = [:]
        // Два режима: верх из одного слоя и из двух; неполный последний батч.
        for (tag, freeze, n) in [("f2", 2, 10), ("f1", 1, 14)] {
            let (bb, cfg) = TinyBackbone.make(nLayer: 3, seed: 42)
            X070PartialFinetune.clearCache()
            let ex: (Int, UInt64) -> [X070Example] = { count, s in
                (0 ..< count).map { i in
                    let label = i % 3
                    var ids = TinyBackbone.ids(1, 16, seed: s &+ UInt64(i)).asArray(Int32.self).map { Int($0) }
                    ids[0] = label
                    return X070Example(ids: ids, label: label)
                }
            }
            var losses: [Float] = [], accs: [Float] = []
            let r = X070PartialFinetune.run(
                backbone: bb, cfg: cfg, numClasses: 3, trainSet: ex(n, 3), valSet: ex(6, 99),
                freeze: freeze, ctxLen: 16, epochs: 3, batchSize: 4, lr: 1e-3, seed: 11,
                onStep: { _, l, _ in losses.append(l) }, onEpoch: { _, a in accs.append(a) })
            for (k, v) in r.params { out["\(tag)/\(k)"] = v }
            out["\(tag)/_losses"] = MLXArray(losses)
            out["\(tag)/_accs"] = MLXArray(accs + [r.valAcc])
        }
        X070PartialFinetune.clearCache()
        eval(Array(out.values))
        try MLX.save(arrays: out, url: URL(fileURLWithPath: path))
    }
}
