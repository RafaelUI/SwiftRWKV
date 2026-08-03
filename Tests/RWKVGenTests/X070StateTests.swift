import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Граничное состояние на уровне МОДЕЛИ (не ядра).
//
//  Ядро уже проверено отдельно (WKV7StateTests). Здесь проверяется то, что
//  добавляет модель поверх него и что ядро проверить не может:
//    • перенос token-shift между кусками (без него продолжение расходится
//      на первом токене КАЖДОГО слоя — маленькое правдоподобное расхождение);
//    • нейтральность right-паддинга (w←1, k←0, b←0);
//    • независимость строки от соседей по батчу.
// ───────────────────────────────────────────────────────────────────────

final class X070StateTests: XCTestCase {

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        eval(ref, got)
        let d = MLX.abs(ref - got).max().item(Float.self)
        let m = MLX.abs(ref).max().item(Float.self)
        return d / (m + 1e-9)
    }

    // ── Совместимость со старым путём ────────────────────────────────

    /// bodyWithState без состояния и без маски обязан дать ровно то же, что
    /// body(). Иначе вся уже проверенная функциональность поехала.
    func testBodyWithStateMatchesBody() {
        let (bb, _) = TinyBackbone.make()
        let ids = TinyBackbone.ids(2, 32)
        let a = bb.body(ids)
        let (b, _) = bb.bodyWithState(ids)
        XCTAssertLessThan(relDiff(a, b), 1e-6,
                          "bodyWithState разошёлся с body на пустом состоянии")
    }

    // ── Непрерывность состояния ──────────────────────────────────────

    /// Разрезанный проход через модель == сплошной. Главный тест шага:
    /// именно на этом свойстве стоит префикс-кэш.
    func testStateContinuitySplit() {
        let (bb, _) = TinyBackbone.make()
        let T = 32, cut = 16
        let ids = TinyBackbone.ids(2, T)
        let (full, hFull) = bb.bodyWithState(ids)

        let (o1, s1) = bb.bodyWithState(ids[0..., 0 ..< cut])
        let (o2, s2) = bb.bodyWithState(ids[0..., cut ..< T], state: s1)
        let joined = concatenated([o1, o2], axis: 1)

        XCTAssertLessThan(relDiff(full, joined), 1e-5,
                          "продолжение разошлось со сплошным проходом")
        XCTAssertLessThan(relDiff(hFull.wkv, s2.wkv), 1e-5,
                          "wkv-состояние разошлось")
        XCTAssertLessThan(relDiff(hFull.tmixShift, s2.tmixShift), 1e-5,
                          "tmix-сдвиг разошёлся")
        XCTAssertLessThan(relDiff(hFull.cmixShift, s2.cmixShift), 1e-5,
                          "cmix-сдвиг разошёлся")
    }

    /// Три куска подряд — состояние переживает многократную передачу.
    func testStateContinuityThreeWay() {
        let (bb, _) = TinyBackbone.make()
        let T = 48
        let ids = TinyBackbone.ids(2, T)
        let (full, _) = bb.bodyWithState(ids)

        var st: RWKVBatchState? = nil
        var parts: [MLXArray] = []
        for (from, to) in [(0, 16), (16, 32), (32, 48)] {
            let (o, s) = bb.bodyWithState(ids[0..., from ..< to], state: st)
            parts.append(o); st = s
        }
        XCTAssertLessThan(relDiff(full, concatenated(parts, axis: 1)), 1e-5,
                          "трёхчастное продолжение разошлось")
    }

    /// Перенос token-shift — отдельно. Если бы сдвиги не переносились,
    /// расхождение сидело бы ТОЛЬКО на первом токене второго куска: там
    /// token-shift тянется за предыдущим токеном и нашёл бы ноль вместо
    /// последнего токена первого куска. Тест смотрит именно на эту позицию.
    func testTokenShiftCarriedAcrossSplit() {
        let (bb, _) = TinyBackbone.make()
        let T = 32, cut = 16
        let ids = TinyBackbone.ids(2, T)
        let (full, _) = bb.bodyWithState(ids)
        let (_, s1) = bb.bodyWithState(ids[0..., 0 ..< cut])
        let (o2, _) = bb.bodyWithState(ids[0..., cut ..< T], state: s1)

        let firstOfSecond = full[0..., cut ..< (cut + 1)]
        XCTAssertLessThan(relDiff(firstOfSecond, o2[0..., 0 ..< 1]), 1e-5,
                          "первый токен продолжения разошёлся — сдвиг не перенесён")
    }

    // ── Right-padding ────────────────────────────────────────────────

    /// Состояние строки не зависит от числа пад-токенов после неё.
    /// Считаем одну и ту же строку дважды: без паддинга и с паддингом до
    /// большей длины — конечные состояния обязаны совпасть.
    func testPaddingDoesNotAffectState() {
        let (bb, cfg) = TinyBackbone.make()
        let real = 20, padded = 32
        let ids = TinyBackbone.ids(1, padded)

        // без паддинга: только реальная часть, маски нет
        let sShort = bb.states(ids[0..., 0 ..< real])

        // с паддингом: полная длина + маска + endIdx на последнем реальном
        let mask = buildMask(lengths: [real], total: padded)
        let end  = lastRealIndex(lengths: [real])
        let sLong = bb.states(ids, mask: mask, endIdx: end)

        XCTAssertLessThan(relDiff(sShort.wkv, sLong.wkv), 1e-5,
                          "паддинг изменил wkv-состояние")
        XCTAssertLessThan(relDiff(sShort.tmixShift, sLong.tmixShift), 1e-5,
                          "паддинг изменил tmix-сдвиг")
        XCTAssertLessThan(relDiff(sShort.cmixShift, sLong.cmixShift), 1e-5,
                          "паддинг изменил cmix-сдвиг")
        XCTAssertEqual(sLong.nLayer, cfg.nLayer)
    }

    /// Содержимое пад-позиций не влияет на состояние: заполняем «хвост»
    /// разным мусором, состояние обязано остаться тем же.
    func testPaddingContentIrrelevant() {
        let (bb, _) = TinyBackbone.make()
        let real = 12, total = 32
        let base = TinyBackbone.ids(1, total, seed: 1)
        let other = TinyBackbone.ids(1, total, seed: 999)
        // одинаковое начало, разный хвост
        let mixed = concatenated([base[0..., 0 ..< real], other[0..., real ..< total]],
                                 axis: 1)

        let mask = buildMask(lengths: [real], total: total)
        let end  = lastRealIndex(lengths: [real])
        let s1 = bb.states(base,  mask: mask, endIdx: end)
        let s2 = bb.states(mixed, mask: mask, endIdx: end)

        XCTAssertLessThan(relDiff(s1.wkv, s2.wkv), 1e-5,
                          "содержимое паддинга протекло в состояние")
        XCTAssertLessThan(relDiff(s1.tmixShift, s2.tmixShift), 1e-5,
                          "содержимое паддинга протекло в tmix-сдвиг")
    }

    /// Строка в батче не зависит от соседей: та же строка, посчитанная одна и
    /// в компании строк другой длины, обязана дать то же состояние.
    func testRowIndependentOfBatchNeighbours() {
        let (bb, _) = TinyBackbone.make()
        let total = 32
        let lens = [28, 9, 17]
        let ids = TinyBackbone.ids(3, total, seed: 5)

        let mask = buildMask(lengths: lens, total: total)
        let end  = lastRealIndex(lengths: lens)
        let batched = bb.states(ids, mask: mask, endIdx: end)

        for (row, len) in lens.enumerated() {
            let alone = bb.states(ids[row ..< (row + 1), 0 ..< len])
            XCTAssertLessThan(relDiff(alone.wkv, batched[row].wkv), 1e-5,
                              "строка \(row) (len=\(len)) зависит от соседей по батчу")
            XCTAssertLessThan(relDiff(alone.tmixShift, batched[row].tmixShift), 1e-5,
                              "строка \(row): tmix-сдвиг зависит от соседей")
        }
    }

    /// Скрытые состояния РЕАЛЬНЫХ токенов не зависят от маски: модель
    /// каузальна, паддинг идёт справа, поэтому испортить он может только
    /// конечное состояние, но не активации до него.
    func testMaskDoesNotAffectRealTokenOutputs() {
        let (bb, _) = TinyBackbone.make()
        let real = 20, total = 32
        let ids = TinyBackbone.ids(1, total)
        let mask = buildMask(lengths: [real], total: total)

        let (withMask, _)    = bb.bodyWithState(ids, mask: mask,
                                                endIdx: lastRealIndex(lengths: [real]))
        let (withoutMask, _) = bb.bodyWithState(ids)

        XCTAssertLessThan(relDiff(withoutMask[0..., 0 ..< real],
                                  withMask[0..., 0 ..< real]), 1e-5,
                          "маска повлияла на выходы реальных токенов")
    }

    // ── Операции над состоянием ──────────────────────────────────────

    /// repeated(): один документ → много запросов. Размноженное состояние
    /// обязано дать то же, что честный проход с батчем из копий.
    func testRepeatedStateMatchesManualBatch() {
        let (bb, _) = TinyBackbone.make()
        let doc = TinyBackbone.ids(1, 16, seed: 3)
        let sDoc = bb.states(doc)

        let n = 3
        let queries = TinyBackbone.ids(n, 16, seed: 4)
        let (oRep, _) = bb.bodyWithState(queries, state: sDoc.repeated(n))

        // эталон: тот же документ, продублированный вручную в батч
        let docBatch = concatenated(Array(repeating: doc, count: n), axis: 0)
        let sBatch = bb.states(docBatch)
        let (oManual, _) = bb.bodyWithState(queries, state: sBatch)

        XCTAssertLessThan(relDiff(oManual, oRep), 1e-5,
                          "repeated() разошёлся с честным батчем")
    }

    func testStateShapesAndSlicing() {
        let (bb, cfg) = TinyBackbone.make()
        let s = bb.states(TinyBackbone.ids(4, 16))
        XCTAssertEqual(s.wkv.shape, [cfg.nLayer, 4, cfg.nHead, cfg.headSize, cfg.headSize])
        XCTAssertEqual(s.tmixShift.shape, [cfg.nLayer, 4, 1, cfg.nEmbd])
        XCTAssertEqual(s.batch, 4)
        XCTAssertEqual(s.nLayer, cfg.nLayer)

        XCTAssertEqual(s[1].batch, 1)
        XCTAssertEqual(s[1 ..< 3].batch, 2)
        XCTAssertEqual(s.gather([3, 0]).batch, 2)
        XCTAssertEqual(RWKVBatchState.concatenated([s[0], s[1]]).batch, 2)
        XCTAssertEqual(s[0].repeated(5).batch, 5)

        // gather([3,0]) обязан отдать именно строки 3 и 0, а не что попало
        XCTAssertEqual(maxAbsDiff(s.gather([3, 0]).wkv[0..., 0 ..< 1], s[3].wkv), 0)
        XCTAssertEqual(maxAbsDiff(s.gather([3, 0]).wkv[0..., 1 ..< 2], s[0].wkv), 0)

        // wkv всегда fp32, даже если сдвиги приведены к bf16
        let cast = s.asType(.bfloat16)
        XCTAssertEqual(cast.wkv.dtype, .float32)
        XCTAssertEqual(cast.tmixShift.dtype, .bfloat16)
    }

    func testBuildMaskAndLastRealIndex() {
        let m = buildMask(lengths: [3, 1], total: 4)
        eval(m)
        XCTAssertEqual(m.shape, [2, 4])
        XCTAssertEqual(m.asType(.int32).sum().item(Int32.self), 4)   // 3 + 1
        let e = lastRealIndex(lengths: [3, 1])
        eval(e)
        XCTAssertEqual(e[0].item(Int32.self), 2)
        XCTAssertEqual(e[1].item(Int32.self), 0)
    }
}
