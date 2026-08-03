//
//  WKV7StateTests.swift
//  Граничное состояние WKV-7: непрерывность, паритет с эталоном, градиент по h_in.
//
//  Зачем эти тесты. Ядро всегда умело переносить состояние между чанками, но
//  наружу его не отдавало: wkv7Forward стартовал с нулей и выбрасывал финальное
//  h, а wkv7Train вдобавок отбрасывал dh_in. Теперь и то, и другое доступно —
//  и на этом стоит весь префикс-кэш (документ сворачивается один раз, запрос
//  продолжает с готового состояния) и обучение головы поверх состояния.
//
//  Главное свойство, которое здесь проверяется: РАЗРЕЗАННЫЙ проход через
//  состояние обязан совпасть со СПЛОШНЫМ. Если он не совпадает, кэш молча
//  портит результат — модель продолжает выдавать правдоподобный текст, просто
//  не тот. Такую ошибку без теста не видно.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVKernel

final class WKV7StateTests: XCTestCase {

    typealias Inputs = (r: MLXArray, w: MLXArray, k: MLXArray,
                        v: MLXArray, a: MLXArray, b: MLXArray, g: MLXArray)

    /// Те же распределения, что в WKV7KernelParityTests: w — затухание из
    /// exp(-0.606531·sigmoid), a/b — DPLR-пара из нормированного k.
    func makeInputs(B: Int = 2, T: Int = 64, H: Int = 4, D: Int = 64,
                    seed: UInt64 = 0) -> Inputs {
        MLXRandom.seed(seed)
        func rnd(_ s: Float) -> MLXArray { MLXRandom.normal([B, T, H, D]) * s }
        let r = rnd(0.5), v = rnd(0.5), k = rnd(0.5)
        let kk = k / sqrt((k * k).sum(axis: -1, keepDims: true) + 1e-12)
        let iclr = sigmoid(rnd(1.0))
        let a = -kk
        let b = kk * iclr
        let w = exp(-0.606531 * sigmoid(rnd(1.0)))
        let g = rnd(1.0)
        eval(r, w, k, v, a, b, g)
        return (r, w, k, v, a, b, g)
    }

    /// Случайное НЕнулевое состояние. Масштаб 0.1 — состояние WKV не бывает
    /// большим: w < 1 на каждом шаге, так что рекуррентность его сжимает.
    func makeState(B: Int = 2, H: Int = 4, D: Int = 64, seed: UInt64 = 7) -> MLXArray {
        MLXRandom.seed(seed)
        let h = MLXRandom.normal([B, H, D, D]) * 0.1
        eval(h)
        return h
    }

    func slice(_ x: MLXArray, _ from: Int, _ to: Int) -> MLXArray {
        x[0..., from ..< to]
    }

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

    // ─────────────────────────────────────────────────────────────────
    //  Непрерывность состояния: split == contiguous
    // ─────────────────────────────────────────────────────────────────

    /// Forward, разрез по границе чанка (32 = 2×CHUNK).
    func testForwardStateContinuityAligned() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 64)
        let (full, hFull) = wkv7ForwardWithState(r, w, k, v, a, b, nil)

        let (o1, h1) = wkv7ForwardWithState(slice(r, 0, 32), slice(w, 0, 32),
                                            slice(k, 0, 32), slice(v, 0, 32),
                                            slice(a, 0, 32), slice(b, 0, 32), nil)
        let (o2, h2) = wkv7ForwardWithState(slice(r, 32, 64), slice(w, 32, 64),
                                            slice(k, 32, 64), slice(v, 32, 64),
                                            slice(a, 32, 64), slice(b, 32, 64), h1)
        let joined = concatenated([o1, o2], axis: 1)

        XCTAssertEqual(maxAbsDiff(full, joined), 0,
                       "разрез по границе чанка обязан быть бит-в-бит")
        XCTAssertEqual(maxAbsDiff(hFull, h2), 0,
                       "конечное состояние разрезанного прохода разошлось")
    }

    /// Forward, разрез НЕ по границе чанка. Тоже обязан быть бит-в-бит, и это
    /// не самоочевидно — стоит объяснить, почему выравнивание не требуется.
    ///
    /// Рекуррентность строго последовательна внутри потока (batch, head, dv),
    /// а не редукция: арифметика на токен — фиксированная цепочка fp32-операций
    /// над h_row, одинаковая независимо от позиции токена внутри чанка.
    /// Переход через границу чанка (h_row → h_out → h_in) — чистое копирование,
    /// без арифметики. Добивающий токен точно нейтрален:
    ///     h' = 1.0f·h + 0.0f·0.0f + 0.0f·0.0f
    /// что по IEEE-754 возвращает h неизменным для любого конечного h.
    /// Переупорядочивать, таким образом, нечего — расхождению взяться неоткуда.
    ///
    /// Практическое следствие: префикс-кэш НЕ обязан выравнивать документы по
    /// CHUNK. Если этот тест когда-нибудь упадёт — сломалась нейтральность
    /// паддинга либо в перенос состояния затесалась арифметика.
    func testForwardStateContinuityUnaligned() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 64)
        let (full, hFull) = wkv7ForwardWithState(r, w, k, v, a, b, nil)

        // 20 и 5 — обе не кратны CHUNK=16; 5 вдобавок короче одного чанка.
        for cut in [20, 5] {
            let (o1, h1) = wkv7ForwardWithState(
                slice(r, 0, cut), slice(w, 0, cut), slice(k, 0, cut),
                slice(v, 0, cut), slice(a, 0, cut), slice(b, 0, cut), nil)
            let (o2, h2) = wkv7ForwardWithState(
                slice(r, cut, 64), slice(w, cut, 64), slice(k, cut, 64),
                slice(v, cut, 64), slice(a, cut, 64), slice(b, cut, 64), h1)
            let joined = concatenated([o1, o2], axis: 1)

            XCTAssertEqual(maxAbsDiff(full, joined), 0,
                           "разрез \(cut)+\(64 - cut) не бит-в-бит")
            XCTAssertEqual(maxAbsDiff(hFull, h2), 0,
                           "состояние при разрезе \(cut)+\(64 - cut) разошлось")
        }
    }

    /// Три куска подряд — состояние должно переживать многократную передачу,
    /// а не только одну (накопление ошибки).
    func testForwardStateContinuityThreeWay() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 64)
        let (full, hFull) = wkv7ForwardWithState(r, w, k, v, a, b, nil)

        var h: MLXArray? = nil
        var parts: [MLXArray] = []
        for (from, to) in [(0, 16), (16, 48), (48, 64)] {
            let (o, hNext) = wkv7ForwardWithState(
                slice(r, from, to), slice(w, from, to), slice(k, from, to),
                slice(v, from, to), slice(a, from, to), slice(b, from, to), h)
            parts.append(o)
            h = hNext
        }
        XCTAssertEqual(maxAbsDiff(full, concatenated(parts, axis: 1)), 0,
                       "трёхчастный разрез по границам чанков разошёлся")
        XCTAssertEqual(maxAbsDiff(hFull, h!), 0, "состояние после трёх передач разошлось")
    }

    /// Train-путь: тот же инвариант. T каждого куска кратна CHUNK.
    func testTrainStateContinuity() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 64)
        let (full, hFull) = wkv7TrainWithState(r, w, k, v, a, b, nil)

        let (o1, h1) = wkv7TrainWithState(slice(r, 0, 32), slice(w, 0, 32),
                                          slice(k, 0, 32), slice(v, 0, 32),
                                          slice(a, 0, 32), slice(b, 0, 32), nil)
        let (o2, h2) = wkv7TrainWithState(slice(r, 32, 64), slice(w, 32, 64),
                                          slice(k, 32, 64), slice(v, 32, 64),
                                          slice(a, 32, 64), slice(b, 32, 64), h1)

        XCTAssertEqual(maxAbsDiff(full, concatenated([o1, o2], axis: 1)), 0,
                       "train: разрезанный проход разошёлся со сплошным")
        XCTAssertEqual(maxAbsDiff(hFull, h2), 0, "train: конечное состояние разошлось")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Согласие путей и паритет с эталоном
    // ─────────────────────────────────────────────────────────────────

    /// frozen-путь и train-путь обязаны давать одно и то же состояние — иначе
    /// кэш, собранный frozen-проходом, незаконно использовать для обучения на
    /// нём (а на этом стоит вся экономия реранкера).
    ///
    /// Это РАЗНЫЕ ядра: wkv7ChunkForward — один launch на чанк, ckptFwdKernel —
    /// один launch на весь T с внутренним циклом по чанкам. Совпадение бит-в-бит
    /// не обязано выполняться по построению, но выполняется: тело шага
    /// посимвольно одно и то же, а порядок шагов последовательный в обоих.
    /// Проверяем строгим равенством намеренно — ослабление допуска здесь
    /// означало бы, что кэш и обучение разъехались, и это надо заметить.
    func testForwardAndTrainAgreeOnState() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 64)
        let h0 = makeState()
        let (oF, hF) = wkv7ForwardWithState(r, w, k, v, a, b, h0)
        let (oT, hT) = wkv7TrainWithState(r, w, k, v, a, b, h0)

        XCTAssertEqual(maxAbsDiff(oF, oT), 0, "выходы forward/train разошлись")
        XCTAssertEqual(maxAbsDiff(hF, hT), 0, "состояния forward/train разошлись")
    }

    /// Ненулевое hIn против наивного эталона: и выход, и конечное состояние.
    func testNonZeroStateParityWithReference() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 32)
        let h0 = makeState()
        let (oRef, hRef) = wkv7ReferenceWithState(r, w, k, v, a, b, h0)
        let (oKrn, hKrn) = wkv7TrainWithState(r, w, k, v, a, b, h0)

        XCTAssertLessThan(relDiff(oRef, oKrn), 1e-4,
                          "выход с ненулевым состоянием разошёлся с эталоном")
        XCTAssertLessThan(relDiff(hRef, hKrn), 1e-4,
                          "конечное состояние разошлось с эталоном")
    }

    /// Совместимые обёртки не изменили поведение: нулевое состояние даёт
    /// ровно то же, что раньше.
    func testZeroStateMatchesLegacyWrappers() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 32)
        XCTAssertEqual(maxAbsDiff(wkv7Forward(r, w, k, v, a, b),
                                  wkv7ForwardWithState(r, w, k, v, a, b, nil).0), 0)
        XCTAssertEqual(maxAbsDiff(wkv7Train(r, w, k, v, a, b),
                                  wkv7TrainWithState(r, w, k, v, a, b, nil).0), 0)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Градиенты
    // ─────────────────────────────────────────────────────────────────

    /// dh_in против автограда эталона. Раньше это значение ядро считало и
    /// выбрасывало — теперь оно доходит до вызывающего, и его надо проверить.
    func testGradientWrtStateParity() {
        let (r, w, k, v, a, b, g) = makeInputs(T: 32)
        let h0 = makeState()

        func lossRef(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7ReferenceWithState(r, w, k, v, a, b, inp[0]).0 * g).sum()]
        }
        func lossKrn(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(r, w, k, v, a, b, inp[0]).0 * g).sum()]
        }
        let dRef = grad(lossRef, argumentNumbers: [0])([h0])
        let dKrn = grad(lossKrn, argumentNumbers: [0])([h0])
        eval(dRef); eval(dKrn)

        XCTAssertLessThan(relDiff(dRef[0], dKrn[0]),
                          1e-3, "dh_in разошёлся с эталоном")
        XCTAssertGreaterThan(MLX.abs(dKrn[0]).max().item(Float.self), 0,
                             "dh_in нулевой — градиент по состоянию не течёт")
    }

    /// Градиент, приходящий ЧЕРЕЗ h_out (второй котангент). Проверяет, что
    /// cotangents[1] действительно используется, а не игнорируется: лосс здесь
    /// зависит ТОЛЬКО от конечного состояния.
    func testGradientThroughOutputState() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 32)
        let gh = makeState(seed: 11)

        func lossRef(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7ReferenceWithState(inp[0], w, k, v, a, b, nil).1 * gh).sum()]
        }
        func lossKrn(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(inp[0], w, k, v, a, b, nil).1 * gh).sum()]
        }
        let dRef = grad(lossRef, argumentNumbers: [0])([r])
        let dKrn = grad(lossKrn, argumentNumbers: [0])([r])
        eval(dRef); eval(dKrn)

        // y = S·r читает состояние ПОСЛЕ обновления, но само обновление от r
        // не зависит ⇒ dr через h_out строго нулевой. Эталон это подтверждает,
        // и ядро обязано согласиться (а не выдать мусор из неинициализированного
        // d_h_out).
        XCTAssertEqual(MLX.abs(dRef[0]).max().item(Float.self), 0,
                       "эталон: dr через h_out должен быть нулевым")
        XCTAssertEqual(maxAbsDiff(dRef[0], dKrn[0]), 0,
                       "ядро: dr через h_out разошёлся с эталоном")
    }

    /// То же, но по параметру, который на состояние ВЛИЯЕТ (k входит в
    /// обновление S). Здесь градиент обязан быть ненулевым и совпасть.
    func testGradientThroughOutputStateNonZero() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 32)
        let gh = makeState(seed: 11)

        func lossRef(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7ReferenceWithState(r, w, inp[0], v, a, b, nil).1 * gh).sum()]
        }
        func lossKrn(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(r, w, inp[0], v, a, b, nil).1 * gh).sum()]
        }
        let dRef = grad(lossRef, argumentNumbers: [0])([k])
        let dKrn = grad(lossKrn, argumentNumbers: [0])([k])
        eval(dRef); eval(dKrn)

        XCTAssertGreaterThan(MLX.abs(dRef[0]).max().item(Float.self), 0,
                             "эталон: dk через h_out не должен быть нулевым")
        XCTAssertLessThan(relDiff(dRef[0], dKrn[0]), 1e-3,
                          "dk через h_out разошёлся с эталоном")
    }

    /// Градиенты по r..b при НЕнулевом начальном состоянии: backward должен
    /// корректно раскручиваться с произвольного h_in, а не только с нуля.
    func testParameterGradientsWithNonZeroState() {
        let (r, w, k, v, a, b, g) = makeInputs(T: 32)
        let h0 = makeState()

        func lossRef(_ i: [MLXArray]) -> [MLXArray] {
            [(wkv7ReferenceWithState(i[0], i[1], i[2], i[3], i[4], i[5], h0).0 * g).sum()]
        }
        func lossKrn(_ i: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(i[0], i[1], i[2], i[3], i[4], i[5], h0).0 * g).sum()]
        }
        let args = Array(0 ..< 6)
        let gRef = grad(lossRef, argumentNumbers: args)([r, w, k, v, a, b])
        let gKrn = grad(lossKrn, argumentNumbers: args)([r, w, k, v, a, b])
        eval(gRef); eval(gKrn)

        let names = ["dr", "dw", "dk", "dv", "da", "db"]
        for i in 0 ..< 6 {
            XCTAssertLessThan(relDiff(gRef[i], gKrn[i]), 1e-3,
                              "\(names[i]) при ненулевом h_in разошёлся с эталоном")
        }
    }
}
