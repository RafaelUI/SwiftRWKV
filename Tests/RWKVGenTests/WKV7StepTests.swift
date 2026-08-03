//
//  WKV7StepTests.swift
//  Один шаг WKV-7 на чистых MLX-операциях: согласие с ядром и градиенты.
//
//  Зачем это нужно. Голова реранкера прогоняет РОВНО ОДИН обучаемый токен
//  на блок поверх состояния базы. Через Metal-ядро это стоило бы CHUNK=16
//  шагов вместо одного (длина добивается no-op'ами) плюс checkpoint-ядро с
//  ручным backward. wkv7Step разворачивает шаг в обычные операции: та же
//  математика, автоградиент бесплатно, в 16 раз меньше работы.
//
//  «Та же математика» — ровно то, что здесь проверяется. Раскладка состояния
//  h[dv, dk] (первый индекс — value, второй — key) не самоочевидна, а
//  перепутать оси местами легко: формы совпадают (D×D), ошибка молчит, и
//  голова просто учится чему-то другому. Ловится цепочкой шагов против
//  сплошного прохода — при перепутанных осях рекуррентность разъезжается
//  уже на втором токене.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVKernel

final class WKV7StepTests: XCTestCase {

    typealias Inputs = (r: MLXArray, w: MLXArray, k: MLXArray,
                        v: MLXArray, a: MLXArray, b: MLXArray, g: MLXArray)

    /// Те же распределения, что в WKV7StateTests: w — затухание из
    /// exp(-0.606531·sigmoid), a/b — DPLR-пара из нормированного k.
    func makeInputs(B: Int = 2, T: Int = 1, H: Int = 4, D: Int = 64,
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

    func makeState(B: Int = 2, H: Int = 4, D: Int = 64, seed: UInt64 = 7) -> MLXArray {
        MLXRandom.seed(seed)
        let h = MLXRandom.normal([B, H, D, D]) * 0.1
        eval(h)
        return h
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        eval(ref, got)
        let d = MLX.abs(ref - got).max().item(Float.self)
        let m = MLX.abs(ref).max().item(Float.self)
        return d / (m + 1e-9)
    }

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Согласие с ядром
    // ─────────────────────────────────────────────────────────────────

    /// Один шаг против forward-ядра с добивкой до CHUNK, ненулевое состояние.
    ///
    /// Допуск, а не строгое равенство: ядро складывает sa и y последовательным
    /// циклом по dk, MLX-редукция — деревом. Порядок сложения fp32 разный,
    /// расхождение неизбежно и мало.
    ///
    /// Границы назначены ПОСЛЕ замера, а не до. Замерено (относительное,
    /// max|Δ| / max|ref|): выход 8.7e-8, состояние 4.9e-8, нулевое
    /// состояние 2.2e-7. Граница 1e-6 — примерно впятеро выше худшего из них.
    func testStepMatchesForwardKernel() {
        let (r, w, k, v, a, b, _) = makeInputs()
        let h0 = makeState()
        let (oKrn, hKrn) = wkv7ForwardWithState(r, w, k, v, a, b, h0)
        let (oStp, hStp) = wkv7Step(r, w, k, v, a, b, h0)

        XCTAssertLessThan(relDiff(oKrn, oStp), 1e-6, "выход шага разошёлся с ядром")
        XCTAssertLessThan(relDiff(hKrn, hStp), 1e-6, "состояние шага разошлось с ядром")
    }

    /// То же с нулевым начальным состоянием: hIn=nil — отдельная ветка.
    func testStepMatchesForwardKernelZeroState() {
        let (r, w, k, v, a, b, _) = makeInputs(seed: 3)
        let (oKrn, hKrn) = wkv7ForwardWithState(r, w, k, v, a, b, nil)
        let (oStp, hStp) = wkv7Step(r, w, k, v, a, b, nil)

        XCTAssertLessThan(relDiff(oKrn, oStp), 1e-6, "выход при нулевом состоянии разошёлся")
        XCTAssertLessThan(relDiff(hKrn, hStp), 1e-6, "состояние при нулевом hIn разошлось")
    }

    /// Против наивного эталона (wkv7ReferenceWithState) при T=1.
    ///
    /// ЧЕСТНАЯ ОГОВОРКА: это НЕ независимая проверка формулы. Эталон при T=1
    /// строит ровно тот же граф MLX-операций в том же порядке, поэтому
    /// равенство здесь побитовое и по построению. Тест держится как
    /// детектор расхождения двух реализаций (если однажды правку внесут в
    /// одну и забудут в другой), а роль независимой проверки играют тесты
    /// против ЯДРА выше и цепочка шагов ниже.
    func testStepMatchesReference() {
        let (r, w, k, v, a, b, _) = makeInputs(seed: 5)
        let h0 = makeState(seed: 13)
        let (oRef, hRef) = wkv7ReferenceWithState(r, w, k, v, a, b, h0)
        let (oStp, hStp) = wkv7Step(r, w, k, v, a, b, h0)

        XCTAssertEqual(maxAbsDiff(oRef, oStp), 0, "выход разошёлся с эталоном")
        XCTAssertEqual(maxAbsDiff(hRef, hStp), 0, "состояние разошлось с эталоном")
    }

    /// ГЛАВНЫЙ тест: T шагов подряд == один сплошной проход.
    ///
    /// Именно он различает раскладку h[dv,dk] и h[dk,dv]. При перепутанных
    /// осях один шаг из нулевого состояния ещё может совпасть (h=0 симметрично),
    /// но со второго токена рекуррентность разъезжается необратимо. Поэтому
    /// цепочка длиной 32, а не 2.
    func testChainedStepsMatchContiguousForward() {
        let T = 32
        let (r, w, k, v, a, b, _) = makeInputs(T: T, seed: 21)
        let h0 = makeState(seed: 23)
        let (oFull, hFull) = wkv7ForwardWithState(r, w, k, v, a, b, h0)

        var h = h0
        var outs: [MLXArray] = []
        for t in 0 ..< T {
            func at(_ x: MLXArray) -> MLXArray { x[0..., t ..< (t + 1)] }
            let (o, hNext) = wkv7Step(at(r), at(w), at(k), at(v), at(a), at(b), h)
            outs.append(o)
            h = hNext
        }
        let joined = concatenated(outs, axis: 1)

        // Замерено: выход 2.1e-7, состояние 8.6e-8 — то есть 32 шага почти
        // НЕ накапливают расхождение (на одном шаге было 8.7e-8). Так и
        // должно быть: w < 1 сжимает состояние, старая ошибка затухает
        // вместе с ним. Граница 1e-5 с запасом на другие сиды.
        XCTAssertLessThan(relDiff(oFull, joined), 1e-5,
                          "цепочка шагов разошлась со сплошным проходом")
        XCTAssertLessThan(relDiff(hFull, h), 1e-5,
                          "состояние после \(T) шагов разошлось")
    }

    /// Ранг входа сохраняется: [B,H,D] и [B,1,H,D] дают одно и то же
    /// с точностью до оси времени.
    func testRankPreservedBothWays() {
        let (r, w, k, v, a, b, _) = makeInputs(seed: 31)
        let h0 = makeState(seed: 37)
        let (o4, h4) = wkv7Step(r, w, k, v, a, b, h0)
        func sq(_ x: MLXArray) -> MLXArray { x[0..., 0] }
        let (o3, h3) = wkv7Step(sq(r), sq(w), sq(k), sq(v), sq(a), sq(b), h0)

        XCTAssertEqual(o4.shape, [r.shape[0], 1, r.shape[2], r.shape[3]])
        XCTAssertEqual(o3.shape, [r.shape[0], r.shape[2], r.shape[3]])
        XCTAssertEqual(maxAbsDiff(sq(o4), o3), 0, "ранг входа изменил результат")
        XCTAssertEqual(maxAbsDiff(h4, h3), 0, "ранг входа изменил состояние")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Градиенты
    // ─────────────────────────────────────────────────────────────────

    // Сверять градиенты приходится на ЦЕПОЧКЕ из CHUNK шагов, а не на одном:
    // checkpoint-ядро требует T кратного 16 и на T=1 просто падает по
    // precondition. Это не обход неудобства, а ровно тот сценарий, ради
    // которого шаг написан, — только развёрнутый до длины, которую ядро
    // вообще умеет считать.
    private func chainedSteps(_ r: MLXArray, _ w: MLXArray, _ k: MLXArray,
                              _ v: MLXArray, _ a: MLXArray, _ b: MLXArray,
                              _ h0: MLXArray?) -> (MLXArray, MLXArray) {
        let T = r.shape[1]
        var h: MLXArray? = h0
        var outs: [MLXArray] = []
        for t in 0 ..< T {
            func at(_ x: MLXArray) -> MLXArray { x[0..., t ..< (t + 1)] }
            let (o, hNext) = wkv7Step(at(r), at(w), at(k), at(v), at(a), at(b), h)
            outs.append(o)
            h = hNext
        }
        return (concatenated(outs, axis: 1), h!)
    }

    /// Градиент по h_in против dh_in checkpoint-ядра. Это то, ради чего шаг и
    /// существует: голова читает состояние, и без градиента по нему ничему бы
    /// не научилась — точнее, научилась бы, но только своим весам, а не чтению.
    func testGradientWrtStateMatchesKernel() {
        let (r, w, k, v, a, b, g) = makeInputs(T: 16, seed: 41)
        let h0 = makeState(seed: 43)

        func lossStep(_ i: [MLXArray]) -> [MLXArray] {
            [(chainedSteps(r, w, k, v, a, b, i[0]).0 * g).sum()]
        }
        func lossKrn(_ i: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(r, w, k, v, a, b, i[0]).0 * g).sum()]
        }
        let dStp = grad(lossStep, argumentNumbers: [0])([h0])
        let dKrn = grad(lossKrn, argumentNumbers: [0])([h0])
        eval(dStp); eval(dKrn)

        XCTAssertGreaterThan(MLX.abs(dStp[0]).max().item(Float.self), 0,
                             "dh_in нулевой — градиент по состоянию не течёт")
        XCTAssertLessThan(relDiff(dKrn[0], dStp[0]), 1e-4,
                          "dh_in разошёлся с ядром")
    }

    /// Градиенты по r..b при ненулевом h_in — против того же checkpoint-ядра.
    /// Ядро считает backward своей ручной обратной рекуррентностью с делением
    /// на w; шаг — автоградом MLX. Совпадение означает, что ручной вывод верен.
    ///
    /// Замерено: dr 5.9e-6, dw 1.7e-5, dk 2.2e-7, dv 1.8e-7, da 1.2e-5,
    /// db 3.0e-7. dw и da — самые шумные, и это ожидаемо: обратная
    /// рекуррентность ядра делит на w, а w здесь порядка exp(-0.6·σ) ≈ 0.55…1,
    /// так что деление подтягивает младшие разряды. Граница 1e-4 — вшестеро
    /// выше худшего замера.
    func testParameterGradientsMatchKernel() {
        let (r, w, k, v, a, b, g) = makeInputs(T: 16, seed: 47)
        let h0 = makeState(seed: 53)

        func lossStep(_ i: [MLXArray]) -> [MLXArray] {
            [(chainedSteps(i[0], i[1], i[2], i[3], i[4], i[5], h0).0 * g).sum()]
        }
        func lossKrn(_ i: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(i[0], i[1], i[2], i[3], i[4], i[5], h0).0 * g).sum()]
        }
        let args = Array(0 ..< 6)
        let gStp = grad(lossStep, argumentNumbers: args)([r, w, k, v, a, b])
        let gKrn = grad(lossKrn, argumentNumbers: args)([r, w, k, v, a, b])
        eval(gStp); eval(gKrn)

        let names = ["dr", "dw", "dk", "dv", "da", "db"]
        for i in 0 ..< 6 {
            XCTAssertGreaterThan(MLX.abs(gStp[i]).max().item(Float.self), 0,
                                 "\(names[i]) нулевой — градиент не течёт")
            XCTAssertLessThan(relDiff(gKrn[i], gStp[i]), 1e-4,
                              "\(names[i]) разошёлся с ядром")
        }
    }

    /// Градиент, приходящий ЧЕРЕЗ h_out: лосс зависит ТОЛЬКО от конечного
    /// состояния. Голова реранкера ставит блоки друг на друга — выход одного
    /// блока идёт следующему, а состояние читается каждым, — поэтому этот
    /// путь не гипотетический.
    func testGradientThroughOutputState() {
        let (r, w, k, v, a, b, _) = makeInputs(T: 16, seed: 59)
        let gh = makeState(seed: 61)

        func lossStep(_ i: [MLXArray]) -> [MLXArray] {
            [(chainedSteps(r, w, i[0], v, a, b, nil).1 * gh).sum()]
        }
        func lossKrn(_ i: [MLXArray]) -> [MLXArray] {
            [(wkv7TrainWithState(r, w, i[0], v, a, b, nil).1 * gh).sum()]
        }
        let dStp = grad(lossStep, argumentNumbers: [0])([k])
        let dKrn = grad(lossKrn, argumentNumbers: [0])([k])
        eval(dStp); eval(dKrn)

        XCTAssertGreaterThan(MLX.abs(dStp[0]).max().item(Float.self), 0,
                             "dk через h_out нулевой")
        XCTAssertLessThan(relDiff(dKrn[0], dStp[0]), 1e-4,
                          "dk через h_out разошёлся с ядром")
    }
}
