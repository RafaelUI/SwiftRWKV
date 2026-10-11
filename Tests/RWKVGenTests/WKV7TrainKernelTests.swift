import XCTest
import MLX
import MLXRandom
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Ядра обучаемого WKV: прямое со стейджингом, обратное по схеме Бо Пэна
//  (`wkv7BackwardV2`) и удержание выходов прямого ядра до обратного прохода
//  (`wkv7KeepForwardOutputs`) вместо пересчёта внутри VJP.
//
//  Лосс везде берёт ОБА выхода — out и h_out — и дифференцируется по всем
//  семи входам, включая начальное состояние: у обратного ядра семь выходов и
//  два котангента, и дефект в любом из них обязан быть виден.
// ───────────────────────────────────────────────────────────────────────

final class WKV7TrainKernelTests: XCTestCase {

    struct Case {
        let ins: [MLXArray]      // r, w, k, v, a, b, h0
        let g: MLXArray, gh: MLXArray
    }

    func makeCase(B: Int = 2, T: Int = 48, H: Int = 3, seed: UInt64 = 0) -> Case {
        let D = WKV7_HEAD_SIZE
        MLXRandom.seed(seed)
        func rnd(_ s: Float) -> MLXArray { MLXRandom.normal([B, T, H, D]) * s }
        let r = rnd(0.5), v = rnd(0.5), k = rnd(0.5)
        let kk = k / sqrt((k * k).sum(axis: -1, keepDims: true) + 1e-12)
        let a = -kk, b = kk * sigmoid(rnd(1.0))
        let w = exp(-0.606531 * sigmoid(rnd(1.0)))
        // состояние и котангент по нему — НЕсимметричные матрицы: перепутанные
        // строка и колонка на симметричной были бы не видны
        let h0 = MLXRandom.normal([B, H, D, D]) * 0.1
        let g = rnd(1.0), gh = MLXRandom.normal([B, H, D, D]) * 0.3
        eval(r, w, k, v, a, b, h0, g, gh)
        return Case(ins: [r, w, k, v, a, b, h0], g: g, gh: gh)
    }

    static let names = ["dr", "dw", "dk", "dv", "da", "db", "dh_in"]

    func kernelGrads(_ c: Case, v2: Bool, keep: Bool) -> [MLXArray] {
        wkv7BackwardV2 = v2; wkv7KeepForwardOutputs = keep
        defer { wkv7BackwardV2 = true; wkv7KeepForwardOutputs = true }
        let f = grad({ (i: [MLXArray]) -> [MLXArray] in
            let (o, h) = wkv7TrainWithState(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
            return [(o * c.g).sum() + (h * c.gh).sum()]
        }, argumentNumbers: Array(0 ..< 7))
        let d = f(c.ins); eval(d); return d
    }

    func referenceGrads(_ c: Case) -> [MLXArray] {
        let f = grad({ (i: [MLXArray]) -> [MLXArray] in
            let (o, h) = wkv7ReferenceWithState(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
            return [(o * c.g).sum() + (h * c.gh).sum()]
        }, argumentNumbers: Array(0 ..< 7))
        let d = f(c.ins); eval(d); return d
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        MLX.abs(ref - got).max().item(Float.self) / (MLX.abs(ref).max().item(Float.self) + 1e-9)
    }
    func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        Int((a .!= b).asType(.int32).sum().item(Int32.self))
    }

    /// Оба обратных ядра сходятся с автоградом эталона по всем семи выходам.
    func testBothBackwardKernelsMatchReference() {
        let c = makeCase()
        let ref = referenceGrads(c)
        for v2 in [true, false] {
            let got = kernelGrads(c, v2: v2, keep: true)
            let errs = zip(ref, got).map { rel($0, $1) }
            print("ОБРАТНОЕ ЯДРО \(v2 ? "по схеме Бо Пэна" : "прежнее") к эталону: "
                  + zip(Self.names, errs).map { String(format: "%@ %.1e", $0, $1) }.joined(separator: ", "))
            for (n, e) in zip(Self.names, errs) {
                XCTAssertLessThan(e, 1e-3, "\(n), ядро \(v2 ? "новое" : "прежнее")")
            }
            for (n, gk) in zip(Self.names, got) {
                XCTAssertGreaterThan(MLX.abs(gk).max().item(Float.self), 0, "\(n) нулевой")
            }
        }
    }

    /// Ядра считают одно и то же (порядок сложения разный), а переключатель
    /// действительно их меняет.
    func testBackwardKernelsAgreeAndTheSwitchWorks() {
        let c = makeCase(seed: 3)
        let new = kernelGrads(c, v2: true, keep: true), old = kernelGrads(c, v2: false, keep: true)
        let errs = zip(old, new).map { rel($0, $1) }
        print("ОБРАТНЫЕ ЯДРА между собой: "
              + zip(Self.names, errs).map { String(format: "%@ %.1e", $0, $1) }.joined(separator: ", "))
        for (n, e) in zip(Self.names, errs) { XCTAssertLessThan(e, 1e-4, n) }
        XCTAssertGreaterThan(zip(old, new).map { mismatches($0, $1) }.reduce(0, +), 0,
                             "переключатель wkv7BackwardV2 ничего не меняет")
    }

    /// Удержание выходов прямого ядра — тот же счёт, что пересчёт: градиенты
    /// ПОБИТНО равны. И пересчёта действительно нет: прямое ядро ставится в
    /// граф один раз на вызов, а не два.
    func testKeptForwardOutputsGiveTheSameGradientsWithoutRecompute() {
        for v2 in [true, false] {
            let c = makeCase(seed: 5)
            var n = wkv7TrainForwardLaunches
            let kept = kernelGrads(c, v2: v2, keep: true)
            XCTAssertEqual(wkv7TrainForwardLaunches - n, 1, "удержание: прямое ядро пересчитано")
            n = wkv7TrainForwardLaunches
            let recomputed = kernelGrads(c, v2: v2, keep: false)
            XCTAssertEqual(wkv7TrainForwardLaunches - n, 2, "пересчёт: прямое ядро не пересчитано")
            for (name, (x, y)) in zip(Self.names, zip(kept, recomputed)) {
                XCTAssertEqual(mismatches(x, y), 0, "\(name), v2 = \(v2)")
            }
        }
    }

    /// Прямой проход: оба режима дают побитно одно и то же и сходятся с
    /// эталоном; стейджинг и разорванные суммы числа не портят.
    func testForwardValueIsTheSameInBothModes() {
        let c = makeCase(seed: 9)
        func run(_ keep: Bool) -> (MLXArray, MLXArray) {
            wkv7KeepForwardOutputs = keep
            defer { wkv7KeepForwardOutputs = true }
            let o = wkv7TrainWithState(c.ins[0], c.ins[1], c.ins[2], c.ins[3], c.ins[4], c.ins[5], c.ins[6])
            eval(o.0, o.1); return o
        }
        let a = run(true), b = run(false)
        XCTAssertEqual(mismatches(a.0, b.0), 0)
        XCTAssertEqual(mismatches(a.1, b.1), 0)
        let ref = wkv7ReferenceWithState(c.ins[0], c.ins[1], c.ins[2], c.ins[3], c.ins[4], c.ins[5], c.ins[6])
        print(String(format: "ПРЯМОЕ ОБУЧАЕМОЕ ЯДРО к эталону: out %.1e, h_out %.1e", rel(ref.0, a.0), rel(ref.1, a.1)))
        XCTAssertLessThan(rel(ref.0, a.0), 1e-4)
        XCTAssertLessThan(rel(ref.1, a.1), 1e-4)
        // и с замороженным ядром (одним запуском на всю длину)
        let seq = wkv7ForwardWithState(c.ins[0], c.ins[1], c.ins[2], c.ins[3], c.ins[4], c.ins[5], c.ins[6])
        XCTAssertLessThan(rel(seq.0, a.0), 1e-4)
        XCTAssertLessThan(rel(seq.1, a.1), 1e-4)
    }

    /// Градиент доходит до параметра ВЫШЕ ядра и равен цепному правилу; через
    /// удержанные выходы прямого ядра (они под stopGradient) он не идёт.
    func testGradientReachesAParameterAboveTheKernel() {
        let c = makeCase(seed: 13)
        let dr = kernelGrads(c, v2: true, keep: true)[0]
        for keep in [true, false] {
            wkv7KeepForwardOutputs = keep
            defer { wkv7KeepForwardOutputs = true }
            let f = grad({ (p: [MLXArray]) -> [MLXArray] in
                let (o, h) = wkv7TrainWithState(c.ins[0] * p[0], c.ins[1], c.ins[2], c.ins[3],
                                                c.ins[4], c.ins[5], c.ins[6])
                return [(o * c.g).sum() + (h * c.gh).sum()]
            }, argumentNumbers: [0])
            let got = f([MLXArray(Float(1))])[0].item(Float.self)
            let want = (dr * c.ins[0]).sum().item(Float.self)
            XCTAssertEqual(got, want, accuracy: Swift.abs(want) * 1e-4 + 1e-4, "keep = \(keep)")
            XCTAssertNotEqual(got, 0)
        }
    }

    /// Абляция выключена по умолчанию: подмена обратного прохода нулями —
    /// только прибор замера.
    func testAblationIsOffByDefault() {
        XCTAssertFalse(RWKVAblation.skipWKVBackward)
        XCTAssertFalse(RWKVAblation.skipProjectionBackward)
        XCTAssertTrue(wkv7BackwardV2)
        XCTAssertTrue(wkv7KeepForwardOutputs)
    }
}
