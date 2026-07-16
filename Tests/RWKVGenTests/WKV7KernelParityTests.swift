//
//  WKV7KernelParityTests.swift
//  Паритет checkpoint-ядра (wkv7Train) с автоград-эталоном (wkv7Reference).
//  Самодостаточен: случайные данные с фиксированным seed, без ресурсов.
//  Регрессионный тест для гонки threadgroup-памяти в backward
//  (C_row = C_row*w_sh + dsa*a_sh без барьера перед следующим шагом).
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVKernel

final class WKV7KernelParityTests: XCTestCase {

    func makeInputs(B: Int = 2, T: Int = 64, H: Int = 4, D: Int = 64)
        -> (r: MLXArray, w: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray, g: MLXArray) {
        MLXRandom.seed(0)
        func rnd(_ s: Float) -> MLXArray {
            MLXRandom.normal([B, T, H, D]) * s
        }
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

    func testForwardParity() {
        let (r, w, k, v, a, b, _) = makeInputs()
        let oRef = wkv7Reference(r, w, k, v, a, b)
        let oKrn = wkv7Train(r, w, k, v, a, b)
        eval(oRef, oKrn)
        let maxDiff = MLX.abs(oRef - oKrn).max().item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-4, "forward расходится с эталоном")
    }

    func testBackwardParity() {
        let (r, w, k, v, a, b, g) = makeInputs()

        func lossRef(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7Reference(inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]) * g).sum()]
        }
        func lossKrn(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7Train(inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]) * g).sum()]
        }
        let args = Array(0 ..< 6)
        let gRef = grad(lossRef, argumentNumbers: args)([r, w, k, v, a, b])
        let gKrn = grad(lossKrn, argumentNumbers: args)([r, w, k, v, a, b])
        eval(gRef); eval(gKrn)

        let names = ["dr", "dw", "dk", "dv", "da", "db"]
        for i in 0 ..< 6 {
            let d = MLX.abs(gRef[i] - gKrn[i]).max().item(Float.self)
            let m = MLX.abs(gRef[i]).max().item(Float.self)
            let rel = d / (m + 1e-9)
            XCTAssertLessThan(rel, 1e-3, "\(names[i]): rel=\(rel), max|d|=\(d)")
        }
    }

    /// Детерминизм backward: гонка проявлялась дрейфом градиентов ~1–2%
    /// между идентичными запусками.
    func testBackwardDeterminism() {
        let (r, w, k, v, a, b, g) = makeInputs()
        func lossKrn(_ inp: [MLXArray]) -> [MLXArray] {
            [(wkv7Train(inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]) * g).sum()]
        }
        let f = grad(lossKrn, argumentNumbers: Array(0 ..< 6))
        let g1 = f([r, w, k, v, a, b]); eval(g1)
        let g2 = f([r, w, k, v, a, b]); eval(g2)
        for i in 0 ..< 6 {
            let d = MLX.abs(g1[i] - g2[i]).max().item(Float.self)
            XCTAssertEqual(d, 0, "недетерминизм в градиенте #\(i): max|d|=\(d)")
        }
    }
}
