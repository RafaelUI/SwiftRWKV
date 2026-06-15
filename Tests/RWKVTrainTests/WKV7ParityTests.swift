import XCTest
import MLX
@testable import RWKVTrain
@testable import RWKVKernel

/// Паритет кастомного Metal-ядра WKV-7 против эталона из Python-референса
/// (`rwkv-metal`). Эталон — один forward-чанк (B=1, T=CHUNK=32, H=4, D=64):
/// входы `r,w,k,v,a,b,h_in` и эталонные выходы `out,h_out,sa_out`.
final class WKV7ParityTests: XCTestCase {

    /// Загружает эталонные тензоры из ресурса теста.
    private func loadParity() throws -> [String: MLXArray] {
        guard let url = Bundle.module.url(
            forResource: "wkv7_kernel_parity", withExtension: "safetensors")
        else {
            throw XCTSkip("Эталон wkv7_kernel_parity.safetensors не найден в ресурсах теста")
        }
        return try loadArrays(url: url)
    }

    /// Максимум модуля поэлементной разницы двух массивов.
    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32))
        let m = d.max()
        m.eval()
        return m.item(Float.self)
    }

    /// Один forward-чанк ядра должен совпасть с эталоном.
    /// fp32, ожидаем (почти) битовое совпадение — допуск очень мал.
    func testChunkForwardMatchesReference() throws {
        let ref = try loadParity()
        for key in ["r", "w", "k", "v", "a", "b", "h_in", "out", "h_out", "sa_out"] {
            XCTAssertNotNil(ref[key], "в эталоне нет тензора '\(key)'")
        }

        let r = ref["r"]!, w = ref["w"]!, k = ref["k"]!
        let v = ref["v"]!, a = ref["a"]!, b = ref["b"]!
        let hIn = ref["h_in"]!

        // Геометрия эталона должна совпасть с константами ядра.
        XCTAssertEqual(r.shape[1], WKV7_CHUNK, "T эталона должен равняться CHUNK")
        XCTAssertEqual(r.shape[3], WKV7_HEAD_SIZE, "D эталона должен равняться HEAD_SIZE")

        let (out, hOut, saOut) = wkv7ChunkForward(r, w, k, v, a, b, hIn)
        eval(out, hOut, saOut)

        let tol: Float = 1e-4
        let dOut = maxAbsDiff(out, ref["out"]!)
        let dH   = maxAbsDiff(hOut, ref["h_out"]!)
        let dSA  = maxAbsDiff(saOut, ref["sa_out"]!)

        XCTAssertLessThan(dOut, tol, "out: Δ=\(dOut)")
        XCTAssertLessThan(dH,   tol, "h_out: Δ=\(dH)")
        XCTAssertLessThan(dSA,  tol, "sa_out: Δ=\(dSA)")

        print(String(format: "[parity] chunk forward: out Δ=%.2e  h_out Δ=%.2e  sa_out Δ=%.2e",
                     dOut, dH, dSA))
    }

    /// Полносеквенсный `wkv7Forward` на одном чанке должен совпасть с
    /// `wkv7ChunkForward` (h_in = 0). Проверяет согласованность двух путей.
    func testFullForwardMatchesChunkOnSingleChunk() throws {
        let ref = try loadParity()
        let r = ref["r"]!, w = ref["w"]!, k = ref["k"]!
        let v = ref["v"]!, a = ref["a"]!, b = ref["b"]!

        let B = r.shape[0], H = r.shape[2], D = r.shape[3]
        let hZero = MLXArray.zeros([B, H, D, D], dtype: .float32)

        let (chunkOut, _, _) = wkv7ChunkForward(r, w, k, v, a, b, hZero)
        let fullOut = wkv7Forward(r, w, k, v, a, b)   // h_in = 0 внутри
        eval(chunkOut, fullOut)

        let diff = maxAbsDiff(fullOut, chunkOut)
        XCTAssertLessThan(diff, 1e-4, "wkv7Forward vs wkv7ChunkForward: Δ=\(diff)")
        print(String(format: "[parity] full vs chunk (1 chunk): Δ=%.2e", diff))
    }

    /// Дифференцируемый `wkv7Train` (forward-часть) должен совпасть с
    /// `wkv7Forward` на тех же входах.
    func testTrainForwardMatchesForward() throws {
        let ref = try loadParity()
        let r = ref["r"]!, w = ref["w"]!, k = ref["k"]!
        let v = ref["v"]!, a = ref["a"]!, b = ref["b"]!

        let fwd = wkv7Forward(r, w, k, v, a, b)
        let trn = wkv7Train(r, w, k, v, a, b)
        eval(fwd, trn)

        let diff = maxAbsDiff(trn, fwd)
        XCTAssertLessThan(diff, 1e-4, "wkv7Train vs wkv7Forward: Δ=\(diff)")
        print(String(format: "[parity] train-forward vs forward: Δ=%.2e", diff))
    }
}
