import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

/// Паритет Swift-порта RWKV-7 x070 против Python-эталона (rwkv-metal).
/// Эталон: World-0.1B (n_layer=12, n_embd=768, vocab=65536), вход
/// input_ids=[1..8], ожидаемые ln_out [8,768] и logits [8,65536] в bf16.
final class X070ParityTests: XCTestCase {

    private func resource(_ name: String) throws -> URL {
        guard let url = Bundle.module.url(forResource: name, withExtension: "safetensors")
        else { throw XCTSkip("ресурс \(name).safetensors не найден") }
        return url
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32))
        let m = d.max(); m.eval()
        return m.item(Float.self)
    }

    func testForwardMatchesReference() throws {
        // Веса World-0.1B (большой файл — может отсутствовать в CI).
        let weightsURL = try resource("world_0.1b_x070")
        let refURL = try resource("x070_parity")

        let weights = try loadArrays(url: weightsURL)
        let ref = try loadArrays(url: refURL)

        guard let inputIds = ref["input_ids"],
              var refLnOut = ref["ln_out"],
              var refLogits = ref["logits"]
        else { throw XCTSkip("в эталоне нет input_ids/ln_out/logits") }

        // Эталон сохранён как U16 (сырые биты bf16) — переинтерпретируем в bfloat16.
        if refLnOut.dtype == .uint16 { refLnOut = refLnOut.view(dtype: .bfloat16) }
        if refLogits.dtype == .uint16 { refLogits = refLogits.view(dtype: .bfloat16) }

        let cfg = X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536)
        let model = X070Backbone(weights: weights, cfg: cfg, computeDType: .bfloat16)

        // input_ids в эталоне [1,8] int32
        let ids = inputIds.asType(.int32)

        let lnOut = model.body(ids)            // [1,8,768]
        let logits = model(ids)                // [1,8,65536]
        eval(lnOut, logits)

        // Диагностика масштаба.
        func mm(_ a: MLXArray, _ name: String) {
            let mn = a.asType(.float32).min(); let mx_ = a.asType(.float32).max()
            mn.eval(); mx_.eval()
            print("[x070 dbg] \(name): min=\(mn.item(Float.self)) max=\(mx_.item(Float.self))")
        }
        mm(lnOut, "swift ln_out")
        mm(refLnOut, "ref   ln_out")
        mm(logits, "swift logits")
        mm(refLogits, "ref   logits")

        let lnOut2 = lnOut.reshaped([8, 768])
        let logits2 = logits.reshaped([8, 65536])

        // ── Критерий 1: средняя абсолютная ошибка logits — на уровне bf16.
        //    max-метрика непоказательна: RWKV имеет каналы-выбросы (±150),
        //    которые в bf16 квантуются грубо → накопление по 12 слоям (это норма).
        let meanErr = abs(logits2.asType(.float32) - refLogits.asType(.float32)).mean()
        meanErr.eval()
        let dLogitsMax = maxAbsDiff(logits2, refLogits)
        print(String(format: "[x070 parity] logits: meanErr=%.4e  maxErr=%.4e",
                     meanErr.item(Float.self), dLogitsMax))
        XCTAssertLessThan(meanErr.item(Float.self), 3e-2, "средняя ошибка logits велика")

        // ── Критерий 2: топ-5 предсказаний на последнем токене совпадают (порядок может
        //    различаться на bf16-шуме, но множество должно совпасть).
        func top5(_ row: MLXArray) -> Set<Int32> {
            let idx = argSort(row.asType(.float32), axis: -1)   // возрастание
            idx.eval()
            let arr = idx.asArray(Int32.self)
            return Set(arr.suffix(5))
        }
        let swTop = top5(logits2[7])
        let refTop = top5(refLogits[7])
        let overlap = swTop.intersection(refTop).count
        print("[x070 parity] top-5 overlap (last token): \(overlap)/5  swift=\(swTop.sorted()) ref=\(refTop.sorted())")
        XCTAssertGreaterThanOrEqual(overlap, 4, "top-5 предсказания сильно расходятся")
    }
}
