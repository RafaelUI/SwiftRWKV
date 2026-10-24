//
//  StateCastDump.swift
//  Градиент начального состояния при разных типах вычислений (bf16, fp16) и
//  способах приведения (без / вход проекций / выход WKV) против счёта целиком
//  в fp32 — на нескольких окнах и двух стартовых состояниях. Замер, не
//  проверка — печатает числа и пишет JSON.
//
//      RWKV_STATE_CAST_MODEL=…/g1j_1.5b.safetensors \
//      RWKV_STATE_CAST_TOKENS=…/windows16.txt [RWKV_STATE_CAST_OUT=…/grad.json] \
//          swift test -c release --filter StateCastDump
//
//  TOKENS — окна по строке (номера токенов через пробел, T+1 штук, T кратна 16).
//  Без RWKV_STATE_CAST_MODEL тест пропускается.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

final class StateCastDump: XCTestCase {
    func testDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let mp = env["RWKV_STATE_CAST_MODEL"], let tp = env["RWKV_STATE_CAST_TOKENS"] else {
            throw XCTSkip("RWKV_STATE_CAST_MODEL / RWKV_STATE_CAST_TOKENS не заданы — замер вручную")
        }
        let windows: [[Int32]] = try String(contentsOfFile: tp, encoding: .utf8).split(separator: "\n")
            .map { $0.split(separator: " ").compactMap { Int32($0) } }.filter { $0.count > 16 }
        let batches: [LoRABatch] = windows.map { ids in
            let T = ids.count - 1
            precondition(T % 16 == 0, "T = \(T) не кратна 16")
            return (x: MLXArray(Array(ids.dropLast()), [1, T]), y: MLXArray(Array(ids.dropFirst()), [1, T]))
        }
        // подкачка весов: в памяти один блок — плечо fp32 на 1.5B иначе уходит в своп (числа те же до бита)
        let url = URL(fileURLWithPath: mp)
        let cfg = try X070Backbone.loadStreaming(safetensors: url, computeDType: .float32).cfg
        let nL = cfg.nLayer
        let S = cfg.headSize, H = cfg.nEmbd / S

        func start(_ random: Bool, _ i: Int) -> RWKVBatchState? {
            guard random else { return nil }
            MLXRandom.seed(UInt64(5 + i))
            return RWKVBatchState(wkv: MLXRandom.normal([nL, 1, H, S, S]) * 0.02,
                                  tmixShift: MLXArray.zeros([nL, 1, 1, cfg.nEmbd]),
                                  cmixShift: MLXArray.zeros([nL, 1, 1, cfg.nEmbd]))
        }
        // Плечо: тип вычислений и приведение. Градиент — обратным проходом по блокам, как в StateTuning.
        func gradients(_ dt: DType, wkv: Bool, lin: Bool) -> [(loss: Float, g: MLXArray)] {
            let bb = try! X070Backbone.loadStreaming(safetensors: url, computeDType: dt)
            bb.castWKVOutputToComputeDType = wkv
            bb.castDenseProjectionInput = lin
            var out: [(Float, MLXArray)] = []
            for random in [false, true] {
                for (i, b) in batches.enumerated() {
                    bb.initialState = nil
                    let set = StateTrainableSet(bb, initial: start(random, i))
                    let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: set.initialParameters(),
                                                             batch: b, lossChunk: 256)
                    let g = concatenated(r.gradients.map { $0.asType(.float32).reshaped([-1]) })
                    eval(g, r.loss)
                    out.append((r.loss.item(Float.self), g))
                }
            }
            return out
        }
        let ref = gradients(.float32, wkv: false, lin: false)
        let arms: [(String, DType, Bool, Bool)] = [
            ("bf16 без", .bfloat16, false, false), ("bf16 вход проекций", .bfloat16, false, true),
            ("bf16 выход WKV", .bfloat16, true, false),
            ("fp16 без", .float16, false, false), ("fp16 вход проекций", .float16, false, true),
            ("fp16 выход WKV", .float16, true, false)]
        var json: [[String: Any]] = []
        let K = batches.count
        for (name, dt, wkv, lin) in arms {
            let got = gradients(dt, wkv: wkv, lin: lin)
            var rels: [Float] = [], coss: [Float] = [], dl: [Float] = []
            for (a, r) in zip(got, ref) {
                rels.append((norm(a.g - r.g) / norm(r.g)).item(Float.self))
                coss.append(((a.g * r.g).sum() / (norm(a.g) * norm(r.g))).item(Float.self))
                dl.append(a.loss - r.loss)
            }
            func mean(_ x: ArraySlice<Float>) -> Float { x.reduce(0, +) / Float(x.count) }
            print(String(format: "STATE-DTYPE %@ | ‖g−g32‖/‖g32‖: нулевое S0 %.4f, случайное %.4f | cos %.6f / %.6f | лосс − fp32 %.2e",
                         name, mean(rels[0 ..< K]), mean(rels[K...]), mean(coss[0 ..< K]), mean(coss[K...]), mean(dl[...])))
            json.append(["arm": name, "rel": rels, "cos": coss, "dloss": dl, "windows": K])
        }
        print("STATE-DTYPE fp32 лоссы: " + ref.map { String(format: "%.4f", $0.loss) }.joined(separator: " "))
        if let o = env["RWKV_STATE_CAST_OUT"] {
            try JSONSerialization.data(withJSONObject: ["arms": json, "ref_loss": ref.map { $0.loss }],
                                       options: [.prettyPrinted]).write(to: URL(fileURLWithPath: o))
        }
    }
}
