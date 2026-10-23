//
//  StateCastDump.swift
//  Градиент начального состояния при приведении выхода WKV к типу вычислений
//  (`castWKVOutputToComputeDType`) против точного счёта: какое плечо ближе к
//  модели целиком в fp32. Замер, не проверка — печатает числа.
//
//      RWKV_STATE_CAST_MODEL=…/g1j_1.5b.safetensors \
//      RWKV_STATE_CAST_TOKENS=…/tokens_321.txt  swift test -c release --filter StateCastDump
//
//  Без RWKV_STATE_CAST_MODEL тест пропускается.
//
import XCTest
import MLX
@testable import RWKVGen

final class StateCastDump: XCTestCase {
    func testDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let mp = env["RWKV_STATE_CAST_MODEL"], let tp = env["RWKV_STATE_CAST_TOKENS"] else {
            throw XCTSkip("RWKV_STATE_CAST_MODEL / RWKV_STATE_CAST_TOKENS не заданы — замер вручную")
        }
        let ids = try String(contentsOfFile: tp, encoding: .utf8)
            .split(whereSeparator: { $0 == " " || $0 == "\n" }).compactMap { Int32($0) }
        let T = ids.count - 1
        let batch: LoRABatch = (x: MLXArray(Array(ids.dropLast()), [1, T]), y: MLXArray(Array(ids.dropFirst()), [1, T]))
        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil }.max()! + 1
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1], vocab: w["head.weight"]!.shape[0])

        // Плечо: тип вычислений, приведение, стартовое состояние (нули либо случайное малое — как после
        // нескольких шагов обучения). Градиент — обратным проходом по блокам, как в StateTuning.
        func gradient(_ dt: DType, cast: Bool, random: Bool, lin: Bool = false) -> (Float, MLXArray) {
            let bb = X070Backbone(weights: w, cfg: cfg, computeDType: dt)
            bb.castWKVOutputToComputeDType = cast
            bb.castDenseProjectionInput = lin
            var initial: RWKVBatchState? = nil
            if random {
                MLXRandom.seed(5)
                let S = cfg.headSize, H = cfg.nEmbd / S
                initial = RWKVBatchState(wkv: MLXRandom.normal([nL, 1, H, S, S]) * 0.02,
                                         tmixShift: MLXArray.zeros([nL, 1, 1, cfg.nEmbd]),
                                         cmixShift: MLXArray.zeros([nL, 1, 1, cfg.nEmbd]))
            }
            let set = StateTrainableSet(bb, initial: initial)
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: set.initialParameters(),
                                                     batch: batch, lossChunk: 256)
            let g = concatenated(r.gradients.map { $0.asType(.float32).reshaped([-1]) })
            eval(g, r.loss)
            return (r.loss.item(Float.self), g)
        }
        func rel(_ a: MLXArray, _ ref: MLXArray) -> Float { (norm(a - ref) / norm(ref)).item(Float.self) }
        func cos(_ a: MLXArray, _ b: MLXArray) -> Float { ((a * b).sum() / (norm(a) * norm(b))).item(Float.self) }
        for random in [false, true] {
            let (l32, g32) = gradient(.float32, cast: false, random: random)
            let (lOff, gOff) = gradient(.bfloat16, cast: false, random: random)
            let (lOn, gOn) = gradient(.bfloat16, cast: true, random: random)
            let (lLin, gLin) = gradient(.bfloat16, cast: false, random: random, lin: true)
            print(String(format: "STATE-CAST %@ | приведение входа проекций: лосс %.6f, ‖g−g32‖/‖g32‖ %.3e, cos %.6f",
                         random ? "случайное S0" : "нулевое S0", lLin, rel(gLin, g32), cos(gLin, g32)))
            print(String(format: "STATE-CAST %@ T=%d | лосс: fp32 %.6f, bf16 без приведения %.6f, bf16 с приведением %.6f | "
                         + "‖g−g32‖/‖g32‖: без %.3e, с %.3e | cos к g32: без %.6f, с %.6f | без − с: %.3e",
                         random ? "случайное S0" : "нулевое S0", T, l32, lOff, lOn,
                         rel(gOff, g32), rel(gOn, g32), cos(gOff, g32), cos(gOn, g32), rel(gOn, gOff)))
        }
    }
}
