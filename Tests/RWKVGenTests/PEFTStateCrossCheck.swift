//
//  PEFTStateCrossCheck.swift
//  Раскладка состояния RWKV-PEFT против mlx-lm на настоящей модели. Замер,
//  не проверка — печатает числа; сверку делает
//  swr_bench_keep/w1023/peft/crosscheck.py.
//
//      RWKV_PEFT_MODEL=…/g1j_1.5b.safetensors RWKV_PEFT_TOKENS=…/tokens_321.txt RWKV_PEFT_N=64 \
//      RWKV_PEFT_OUT=…/swift_fold.pth [RWKV_PEFT_IN=…/mlxlm_fold.pth] \
//          swift test -c release --filter PEFTStateCrossCheck
//
//  OUT — свёртка первых N токенов здесь (fp32), записанная exportPEFT.
//  IN — свёртка тех же токенов в mlx-lm, записанная torch.save: читается
//  importPEFT и сравнивается со своей свёрткой как есть и транспонированной.
//  Без RWKV_PEFT_MODEL тест пропускается.
//
import XCTest
import MLX
@testable import RWKVGen

final class PEFTStateCrossCheck: XCTestCase {
    func testCrossCheck() throws {
        let env = ProcessInfo.processInfo.environment
        guard let mp = env["RWKV_PEFT_MODEL"], let tp = env["RWKV_PEFT_TOKENS"], let out = env["RWKV_PEFT_OUT"] else {
            throw XCTSkip("RWKV_PEFT_MODEL / RWKV_PEFT_TOKENS / RWKV_PEFT_OUT не заданы — сверка вручную")
        }
        let n = Int(env["RWKV_PEFT_N"] ?? "64") ?? 64
        let ids = try String(contentsOfFile: tp, encoding: .utf8)
            .split(whereSeparator: { $0 == " " || $0 == "\n" }).compactMap { Int($0) }.prefix(n)
        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil }.max()! + 1
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1], vocab: w["head.weight"]!.shape[0])
        let bb = X070Backbone(weights: w, cfg: cfg, computeDType: .float32)
        let fold = bb.foldPrompt(Array(ids))
        try StateFile.exportPEFT(fold, to: URL(fileURLWithPath: out), dropShifts: true)
        print("PEFT-CROSS: свёртка \(ids.count) токенов записана: \(out), wkv \(fold.wkv.shape)")
        if let inp = env["RWKV_PEFT_IN"] {
            let (theirs, ignored) = try StateFile.importPEFT(URL(fileURLWithPath: inp), base: bb)
            func rel(_ a: MLXArray) -> Float { (norm(a - fold.wkv) / norm(fold.wkv)).item(Float.self) }
            print(String(format: "PEFT-CROSS: mlx-lm (torch.save) против своей свёртки: как есть %.3e, "
                         + "транспонировано %.3e; лишних ключей %d",
                         rel(theirs.wkv), rel(theirs.wkv.swappedAxes(-1, -2)), ignored.count))
        }
    }
}
