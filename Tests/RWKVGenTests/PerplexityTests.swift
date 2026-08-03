//
//  PerplexityTests.swift
//  Перплексия под флагом приведения выхода WKV — то самое число, которого
//  не хватало для решения.
//
//  Скорость под флагом замерена (release, ×2.82 на плотной 0.1B), цена по
//  точности тоже (согласие путей 3.75e-6 → 9.90e-3). Осталось единственное:
//  ЗНАЧИТ ли расхождение 7e-3 на логитах что-нибудь для качества.
//
//  Дисциплина замера:
//   • обе ветки — В ОДНОМ ПРОЦЕССЕ, на одной модели, на одном корпусе;
//     флаг переключается на живом объекте. Разными прогонами перплексию
//     сравнивать нельзя по той же причине, по которой нельзя скорость;
//   • чанки НЕЗАВИСИМЫЕ, каждый со своего нулевого состояния. RWKV
//     stateful, и «продолжать» состояние между чанками значило бы мерить
//     ещё и длину контекста;
//   • лосс считается в fp32 ВСЕГДА, независимо от типа логитов. Иначе
//     сравнивались бы не модели, а точность самого суммирования;
//   • корпус ~/Develop/test.txt — разные домены, русский, сербский,
//     английский. Русский тут важен отдельно: он даёт длинные цепочки
//     байтовых токенов, где потеря разрядов активации видна сильнее.
//
//  Пропускается без модели или корпуса.
//
import XCTest
import MLX
@testable import RWKVGen

final class PerplexityTests: XCTestCase {

    /// Средняя кросс-энтропия по чанкам. Возвращает (ppl, число позиций).
    func perplexity(_ bb: X070Backbone, _ chunks: [[Int]]) -> (Double, Int) {
        var total = 0.0
        var count = 0
        for ids in chunks {
            let idx = MLXArray(ids.map { Int32($0) }, [1, ids.count])
            // Логиты в fp32 независимо от того, в чём считала модель.
            let logits = bb(idx)[0].asType(.float32)          // [T, V]
            // Предсказание позиции t относится к токену t+1.
            let pred = logits[0 ..< (ids.count - 1)]
            let target = MLXArray(ids[1...].map { Int32($0) })
            let lse = MLX.logSumExp(pred, axis: -1)
            let picked = MLX.takeAlong(pred, target.reshaped([-1, 1]), axis: 1)
                .reshaped([-1])
            let nll = (lse - picked).sum()
            eval(nll)
            total += Double(nll.item(Float.self))
            count += ids.count - 1
        }
        return (exp(total / Double(count)), count)
    }

    func testPerplexityUnderWKVCastFlag() throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let vp = env["RWKV_WORLD_VOCAB"] ?? home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        let cp = env["RWKV_PPL_CORPUS"] ?? home.appendingPathComponent(
            "Develop/test.txt").path
        for p in [mp, vp, cp] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет фикстуры \(p)")
        }

        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: w["head.weight"]!.shape[0])
        let bb = X070Backbone(weights: w, cfg: cfg)
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: vp))
        else { throw XCTSkip("словарь не разобрался") }

        let text = try String(contentsOfFile: cp, encoding: .utf8)
        let ids = tok.encode(text)
        let T = 512
        var chunks: [[Int]] = []
        var s = 0
        while s + T <= ids.count { chunks.append(Array(ids[s ..< s + T])); s += T }
        try XCTSkipIf(chunks.count < 4, "корпус слишком мал: \(chunks.count) чанков")
        print("корпус: \(text.count) символов → \(ids.count) токенов, "
              + "\(chunks.count) чанков по \(T)")

        // A/B в одном процессе. Порядок off→on→off: если бы что-то дрейфовало
        // между прогонами, два «off» разошлись бы, и это было бы видно.
        bb.castWKVOutputToComputeDType = false
        let (pplOff1, n) = perplexity(bb, chunks)
        bb.castWKVOutputToComputeDType = true
        let (pplOn, _) = perplexity(bb, chunks)
        bb.castWKVOutputToComputeDType = false
        let (pplOff2, _) = perplexity(bb, chunks)

        let delta = (pplOn - pplOff1) / pplOff1 * 100
        print(String(format: """

        ── ПЕРПЛЕКСИЯ 0.1B на test.txt (%d позиций) ──
        выкл (fp32-активация): %.4f   [повтор: %.4f]
        вкл  (bf16-активация): %.4f
        разница: %+.3f%%
        """, n, pplOff1, pplOff2, pplOn, delta))

        XCTAssertEqual(pplOff1, pplOff2, accuracy: 1e-9,
                       "два одинаковых прогона разошлись — замер недетерминирован")
        XCTAssertTrue(pplOn.isFinite && pplOn > 1, "перплексия под флагом невалидна")
        // Граница-сторож, а не решение: если приведение СЛОМАЕТ модель, ppl
        // улетит в разы. Решение о включении принимается по напечатанному
        // числу человеком, а не этим порогом.
        XCTAssertLessThan(abs(delta), 10.0,
                          "перплексия сдвинулась на \(delta)% — это уже не округление")
    }
}
