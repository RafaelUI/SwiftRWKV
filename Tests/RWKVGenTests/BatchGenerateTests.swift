//
//  BatchGenerateTests.swift
//  Батчевый декод (25.10): префилл и шаг батчем против одиночного пути,
//  независимость строк от соседей, генерация батчем против `generate`.
//
//  Модель — крошечная x070 в fp32 (приведение выхода WKV там тождественно),
//  токенизатор — свой: id → одна буква, без фикстур.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

/// Токенизатор на 64 id: id → буква (a…z и A…Z, дальше — цифры).
final class LetterTokenizer: RWKVTokenizer {
    static let alphabet = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;!".utf8)
    var count: Int { 64 }
    func encode(_ text: String) -> [Int] { text.utf8.compactMap { Self.alphabet.firstIndex(of: $0) } }
    func decode(_ tokens: [Int]) -> String { String(decoding: tokens.compactMap { rawBytes($0) }.flatMap { $0 }, as: UTF8.self) }
    func rawBytes(_ id: Int) -> [UInt8]? { id >= 0 && id < 64 ? [Self.alphabet[id]] : nil }
}

final class BatchGenerateTests: XCTestCase {

    func model(seed: UInt64 = 11) -> X070Backbone {
        TinyBackbone.make(nLayer: 3, nEmbd: 128, vocab: 64, seed: seed).0
    }

    // длины вокруг границы куска ядра (16)
    let prompts: [[Int]] = [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12, 13, 14, 15], [42],
                            Array(0 ..< 16).map { ($0 * 5) % 64 }, Array(0 ..< 17).map { ($0 * 3 + 1) % 64 }]

    func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
        (MLX.abs(a - b).max() / (MLX.abs(b).max() + 1e-12)).item(Float.self)
    }

    /// Префилл и три шага батчем — те же логиты, что у одиночного пути на
    /// каждой строке (в пределах расхождения параллельного и рекуррентного
    /// ядер), и строка не зависит от соседей по батчу.
    func testBatchMatchesSingleAndRowsAreIndependent() {
        let bb = model()
        var (logits, state) = bb.prefillBatch(prompts)
        XCTAssertEqual(logits.shape, [prompts.count, 64])
        var singles = prompts.map { p -> (MLXArray, RWKVState) in
            var st = RWKVState(cfg: bb.cfg)
            let l = bb.prefill(p, state: &st)
            return (l, st)
        }
        var worst: Float = 0
        for step in 0 ..< 3 {
            for i in prompts.indices { worst = max(worst, rel(logits[i], singles[i].0)) }
            let next = prompts.indices.map { (i: Int) in (i * 7 + step * 3) % 64 }
            logits = bb.stepBatch(next, state: &state)
            for i in prompts.indices { singles[i].0 = bb.step(next[i], state: &singles[i].1) }
        }
        XCTAssertLessThan(worst, 1e-5, "батч разошёлся с одиночным путём: \(worst)")

        // соседи: строка в батче против той же строки одной
        let (alone, _) = bb.prefillBatch([prompts[1]])
        let (full, _) = bb.prefillBatch(prompts)
        XCTAssertLessThan(rel(full[1], alone[0]), 1e-5)
        // различающее: строки действительно разные
        XCTAssertGreaterThan(rel(full[0], full[1]), 1e-2)
    }

    /// Жадная генерация батчем — те же токены, текст и причина остановки,
    /// что у `generate` по каждой строке (с конвейером и без), в том числе
    /// при стоп-токене и стоп-строке на разных шагах у разных строк.
    func testGreedyBatchEqualsSingleGenerate() {
        let bb = model()
        let tok = LetterTokenizer()
        // стоп-токен и стоп-строка — из того, что модель реально выдаёт
        var st = RWKVState(cfg: bb.cfg)
        let probe = bb.generate(prompt: prompts[0], config: GenerationConfig(maxTokens: 12, pipelined: false),
                                tokenizer: tok, state: &st)
        let stopTok = probe.tokens[5]
        let stopStr = String(probe.text.dropFirst(2).prefix(2))
        for pipelined in [false, true] {
            let cfg = GenerationConfig(maxTokens: 12, stopStrings: [stopStr], stopTokens: [stopTok], pipelined: pipelined)
            let batch = bb.generateBatch(prompts: prompts, config: cfg, tokenizer: tok)
            var stops = Set<String>()
            for (i, p) in prompts.enumerated() {
                var s = RWKVState(cfg: bb.cfg)
                let one = bb.generate(prompt: p, config: cfg, tokenizer: tok, state: &s)
                XCTAssertEqual(batch[i].tokens, one.tokens, "строка \(i), конвейер \(pipelined)")
                XCTAssertEqual(batch[i].text, one.text, "строка \(i)")
                XCTAssertEqual(batch[i].stopReason, one.stopReason, "строка \(i)")
                stops.insert("\(one.stopReason)")
            }
            XCTAssertGreaterThan(stops.count, 1, "все строки остановились одинаково — выбытие из батча не проверено")
        }
    }

    /// Сэмплирование: строка i — тот же текст, что `generate` (CPU-сэмплер)
    /// с сидом seed + i.
    func testSampledRowUsesSeedPlusRow() {
        let bb = model()
        let tok = LetterTokenizer()
        var sc = SamplingConfig(temperature: 1.0, topP: 0.9)
        sc.seed = 100
        let batch = bb.generateBatch(prompts: Array(prompts.prefix(3)),
                                     config: GenerationConfig(maxTokens: 10, sampling: sc, pipelined: false),
                                     tokenizer: tok)
        for i in 0 ..< 3 {
            var s = sc; s.seed = 100 + UInt64(i)
            var st = RWKVState(cfg: bb.cfg)
            let one = bb.generate(prompt: prompts[i], config: GenerationConfig(maxTokens: 10, sampling: s, pipelined: false),
                                  tokenizer: tok, state: &st)
            XCTAssertEqual(batch[i].tokens, one.tokens, "строка \(i)")
        }
        XCTAssertNotEqual(batch[0].tokens, batch[1].tokens)
    }

    /// Штрафы с историей промпта: строка — как `generate` с теми же штрафами.
    func testPenaltiesWithPromptHistory() {
        let bb = model()
        let tok = LetterTokenizer()
        var sc = SamplingConfig()
        sc.presencePenalty = 2.0
        let cfg = GenerationConfig(maxTokens: 10, penalizePrompt: true, sampling: sc, pipelined: false)
        let batch = bb.generateBatch(prompts: prompts, config: cfg, tokenizer: tok)
        var plainCfg = cfg; plainCfg.penalizePrompt = false
        let plain = bb.generateBatch(prompts: prompts, config: plainCfg, tokenizer: tok)
        var differs = false
        for (i, p) in prompts.enumerated() {
            var st = RWKVState(cfg: bb.cfg)
            XCTAssertEqual(batch[i].tokens, bb.generate(prompt: p, config: cfg, tokenizer: tok, state: &st).tokens, "строка \(i)")
            differs = differs || batch[i].tokens != plain[i].tokens
        }
        XCTAssertTrue(differs, "история промпта ничего не изменила — тест вырожден")
    }

    /// Подключённое начальное состояние действует на все строки батча, как на
    /// одиночный путь.
    func testAttachedStateAppliesToEveryRow() {
        let bb = model()
        let c = bb.cfg
        MLXRandom.seed(5)
        bb.initialState = RWKVBatchState(
            wkv: MLXRandom.normal([c.nLayer, 1, c.nHead, c.headSize, c.headSize]) * 0.3,
            tmixShift: MLXRandom.normal([c.nLayer, 1, 1, c.nEmbd]) * 0.3,
            cmixShift: MLXRandom.normal([c.nLayer, 1, 1, c.nEmbd]) * 0.3)
        let (logits, _) = bb.prefillBatch(prompts)
        for (i, p) in prompts.enumerated() {
            var st = RWKVState(cfg: c)
            XCTAssertLessThan(rel(logits[i], bb.prefill(p, state: &st)), 1e-5, "строка \(i)")
        }
        bb.initialState = nil
        let (plain, _) = bb.prefillBatch(prompts)
        XCTAssertGreaterThan(rel(plain[0], logits[0]), 1e-3, "состояние ничего не изменило — тест вырожден")
    }
}
