import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Конвейерный жадный декод (GreedyDecodePipeline, GenerationConfig.pipelined).
//
//  Конвейер меняет РАСПИСАНИЕ: шаг t+1 ставится в очередь, пока GPU считает
//  токен t. Граф и его входы те же, поэтому всё наблюдаемое — токены, текст,
//  причина остановки и СОСТОЯНИЕ после генерации — обязано совпасть с
//  синхронным путём побитно, при любой причине остановки. Состояние здесь
//  главное: конвейер всегда считает на шаг вперёд, и ошибка на единицу в
//  том, какой токен оно уже поглотило, не видна ни в тексте, ни в токенах —
//  только в следующем продолжении.
// ───────────────────────────────────────────────────────────────────────

final class GeneratePipelineTests: XCTestCase {

    static func tokenizer() throws -> WorldTokenizer {
        let vp = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipIf(!FileManager.default.fileExists(atPath: vp), "нет словаря World — тест пропущен")
        return try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: vp)))
    }

    struct Run { let result: GenerationResult; let state: RWKVState; let calls: Int }

    static func run(_ bb: X070Backbone, _ tok: WorldTokenizer, _ prompt: [Int],
                    _ config: GenerationConfig, cancelAfter: Int? = nil) -> Run {
        var st = RWKVState(cfg: bb.cfg)
        let before = bb.pipelinedSteps
        var seen = 0
        let r = bb.generate(prompt: prompt, config: config, tokenizer: tok, state: &st) { _, _ in
            seen += 1
            return cancelAfter.map { seen < $0 } ?? true
        }
        st.eval()
        return Run(result: r, state: st, calls: bb.pipelinedSteps - before)
    }

    static func assertSame(_ a: Run, _ b: Run, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.result.tokens, b.result.tokens, "\(what): токены", file: file, line: line)
        XCTAssertEqual(a.result.text, b.result.text, "\(what): текст", file: file, line: line)
        XCTAssertEqual(a.result.stopReason, b.result.stopReason, "\(what): причина остановки", file: file, line: line)
        func bad(_ x: [MLXArray], _ y: [MLXArray]) -> Int {
            zip(x, y).map { RwkvqFileTests.mismatches($0.asType(.float32), $1.asType(.float32)) }.reduce(0, +)
        }
        XCTAssertEqual(bad(a.state.wkv, b.state.wkv), 0, "\(what): состояние wkv", file: file, line: line)
        XCTAssertEqual(bad(a.state.tmixPrev, b.state.tmixPrev), 0, "\(what): сдвиг tmix", file: file, line: line)
        XCTAssertEqual(bad(a.state.cmixPrev, b.state.cmixPrev), 0, "\(what): сдвиг cmix", file: file, line: line)
    }

    func models() throws -> [(String, X070Backbone)] {
        var out: [(String, X070Backbone)] = []
        for preset in RwkvqFileTests.presets {
            out.append((preset, try X070Backbone.load(fromRwkvq: try RwkvqModelTests.fixture(preset).path)))
        }
        let dense = try loadArrays(url: RwkvqFileTests.testdata.appendingPathComponent("world_0.1b_x070.safetensors"))
        out.append(("плотная bf16", X070Backbone(weights: dense, cfg: X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536))))
        return out
    }

    /// Все причины остановки: лимит, стоп-токен, стоп-строка, отмена.
    func testPipelinedGenerationEqualsSynchronousOne() throws {
        let tok = try Self.tokenizer()
        let prompt = tok.encode("The capital of France is")
        for (name, bb) in try models() {
            func both(_ what: String, _ make: (Bool) -> GenerationConfig, cancelAfter: Int? = nil) -> Run {
                let sync = Self.run(bb, tok, prompt, make(false), cancelAfter: cancelAfter)
                let pipe = Self.run(bb, tok, prompt, make(true), cancelAfter: cancelAfter)
                XCTAssertEqual(sync.calls, 0, "\(name), \(what): синхронный путь пошёл конвейером")
                XCTAssertEqual(pipe.calls, pipe.result.tokens.count, "\(name), \(what): конвейер не включился")
                Self.assertSame(pipe, sync, "\(name), \(what)")
                return sync
            }
            let full = both("лимит") { GenerationConfig(maxTokens: 24, pipelined: $0) }
            XCTAssertEqual(full.result.tokens.count, 24)
            XCTAssertEqual(full.result.stopReason, .maxTokens)

            let stopAt = full.result.tokens[7]
            let cut = both("стоп-токен") { GenerationConfig(maxTokens: 24, stopTokens: [stopAt], pipelined: $0) }
            XCTAssertEqual(cut.result.stopReason, .stopToken(stopAt))
            XCTAssertLessThan(cut.result.tokens.count, 24)

            let cancelled = both("отмена", { GenerationConfig(maxTokens: 24, pipelined: $0) }, cancelAfter: 5)
            XCTAssertEqual(cancelled.result.stopReason, .cancelled)
            XCTAssertEqual(cancelled.result.tokens.count, 5)

            // стоп-строка: кусок из середины выданного текста
            let text = full.result.text
            if text.count >= 12 {
                let from = text.index(text.startIndex, offsetBy: 6)
                let needle = String(text[from ..< text.index(from, offsetBy: 4)])
                let s = both("стоп-строка") { GenerationConfig(maxTokens: 24, stopStrings: [needle], pipelined: $0) }
                XCTAssertEqual(s.result.stopReason, .stopString(needle))
            }
        }
    }

    /// Состояние после конвейера годится для продолжения: вторая генерация с
    /// него выдаёт то же, что после синхронной первой.
    func testStateAfterPipelineContinuesTheSameText() throws {
        let tok = try Self.tokenizer()
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqModelTests.fixture("compression").path)
        let prompt = tok.encode("Пчёлы собирают нектар")
        func twice(_ pipelined: Bool) -> ([Int], [Int]) {
            var st = RWKVState(cfg: bb.cfg)
            let cfg = GenerationConfig(maxTokens: 10, pipelined: pipelined)
            let a = bb.generate(prompt: prompt, config: cfg, tokenizer: tok, state: &st)
            let b = bb.generate(prompt: [a.tokens.last!], config: cfg, tokenizer: tok, state: &st)
            return (a.tokens, b.tokens)
        }
        let p = twice(true), s = twice(false)
        XCTAssertEqual(p.0, s.0); XCTAssertEqual(p.1, s.1)
    }

    /// Сэмплер с температурой или штрафами живёт на CPU — конвейер для него
    /// не включается, сколько ни проси.
    func testPipelineIsOnlyForPlainGreedy() throws {
        let tok = try Self.tokenizer()
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqModelTests.fixture("reduction").path)
        let prompt = tok.encode("Hello")
        let hot = Self.run(bb, tok, prompt, GenerationConfig(maxTokens: 6, sampling: SamplingConfig(temperature: 0.8, seed: 3)))
        XCTAssertEqual(hot.calls, 0, "сэмплер с температурой пошёл конвейером")
        let pen = Self.run(bb, tok, prompt, GenerationConfig(maxTokens: 6, sampling: SamplingConfig(repetitionPenalty: 1.2)))
        XCTAssertEqual(pen.calls, 0, "жадный выбор со штрафом пошёл конвейером")
        let plain = Self.run(bb, tok, prompt, GenerationConfig(maxTokens: 6))
        XCTAssertEqual(plain.calls, 6, "конвейер по умолчанию выключен")
    }
}
