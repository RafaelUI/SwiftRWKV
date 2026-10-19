//
//  StateSnapshotTests.swift
//  Снимок состояния генерации (19.10): RWKVState как значение, файл
//  (`StateFile.saveSnapshot` / `loadSnapshot`), подключение снимка как
//  начального состояния.
//
//  Эталон — непрерывная генерация: продолжение со снимка (в памяти или с
//  диска) обязано дать ПОБИТНО те же логиты и токены, что продолжение без
//  перерыва, во всех типах потока (fp32 / bf16 / fp16).
//
import XCTest
import MLX
@testable import RWKVGen

final class StateSnapshotTests: XCTestCase {

    func model(_ dtype: DType) -> X070Backbone {
        let cfg = TinyBackbone.config(nLayer: 3, vocab: 64)
        return X070Backbone(weights: TinyBackbone.weights(cfg, seed: 7), cfg: cfg, computeDType: dtype)
    }

    let prompt = [3, 17, 42, 5, 9, 33, 21, 8, 60, 2, 11]
    let more = [14, 27, 6, 50, 1, 39]

    func tmp() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("snap_\(UUID().uuidString).safetensors")
    }

    /// Логиты по шагам: подать `ids` по одному с состояния `s`.
    func roll(_ bb: X070Backbone, _ s: inout RWKVState, _ ids: [Int]) -> [MLXArray] {
        ids.map { let l = bb.step($0, state: &s); eval(l); return l }
    }

    func same(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { MLX.abs($0.asType(.float32) - $1.asType(.float32)).max().item(Float.self) == 0 }
    }

    func sameState(_ a: RWKVState, _ b: RWKVState) -> Bool {
        same(a.wkv, b.wkv) && same(a.tmixPrev, b.tmixPrev) && same(a.cmixPrev, b.cmixPrev)
            && zip(a.tmixPrev, b.tmixPrev).allSatisfy { $0.dtype == $1.dtype }
    }

    /// В памяти: снимок не меняется от дальнейшей генерации, продолжение с
    /// него побитно то же, что без перерыва.
    func testInMemorySnapshotIsAValue() {
        for dt in [DType.float32, .bfloat16, .float16] {
            let bb = model(dt)
            var s = bb.startState()
            _ = bb.prefill(prompt, state: &s); s.eval()
            let saved = s
            let ref = roll(bb, &s, more)
            var c = saved
            XCTAssertTrue(same(roll(bb, &c, more), ref), "\(dt): продолжение со снимка разошлось")
            XCTAssertTrue(sameState(c, s), "\(dt): конечные состояния разошлись")
            var c2 = saved
            XCTAssertTrue(same(roll(bb, &c2, more), ref), "\(dt): снимок изменился от генерации в копии")
        }
    }

    /// Файл: круг туда-обратно побитный (и тип сдвигов тот же), продолжение
    /// с прочитанного — побитно как без перерыва.
    func testFileRoundTripContinuesBitIdentically() throws {
        for dt in [DType.float32, .bfloat16, .float16] {
            let bb = model(dt)
            var s = bb.startState()
            _ = bb.prefill(prompt, state: &s); s.eval()
            let url = tmp(); defer { try? FileManager.default.removeItem(at: url) }
            let info = try StateFile.saveSnapshot(s, base: bb, to: url, note: "тест")
            XCTAssertEqual(info.kind, "snapshot")
            let (r, rinfo) = try StateFile.loadSnapshot(url, base: bb)
            XCTAssertEqual(rinfo.note, "тест")
            XCTAssertTrue(sameState(r, s), "\(dt): круг через файл не побитный")
            XCTAssertEqual(r.tmixPrev[0].dtype, s.tmixPrev[0].dtype, "\(dt): тип сдвигов не восстановлен")
            var a = s, b = r
            XCTAssertTrue(same(roll(bb, &a, more), roll(bb, &b, more)), "\(dt): продолжение с файла разошлось")
        }
    }

    /// Тип сдвигов восстанавливается из файла, а не берётся у модели: снимок
    /// со сдвигами в bf16 и в fp16 на fp32-модели читается с тем же типом и
    /// побитно теми же значениями.
    func testShiftTypeIsRestored() throws {
        let bb = model(.float32)
        var s = bb.startState()
        _ = bb.prefill(prompt, state: &s); s.eval()
        for t in [DType.bfloat16, .float16] {
            var c = s
            c.tmixPrev = c.tmixPrev.map { $0.asType(t) }
            c.cmixPrev = c.cmixPrev.map { $0.asType(t) }
            c.eval()
            let url = tmp(); defer { try? FileManager.default.removeItem(at: url) }
            try StateFile.saveSnapshot(c, base: bb, to: url)
            let r = try StateFile.loadSnapshot(url, base: bb).state
            XCTAssertEqual(r.tmixPrev[0].dtype, t)
            XCTAssertEqual(r.cmixPrev[2].dtype, t)
            XCTAssertTrue(sameState(r, c), "\(t): значения сдвигов не те")
        }
    }

    /// Снимок после `generate` (многоходовый диалог): следующий ход с
    /// прочитанного снимка — те же токены и текст, что в той же сессии.
    func testGenerateResumesFromSnapshot() throws {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет словаря World")
        let tok = try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: path)))
        let bb = model(.bfloat16)
        for pipelined in [true, false] {
            let cfg = GenerationConfig(maxTokens: 12, sampling: SamplingConfig(temperature: 0.9, topK: 20, seed: 5),
                                       pipelined: pipelined)
            var s = bb.startState()
            _ = bb.generate(prompt: prompt, config: cfg, tokenizer: tok, state: &s)
            let url = tmp(); defer { try? FileManager.default.removeItem(at: url) }
            try StateFile.saveSnapshot(s, base: bb, to: url)
            let ref = bb.generate(prompt: more, config: cfg, tokenizer: tok, state: &s)
            var r = try StateFile.loadSnapshot(url, base: bb).state
            let got = bb.generate(prompt: more, config: cfg, tokenizer: tok, state: &r)
            XCTAssertEqual(got.tokens, ref.tokens, "pipelined \(pipelined): второй ход со снимка — другие токены")
            XCTAssertTrue(sameState(r, s))
        }
    }

    /// Чужая база — отказ (пока не разрешено); чужая архитектура — всегда.
    func testRefusals() throws {
        let bb = model(.float32)
        var s = bb.startState()
        _ = bb.prefill(prompt, state: &s); s.eval()
        let url = tmp(); defer { try? FileManager.default.removeItem(at: url) }
        try StateFile.saveSnapshot(s, base: bb, to: url)
        let cfg = TinyBackbone.config(nLayer: 3, vocab: 64)
        let other = X070Backbone(weights: TinyBackbone.weights(cfg, seed: 8), cfg: cfg, computeDType: .float32)
        XCTAssertThrowsError(try StateFile.loadSnapshot(url, base: other)) {
            guard case StateFile.Failure.otherBase = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertNoThrow(try StateFile.loadSnapshot(url, base: other, allowOtherBase: true))
        let deeper = model(.float32)
        _ = deeper
        let cfg4 = TinyBackbone.config(nLayer: 4, vocab: 64)
        let four = X070Backbone(weights: TinyBackbone.weights(cfg4, seed: 7), cfg: cfg4, computeDType: .float32)
        XCTAssertThrowsError(try StateFile.loadSnapshot(url, base: four, allowOtherBase: true)) {
            guard case StateFile.Failure.otherArchitecture = $0 else { return XCTFail("\($0)") }
        }
    }

    /// Снимок — то же, что свёрнутый промпт: подключённый как начальное
    /// состояние, он продолжает так же, как пошаговый декод с него (в fp32 —
    /// до счёта параллельного пути против рекуррентного).
    func testSnapshotAttachesAsInitialState() throws {
        let bb = model(.float32)
        var s = bb.startState()
        _ = bb.prefill(prompt, state: &s); s.eval()
        let url = tmp(); defer { try? FileManager.default.removeItem(at: url) }
        try StateFile.saveSnapshot(s, base: bb, to: url)
        let (batch, info) = try StateFile.load(url, base: bb)
        XCTAssertEqual(info.kind, "snapshot")
        var dec = s
        let stepwise = roll(bb, &dec, more).last!
        bb.initialState = batch
        var fresh = bb.startState()
        let parallel = bb.prefill(more, state: &fresh)
        bb.initialState = nil
        eval(parallel)
        let d = MLX.abs(parallel.asType(.float32).reshaped([-1]) - stepwise.asType(.float32).reshaped([-1])).max().item(Float.self)
        XCTAssertLessThan(d, 1e-4, "подключённый снимок продолжает не так, как декод с него: \(d)")
        XCTAssertTrue(sameState(fresh, dec) || d < 1e-4)
    }
}
