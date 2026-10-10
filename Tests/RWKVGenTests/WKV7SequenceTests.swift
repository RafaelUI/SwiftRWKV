import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Замороженный WKV одним запуском на всю длину (`wkv7SequenceForward`)
//  против нарезки по CHUNK (`wkv7ForwardChunked`): выход и состояние обязаны
//  совпасть ПОБИТНО на любой длине — кратной CHUNK, некратной, единичной,
//  с начальным состоянием и без, при батче.
// ───────────────────────────────────────────────────────────────────────

final class WKV7SequenceTests: XCTestCase {

    static func inputs(B: Int, T: Int, H: Int, seed: UInt64) -> [MLXArray] {
        MLXRandom.seed(seed)
        var x = (0 ..< 6).map { _ in MLXRandom.normal([B, T, H, WKV7_HEAD_SIZE]) * 0.3 }
        x[1] = exp(-abs(x[1]))          // затухание в (0, 1]
        eval(x)
        return x
    }

    static func bad(_ a: MLXArray, _ b: MLXArray) -> Int {
        guard a.shape == b.shape else { return -1 }
        return (a .!= b).asType(.int32).sum().item(Int.self)
    }

    func testWholeSequenceEqualsChunkedBitwise() throws {
        for (B, T, H) in [(1, 1, 2), (1, 5, 3), (1, 16, 2), (1, 17, 4), (2, 53, 3), (1, 512, 12), (3, 32, 1)] {
            let x = Self.inputs(B: B, T: T, H: H, seed: UInt64(1000 + T))
            let h0 = MLXRandom.normal([B, H, WKV7_HEAD_SIZE, WKV7_HEAD_SIZE]) * 0.1
            for hIn in [nil, h0] {
                let (o1, s1) = wkv7SequenceForward(x[0], x[1], x[2], x[3], x[4], x[5], hIn)
                let (o2, s2) = wkv7ForwardChunked(x[0], x[1], x[2], x[3], x[4], x[5], hIn)
                eval(o1, s1, o2, s2)
                let what = "B=\(B) T=\(T) H=\(H) состояние \(hIn == nil ? "нулевое" : "заданное")"
                XCTAssertEqual(o1.shape, [B, T, H, WKV7_HEAD_SIZE], what)
                XCTAssertEqual(Self.bad(o1, o2), 0, "\(what): выход")
                XCTAssertEqual(Self.bad(s1, s2), 0, "\(what): состояние")
                XCTAssertGreaterThan(abs(o1).sum().item(Float.self), 0, "\(what): выход нулевой")
                XCTAssertFalse(o1.sum().item(Float.self).isNaN, what)
                // последняя позиция и последняя строка батча посчитаны
                XCTAssertGreaterThan(abs(o1[B - 1, T - 1]).sum().item(Float.self), 0, "\(what): хвост")
            }
        }
    }

    /// Продолжение с состояния: два куска подряд == один проход.
    func testStateCarriesAcrossCalls() throws {
        let x = Self.inputs(B: 1, T: 41, H: 3, seed: 7)
        let (whole, hWhole) = wkv7SequenceForward(x[0], x[1], x[2], x[3], x[4], x[5])
        func cut(_ r: Range<Int>) -> [MLXArray] { x.map { $0[0..., r] } }
        let p = cut(0 ..< 19), q = cut(19 ..< 41)
        let (o1, h1) = wkv7SequenceForward(p[0], p[1], p[2], p[3], p[4], p[5])
        let (o2, h2) = wkv7SequenceForward(q[0], q[1], q[2], q[3], q[4], q[5], h1)
        XCTAssertEqual(Self.bad(concatenated([o1, o2], axis: 1), whole), 0)
        XCTAssertEqual(Self.bad(h2, hWhole), 0)
        // и состояние действительно участвует
        let (o3, _) = wkv7SequenceForward(q[0], q[1], q[2], q[3], q[4], q[5])
        XCTAssertGreaterThan(Self.bad(o2, o3), 0)
    }

    /// Выключатель и счётчик: общий вход `wkv7ForwardWithState` идёт одним
    /// запуском по умолчанию и нарезкой при выключателе; выход тот же.
    func testSwitchSelectsThePath() throws {
        let x = Self.inputs(B: 1, T: 40, H: 2, seed: 3)
        XCTAssertTrue(wkv7WholeSequenceForward, "одним запуском — умолчание")
        var before = wkv7SequenceLaunches
        let (a, ha) = wkv7ForwardWithState(x[0], x[1], x[2], x[3], x[4], x[5])
        XCTAssertEqual(wkv7SequenceLaunches - before, 1)
        wkv7WholeSequenceForward = false
        defer { wkv7WholeSequenceForward = true }
        before = wkv7SequenceLaunches
        let (b, hb) = wkv7ForwardWithState(x[0], x[1], x[2], x[3], x[4], x[5])
        XCTAssertEqual(wkv7SequenceLaunches, before, "выключатель не действует")
        XCTAssertEqual(Self.bad(a, b), 0); XCTAssertEqual(Self.bad(ha, hb), 0)
    }

    /// Префилл модели идёт одним запуском на слой, а не T / CHUNK.
    func testPrefillLaunchesOnceALayer() throws {
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("compression"))
        var st = RWKVState(cfg: bb.cfg)
        let before = wkv7SequenceLaunches
        let lg = bb.prefill((0 ..< 70).map { $0 * 31 + 200 }, state: &st)
        eval(lg); st.eval()
        XCTAssertEqual(wkv7SequenceLaunches - before, bb.cfg.nLayer)
    }
}
