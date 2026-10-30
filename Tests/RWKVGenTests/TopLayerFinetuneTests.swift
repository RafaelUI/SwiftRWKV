import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  TopLayerFinetune (30.10): верхние слои языковым лоссом — сильный контроль
//  к обучению состояния в gen-run --distill. Свойства, а не числа:
//    1. обучаются ТОЛЬКО слои [from..<nLayer] и ln_out; нижние, emb, head —
//       байт-в-байт прежние;
//    2. лосс падает на фиксированной пачке;
//    3. флаг sequentialBackward не действует (ln_out вне блоков — всегда обычный проход);
//    4. цели < 0 (вопрос) в лосс не входят;
//    5. save → load на свежей базе восстанавливает веса точно; чужая база,
//       чужая форма и не тот файл — отказ.
// ───────────────────────────────────────────────────────────────────────

final class TopLayerFinetuneTests: XCTestCase {

    static let T = 16

    func batch(_ cfg: X070Config, seed: UInt64 = 7, maskFirst: Int = 0) -> LoRABatch {
        let ids = TinyBackbone.ids(2, Self.T + 1, vocab: cfg.vocab, seed: seed)
        var y = ids[0..., 1 ... Self.T]
        if maskFirst > 0 {
            let m = MLXArray((0 ..< Self.T).map { $0 < maskFirst ? Int32(1) : Int32(0) }, [1, Self.T])
            y = MLX.where(m .== 1, MLXArray(Int32(-1)), y)
        }
        return (x: ids[0..., 0 ..< Self.T], y: y)
    }

    func config(steps: Int = 6, sequential: Bool = false) -> LoRAConfig {
        var c = LoRAConfig(lr: 1e-2, gradClip: 1.0, maxSteps: steps, cacheLimitGB: 0, logEvery: 1)
        c.sequentialBackward = sequential
        return c
    }

    func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        eval(a, b)
        return MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // 1
    func testOnlyTopLayersAndLnOutChange() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3, seed: 42)
        let before = bb.w.mapValues { $0 }
        let b = batch(cfg)
        let res = TopLayerFinetune.run(bb, from: 2, nextBatch: { b }, config: config())
        XCTAssertEqual(Set(res.keys), Set(bb.w.keys.filter { $0.hasPrefix("blocks.2.") } + ["ln_out.weight", "ln_out.bias"]))
        var moved = 0
        for (k, v) in before {
            let d = maxDiff(v, bb.w[k]!)
            if res.keys.contains(k) { if d > 0 { moved += 1 } }
            else { XCTAssertEqual(d, 0, "\(k) вне верхних слоёв изменился") }
        }
        XCTAssertGreaterThan(moved, res.keys.count / 2, "обучаемые веса почти не сдвинулись")
        XCTAssertNil(bb.wOverride, "подмена весов осталась после обучения")
        XCTAssertEqual(bb.trainLayers, [], "trainLayers не возвращён")
    }

    // 2
    func testLossDecreases() {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2, seed: 42)
        let b = batch(cfg)
        var losses: [Float] = []
        TopLayerFinetune.run(bb, from: 1, nextBatch: { b }, config: config(steps: 15)) { _, l, _, _ in losses.append(l) }
        XCTAssertEqual(losses.count, 15)
        XCTAssertLessThan(losses.last!, losses.first! * 0.8, "лосс не падает: \(losses.first!) → \(losses.last!)")
    }

    // 3
    func testSequentialBackwardMatchesPlain() {
        let (a, cfg) = TinyBackbone.make(nLayer: 3, seed: 42)
        let (s, _) = TinyBackbone.make(nLayer: 3, seed: 42)
        let b = batch(cfg)
        TopLayerFinetune.run(a, from: 1, nextBatch: { b }, config: config(steps: 3, sequential: false))
        TopLayerFinetune.run(s, from: 1, nextBatch: { b }, config: config(steps: 3, sequential: true))
        for (k, v) in a.w {
            XCTAssertLessThan(maxDiff(v, s.w[k]!), 1e-4, "\(k): последовательный проход расходится с обычным")
        }
    }

    // 4
    func testMaskedTargetsDoNotContribute() {
        // цели вопроса (< 0) не входят в лосс: замена ВХОДОВ на замаскированных позициях
        // меняет лосс (контекст), а замена самих замаскированных целей на другие отрицательные — нет
        let (bb, cfg) = TinyBackbone.make(nLayer: 2, seed: 42)
        let b1 = batch(cfg, seed: 7, maskFirst: 8)
        let m = MLXArray((0 ..< Self.T).map { $0 < 8 ? Int32(1) : Int32(0) }, [1, Self.T])
        let y2 = MLX.where(m .== 1, MLXArray(Int32(-7)), b1.y)
        let l1 = LoRAFinetune.languageModelLoss(bb, b1), l2 = LoRAFinetune.languageModelLoss(bb, (x: b1.x, y: y2))
        XCTAssertEqual(maxDiff(l1, l2), 0)
        // обучение по пачке, где замаскировано всё, кроме последней позиции, двигает веса иначе, чем без маски
        let (u, _) = TinyBackbone.make(nLayer: 2, seed: 42), (v, _) = TinyBackbone.make(nLayer: 2, seed: 42)
        let full = batch(cfg), masked = batch(cfg, maskFirst: Self.T - 1)
        TopLayerFinetune.run(u, from: 1, nextBatch: { full }, config: config(steps: 2))
        TopLayerFinetune.run(v, from: 1, nextBatch: { masked }, config: config(steps: 2))
        XCTAssertGreaterThan(maxDiff(u.w["ln_out.weight"]!, v.w["ln_out.weight"]!), 0, "маска не влияет на обучение")
        // и через сам run: замаскированные цели ≠ цели «0» на тех же местах (иначе маска теряется по дороге)
        let zeros = (x: masked.x, y: MLX.where(masked.y .< 0, MLXArray(Int32(0)), masked.y))
        let (z, _) = TinyBackbone.make(nLayer: 2, seed: 42)
        TopLayerFinetune.run(z, from: 1, nextBatch: { zeros }, config: config(steps: 2))
        XCTAssertGreaterThan(maxDiff(z.w["ln_out.weight"]!, v.w["ln_out.weight"]!), 0, "замаскированные цели учатся как цель 0")
    }

    // 5
    func testSaveLoadRoundTripAndRefusals() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2, seed: 42)
        let b = batch(cfg)
        let res = TopLayerFinetune.run(bb, from: 1, nextBatch: { b }, config: config())
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tlf_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("top.safetensors")
        try TopLayerFinetune.save(bb, result: res, to: url, note: "тест")

        let (fresh, _) = TinyBackbone.make(nLayer: 2, seed: 42)
        let keys = try TopLayerFinetune.load(fresh, from: url)
        XCTAssertEqual(keys, res.keys)
        for (k, v) in bb.w { XCTAssertEqual(maxDiff(v, fresh.w[k]!), 0, "\(k) после load не равен обученному") }
        let ids = TinyBackbone.ids(1, Self.T, vocab: cfg.vocab)
        XCTAssertEqual(maxDiff(bb(ids), fresh(ids)), 0, "логиты после load отличаются")

        // другая база (другой сид весов) — отказ; allowOtherBase — грузит
        let (other, _) = TinyBackbone.make(nLayer: 2, seed: 43)
        XCTAssertThrowsError(try TopLayerFinetune.load(other, from: url)) { e in
            guard case TopLayerFinetune.Error.otherBase = e else { return XCTFail("ждали otherBase, получили \(e)") }
        }
        XCTAssertNoThrow(try TopLayerFinetune.load(other, from: url, allowOtherBase: true))

        // другая форма — отказ, и модель не тронута частично
        let (wide, _) = TinyBackbone.make(nLayer: 2, nEmbd: 64, seed: 42)
        let wideBefore = wide.w.mapValues { $0 }
        XCTAssertThrowsError(try TopLayerFinetune.load(wide, from: url, allowOtherBase: true))
        for (k, v) in wideBefore { XCTAssertEqual(maxDiff(v, wide.w[k]!), 0, "\(k) изменён несмотря на отказ") }

        // файл, где испорчена форма ОДНОГО ключа: отказ, и ни один вес модели не подменён
        var (arrs, md) = try loadArraysAndMetadata(url: url)
        arrs["ln_out.bias"] = MLXArray.zeros([3])
        let bad = dir.appendingPathComponent("bad.safetensors")
        try MLX.save(arrays: arrs, metadata: md, url: bad)
        let (clean, _) = TinyBackbone.make(nLayer: 2, seed: 42)
        let cleanBefore = clean.w.mapValues { $0 }
        XCTAssertThrowsError(try TopLayerFinetune.load(clean, from: bad))
        for (k, v) in cleanBefore { XCTAssertEqual(maxDiff(v, clean.w[k]!), 0, "\(k) подменён до отказа") }

        // файл не того формата (адаптеры LoRA) — отказ
        let (lb, _) = TinyBackbone.make(nLayer: 2, seed: 42)
        LoRA.add(to: lb, spec: LoRASpec(rank: 4))
        let lurl = dir.appendingPathComponent("lora.safetensors")
        try LoRA.save(lb, to: lurl)
        XCTAssertThrowsError(try TopLayerFinetune.load(fresh, from: lurl)) { e in
            guard case TopLayerFinetune.Error.notTopLayers = e else { return XCTFail("ждали notTopLayers, получили \(e)") }
        }
    }
}
