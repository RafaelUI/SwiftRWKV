import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVQuant
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Слитый счёт замороженного параллельного прохода (`FrozenPassKernels`):
//  хвост tmix ядром, нормировки слоя штатным ядром MLX, cmix в типе базы.
//
//    • cmix — ПОБИТНО тот же выход;
//    • хвост и нормировки — в допуске порядка суммирования, и не дальше от
//      точного счёта, чем покомандный путь;
//    • действует только в норме и только у слоёв вне `trainLayers`: прежний
//      путь базы, плотная модель и обучаемые слои считают как раньше.
// ───────────────────────────────────────────────────────────────────────

final class FrozenPassKernelTests: XCTestCase {

    static let ids = (0 ..< 48).map { ($0 * 7919 + 13) % 60000 + 100 }
    static var arr: MLXArray { MLXArray(ids.map { Int32($0) }, [1, ids.count]) }

    static func bad(_ a: MLXArray, _ b: MLXArray) -> Int {
        guard a.shape == b.shape else { return -1 }
        return RwkvqFileTests.mismatches(a.asType(.float32), b.asType(.float32))
    }

    /// Логиты параллельного прохода, скрытое состояние с продолжением и
    /// префилл некратной длины при заданном наборе слитых частей.
    static func outputs(_ bb: X070Backbone, _ kernels: FrozenPassKernels) -> [MLXArray] {
        let saved = bb.frozenPassKernels
        bb.frozenPassKernels = kernels
        defer { bb.frozenPassKernels = saved }
        var out = [bb(arr)]
        let (h, st) = bb.bodyWithState(arr[0..., 0 ..< 29])
        let (h2, st2) = bb.bodyWithState(arr[0..., 29 ..< 48], state: st)
        out += [h, h2, st2.wkv, st2.tmixShift, st2.cmixShift]
        var rs = RWKVState(cfg: bb.cfg)
        out.append(bb.prefill(Array(ids[0 ..< 37]), state: &rs))
        out.append(rs.wkv[bb.cfg.nLayer - 1])
        eval(out)
        return out
    }

    func testAllKernelsAreTheDefault() throws {
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("reduction"))
        XCTAssertEqual(bb.frozenPassKernels, .all)
        XCTAssertEqual(FrozenPassKernels.all, [.tail, .layerNorm, .cmix])
    }

    func testCmixInBaseTypeIsBitwise() throws {
        for preset in RwkvqFileTests.presets {
            let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath(preset))
            var before = bb.cmixFrozenCalls
            let plain = Self.outputs(bb, [])
            XCTAssertEqual(bb.cmixFrozenCalls, before, "\(preset): cmix пошёл слитно без флага")
            before = bb.cmixFrozenCalls
            let fused = Self.outputs(bb, [.cmix])
            // bb(arr), два куска bodyWithState и префилл — по разу на слой
            XCTAssertEqual(bb.cmixFrozenCalls - before, 4 * bb.cfg.nLayer, "\(preset): cmix не пошёл слитно")
            for (i, (a, b)) in zip(plain, fused).enumerated() {
                XCTAssertEqual(Self.bad(a, b), 0, "\(preset): выход \(i)")
            }
        }
    }

    /// Хвост и нормировки: числа другие (иначе флаг не действует), но в
    /// допуске, и к точному счёту не дальше покомандного пути.
    func testTailAndLayerNormStayInTolerance() throws {
        for preset in RwkvqFileTests.presets {
            let f = try RwkvqModelTests.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let plain = Self.outputs(bb, [])
            func klRef(_ k: FrozenPassKernels) -> Float {
                bb.frozenPassKernels = k
                defer { bb.frozenPassKernels = .all }
                return RwkvqModelTests.kl(f.ref["logits_ref"]!, bb(f.ref["input_ids"]!)[0])
            }
            let base = klRef([])
            for (name, k) in [("хвост", FrozenPassKernels.tail), ("нормировки", .layerNorm), ("все", .all)] {
                let got = Self.outputs(bb, k)
                XCTAssertGreaterThan(Self.bad(plain[0], got[0]), 0, "\(preset), \(name): флаг ничего не изменил")
                let kl = RwkvqModelTests.kl(plain[0][0], got[0][0])
                let relHidden = RwkvqModelTests.relErr(got[1], plain[1])
                let relState = RwkvqModelTests.relErr(got[3], plain[3])
                let toRef = klRef(k)
                print(String(format: "СЛИТЫЕ %@ / %@: KL к покомандному %.2e, скрытое %.2e, состояние %.2e; KL к точному %.3e (покомандно %.3e)",
                             preset, name, kl, relHidden, relState, toRef, base))
                // Допуск к покомандному пути — уровень шума fp16-матмулов:
                // любая перестановка суммирования перебрасывает округления
                // fp16 ниже по стеку (замерено 2.6–3.6e-6 при KL к точному
                // счёту 4.4–6.1e-6). Мера — следующая проверка: к точному
                // счёту слитый путь не дальше покомандного (он ближе).
                XCTAssertLessThan(abs(kl), 1e-5, "\(preset), \(name)")
                XCTAssertLessThan(relHidden, 2e-3, "\(preset), \(name)")
                XCTAssertLessThan(relState, 2e-3, "\(preset), \(name)")
                XCTAssertLessThan(toRef, base * 1.1 + 1e-7, "\(preset), \(name): дальше от точного счёта")
                XCTAssertEqual((argMax(plain[0], axis: -1) .!= argMax(got[0], axis: -1)).asType(.int32).sum().item(Int.self),
                               0, "\(preset), \(name): сменился top-1")
            }
        }
    }

    /// Вне нормы и у обучаемых слоёв слитого счёта нет: выход побитно не
    /// зависит от флага.
    func testOnlyFrozenLayersInTheNorm() throws {
        let path = try RwkvqFileTests.sourcePath("reduction")
        // прежний путь базы
        let legacy = try X070Backbone.load(fromRwkvq: path, paramDType: .bfloat16, baseDType: nil)
        for (i, (a, b)) in zip(Self.outputs(legacy, []), Self.outputs(legacy, .all)).enumerated() {
            XCTAssertEqual(Self.bad(a, b), 0, "прежний путь базы: выход \(i)")
        }
        XCTAssertEqual(legacy.cmixFrozenCalls, 0)
        // плотная модель
        let dense = X070Backbone(
            weights: try loadArrays(url: RwkvqFileTests.testdata.appendingPathComponent("world_0.1b_x070.safetensors")),
            cfg: X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536))
        for (i, (a, b)) in zip(Self.outputs(dense, []), Self.outputs(dense, .all)).enumerated() {
            XCTAssertEqual(Self.bad(a, b), 0, "плотная модель: выход \(i)")
        }
        // все слои обучаемые: дифференцируемое ядро, покомандный блок
        let bb = try X070Backbone.load(fromRwkvq: path)
        bb.trainLayers = Set(0 ..< bb.cfg.nLayer)
        func logits(_ k: FrozenPassKernels) -> MLXArray {
            bb.frozenPassKernels = k
            let y = bb(Self.arr); eval(y); return y
        }
        let before = bb.cmixFrozenCalls
        XCTAssertEqual(Self.bad(logits([]), logits(.all)), 0, "обучаемые слои посчитаны слитно")
        XCTAssertEqual(bb.cmixFrozenCalls, before)
        // половина слоёв заморожена: слитно идут только они
        bb.trainLayers = Set(6 ..< bb.cfg.nLayer)
        _ = logits(.all)
        XCTAssertEqual(bb.cmixFrozenCalls - before, 6)
    }

    /// Адаптер на cmix (без обучения слоя): слитый cmix обязан уступить
    /// общему пути, иначе адаптер молча не участвует в счёте.
    func testCmixWithAdapterTakesTheGeneralPath() throws {
        let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("reduction"))
        let plain = bb(Self.arr); eval(plain)
        LoRA.add(to: bb, spec: LoRASpec(rank: 4, alpha: 4, tmixTargets: [], cmixTargets: ["key"], layers: 2 ..< 3))
        let t = "blocks.2.cmix.key"
        let (a, b) = bb.loraAdapter(t)
        bb.setLoraAdapter(t, a, MLXArray.ones(b.shape, dtype: b.dtype) * 0.05)     // B ненулевой: адаптер слышен
        bb.trainLayers = []
        let before = bb.cmixFrozenCalls
        let withAdapter = bb(Self.arr); eval(withAdapter)
        XCTAssertEqual(bb.cmixFrozenCalls - before, bb.cfg.nLayer - 1, "слой с адаптером посчитан слитно")
        XCTAssertGreaterThan(RwkvqModelTests.relErr(withAdapter, plain), 1e-4, "адаптер cmix не участвует в счёте")
    }

    /// Хвост ядром на N строк равен хвосту по одной строке.
    func testTailKernelRowsEqualSingleRows() throws {
        let H = 3, S = 64, N = 5, D = H * S
        MLXRandom.seed(11)
        let x = (0 ..< 5).map { _ in MLXRandom.normal([N, D]) }
        let p = (0 ..< 3).map { _ in MLXRandom.normal([D]) }
        let all = wkvTail(x[0], r: x[1], k: x[2], v: x[3], rK: p[0], lnWeight: p[1], lnBias: p[2],
                          gate: x[4], nHead: H, headSize: S, rows: N)
        XCTAssertEqual(all.shape, [N, D])
        for n in 0 ..< N {
            let one = wkvTail(x[0][n], r: x[1][n], k: x[2][n], v: x[3][n], rK: p[0], lnWeight: p[1],
                              lnBias: p[2], gate: x[4][n], nHead: H, headSize: S)
            XCTAssertEqual(Self.bad(all[n ..< (n + 1)], one), 0, "строка \(n)")
        }
        XCTAssertGreaterThan(Self.bad(all[0], all[1]), 0)
    }
}
