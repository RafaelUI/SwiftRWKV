import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  GEMV-ядро раскладки sym: путь декода без плотной матрицы.
//
//  Эталон — само ядро Metal-бэкенда rwkv-quant (SymQuantLinear._gemv):
//      cd ~/Develop/rwkv-metal && .venv/bin/python \
//          ~/Develop/SwiftRWKV/Scripts/dump_rwkvq2_gemv.py
//  -> .testdata/rwkvq2_reduction_gemv.safetensors
//  Исходник и порядок суммирования перенесены дословно, поэтому требуется
//  РАВЕНСТВО, а не допуск. Вторая опора, не зависящая от соседнего
//  репозитория: точный матмул с весами codec в fp32 — на всех sym-тензорах.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqSymGEMVTests: XCTestCase {

    static func gemvReference() throws -> [String: MLXArray] {
        let p = RwkvqFileTests.testdata.appendingPathComponent("rwkvq2_reduction_gemv.safetensors")
        try XCTSkipIf(!FileManager.default.fileExists(atPath: p.path),
                      "нет эталона GEMV \(p.lastPathComponent) — тест пропущен (см. шапку файла)")
        return try loadArrays(url: p)
    }

    func testKernelIsBitExactAgainstRwkvQuant() throws {
        let s = try RwkvqFileTests.base("reduction")
        let ref = try Self.gemvReference()
        var checked = 0, bits: Set<Int> = []
        for name in ref.keys.sorted() where name.hasPrefix("x/") {
            let tag = String(name.dropFirst(2))
            let key = tag.replacingOccurrences(of: "/", with: ".")
            let info = try XCTUnwrap(s.tensors[key], key)
            let x = ref[name]!
            for n in [1, 3, 4] {
                let want = try XCTUnwrap(ref["y\(n)/\(tag)"], "\(key): нет y\(n)")
                let got = try XCTUnwrap(s.symGEMV(key, x[0 ..< n]), "\(key): ядро не взяло форму")
                XCTAssertEqual(got.shape, want.shape, key)
                XCTAssertEqual(got.dtype, .float32)
                XCTAssertEqual(RwkvqFileTests.mismatches(got, want), 0,
                               "\(key), строк \(n): выход ядра разошёлся с rwkv-quant")
                checked += 1
            }
            bits.insert(info.bits)
        }
        XCTAssertGreaterThanOrEqual(checked, 15)
        XCTAssertEqual(bits, [6, 8], "эталон обязан покрыть обе битности")
    }

    /// Против точного матмула с весами codec (fp32) — на ВСЕХ sym-тензорах.
    /// Допуск — только на порядок суммирования в fp32.
    func testKernelMatchesExactMatmulOnAllTensors() throws {
        let s = try RwkvqFileTests.base("reduction")
        MLXRandom.seed(7)
        var n = 0
        var worst: Float = 0
        for key in s.keys {
            guard let info = s.tensors[key], info.kind == .sym else { continue }
            let x = MLXRandom.normal([2, info.inFeatures])
            let got = try XCTUnwrap(s.symGEMV(key, x), key)
            let w = try s.dequantize(key, dtype: .float32)
            let want = matmul(x, w.transposed())
            let e = RwkvqModelTests.relErr(got, want)
            worst = max(worst, e)
            XCTAssertLessThan(e, 2e-5, "\(key): GEMV разошёлся с точным матмулом на \(e)")
            n += 1
        }
        XCTAssertEqual(n, 73)
        print(String(format: "GEMV против точного матмула: 73 тензора, худшая отн. ошибка %.2e", worst))
    }

    /// Чего ядро не берёт — возвращает nil, а не считает мусор.
    func testKernelDeclinesWhatItCannotTake() throws {
        let s = try RwkvqFileTests.base("reduction")
        let key = "blocks.1.att.key.weight"
        XCTAssertNil(s.symGEMV(key, MLXArray.zeros([5, 768])), "больше 4 строк — не один запуск")
        XCTAssertNil(s.symGEMV(key, MLXArray.zeros([1, 512])), "чужая ширина входа")
        XCTAssertNil(s.symGEMV("blocks.0.att.w1", MLXArray.zeros([1, 768])), "asym — не sym")
        XCTAssertNil(s.symGEMV("blocks.0.att.output.weight", MLXArray.zeros([1, 768])), "плотная цель")
        XCTAssertNil(try RwkvqFileTests.base("compression").symGEMV(key, MLXArray.zeros([1, 768])), "sb6")
        XCTAssertFalse(rwkvqSymGEMVApplies(outFeatures: 24, inFeatures: 256, bits: 8, rows: 1))
        XCTAssertFalse(rwkvqSymGEMVApplies(outFeatures: 16, inFeatures: 48, bits: 8, rows: 1))
        XCTAssertTrue(rwkvqSymGEMVApplies(outFeatures: 16, inFeatures: 256, bits: 6, rows: 4))
    }

    // ── В модели ─────────────────────────────────────────────────────

    /// Ядро работает на рекуррентном шаге и ТОЛЬКО на нём: параллельный
    /// проход (через него идёт градиент) считает «деквант + матмул».
    func testStepGoesThroughTheKernelAndParallelPassDoesNot() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let ids = f.ref["input_ids"]!
        _ = bb(ids).sum().item(Float.self)
        let (_, st) = bb.bodyWithState(ids)
        XCTAssertEqual(bb.rwkvqSymGEMVCalls, 0, "параллельный проход ушёл в ядро декода")
        // и на коротком входе, который ядро ВЗЯЛО БЫ по форме (до 4 строк):
        // путь с состоянием по одному токену и параллельный проход по трём
        _ = bb.bodyWithState(ids[0..., 0 ..< 1], state: st).0.sum().item(Float.self)
        _ = bb(ids[0..., 0 ..< 3]).sum().item(Float.self)
        XCTAssertEqual(bb.rwkvqSymGEMVCalls, 0, "короткий параллельный проход ушёл в ядро декода")

        var rs = RWKVState(st, row: 0)
        _ = bb.step(ids[0, 0].item(Int.self), state: &rs).sum().item(Float.self)
        XCTAssertEqual(bb.rwkvqSymGEMVCalls, 72, "за шаг — все 72 квантованные проекции")

        bb.rwkvqSymDecodeKernel = false
        _ = bb.step(ids[0, 1].item(Int.self), state: &rs).sum().item(Float.self)
        XCTAssertEqual(bb.rwkvqSymGEMVCalls, 72, "выключенный флаг ядро не зовёт")

        // COMPRESSION: sym там нет, счётчик обязан молчать
        let c = try X070Backbone.load(fromRwkvq: try RwkvqModelTests.fixture("compression").path)
        var cs = RWKVState(cfg: c.cfg)
        _ = c.step(5, state: &cs).sum().item(Float.self)
        XCTAssertEqual(c.rwkvqSymGEMVCalls, 0)
    }

    /// Декод ядром против декода «деквант + матмул» и против эталонов:
    /// ядро точнее (вес не округлён до fp16, сумма в fp32), поэтому к
    /// точному счёту оно обязано стоять не дальше прежнего пути.
    func testKernelDecodeAgainstReferences() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let (_, st) = bb.bodyWithState(f.ref["input_ids"]!)
        let dec = f.ref["decode_ids"]!
        func run(_ kernel: Bool) -> MLXArray {
            bb.rwkvqSymDecodeKernel = kernel
            var rs = RWKVState(st, row: 0)
            var rows: [MLXArray] = []
            for i in 0 ..< dec.shape[1] {
                rows.append(bb.step(dec[0, i].item(Int.self), state: &rs).asType(.float32))
            }
            return stacked(rows, axis: 0).reshaped([dec.shape[1], -1])
        }
        let viaKernel = run(true), viaDequant = run(false)
        let exact = f.ref["decode_logits_ref"]!
        let klK = RwkvqModelTests.kl(exact, viaKernel), klD = RwkvqModelTests.kl(exact, viaDequant)
        let klBetween = RwkvqModelTests.kl(viaDequant, viaKernel)
        let same = (argMax(viaKernel, axis: -1) .== argMax(viaDequant, axis: -1))
            .asType(.int32).sum().item(Int.self)
        print(String(format: "ДЕКОД sym, KL к точному счёту: ядро %.3e, деквант+матмул %.3e; "
                     + "KL между путями %.3e; top-1 совпало %d из %d; KL(ref‖metal) %.3e",
                     klK, klD, klBetween, same, dec.shape[1], f.klRefMetalDecode))
        XCTAssertGreaterThan(RwkvqFileTests.mismatches(viaKernel, viaDequant), 0,
                             "пути неразличимы — ядро не включилось")
        XCTAssertLessThan(klK, 1.25 * klD, "ядро дальше от точного счёта, чем прежний путь")
        XCTAssertLessThan(klBetween, f.klRefMetalDecode)
        XCTAssertEqual(same, dec.shape[1])
    }
}
