import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Ядра декода (WKV7DecodeKernels.swift): шаг WKV на один токен и хвост
//  tmix одним запуском — и то, как они и ядро GEMV sb6 включаются в модели.
// ───────────────────────────────────────────────────────────────────────

final class DecodeKernelTests: XCTestCase {

    static func wkvInputs(H: Int, seed: UInt64) -> [MLXArray] {
        MLXRandom.seed(seed)
        func rnd(_ s: Float) -> MLXArray { MLXRandom.normal([H, 64]) * s }
        let r = rnd(0.5), k = rnd(0.5), v = rnd(0.5), a = rnd(0.1), b = rnd(0.1)
        let w = exp(-0.606531 * sigmoid(rnd(1.0)))
        let h = MLXRandom.normal([H, 64, 64]) * 0.3
        eval(r, w, k, v, a, b, h)
        return [r, w, k, v, a, b, h]
    }

    /// Ядро шага против прямого ядра параллельного пути на T = 1: та же
    /// арифметика в том же порядке — обязано быть побитно.
    func testStepKernelEqualsForwardKernelOnOneToken() {
        for (H, seed) in [(12, UInt64(1)), (3, 2), (32, 3)] {
            let i = Self.wkvInputs(H: H, seed: seed)
            let (out, hNew) = wkv7StepForward(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
            func b(_ x: MLXArray) -> MLXArray { x.reshaped([1, 1, H, 64]) }
            let (refOut, refH) = wkv7ForwardWithState(b(i[0]), b(i[1]), b(i[2]), b(i[3]), b(i[4]), b(i[5]),
                                                      i[6].reshaped([1, H, 64, 64]))
            XCTAssertEqual(out.shape, [H, 64]); XCTAssertEqual(hNew.shape, [H, 64, 64])
            XCTAssertEqual(RwkvqFileTests.mismatches(out, refOut.reshaped([H, 64])), 0, "H=\(H): выход")
            XCTAssertEqual(RwkvqFileTests.mismatches(hNew, refH.reshaped([H, 64, 64])), 0, "H=\(H): состояние")
            XCTAssertGreaterThan(MLX.abs(out).max().item(Float.self), 0)
        }
    }

    /// Тип хранения входов не меняет счёт и не меняет ядро: bf16 / fp16 /
    /// fp32 в одном графе дают то же, что по отдельности.
    func testStepKernelTakesAnyInputType() {
        let i = Self.wkvInputs(H: 12, seed: 5)
        func run(_ t: DType) -> MLXArray {
            let c = i.map { $0.asType(t) }
            return wkv7StepForward(c[0], c[1], c[2], c[3], c[4], c[5], i[6]).0
        }
        let alone = [DType.bfloat16, .float32, .float16].map { t -> MLXArray in let o = run(t); eval(o); return o }
        let lazy = [DType.bfloat16, .float32, .float16].map { run($0) }
        eval(lazy)
        for j in 0 ..< 3 { XCTAssertEqual(RwkvqFileTests.mismatches(lazy[j], alone[j]), 0, "тип \(j)") }
    }

    /// Хвост одним ядром против покомандных формул: group_norm в два
    /// прохода, bonus, gate. Допуск — на порядок суммирования.
    func testTailKernelMatchesTheFormulas() {
        for (H, seed) in [(12, UInt64(7)), (32, 8)] {
            let S = 64, D = H * S
            MLXRandom.seed(seed)
            let wkv = MLXRandom.normal([H, S]) * 3, r = MLXRandom.normal([H, S]), k = MLXRandom.normal([H, S])
            let v = MLXRandom.normal([H, S]), rk = MLXRandom.normal([H, S]) * 0.1
            let lnw = MLXRandom.normal([D]) * 0.5 + 1, lnb = MLXRandom.normal([D]) * 0.1
            let g = MLXRandom.normal([1, D])
            let got = wkvTail(wkv, r: r, k: k, v: v, rK: rk, lnWeight: lnw, lnBias: lnb, gate: g,
                              nHead: H, headSize: S)
            let mean = wkv.mean(axis: -1, keepDims: true)
            let varc = (wkv - mean).square().mean(axis: -1, keepDims: true)
            let normed = ((wkv - mean) / sqrt(varc + 64e-5)).reshaped([1, D]) * lnw + lnb
            let bonus = ((r * k * rk).sum(axis: -1, keepDims: true) * v).reshaped([1, D])
            let want = (normed + bonus) * g
            XCTAssertEqual(got.shape, [1, D])
            XCTAssertLessThan(RwkvqModelTests.relErr(got, want), 1e-5, "H=\(H)")
        }
        XCTAssertFalse(wkvTailApplies(headSize: 48))
    }

    // ── В модели ─────────────────────────────────────────────────────

    static func steps(_ bb: X070Backbone, _ ids: [Int]) -> MLXArray {
        var st = RWKVState(cfg: bb.cfg)
        var rows: [MLXArray] = []
        for id in ids { rows.append(bb.step(id, state: &st).asType(.float32)); st.eval() }
        let l = stacked(rows, axis: 0); eval(l); return l
    }

    /// Ядра шага на обоих пресетах: зовутся на каждом слое, числа те же, что
    /// у покомандного шага; у плотной модели и при приведении выхода WKV —
    /// не зовутся.
    func testStepKernelsInTheModel() throws {
        for preset in RwkvqFileTests.presets {
            let f = try RwkvqModelTests.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            bb.useCompiledStep = false                       // покомандно: счётчики на каждый шаг
            let ids = f.ref["input_ids"]![0, 0 ..< 8].asArray(Int32.self).map(Int.init)
            bb.useStepKernels = false
            let plain = Self.steps(bb, ids)
            XCTAssertEqual(bb.stepKernelCalls, 0)
            bb.useStepKernels = true
            let fused = Self.steps(bb, ids)
            XCTAssertEqual(bb.stepKernelCalls, ids.count * 12, "\(preset): ядра шага на каждом слое")
            // Ветки gate / v / a / w при fp32-входе считаются с приведением
            // ВХОДА: 8 матмулов на слой, кроме слоя 0 (вход ещё в типе матриц,
            // приводить нечего; ветки v у него нет). Числа от этого не выходят
            // за допуск, поэтому стережёт ускорение только этот счётчик.
            XCTAssertEqual(bb.narrowBranchCalls, ids.count * 11 * 8, "\(preset): ветки считаются с повышением матрицы")
            let kl = RwkvqModelTests.kl(plain, fused)
            let same = (argMax(plain, axis: -1) .== argMax(fused, axis: -1)).asType(.int32).sum().item(Int.self)
            print(String(format: "ЯДРА ШАГА %@: KL(покомандный‖ядра) %.3e, top-1 %d из %d", preset, kl, same, ids.count))
            XCTAssertLessThan(kl, 2e-6, preset)
            XCTAssertEqual(same, ids.count)
            XCTAssertGreaterThan(RwkvqFileTests.mismatches(plain, fused), 0, "пути неразличимы — ядра не включились")
            bb.castWKVOutputToComputeDType = true
            _ = Self.steps(bb, Array(ids[0 ..< 2]))
            XCTAssertEqual(bb.stepKernelCalls, ids.count * 12, "при приведении выхода WKV ядра шага не зовутся")
        }
        let (dense, cfg) = TinyBackbone.make(nLayer: 2)
        _ = Self.steps(dense, [1 % cfg.vocab, 2 % cfg.vocab])
        XCTAssertEqual(dense.stepKernelCalls, 0, "плотная модель пошла ядрами шага")
    }

    /// COMPRESSION без родного контейнера: K3 остаётся в памяти, шаг идёт
    /// ядром GEMV sb6, параллельный проход — деквантом. С родным контейнером
    /// ядро не зовётся (K3 отпущен).
    func testSb6StepGoesThroughGEMVWhenBaseKeepsK3() throws {
        let f = try RwkvqModelTests.fixture("compression")
        let ids = f.ref["input_ids"]!
        let k3 = try X070Backbone.load(fromRwkvq: f.path, useNativeKernel: false)
        k3.useCompiledStep = false
        XCTAssertTrue(k3.isRawSidecarRetained)
        XCTAssertTrue(k3.rwkvqBackedKeys.allSatisfy { !k3.hasNativeRepack($0) })
        XCTAssertEqual(k3.rwkvqSidecar?.tensors.count, 72)
        _ = k3(ids).sum().item(Float.self)
        let (_, st) = k3.bodyWithState(ids[0..., 0 ..< 1])
        XCTAssertEqual(k3.rwkvqSb6GEMVCalls, 0, "параллельный проход ушёл в ядро декода")
        var rs = RWKVState(st, row: 0)
        _ = k3.step(ids[0, 1].item(Int.self), state: &rs).sum().item(Float.self)
        XCTAssertEqual(k3.rwkvqSb6GEMVCalls, 72, "за шаг — все 72 квантованные проекции")
        k3.rwkvqSb6DecodeKernel = false
        _ = k3.step(ids[0, 2].item(Int.self), state: &rs).sum().item(Float.self)
        XCTAssertEqual(k3.rwkvqSb6GEMVCalls, 72, "выключенный флаг ядро не зовёт")

        let native = try X070Backbone.load(fromRwkvq: f.path, useNativeKernel: true)
        native.useCompiledStep = false
        var ns = RWKVState(cfg: native.cfg)
        _ = native.step(5, state: &ns).sum().item(Float.self)
        XCTAssertEqual(native.rwkvqSb6GEMVCalls, 0)
        XCTAssertFalse(native.isRawSidecarRetained)
    }

    /// Декод COMPRESSION тремя путями против эталонов: ядро GEMV, деквант +
    /// матмул, родной контейнер.
    func testSb6DecodePathsAgainstReferences() throws {
        let f = try RwkvqModelTests.fixture("compression")
        let dec = f.ref["decode_ids"]!
        let exact = f.ref["decode_logits_ref"]!
        func run(_ bb: X070Backbone, kernel: Bool) -> MLXArray {
            bb.useCompiledStep = false
            bb.rwkvqSb6DecodeKernel = kernel
            let (_, st) = bb.bodyWithState(f.ref["input_ids"]!)
            var rs = RWKVState(st, row: 0)
            var rows: [MLXArray] = []
            for i in 0 ..< dec.shape[1] { rows.append(bb.step(dec[0, i].item(Int.self), state: &rs).asType(.float32)) }
            let l = stacked(rows, axis: 0).reshaped([dec.shape[1], -1]); eval(l); return l
        }
        let k3 = try X070Backbone.load(fromRwkvq: f.path, useNativeKernel: false)
        let viaKernel = run(k3, kernel: true), viaDequant = run(k3, kernel: false)
        let viaNative = run(try X070Backbone.load(fromRwkvq: f.path, useNativeKernel: true), kernel: true)
        let klK = RwkvqModelTests.kl(exact, viaKernel), klD = RwkvqModelTests.kl(exact, viaDequant)
        let klN = RwkvqModelTests.kl(exact, viaNative)
        let same = (argMax(viaKernel, axis: -1) .== argMax(viaNative, axis: -1)).asType(.int32).sum().item(Int.self)
        print(String(format: "ДЕКОД sb6, KL к точному счёту: ядро GEMV %.3e, деквант+матмул %.3e, родной контейнер %.3e; "
                     + "KL(ref‖metal) %.3e; top-1 ядро == родной: %d из %d",
                     klK, klD, klN, f.klRefMetalDecode, same, dec.shape[1]))
        XCTAssertGreaterThan(RwkvqFileTests.mismatches(viaKernel, viaDequant), 0, "пути неразличимы — ядро не включилось")
        XCTAssertLessThan(klK, 4 * f.klRefMetalDecode)
        XCTAssertLessThan(klK, 1.25 * klD, "ядро дальше от точного счёта, чем деквант + матмул")
        XCTAssertEqual(same, dec.shape[1])
    }
}
