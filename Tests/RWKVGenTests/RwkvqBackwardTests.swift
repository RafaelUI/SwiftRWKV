import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Обучение над квантованной базой из .rwkvq: адаптеры навешиваются,
//  градиент идёт сквозь базу, обратный проход при базе fp16 — в bf16.
//
//  Порт гейта rwkv-metal tests/test_base_dtype.py в той части, что касается
//  обратного прохода: прямой проход от типа обратного не зависит, обратный
//  побитно равен формуле dX = dY · W, и на модели порядок ошибок градиентов
//  адаптеров тот, ради которого свой VJP и заведён.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqBackwardTests: XCTestCase {

    /// (x, y) из входа эталона. T = 16 — ОДИН чанк обучаемого ядра WKV, и это
    /// не экономия. MLX кладёт до 40 операций в один командный буфер Metal;
    /// с обратным ядром WKV внутри буфер на настоящей модели при T >= 32
    /// идёт так долго, что GPU сбрасывается сторожевым таймером
    /// (kIOGPUCommandBufferCallbackErrorHang / PageFault), и градиенты нижних
    /// слоёв выходят NaN — на плотной bf16-модели так же, как на квантованной.
    /// Лечится MLX_MAX_OPS_PER_BUFFER=4 (найдено 09.10, см. docs/Train.md).
    /// Свойства обратного прохода через базу от T не зависят.
    static func batch(_ f: RwkvqModelTests.Fixture, T: Int = 16) -> (MLXArray, MLXArray) {
        let ids = f.ref["input_ids"]!
        return (ids[0..., 0 ..< T], ids[0..., 1 ..< (T + 1)])
    }

    static func loss(_ bb: X070Backbone, _ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let l = bb(x).asType(.float32)
        let lp = l - logSumExp(l, axis: -1, keepDims: true)
        return -takeAlong(lp, y.expandedDimensions(axis: -1), axis: -1).mean()
    }

    /// Градиенты по всем адаптерам, одним вектором fp32 (порядок ключей фиксирован).
    static func adapterGrads(_ bb: X070Backbone, _ x: MLXArray, _ y: MLXArray,
                             lossScale: Float = 1) -> MLXArray {
        let keysA = bb.loraA.keys.sorted(), keysB = bb.loraB.keys.sorted()
        let params = keysA.map { bb.loraA[$0]! } + keysB.map { bb.loraB[$0]! }
        func f(_ ps: [MLXArray]) -> [MLXArray] {
            for (i, k) in keysA.enumerated() { bb.loraA[k] = ps[i] }
            for (i, k) in keysB.enumerated() { bb.loraB[k] = ps[keysA.count + i] }
            return [loss(bb, x, y) * lossScale]
        }
        let g = grad(f, argumentNumbers: Array(0 ..< params.count))(params)
        // вернуть исходные массивы: внутри f в модель были подставлены трассируемые
        for (i, k) in keysA.enumerated() { bb.loraA[k] = params[i] }
        for (i, k) in keysB.enumerated() { bb.loraB[k] = params[keysA.count + i] }
        let flat = concatenated(g.map { $0.asType(.float32).reshaped([-1]) }, axis: 0)
        eval(flat)
        return flat
    }

    /// Модель с адаптерами, у которых B не ноль (иначе градиент по A тождественно нулевой).
    static func withAdapters(_ path: String, native: Bool = true) throws -> X070Backbone {
        let bb = try X070Backbone.load(fromRwkvq: path, useNativeKernel: native)
        MLXRandom.seed(11)
        LoRA.add(to: bb, spec: LoRASpec(rank: 8, alpha: 16))
        for k in bb.loraB.keys.sorted() {
            bb.loraB[k] = (MLXRandom.normal(bb.loraB[k]!.shape) * 0.001).asType(bb.loraB[k]!.dtype)
        }
        eval(Array(bb.loraB.values))
        return bb
    }

    // ── Адаптеры на базе из .rwkvq ───────────────────────────────────

    /// На базе из .rwkvq плотных копий проекций нет. Адаптеры обязаны
    /// навеситься на ВСЕ цели — и на квантованные, и на плотную.
    func testAdaptersAttachToEveryTarget() throws {
        for preset in RwkvqFileTests.presets {
            let bb = try X070Backbone.load(fromRwkvq: try RwkvqModelTests.fixture(preset).path)
            let info = LoRA.add(to: bb, spec: LoRASpec(rank: 8, alpha: 16))
            XCTAssertEqual(info.numAdapters, 48, "\(preset): 4 проекции tmix на 12 слоях")
            XCTAssertEqual(bb.loraA["blocks.5.tmix.k_proj"]?.shape, [8, 768])
            XCTAssertEqual(bb.loraB["blocks.0.tmix.o_proj"]?.shape, [768, 8], "плотная цель")
            // модель fp16, но адаптеры в fp16 не заводятся: котангенты слоя 0 переполнились бы
            XCTAssertEqual(bb.loraA["blocks.0.tmix.k_proj"]?.dtype, .bfloat16, preset)
        }
    }

    /// Градиент доходит до адаптеров НИЖНИХ слоёв сквозь квантованную базу
    /// верхних — на обоих пресетах (sym: свой VJP; sb6: родной quantizedMM).
    func testGradientReachesTheLowestAdapters() throws {
        for preset in RwkvqFileTests.presets {
            let f = try RwkvqModelTests.fixture(preset)
            let bb = try Self.withAdapters(f.path)
            let (x, y) = Self.batch(f)
            let keysA = bb.loraA.keys.sorted()
            let params = keysA.map { bb.loraA[$0]! }
            func fn(_ ps: [MLXArray]) -> [MLXArray] {
                for (i, k) in keysA.enumerated() { bb.loraA[k] = ps[i] }
                return [Self.loss(bb, x, y)]
            }
            let g = grad(fn, argumentNumbers: Array(0 ..< params.count))(params)
            for (i, k) in keysA.enumerated() where k.hasPrefix("blocks.0.") || k.hasPrefix("blocks.11.") {
                let n = sqrt(g[i].asType(.float32).square().sum()).item(Float.self)
                XCTAssertTrue(n.isFinite && n > 0, "\(preset): градиент адаптера \(k) = \(n)")
            }
        }
    }

    // ── Свой VJP ─────────────────────────────────────────────────────

    /// Прямой проход от типа обратного не зависит — побитно.
    func testForwardDoesNotDependOnBackwardType() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let ids = f.ref["input_ids"]!
        // Считаются ВСЕ проекции: слитый cmix замороженного слоя идёт мимо
        // обёртки со своим VJP (градиент через такой слой не идёт) — он
        // проверен в конце.
        bb.frozenPassKernels = []
        let before = bb.rwkvqWideBackwardCalls
        let a = bb(ids); eval(a)
        XCTAssertEqual(bb.rwkvqWideBackwardCalls - before, 73, "72 квантованные проекции и плотная цель")
        bb.rwkvqBackwardDType = nil
        let b = bb(ids); eval(b)
        XCTAssertEqual(bb.rwkvqWideBackwardCalls - before, 73, "при nil свой VJP не заводится")
        bb.rwkvqBackwardDType = .float32
        let c = bb(ids); eval(c)
        XCTAssertEqual(RwkvqFileTests.mismatches(a, b), 0)
        XCTAssertEqual(RwkvqFileTests.mismatches(a, c), 0)
        // умолчание: cmix двенадцати замороженных слоёв (24 проекции) — без обёртки
        bb.frozenPassKernels = .all
        bb.rwkvqBackwardDType = .bfloat16
        let n = bb.rwkvqWideBackwardCalls
        let d = bb(ids); eval(d)
        XCTAssertEqual(bb.rwkvqWideBackwardCalls - n, 49, "48 проекций tmix и плотная цель")
    }

    /// Обратный проход через одну проекцию побитно равен формуле
    /// dX = (dY в типе обратного) · (W в типе обратного), приведённой к типу x:
    /// на sym, на sb6 своим ядром декванта и на плотной цели.
    func testBackwardIsTheFormula() throws {
        struct Case { let preset: String; let x070: String; let world: String; let native: Bool }
        let cases = [
            Case(preset: "reduction", x070: "blocks.1.tmix.k_proj.weight", world: "blocks.1.att.key.weight", native: true),
            Case(preset: "reduction", x070: "blocks.2.cmix.key.weight", world: "blocks.2.ffn.key.weight", native: true),
            Case(preset: "reduction", x070: "blocks.0.tmix.o_proj.weight", world: "blocks.0.att.output.weight", native: true),
            Case(preset: "compression", x070: "blocks.1.tmix.k_proj.weight", world: "blocks.1.att.key.weight", native: false),
        ]
        MLXRandom.seed(3)
        for cs in cases {
            let bb = try X070Backbone.load(fromRwkvq: try RwkvqModelTests.fixture(cs.preset).path,
                                           useNativeKernel: cs.native)
            let direct = try RwkvqFileTests.base(cs.preset)
            let info = direct.tensors[cs.world]!
            let x = MLXRandom.normal([1, 16, info.inFeatures])
            let c = MLXRandom.normal([1, 16, info.outFeatures])
            func g() -> MLXArray {
                let fn: ([MLXArray]) -> [MLXArray] = { ins in
                    [(bb.projectForBlock(ins[0], cs.x070, lora: nil) * c).sum()]
                }
                let out = grad(fn, argumentNumbers: [0])([x])[0]
                eval(out)
                return out
            }
            var got: [String: MLXArray] = [:]
            for (tag, bwd) in [("bf16", DType.bfloat16), ("fp32", .float32)] {
                bb.rwkvqBackwardDType = bwd
                let w = try direct.dequantize(cs.world, dtype: bwd)
                let want = matmul(c.asType(bwd), w).asType(x.dtype)
                got[tag] = g()
                XCTAssertEqual(got[tag]!.dtype, x.dtype)
                XCTAssertEqual(RwkvqFileTests.mismatches(got[tag]!, want), 0,
                               "\(cs.preset) \(cs.x070): обратный проход \(tag) разошёлся с формулой")
            }
            bb.rwkvqBackwardDType = nil
            let through = g()
            let w16 = try direct.dequantize(cs.world, dtype: .float16)
            XCTAssertEqual(RwkvqFileTests.mismatches(
                through, matmul(c.asType(.float16), w16).asType(x.dtype)), 0,
                "\(cs.x070): автоград в типе базы — это fp16 насквозь")
            XCTAssertGreaterThan(RwkvqFileTests.mismatches(through, got["bf16"]!), 0,
                                 "\(cs.x070): тип обратного прохода ни на что не влияет")
            XCTAssertGreaterThan(RwkvqFileTests.mismatches(got["fp32"]!, got["bf16"]!), 0)
        }
    }

    /// На модели: зачем обратный проход шире прямого. У fp16 узкая экспонента,
    /// и МЕЛКИЕ котангенты в нём теряют разряды; насколько они мелкие, решает
    /// масштаб лосса — длина контекста и размер батча, по которым он усреднён.
    /// Поэтому свойство проверяется на двух масштабах: при обычном и при лоссе,
    /// уменьшенном в 4096 раз (как у длинного батча). Относительная ошибка
    /// градиентов адаптеров к плечу без приведения (fp32-матмул базы):
    ///   • у «fp16 + обратный bf16» от масштаба не зависит;
    ///   • у «fp16 насквозь» на мелком масштабе обязана быть хуже.
    func testWideBackwardSurvivesSmallCotangents() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try Self.withAdapters(f.path)
        let (x, y) = Self.batch(f)
        func arm(base: DType?, bwd: DType?, scale: Float) -> MLXArray {
            bb.rwkvqBaseDType = base
            bb.rwkvqBackwardDType = bwd
            return Self.adapterGrads(bb, x, y, lossScale: scale)
        }
        var e: [String: [Float]] = [:]
        let scales: [Float] = [1, 1.0 / 4096]
        for scale in scales {
            let exact = arm(base: nil, bwd: nil, scale: scale)   // деквант в тип входа, fp32-матмул
            XCTAssertGreaterThan(sqrt(exact.square().sum()).item(Float.self), 0)
            let arms: [(String, DType?, DType?)] = [
                ("насквозь", .float16, nil), ("обратный bf16", .float16, .bfloat16),
                ("обратный fp32", .float16, .float32), ("bf16 насквозь", .bfloat16, nil)]
            for (tag, base, bwd) in arms {
                e[tag, default: []].append(RwkvqModelTests.relErr(arm(base: base, bwd: bwd, scale: scale), exact))
            }
        }
        print("ГРАДИЕНТЫ адаптеров, отн. L2 к плечу без приведения (лосс ×1 / ×1/4096): "
              + e.sorted { $0.key < $1.key }.map { String(format: "%@ %.2e / %.2e", $0.key, $0.value[0], $0.value[1]) }
                .joined(separator: " | "))
        let wide = e["обратный bf16"]!, through = e["насквозь"]!
        XCTAssertLessThan(abs(wide[1] / wide[0] - 1), 0.25, "обратный bf16 зависит от масштаба лосса")
        XCTAssertLessThan(wide[1], through[1], "на мелких котангентах обратный bf16 не лучше fp16 насквозь")
        XCTAssertLessThan(e["обратный fp32"]![1], wide[1] * 1.05)
        XCTAssertLessThan(wide[0], 0.05)
    }

    /// Два чанка под блочным чекпоинтом: свой VJP работает и внутри
    /// пересчитываемого блока, градиенты конечны и повторяемы.
    func testWideBackwardInsideBlockCheckpoint() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try Self.withAdapters(f.path)
        bb.useBlockCheckpoint = true
        let (x, y) = Self.batch(f, T: 32)
        let a = Self.adapterGrads(bb, x, y), b = Self.adapterGrads(bb, x, y)
        XCTAssertEqual((a .!= a).asType(.int32).sum().item(Int.self), 0, "NaN в градиентах")
        XCTAssertGreaterThan(sqrt(a.square().sum()).item(Float.self), 0)
        XCTAssertLessThan(RwkvqModelTests.relErr(b, a), 1e-5, "градиент не повторяется")
    }

    // ── Слияние ──────────────────────────────────────────────────────

    /// После слияния адаптеров вес становится плотным, квантованная
    /// подложка снимается, а модель считает то же, что считала с адаптерами.
    func testMergeTurnsQuantizedTargetsDense() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try Self.withAdapters(f.path)
        let ids = f.ref["input_ids"]!
        let before = bb(ids)[0]; eval(before)
        XCTAssertEqual(bb.rwkvqBackedKeys.count, 72)
        LoRA.merge(bb)
        XCTAssertFalse(bb.hasLoRAAdapters)
        // 4 проекции tmix × 12 слоёв слиты: 47 квантованных и плотная цель
        XCTAssertEqual(bb.rwkvqBackedKeys.count, 72 - 47)
        XCTAssertTrue(bb.rwkvqDenseTargets.isEmpty)
        XCTAssertTrue(bb.hasWeight("blocks.5.tmix.k_proj.weight"))
        XCTAssertTrue(bb.isRwkvqBacked("blocks.5.cmix.key.weight"), "cmix адаптеров не имел")
        let after = bb(ids)[0]; eval(after)
        let kl = RwkvqModelTests.kl(before, after)
        print(String(format: "СЛИЯНИЕ: KL(с адаптерами ‖ слитая) %.3e", kl))
        XCTAssertLessThan(kl, 1e-4)
    }
}
