import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Короткий замороженный параллельный проход ядрами GEMV
//  (`X070Backbone.shortPassKernelRows`): когда строк входа B·T не больше
//  порога, проекции считаются по сжатым кодам, как на рекуррентном шаге, а
//  не «деквант + fp16-матмул». Быстрее на коротком входе и точнее.
// ───────────────────────────────────────────────────────────────────────

final class ShortPassKernelTests: XCTestCase {

    static let ids = (0 ..< 64).map { ($0 * 7919 + 13) % 60000 + 100 }

    static func kernelCalls(_ bb: X070Backbone) -> Int { bb.rwkvqSymGEMVCalls + bb.rwkvqSb6GEMVCalls }

    static func arr(_ rows: Int, _ T: Int) -> MLXArray {
        MLXArray((0 ..< rows * T).map { Int32(ids[$0 % ids.count]) }, [rows, T])
    }

    func testDefaultThresholdFollowsSizeAndLayout() throws {
        XCTAssertEqual(X070Backbone.defaultShortPassRows(nEmbd: 768, sym: true), 6)
        XCTAssertEqual(X070Backbone.defaultShortPassRows(nEmbd: 2048, sym: true), 16)
        XCTAssertEqual(X070Backbone.defaultShortPassRows(nEmbd: 2048, sym: false), 8)
        XCTAssertEqual(X070Backbone.defaultShortPassRows(nEmbd: 768, sym: false), 4)
        XCTAssertEqual(try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("reduction")).shortPassKernelRows, 6)
        XCTAssertEqual(try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("compression")).shortPassKernelRows, 4)
        // родной контейнер и прежний путь базы: ядра кодов не читают
        XCTAssertEqual(try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("compression"),
                                             useNativeKernel: true).shortPassKernelRows, 0)
        XCTAssertEqual(try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("reduction"),
                                             paramDType: .bfloat16, baseDType: nil).shortPassKernelRows, 0)
    }

    /// Ядрами идут ВСЕ шесть проекций слоя, и ровно до порога; строки — B·T.
    func testKernelsRunUpToTheThresholdInRows() throws {
        for preset in RwkvqFileTests.presets {
            let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath(preset))
            let limit = bb.shortPassKernelRows
            let L = bb.cfg.nLayer
            // o_proj слоя 0 в файле плотная: квантованных проекций 6·L − 1
            let perPass = 6 * L - 1
            func calls(_ rows: Int, _ T: Int) -> Int {
                let before = Self.kernelCalls(bb)
                let (h, st) = bb.bodyWithState(Self.arr(rows, T)); eval(h, st.wkv)
                return Self.kernelCalls(bb) - before
            }
            XCTAssertEqual(calls(1, 1), perPass, "\(preset): T = 1")
            XCTAssertEqual(calls(1, limit), perPass, "\(preset): на пороге")
            XCTAssertEqual(calls(1, limit + 1), 0, "\(preset): за порогом — матмул")
            XCTAssertEqual(calls(2, limit / 2), perPass, "\(preset): строки считаются по батчу")
            XCTAssertEqual(calls(2, limit / 2 + 1), 0, "\(preset): батч за порогом")
            bb.shortPassKernelRows = 0
            XCTAssertEqual(calls(1, 2), 0, "\(preset): выключатель")
            // обучаемый слой — не ядрами: через него идёт градиент
            // (обучаемому ядру WKV нужна длина, кратная 16, — порог поднят под неё)
            bb.shortPassKernelRows = 16
            bb.trainLayers = Set((L - 2) ..< L)
            XCTAssertEqual(calls(1, 16), 6 * (L - 2) - 1, "\(preset): обучаемые слои посчитаны ядрами")
        }
    }

    /// Числа: короткий проход ядрами близок к матмулу, согласен с
    /// рекуррентным шагом и стоит к точному счёту на том же уровне.
    func testShortPassNumbers() throws {
        for preset in RwkvqFileTests.presets {
            let r = try RwkvqQuantReferenceTests.reference(preset)
            let f = r.metal
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let limit = bb.shortPassKernelRows
            let ids = f.ref["input_ids"]!
            func logits(_ rows: Int) -> MLXArray {
                bb.shortPassKernelRows = rows
                defer { bb.shortPassKernelRows = limit }
                let (h, _) = bb.bodyWithState(ids)
                let y = bb.projectForBlock(h, "head.weight", lora: nil)[0].asType(.float32)
                eval(y); return y
            }
            // весь эталонный вход (64 токена) ядрами — только ради сверки
            let kern = logits(1 << 20), mm = logits(0)
            let kl = RwkvqModelTests.kl
            let kRef = kl(f.ref["logits_ref"]!, kern), mRef = kl(f.ref["logits_ref"]!, mm)
            print(String(format: "КОРОТКИЙ ПРОХОД %@: KL(ref‖ядра) %.3e, KL(ref‖матмул) %.3e, KL(ref‖quant ядрами) %.3e, KL(quant‖ядра) %.3e",
                         preset, kRef, mRef, r.klRefQuant, kl(r.quant["logits_quant"]!, kern)))
            XCTAssertGreaterThan(RwkvqFileTests.mismatches(kern, mm), 0, "\(preset): ядра не включились")
            XCTAssertLessThan(kl(mm, kern), 1e-5, preset)
            // К точному счёту ядра стоят на том же уровне, что матмул (замерено
            // 5.2e-6 против 4.4e-6 на REDUCTION, 4.0e-6 против 5.2e-6 на
            // COMPRESSION), а не на уровне rwkv-quant (1.9e-6 / 6.0e-7): шум
            // порта задают не проекции, а fp16-параметры норм и миксов (у
            // rwkv-quant одномерные параметры fp32). Всё это на четыре порядка
            // ниже цены самого квантования (KL к исходной модели ~2e-2).
            XCTAssertLessThan(kRef, 1e-5, preset)
            XCTAssertLessThan(mRef, 1e-5, preset)

            // короткий префилл ядрами против префилла шагами: один и тот же счёт проекций
            var a = RWKVState(cfg: bb.cfg), b = RWKVState(cfg: bb.cfg)
            let prompt = Array(Self.ids[0 ..< limit])
            let before = Self.kernelCalls(bb)
            let p = bb.prefill(prompt, state: &a); eval(p)
            XCTAssertGreaterThan(Self.kernelCalls(bb) - before, 6 * bb.cfg.nLayer - 2)
            let q = bb.prefillRecurrent(prompt, state: &b); eval(q)
            XCTAssertLessThan(kl(q.reshaped([1, -1]), p.reshaped([1, -1])), 1e-5, "\(preset): короткий префилл против шагов")
            XCTAssertEqual(argMax(p, axis: -1).item(Int.self), argMax(q, axis: -1).item(Int.self))
        }
    }

    /// Несколько строк одним вызовом равны тем же строкам по одной: нарезка
    /// входа по строкам ядра (sym — по четыре, sb6 — по одной) ничего не теряет.
    func testRowsAreIndependent() throws {
        for preset in RwkvqFileTests.presets {
            let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath(preset))
            bb.shortPassKernelRows = 64
            let x = Self.arr(7, 3)                       // 21 строка: не кратно четырём
            let (all, _) = bb.bodyWithState(x); eval(all)
            for row in [0, 3, 6] {
                let (one, _) = bb.bodyWithState(x[row ..< (row + 1)]); eval(one)
                XCTAssertLessThan(RwkvqModelTests.relErr(all[row ..< (row + 1)], one), 1e-5, "\(preset): строка \(row)")
            }
            XCTAssertGreaterThan(RwkvqModelTests.relErr(all[0 ..< 1], all[3 ..< 4]), 1e-3)
        }
    }
}
