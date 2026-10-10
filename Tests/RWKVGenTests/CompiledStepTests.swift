import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Рекуррентный шаг через `compile` (useCompiledStep).
//
//  Собранный шаг вшивает веса и переключатели в граф константами. Два
//  свойства, без которых он опасен:
//    1. числа те же, что у покомандного пути (в пределах шума fp16 —
//       слитые цепочки поэлементных операций округляют в другом порядке);
//    2. граф НИКОГДА не устаревает: запись в веса, адаптеры, базу или
//       переключатель пути пересобирает его при следующем вызове.
//  Второе не падает и не выглядит дефектом — дообученная модель просто
//  генерировала бы как недообученная, — поэтому проверяется явно.
// ───────────────────────────────────────────────────────────────────────

final class CompiledStepTests: XCTestCase {

    /// n шагов от пустого состояния; логиты [n, V] fp32 и конечное состояние.
    static func run(_ bb: X070Backbone, _ ids: [Int], compiled: Bool) -> (MLXArray, RWKVState) {
        bb.useCompiledStep = compiled
        var st = RWKVState(cfg: bb.cfg)
        var rows: [MLXArray] = []
        for id in ids {
            rows.append(bb.step(id, state: &st).asType(.float32))
            st.eval()
        }
        let l = stacked(rows, axis: 0)
        eval(l)
        return (l, st)
    }

    func testCompiledStepMatchesEagerOnBothPresets() throws {
        for preset in RwkvqFileTests.presets {
            let f = try RwkvqModelTests.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let ids = f.ref["input_ids"]![0, 0 ..< 12].asArray(Int32.self).map(Int.init)
            let (eager, se) = Self.run(bb, ids, compiled: false)
            let (comp, sc) = Self.run(bb, ids, compiled: true)
            let kl = RwkvqModelTests.kl(eager, comp)
            let same = (argMax(eager, axis: -1) .== argMax(comp, axis: -1)).asType(.int32).sum().item(Int.self)
            let eW = RwkvqModelTests.relErr(stacked(sc.wkv, axis: 0), stacked(se.wkv, axis: 0))
            print(String(format: "СОБРАННЫЙ ШАГ %@: KL(покомандный‖собранный) %.3e, top-1 %d из %d, состояние wkv отн. %.2e",
                         preset, kl, same, ids.count, eW))
            // COMPRESSION на шаге идёт в fp16 от входа родного ядра до его
            // выхода, и слитые цепочки округляют там заметнее
            XCTAssertLessThan(kl, preset == "compression" ? 1e-4 : 1e-5, preset)
            XCTAssertEqual(same, ids.count, preset)
            XCTAssertLessThan(eW, 5e-3, preset)
            XCTAssertEqual(sc.tmixPrev[0].shape, se.tmixPrev[0].shape)
            XCTAssertEqual(sc.tmixPrev[5].dtype, se.tmixPrev[5].dtype)
            XCTAssertNotNil(sc.vFirst)
        }
    }

    /// bf16-модель НЕ собирается никогда: MLX именует слитые ядра fp16 и bf16
    /// одинаково, и собранные в одном процессе они исполнялись бы кодом друг
    /// друга (мусор без ошибки). Проверяется и само правило, и то, ради чего
    /// оно: плотная bf16 и квантованная fp16 в одном процессе, в обоих
    /// порядках, считают верно.
    func testBfloat16ModelsNeverCompileAndMixSafelyWithHalfModels() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let ids = f.ref["input_ids"]![0, 0 ..< 6].asArray(Int32.self).map(Int.init)
        let dense = try loadArrays(url: RwkvqFileTests.testdata.appendingPathComponent("world_0.1b_x070.safetensors"))
        let cfg = X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536)
        func denseCheck(_ when: String) {
            let bb = X070Backbone(weights: dense, cfg: cfg)                 // bf16
            XCTAssertFalse(bb.stepIsCompilable, "bf16-модель признана собираемой")
            let (eager, _) = Self.run(bb, ids, compiled: false)
            let (asked, _) = Self.run(bb, ids, compiled: true)
            XCTAssertEqual(bb.compiledStepCalls, 0, "\(when): bf16-модель пошла собранным путём")
            XCTAssertEqual(RwkvqFileTests.mismatches(asked, eager), 0, "\(when): плотная bf16 считает иначе")
        }
        func halfCheck(_ when: String) throws {
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            XCTAssertTrue(bb.stepIsCompilable)
            let (eager, _) = Self.run(bb, ids, compiled: false)
            let (comp, _) = Self.run(bb, ids, compiled: true)
            XCTAssertEqual(bb.compiledStepCalls, ids.count, when)
            XCTAssertLessThan(RwkvqModelTests.kl(eager, comp), 1e-5, "\(when): собранный шаг fp16-модели разошёлся")
        }
        denseCheck("до"); try halfCheck("после плотной"); denseCheck("после квантованной"); try halfCheck("ещё раз")

        // неслитые адаптеры лежат в bf16 — модель с ними тоже не собирается
        let lora = try X070Backbone.load(fromRwkvq: f.path)
        LoRA.add(to: lora, spec: LoRASpec(rank: 8, alpha: 16))
        XCTAssertFalse(lora.stepIsCompilable)
        // плотные fp32 и fp16 собираются
        XCTAssertTrue(X070Backbone(weights: dense, cfg: cfg, computeDType: .float32).stepIsCompilable)
    }

    /// Запись в веса пересобирает шаг: собранный шаг считает НОВУЮ модель, а
    /// не ту, под которую был собран.
    func testCompiledStepNeverRunsAStaleGraph() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let ids = f.ref["input_ids"]![0, 0 ..< 6].asArray(Int32.self).map(Int.init)
        let (before, _) = Self.run(bb, ids, compiled: true)
        XCTAssertEqual(bb.compiledStepCalls, ids.count)

        // 1. прямая запись в словарь весов
        bb.w["ln_out.bias"] = bb.w["ln_out.bias"]! + 0.5
        let (after, _) = Self.run(bb, ids, compiled: true)
        let (truth, _) = Self.run(bb, ids, compiled: false)
        let moved = RwkvqModelTests.kl(before, truth)
        let off = RwkvqModelTests.kl(truth, after)
        print(String(format: "УСТАРЕВАНИЕ: правка веса сдвинула модель на KL %.3e; собранный шаг от новой модели на %.3e", moved, off))
        XCTAssertGreaterThan(moved, 1e-3, "правка ничего не изменила — тест ничего не проверяет")
        XCTAssertLessThan(off, moved / 100, "собранный шаг считает прежнюю модель")
        XCTAssertEqual(bb.compiledStepCalls, 2 * ids.count)

        // 2. адаптеры: с ними шаг идёт покомандно, после слияния — снова
        //    собранным, и уже со слитыми весами
        MLXRandom.seed(9)
        LoRA.add(to: bb, spec: LoRASpec(rank: 8, alpha: 16))
        for k in bb.loraTargets {
            let (a, b) = bb.loraAdapter(k)
            bb.setLoraAdapter(k, a, (MLXRandom.normal(b.shape) * 0.05).asType(b.dtype))
        }
        let (withAdapters, _) = Self.run(bb, ids, compiled: true)
        XCTAssertEqual(bb.compiledStepCalls, 2 * ids.count, "модель с адаптерами пошла собранным путём")
        LoRA.merge(bb)
        let (merged, _) = Self.run(bb, ids, compiled: true)
        XCTAssertEqual(bb.compiledStepCalls, 3 * ids.count, "после слияния шаг не вернулся к собранному пути")
        let movedByAdapters = RwkvqModelTests.kl(truth, withAdapters)
        XCTAssertGreaterThan(movedByAdapters, 1e-3)
        XCTAssertLessThan(RwkvqModelTests.kl(withAdapters, merged), movedByAdapters / 100)
    }

    /// Переключатель пути пересобирает шаг: видно по счётчику ядра GEMV.
    func testCompiledStepFollowsPathSwitches() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        bb.useCompiledStep = true
        var st = RWKVState(cfg: bb.cfg)
        func step() { _ = bb.step(7, state: &st).sum().item(Float.self); st.eval() }
        step(); step(); step()
        let traced = bb.rwkvqSymGEMVCalls
        XCTAssertGreaterThan(traced, 0)
        XCTAssertLessThanOrEqual(traced, 3 * 72, "трассировок больше, чем шагов")
        bb.rwkvqSymDecodeKernel = false
        step(); step()
        XCTAssertEqual(bb.rwkvqSymGEMVCalls, traced, "после выключения ядро GEMV попало в пересобранный шаг")
        bb.rwkvqSymDecodeKernel = true
        step()
        XCTAssertGreaterThan(bb.rwkvqSymGEMVCalls, traced, "после включения шаг не пересобран")
    }

    // ── Собранный префилл ────────────────────────────────────────────

    static func prefill(_ bb: X070Backbone, _ ids: [Int], compiled: Bool) -> (MLXArray, RWKVState) {
        bb.useCompiledPrefill = compiled
        var st = RWKVState(cfg: bb.cfg)
        let l = bb.prefill(ids, state: &st).asType(.float32)
        eval(l); st.eval()
        return (l.reshaped([1, -1]), st)
    }

    func testCompiledPrefillMatchesPlainOnBothPresets() throws {
        for preset in RwkvqFileTests.presets {
            let f = try RwkvqModelTests.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let ids = f.ref["input_ids"]![0].asArray(Int32.self).map(Int.init)       // 64 токена
            let (plain, sp) = Self.prefill(bb, ids, compiled: false)
            XCTAssertEqual(bb.compiledPrefillCalls, 0)
            let (comp, sc) = Self.prefill(bb, ids, compiled: true)
            XCTAssertEqual(bb.compiledPrefillCalls, 1, preset)
            let kl = RwkvqModelTests.kl(plain, comp)
            let eW = RwkvqModelTests.relErr(stacked(sc.wkv, axis: 0), stacked(sp.wkv, axis: 0))
            print(String(format: "СОБРАННЫЙ ПРЕФИЛЛ %@: KL(обычный‖собранный) %.3e, состояние wkv отн. %.2e", preset, kl, eW))
            XCTAssertLessThan(kl, 1e-5, preset)
            XCTAssertEqual(comp.argMax().item(Int.self), plain.argMax().item(Int.self))
            XCTAssertLessThan(eW, 5e-3, preset)
            XCTAssertEqual(sc.tmixPrev[3].shape, sp.tmixPrev[3].shape)
            // продолжение шагом с собранного состояния — то же, что с обычного
            var a = sp, b = sc
            bb.useCompiledStep = false
            let la = bb.step(ids[5], state: &a).asType(.float32).reshaped([1, -1])
            let lb = bb.step(ids[5], state: &b).asType(.float32).reshaped([1, -1])
            XCTAssertLessThan(RwkvqModelTests.kl(la, lb), 1e-5, "\(preset): состояние после собранного префилла не годится для шага")

            // короткий промпт идёт обычным путём
            _ = Self.prefill(bb, Array(ids[0 ..< 8]), compiled: true)
            XCTAssertEqual(bb.compiledPrefillCalls, 1, "короткий промпт ушёл в собранный путь")
        }
    }

    /// Версия весов и потолок числа длин: устаревший граф не исполняется,
    /// кеш трасс не растёт без предела, bf16-модель не собирается.
    func testCompiledPrefillFollowsWeightsAndBoundsItsCache() throws {
        let f = try RwkvqModelTests.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let ids = f.ref["input_ids"]![0].asArray(Int32.self).map(Int.init)
        let (before, _) = Self.prefill(bb, ids, compiled: true)
        bb.w["ln_out.bias"] = bb.w["ln_out.bias"]! + 0.5
        let (after, _) = Self.prefill(bb, ids, compiled: true)
        let (truth, _) = Self.prefill(bb, ids, compiled: false)
        let moved = RwkvqModelTests.kl(before, truth)
        XCTAssertGreaterThan(moved, 1e-3)
        XCTAssertLessThan(RwkvqModelTests.kl(truth, after), moved / 100, "собранный префилл считает прежнюю модель")

        bb.compiledPrefillMinTokens = 8
        bb.compiledPrefillMaxLengths = 2
        for n in [16, 24, 32, 40, 16] {
            let part = Array(ids[0 ..< n])
            let (c, _) = Self.prefill(bb, part, compiled: true)
            let (p, _) = Self.prefill(bb, part, compiled: false)
            XCTAssertLessThan(RwkvqModelTests.kl(p, c), 1e-5, "длина \(n)")
            XCTAssertLessThanOrEqual(bb.compiledPrefillLengths.count, 2, "кеш длин перерос потолок")
        }

        let dense = try loadArrays(url: RwkvqFileTests.testdata.appendingPathComponent("world_0.1b_x070.safetensors"))
        let bf = X070Backbone(weights: dense, cfg: X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536))
        _ = Self.prefill(bf, ids, compiled: true)
        XCTAssertEqual(bf.compiledPrefillCalls, 0, "bf16-модель пошла собранным префиллом")
    }
}
