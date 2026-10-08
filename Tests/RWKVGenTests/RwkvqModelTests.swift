import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Модель, собранная из .rwkvq нынешних пресетов, против rwkv-metal.
//
//  RwkvqFileTests проверяет ДЕКВАНТ, здесь — ПРОВОДКА и ТИПЫ: что каждая
//  проекция взяла свой тензор, low-rank ветки стоят в своей ориентации,
//  плотная цель подключена, а счёт идёт в норме rwkv-quant (плотные
//  параметры fp16, база fp16).
//
//  Эталон (tools/gen_rwkvq_fixtures.py в rwkv-metal), 0.1B обоих пресетов:
//    logits_metal — путь rwkv-metal по умолчанию (то, что порт повторяет);
//    logits_ref   — точный счёт тех же весов (fp32-параметры, fp32-матмул);
//    blk_out, состояние после префилла, потоковый декод 8 токенов.
//  Порт не обязан совпасть с rwkv-metal побитно: другой порядок суммирования
//  в нормировках слоя 0, а на REDUCTION каждое fp16-округление матмула его
//  разносит дальше. Мера допуска — KL(ref‖metal) из json эталона: порт обязан
//  стоять к точному счёту не дальше, чем сам rwkv-metal (запас ×1.25), и к
//  rwkv-metal — ближе, чем тот к точному счёту. На COMPRESSION, где после
//  слоя 0 всё идёт в fp32, порт совпадает с rwkv-metal до ~1e-8 KL, и это
//  требуется отдельно: по нему видно любую подмену типа.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqModelTests: XCTestCase {

    struct Fixture {
        let path: String
        let ref: [String: MLXArray]
        let klRefMetal: Float
        let klRefMetalDecode: Float
    }

    static func fixture(_ preset: String) throws -> Fixture {
        let dir = RwkvqFileTests.testdata
        let src = try RwkvqFileTests.sourcePath(preset)
        let mp = dir.appendingPathComponent("rwkvq2_\(preset)_model.safetensors")
        try XCTSkipIf(!FileManager.default.fileExists(atPath: mp.path),
                      "нет эталона модели \(mp.lastPathComponent) — тест пропущен")
        let js = try JSONSerialization.jsonObject(with: Data(contentsOf:
            dir.appendingPathComponent("rwkvq2_\(preset).json"))) as! [String: Any]
        return Fixture(path: src, ref: try loadArrays(url: mp),
                       klRefMetal: Float(js["kl_ref_metal"] as! Double),
                       klRefMetalDecode: Float(js["kl_ref_metal_decode"] as! Double))
    }

    /// KL(p‖q) по строкам логитов, среднее.
    static func kl(_ p: MLXArray, _ q: MLXArray) -> Float {
        let lp = p.asType(.float32) - logSumExp(p.asType(.float32), axis: -1, keepDims: true)
        let lq = q.asType(.float32) - logSumExp(q.asType(.float32), axis: -1, keepDims: true)
        return (exp(lp) * (lp - lq)).sum(axis: -1).mean().item(Float.self)
    }

    static func relErr(_ got: MLXArray, _ want: MLXArray) -> Float {
        let g = got.asType(.float32), w = want.asType(.float32)
        return (sqrt((g - w).square().sum()) / sqrt(w.square().sum())).item(Float.self)
    }

    // ── Сборка ───────────────────────────────────────────────────────

    func testLoadsBothPresetsInTheNorm() throws {
        for preset in RwkvqFileTests.presets {
            let f = try Self.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            XCTAssertEqual(bb.cfg.nLayer, 12); XCTAssertEqual(bb.cfg.nEmbd, 768)
            XCTAssertEqual(bb.rwkvqBaseDType, .float16, "\(preset): норма базы — fp16")
            for key in ["emb.weight", "ln0.weight", "blocks.3.tmix.x_r", "blocks.3.tmix.k_k",
                        "blocks.3.tmix.w_lora_A.weight", "blocks.3.tmix.g_lora_B.weight",
                        "blocks.3.cmix.x_k", "ln_out.bias"] {
                XCTAssertEqual(bb.weight(key).dtype, .float16, "\(preset) \(key)")
            }
            // ориентация low-rank: A — [rank, D], B — [D, rank]
            XCTAssertEqual(bb.weight("blocks.3.tmix.w_lora_A.weight").shape, [64, 768])
            XCTAssertEqual(bb.weight("blocks.3.tmix.w_lora_B.weight").shape, [768, 64])
            XCTAssertEqual(bb.weight("blocks.3.tmix.g_lora_A.weight").shape, [128, 768])
            XCTAssertEqual(bb.weight("blocks.3.tmix.v_lora_B.weight").shape, [768, 32])
            // плотная цель: o_proj слоя 0 — как в файле, bf16
            XCTAssertEqual(bb.rwkvqDenseTargets, ["blocks.0.tmix.o_proj.weight"], preset)
            XCTAssertFalse(bb.isRwkvqBacked("blocks.0.tmix.o_proj.weight"))
            XCTAssertEqual(bb.weight("blocks.0.tmix.o_proj.weight").dtype, .bfloat16)
            // всё остальное из целей — квантовано: 12·6 + head − 1 плотная
            XCTAssertEqual(bb.rwkvqBackedKeys.count, 72)
            XCTAssertFalse(bb.hasWeight("blocks.1.tmix.o_proj.weight"))
            XCTAssertFalse(bb.hasWeight("head.weight"))
            if preset == "compression" {
                // sb6 целиком в родном контейнере, сырые буферы отпущены
                XCTAssertFalse(bb.isRawSidecarRetained)
                XCTAssertTrue(bb.rwkvqBackedKeys.allSatisfy { bb.hasNativeRepack($0) })
            } else {
                // у sym родного контейнера нет: буферы держатся, но ТОЛЬКО свои
                XCTAssertTrue(bb.isRawSidecarRetained)
                XCTAssertTrue(bb.rwkvqBackedKeys.allSatisfy { !bb.hasNativeRepack($0) })
                let kept = try XCTUnwrap(bb.rwkvqSidecar)
                XCTAssertEqual(kept.tensors.count, 72)
                XCTAssertNil(kept.arrays["emb.weight::qblk"], "emb развёрнут, его буферы не нужны")
                XCTAssertNil(kept.arrays["blocks.0.att.w1::codes"])
            }
        }
    }

    /// Плотная цель — побитно вес из файла, и подключена она именно к o_proj.
    func testDenseTargetIsTheFileWeight() throws {
        let f = try Self.fixture("reduction")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let direct = try RwkvqFileTests.base("reduction")
        let want = try direct.dequantize("blocks.0.att.output.weight", dtype: .bfloat16)
        XCTAssertEqual(RwkvqFileTests.mismatches(bb.weight("blocks.0.tmix.o_proj.weight"), want), 0)
        let other = try direct.dequantize("blocks.0.att.key.weight", dtype: .bfloat16)
        XCTAssertGreaterThan(RwkvqFileTests.mismatches(want, other), 0)
    }

    /// Вес, оставшийся только в родном контейнере, читается и равен декванту.
    func testWeightReadsBackFromNativeContainer() throws {
        let f = try Self.fixture("compression")
        let bb = try X070Backbone.load(fromRwkvq: f.path)
        let direct = try RwkvqFileTests.base("compression")
        for (x070, world) in [("blocks.5.tmix.k_proj.weight", "blocks.5.att.key.weight"),
                              ("blocks.5.cmix.value.weight", "blocks.5.ffn.value.weight")] {
            XCTAssertTrue(bb.hasNativeRepack(x070))
            let got = bb.weight(x070).asType(.float32)
            let want = try direct.dequantize(world, dtype: .float16).asType(.float32)
            XCTAssertEqual(got.shape, want.shape)
            XCTAssertLessThan(Self.relErr(got, want), 1e-3, x070)
        }
    }

    // ── Главное: сквозной эталон ─────────────────────────────────────

    func testForwardMatchesRwkvMetal() throws {
        for preset in RwkvqFileTests.presets {
            let f = try Self.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let ids = f.ref["input_ids"]!
            let logits = bb(ids)[0].asType(.float32)
            eval(logits)
            XCTAssertEqual(logits.shape, f.ref["logits_metal"]!.shape)

            let klRef = Self.kl(f.ref["logits_ref"]!, logits)
            let klMetal = Self.kl(f.ref["logits_metal"]!, logits)
            let top1 = (argMax(logits, axis: -1) .== argMax(f.ref["logits_metal"]!, axis: -1))
                .asType(.float32).mean().item(Float.self)
            print(String(format: "ЛОГИТЫ %@: KL(ref‖порт) %.3e, KL(metal‖порт) %.3e, "
                         + "KL(ref‖metal) %.3e, top-1 к metal %.3f",
                         preset, klRef, klMetal, f.klRefMetal, top1))
            XCTAssertLessThan(klRef, 1.25 * f.klRefMetal,
                              "\(preset): порт дальше от точного счёта, чем rwkv-metal")
            XCTAssertLessThan(klMetal, f.klRefMetal,
                              "\(preset): порт дальше от rwkv-metal, чем тот от точного счёта")
            if preset == "compression" {
                XCTAssertLessThan(abs(klMetal), 1e-7, "compression: порт обязан совпасть с rwkv-metal")
            }
            XCTAssertGreaterThanOrEqual(top1, 0.98, preset)

            // выход каждого блока — ловит перепутанную проводку послойно
            let per = bb.debugPerLayer(ids)
            var worst: Float = 0
            for layer in 0 ..< 12 {
                let e = Self.relErr(per["after_blk\(layer)"]![0], f.ref["blk_out"]![layer])
                worst = max(worst, e)
                XCTAssertLessThan(e, 5e-3, "\(preset): блок \(layer) разошёлся на \(e)")
            }
            print(String(format: "БЛОКИ %@: худшая отн. ошибка выхода блока %.3e", preset, worst))
        }
    }

    func testPrefillStateAndStreamingDecode() throws {
        for preset in RwkvqFileTests.presets {
            let f = try Self.fixture(preset)
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let (_, st) = bb.bodyWithState(f.ref["input_ids"]!)
            let eW = Self.relErr(st.wkv[0..., 0], f.ref["state_wkv"]!)
            let eT = Self.relErr(st.tmixShift[0..., 0, 0], f.ref["state_tmix"]!)
            let eC = Self.relErr(st.cmixShift[0..., 0, 0], f.ref["state_cmix"]!)
            print(String(format: "СОСТОЯНИЕ %@: wkv %.3e, tmix %.3e, cmix %.3e", preset, eW, eT, eC))
            XCTAssertLessThan(eW, 5e-3); XCTAssertLessThan(eT, 5e-3); XCTAssertLessThan(eC, 5e-3)

            // потоковый декод тем же путём, что в эталоне: по одному токену
            // через путь с состоянием
            let dec = f.ref["decode_ids"]!
            var state = st
            var rows: [MLXArray] = []
            for i in 0 ..< dec.shape[1] {
                let (h, next) = bb.bodyWithState(dec[0..., i ..< (i + 1)], state: state)
                state = next
                rows.append(bb.projectForBlock(h, "head.weight", lora: nil)[0, 0].asType(.float32))
            }
            let got = stacked(rows, axis: 0)
            let klRef = Self.kl(f.ref["decode_logits_ref"]!, got)
            let klMetal = Self.kl(f.ref["decode_logits"]!, got)
            // и рекуррентным шагом — тем, чем порт генерирует
            var rs = RWKVState(st, row: 0)
            var stepRows: [MLXArray] = []
            for i in 0 ..< dec.shape[1] {
                stepRows.append(bb.step(dec[0, i].item(Int.self), state: &rs).asType(.float32))
            }
            let klStep = Self.kl(f.ref["decode_logits"]!, stacked(stepRows, axis: 0).reshaped(got.shape))
            print(String(format: "ДЕКОД %@: KL(ref‖порт) %.3e, KL(metal‖порт) %.3e, "
                         + "KL(metal‖шаг) %.3e, KL(ref‖metal) %.3e",
                         preset, klRef, klMetal, klStep, f.klRefMetalDecode))
            XCTAssertLessThan(klRef, 1.25 * f.klRefMetalDecode, preset)
            XCTAssertLessThan(klMetal, f.klRefMetalDecode, preset)
            XCTAssertLessThan(klStep, 2 * f.klRefMetalDecode, preset)
        }
    }

    // ── Типы: плечи в одном процессе ─────────────────────────────────

    /// Норма не декларация, а измеримое свойство: к точному счёту fp16-база
    /// и fp16-параметры обязаны стоять не дальше, чем bf16.
    func testFp16NormIsCloserToExactThanBf16() throws {
        let f = try Self.fixture("reduction")
        let ids = f.ref["input_ids"]!, exact = f.ref["logits_ref"]!
        let norm = try X070Backbone.load(fromRwkvq: f.path)
        let klNorm = Self.kl(exact, norm(ids)[0])
        norm.rwkvqBaseDType = .bfloat16
        let klBaseBf = Self.kl(exact, norm(ids)[0])
        norm.rwkvqBaseDType = .float16
        XCTAssertEqual(Self.kl(exact, norm(ids)[0]), klNorm, "переключение типа базы обратимо")
        let bf = try X070Backbone.load(fromRwkvq: f.path, paramDType: .bfloat16,
                                       baseDType: .bfloat16)
        XCTAssertEqual(bf.weight("emb.weight").dtype, .bfloat16)
        let klAllBf = Self.kl(exact, bf(ids)[0])
        print(String(format: "ТИПЫ reduction, KL к точному счёту: норма %.3e, база bf16 %.3e, "
                     + "всё bf16 %.3e", klNorm, klBaseBf, klAllBf))
        XCTAssertLessThan(klNorm, klBaseBf)
        XCTAssertLessThan(klNorm, klAllBf)
    }

    // ── Ранги не по формуле ──────────────────────────────────────────

    /// rwkv7-g1d-0.4b: ветка g имеет ранг 128, формула lora_ranks(1024) даёт
    /// 160. Порт берёт ранги из форм тензоров файла — модель обязана
    /// открыться и считать то же, что rwkv-metal.
    /// Эталон: Scripts/dump_rwkvq2_logits.py 0p4b.
    func testOpensModelWhoseRanksAreNotTheFormula() throws {
        let dir = RwkvqFileTests.testdata
        let src01 = try RwkvqFileTests.sourcePath("reduction")
        for preset in RwkvqFileTests.presets {
            let path = (src01 as NSString).deletingLastPathComponent + "/0p4b_\(preset).rwkvq"
            let rp = dir.appendingPathComponent("rwkvq2_0p4b_\(preset)_logits.safetensors")
            try XCTSkipIf(!FileManager.default.fileExists(atPath: path)
                          || !FileManager.default.fileExists(atPath: rp.path),
                          "нет 0.4B (\(path)) или её эталона — тест пропущен")
            let ref = try loadArrays(url: rp)
            let bb = try X070Backbone.load(fromRwkvq: path)
            XCTAssertEqual(bb.cfg.nEmbd, 1024); XCTAssertEqual(bb.cfg.nLayer, 24)
            let ranks = ["w", "a", "v", "g"].map {
                bb.weight("blocks.1.tmix.\($0)_lora_A.weight").shape[0]
            }
            XCTAssertEqual(ranks, ref["ranks"]!.asArray(Int32.self).map(Int.init))
            XCTAssertEqual(ranks[3], 128, "ранг g у 0.4B — из файла, не из формулы (160)")
            XCTAssertEqual(X070InitConfig.LoRARanks.rwkvMetal.ranks(nEmbd: 1024).g, 160,
                           "формула изменилась — свойство теста потеряло смысл")
            let logits = bb(ref["input_ids"]!)[0]
            let kl = Self.kl(ref["logits_metal"]!, logits)
            print(String(format: "0.4B %@: KL(metal‖порт) %.3e", preset, kl))
            XCTAssertLessThan(kl, 1e-5, preset)
        }
    }

    // ── Отказы ───────────────────────────────────────────────────────

    /// Цель, которой в файле нет, — отказ ДО сборки, с раскладкой в тексте.
    func testRefusesBaseWithoutATarget() throws {
        let full = try RwkvqFileTests.base("reduction")
        var keep = Set(full.tensors.keys)
        keep.remove("blocks.7.ffn.key.weight")
        XCTAssertThrowsError(try X070Backbone.load(fromSidecar: full.pruned(keeping: keep))) {
            let text = "\($0)"
            XCTAssertTrue(text.contains("blocks.7.ffn.key.weight"), text)
            XCTAssertTrue(text.contains("НЕТ В ФАЙЛЕ"), text)
        }
    }

    // ── Зажим scale у родного контейнера ─────────────────────────────

    /// Вырожденный блок sb6 (qs·d = 0): формат держит scale на зажиме 1e-8,
    /// в fp16 это ноль. Если коды блока НЕнулевые, обнуление меняет веса —
    /// такой тензор обязан остаться в fp32 с зажимом; если нулевые — fp16.
    /// На 0.1B таких блоков нет вовсе, поэтому буферы синтетические.
    func testNativeScaleStaysFp32OnlyWhereTheClampMatters() throws {
        let OUT = 2, IN = 256, NB = 8
        func base(codeInDeadBlock: UInt8) throws -> RwkvqNativeWeight {
            var qblk = [UInt8](repeating: 0x21, count: OUT * NB * 16)     // коды 1 и 2
            for i in 0 ..< 16 { qblk[3 * 16 + i] = codeInDeadBlock }      // строка 0, блок 3
            var qsqm = [UInt8](repeating: 0, count: OUT * NB * 2)
            for b in 0 ..< OUT * NB { qsqm[2 * b] = 20; qsqm[2 * b + 1] = UInt8(bitPattern: -3) }
            qsqm[2 * 3] = 0                                               // qs = 0: блок вырожден
            let ddm = MLXArray([Float](repeating: 0.01, count: OUT * 2), [OUT, 2]).asType(.float16)
            let info = RwkvqTensorInfo(kind: .sb6, shape: [OUT, IN], bits: 4, xbits: 0,
                                       groupSize: 32, superBlock: 8, packed: false,
                                       outlierCount: 0, transposed: false)
            let s = RwkvqSidecar(
                arrays: ["t::qblk": MLXArray(qblk, [OUT, NB * 16]),
                         "t::qsqm": MLXArray(qsqm, [OUT, NB * 2]), "t::ddm": ddm],
                tensors: ["t": info], naming: "world", nLayer: 0, nEmbd: 0, headSize: 64, vocabSize: 0)
            let legacy = try s.nativeAffine("t", exactScales: false)
            XCTAssertEqual(legacy.scales.dtype, .float16, "прежнее правило: fp16 всегда")
            return try s.nativeAffine("t")
        }
        let live = try base(codeInDeadBlock: 0x21)
        XCTAssertEqual(live.scales.dtype, .float32, "ненулевые коды на зажиме: fp32")
        XCTAssertEqual(live.biases.dtype, .float32)
        XCTAssertEqual(live.scales[0, 3].item(Float.self), 1e-8)
        XCTAssertEqual(live.scales[1, 3].item(Float.self), Float(Float16(20 * Float(Float16(0.01)))))
        let dead = try base(codeInDeadBlock: 0)
        XCTAssertEqual(dead.scales.dtype, .float16, "нулевые коды на зажиме: fp16 ничего не меняет")
        XCTAssertEqual(dead.scales[0, 3].item(Float.self), 0)
    }
}
