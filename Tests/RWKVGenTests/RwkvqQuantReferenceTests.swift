import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Модель из .rwkvq против ЭТАЛОНА ИНФЕРЕНСА — Metal-бэкенда rwkv-quant.
//
//  Решение владельца 12.10: rwkv-quant — эталон инференса, rwkv-metal —
//  обучения. RwkvqModelTests сверяет порт с rwkv-metal; здесь — с тем, что
//  считает сам rwkv-quant, на тех же входах (эталон —
//  Scripts/dump_rwkvq2_quant_model.py, 0.1B обоих пресетов):
//    logits_quant / hidden_quant — параллельный проход, логиты и скрытое
//      состояние после ln_out (им живут векторные модели и реранкер);
//    состояние после промпта; продолжение по одному токену.
//  Точный счёт тех же весов (logits_ref) берётся из эталона rwkv-metal —
//  он от бэкенда не зависит.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqQuantReferenceTests: XCTestCase {

    struct Reference {
        let quant: [String: MLXArray]
        let metal: RwkvqModelTests.Fixture
        /// KL точного счёта к rwkv-quant: ядрами GEMV (его путь при T < 128)…
        let klRefQuant: Float
        /// …и «деквант + fp16-матмул» (его путь на длинных последовательностях).
        let klRefQuantGemm: Float
        let klRefQuantDecode: Float
        /// Считает ли эталон низкоранговые ветки на декоде квантованными.
        let loraQDecode: Bool
    }

    static func reference(_ preset: String) throws -> Reference {
        let dir = RwkvqFileTests.testdata
        let q = dir.appendingPathComponent("rwkvq2_\(preset)_quant.safetensors")
        try XCTSkipIf(!FileManager.default.fileExists(atPath: q.path),
                      "нет эталона \(q.lastPathComponent) — см. Scripts/dump_rwkvq2_quant_model.py")
        let js = try JSONSerialization.jsonObject(with: Data(contentsOf:
            dir.appendingPathComponent("rwkvq2_\(preset)_quant.json"))) as! [String: Any]
        return Reference(quant: try loadArrays(url: q), metal: try RwkvqModelTests.fixture(preset),
                         klRefQuant: Float(js["kl_ref_quant"] as! Double),
                         klRefQuantGemm: Float(js["kl_ref_quant_gemm"] as! Double),
                         klRefQuantDecode: Float(js["kl_ref_quant_decode"] as! Double),
                         loraQDecode: js["lora_q_decode"] as! Bool)
    }

    /// Параллельный проход. Порт на любой длине считает проекции «деквант +
    /// fp16-матмул» — это путь эталона на ДЛИННЫХ последовательностях, и
    /// сравнивать надо с ним. На коротких (T < 128) эталон идёт ядрами GEMV
    /// по сжатым кодам и стоит к точному счёту в 1.7 (REDUCTION) и в 15
    /// (COMPRESSION) раз ближе — этого пути у параллельного прохода порта
    /// нет, и разрыв здесь ЗАПИСАН, а не закрыт (см. NEXT_SESSION).
    func testParallelPassAgainstRwkvQuant() throws {
        for preset in RwkvqFileTests.presets {
            let r = try Self.reference(preset)
            let f = r.metal
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let (h, st) = bb.bodyWithState(f.ref["input_ids"]!)
            let logits = bb.projectForBlock(h, "head.weight", lora: nil)[0].asType(.float32)
            eval(logits, h)
            let kl = RwkvqModelTests.kl, rel = RwkvqModelTests.relErr
            let toRef = kl(f.ref["logits_ref"]!, logits)
            let toGemm = kl(r.quant["logits_quant_gemm"]!, logits)
            let toKern = kl(r.quant["logits_quant"]!, logits)
            let eH = rel(h[0], r.quant["hidden_quant"]!)
            let eW = rel(st.wkv[0..., 0], r.quant["state_wkv"]!)
            let eT = rel(st.tmixShift[0..., 0, 0], r.quant["state_tmix"]!)
            let eC = rel(st.cmixShift[0..., 0, 0], r.quant["state_cmix"]!)
            print(String(format: "ЭТАЛОН-QUANT %@ параллельный: KL(ref‖порт) %.3e при KL(ref‖quant) %.3e матмулом и %.3e ядрами; KL(quant‖порт) %.3e матмулом, %.3e ядрами",
                         preset, toRef, r.klRefQuantGemm, r.klRefQuant, toGemm, toKern))
            print(String(format: "ЭТАЛОН-QUANT %@: скрытое к quant %.3e; состояние wkv %.3e, tmix %.3e, cmix %.3e",
                         preset, eH, eW, eT, eC))
            // к точному счёту — не дальше эталона на том же пути (запас на шум fp16)
            XCTAssertLessThan(toRef, 1.5 * r.klRefQuantGemm, "\(preset): порт дальше от точного счёта, чем rwkv-quant матмулом")
            // к самому эталону: два независимых шумных счёта одной модели
            // расходятся примерно на сумму своих расстояний до точного
            XCTAssertLessThan(toGemm, 1.25 * (r.klRefQuantGemm + toRef), preset)
            XCTAssertLessThan(toKern, 1e-5, preset)
            XCTAssertEqual((argMax(logits, axis: -1) .!= argMax(r.quant["logits_quant"]!, axis: -1))
                .asType(.int32).sum().item(Int.self), 0, "\(preset): top-1 разошёлся с rwkv-quant")
            XCTAssertLessThan(eH, 5e-3, "\(preset): скрытое состояние"); XCTAssertLessThan(eW, 5e-3, preset)
            XCTAssertLessThan(eT, 5e-3, preset); XCTAssertLessThan(eC, 5e-3, preset)
            // разрыв на коротких — факт, а не допуск: когда порт его закроет,
            // эта проверка упадёт и напомнит убрать оговорку
            XCTAssertGreaterThan(toRef, r.klRefQuant, "\(preset): порт сравнялся с ядрами эталона на коротких — обновить оговорку")
        }
    }

    /// Декод по токену с состояния после промпта. У REDUCTION эталон и порт
    /// считают одно. У COMPRESSION эталон по умолчанию считает низкоранговые
    /// ветки КВАНТОВАННЫМИ (lora_q, 8 бит; +0.012% ppl по его замеру) и
    /// стоит от точного счёта в 50 раз дальше порта: порт веток не квантует.
    /// Поэтому там требуется не близость к эталону, а то, что порт ТОЧНЕЕ.
    func testStepDecodeAgainstRwkvQuant() throws {
        for preset in RwkvqFileTests.presets {
            let r = try Self.reference(preset)
            let f = r.metal
            let bb = try X070Backbone.load(fromRwkvq: f.path)
            let (_, st) = bb.bodyWithState(f.ref["input_ids"]!)
            let dec = f.ref["decode_ids"]!
            var rs = RWKVState(st, row: 0)
            var rows: [MLXArray] = []
            for i in 0 ..< dec.shape[1] { rows.append(bb.step(dec[0, i].item(Int.self), state: &rs).asType(.float32)) }
            let got = stacked(rows, axis: 0)
            let kl = RwkvqModelTests.kl
            let toRef = kl(f.ref["decode_logits_ref"]!, got)
            let toQuant = kl(r.quant["decode_logits_quant"]!, got)
            print(String(format: "ЭТАЛОН-QUANT %@ декод: KL(ref‖порт) %.3e, KL(quant‖порт) %.3e, KL(ref‖quant) %.3e, ветки эталона квантованы: %@",
                         preset, toRef, toQuant, r.klRefQuantDecode, r.loraQDecode ? "да" : "нет"))
            XCTAssertEqual((argMax(got, axis: -1) .!= argMax(r.quant["decode_logits_quant"]!, axis: -1))
                .asType(.int32).sum().item(Int.self), 0, "\(preset): top-1 декода разошёлся с rwkv-quant")
            if r.loraQDecode {
                XCTAssertLessThan(toRef, 0.2 * r.klRefQuantDecode, "\(preset): порт без квантованных веток обязан быть точнее эталона")
                XCTAssertLessThan(toQuant, 1.5 * r.klRefQuantDecode, preset)
            } else {
                XCTAssertLessThan(toRef, 3 * r.klRefQuantDecode + 1e-6, preset)
                XCTAssertLessThan(toQuant, 1e-5, preset)
            }
        }
    }
}
