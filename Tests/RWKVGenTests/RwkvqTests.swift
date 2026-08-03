import XCTest
import MLX
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Деквантизация .rwkvq — БИТ-В-БИТ против эталонного ядра rwkv-metal.
//
//  Здесь допуск неуместен принципиально. Пресет REDUCTION откалиброван под
//  конкретную арифметику деквантизации (финальная сборка в fp32, не в half),
//  и «почти совпадает» означает, что калибровка перестала измерять то, что
//  измеряла. Поэтому единственный осмысленный критерий — ровно ноль.
//
//  Фикстуры (в репозиторий не кладутся, ~150 МБ):
//      # 1. настоящий квантованный файл
//      python -c "from rwkv_quant import quantize; \
//                 quantize('model.pth', '/tmp/0.1B_reduction.rwkvq', preset='reduction')"
//      # 2. MLX-сайдкар
//      python -m rwkv_quant.formats.export_mlx /tmp/0.1B_reduction.rwkvq \
//             ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx
//      # 3. эталон деквантизации
//      cd ~/Develop/rwkv-metal && .venv/bin/python \
//          ~/Develop/SwiftRWKV/Scripts/dump_rwkvq_reference.py \
//          --sidecar ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx \
//          --out     ~/Develop/SwiftRWKV/.testdata/rwkvq_dequant_ref.safetensors
// ───────────────────────────────────────────────────────────────────────

final class RwkvqTests: XCTestCase {

    static var sidecarPath: String {
        ProcessInfo.processInfo.environment["RWKV_RWKVQ_SIDECAR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx")
                .path
    }

    static var referencePath: String {
        ProcessInfo.processInfo.environment["RWKV_RWKVQ_REFERENCE"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkvq_dequant_ref.safetensors")
                .path
    }

    func loadSidecar() throws -> RwkvqSidecar? {
        guard FileManager.default.fileExists(atPath: Self.sidecarPath + ".safetensors")
        else { return nil }
        return try RwkvqSidecar(path: Self.sidecarPath)
    }

    func requireSidecar() throws -> RwkvqSidecar {
        let s = try loadSidecar()
        try XCTSkipIf(s == nil, "нет сайдкара .rwkvq_mlx — тест пропущен "
                      + "(см. шапку файла, как собрать фикстуры)")
        return s!
    }

    // ── Манифест ─────────────────────────────────────────────────────

    func testSidecarLoadsManifest() throws {
        let s = try requireSidecar()
        XCTAssertEqual(s.naming, "world")
        XCTAssertGreaterThan(s.nLayer, 0)
        XCTAssertGreaterThan(s.tensors.count, 0)

        // экспортируются только sb6-тензоры; низкоранговые lora — НЕ они
        XCTAssertTrue(s.contains("blocks.0.att.key.weight"))
        XCTAssertTrue(s.contains("emb.weight"))
        XCTAssertFalse(s.contains("blocks.0.att.w2"),
                       "w_lora не должен попадать в сайдкар: по конвенции "
                       + "QLoRA-базы низкоранговые матрицы остаются в fp")

        let info = s.tensors["blocks.0.att.key.weight"]!
        XCTAssertEqual(info.shape, [s.nEmbd, s.nEmbd])
        XCTAssertEqual(info.groupSize, 32)
        XCTAssertEqual(info.superBlock, 8)
        XCTAssertEqual(info.bits, 6)
        XCTAssertEqual(info.xbits, 2, "int6 = 4 бита ниббла + 2 битплоскости")
    }

    func testMissingFilesReported() {
        XCTAssertThrowsError(try RwkvqSidecar(path: "/nonexistent/sidecar")) { e in
            guard case RwkvqError.missingFile = e else {
                return XCTFail("ожидалась missingFile, получено \(e)")
            }
        }
    }

    // ── Главное: бит-в-бит ───────────────────────────────────────────

    /// Деквантизация обязана совпасть с эталоном ТОЧНО, на всех формах.
    func testDequantIsBitExactAgainstReference() throws {
        let s = try requireSidecar()
        try XCTSkipIf(!FileManager.default.fileExists(atPath: Self.referencePath),
                      "нет эталона деквантизации — тест пропущен")
        let ref = try loadArrays(url: URL(fileURLWithPath: Self.referencePath))

        var checked = 0
        for (refKey, expected) in ref {
            guard refKey.hasPrefix("dequant/") else { continue }
            let key = String(refKey.dropFirst("dequant/".count))
            guard s.contains(key) else {
                XCTFail("эталон содержит \(key), а сайдкар — нет")
                continue
            }
            let got = try s.dequantize(key)
            eval(got, expected)
            XCTAssertEqual(got.shape, expected.shape, "\(key): другая форма")

            let diff = MLX.abs(got - expected.asType(.float32)).max().item(Float.self)
            XCTAssertEqual(diff, 0,
                           "\(key): деквантизация разошлась с эталоном на \(diff) "
                           + "— калибровка пресета рассчитана на точное совпадение")
            checked += 1
        }
        XCTAssertGreaterThanOrEqual(checked, 4,
                                    "проверено слишком мало тензоров: \(checked)")
        print("БИТ-В-БИТ: сверено тензоров \(checked)")
    }

    /// Деквантизация детерминирована — повторный вызов даёт то же самое.
    func testDequantIsDeterministic() throws {
        let s = try requireSidecar()
        let key = "blocks.0.att.key.weight"
        let a = try s.dequantize(key), b = try s.dequantize(key)
        eval(a, b)
        XCTAssertEqual(MLX.abs(a - b).max().item(Float.self), 0)
    }

    /// Разные формы идут через РАЗНЫЕ скомпилированные ядра (геометрия зашита
    /// в исходник константами). Прогоняем прямоугольные тензоры и огромный
    /// head: индексация, работающая только при малом OUT, здесь и всплывает.
    func testDequantHandlesAllShapes() throws {
        let s = try requireSidecar()
        for key in ["blocks.0.att.key.weight",      // квадрат D×D
                    "blocks.0.ffn.key.weight",      // 4D×D
                    "blocks.0.ffn.value.weight",    // D×4D
                    "head.weight"] {                // V×D, V = 65536
            guard s.contains(key) else { continue }
            let info = s.tensors[key]!
            let w = try s.dequantize(key)
            eval(w)
            XCTAssertEqual(w.shape, [info.outFeatures, info.inFeatures])
            let m = MLX.abs(w).max().item(Float.self)
            XCTAssertTrue(m.isFinite && m > 0, "\(key): вырожденный результат")
        }
    }

    /// База должна оставаться сжатой: сумма упакованных буферов существенно
    /// меньше, чем те же веса в плотном виде.
    func testPackedBaseIsActuallySmaller() throws {
        let s = try requireSidecar()
        let denseBytes = s.tensors.values.reduce(0) {
            $0 + $1.outFeatures * $1.inFeatures * 2      // bf16
        }
        let ratio = Double(denseBytes) / Double(s.packedBytes)
        print("СЖАТИЕ базы: \(s.packedBytes / 1_000_000) МБ упаковано против "
              + "\(denseBytes / 1_000_000) МБ bf16 = \(String(format: "%.2f", ratio))×")
        XCTAssertGreaterThan(ratio, 2.0,
                             "сжатие всего \(ratio)× — похоже, база не упакована")
    }

    // ── Имена ────────────────────────────────────────────────────────

    /// Имена в сайдкаре — world; SwiftRWKV использует x070. Таблица обязана
    /// разворачиваться в обе стороны без потерь.
    func testNamingRoundTrip() {
        let x070 = ["blocks.0.tmix.r_proj.weight", "blocks.3.tmix.k_proj.weight",
                    "blocks.7.tmix.v_proj.weight", "blocks.1.tmix.o_proj.weight",
                    "blocks.2.cmix.key.weight", "blocks.2.cmix.value.weight",
                    "emb.weight", "head.weight"]
        for key in x070 {
            guard let world = RwkvqNaming.worldKey(forX070: key) else {
                return XCTFail("нет отображения для \(key)")
            }
            XCTAssertEqual(RwkvqNaming.x070Key(forWorld: world), key,
                           "\(key) → \(world) → не вернулось обратно")
        }
        XCTAssertEqual(RwkvqNaming.worldKey(forX070: "blocks.0.tmix.k_proj.weight"),
                       "blocks.0.att.key.weight")
        XCTAssertEqual(RwkvqNaming.worldKey(forX070: "blocks.0.cmix.key.weight"),
                       "blocks.0.ffn.key.weight")
        XCTAssertNil(RwkvqNaming.worldKey(forX070: "blocks.0.tmix.ln_x.weight"))
        XCTAssertNil(RwkvqNaming.worldKey(forX070: "ln_out.weight"))
    }

    /// Отображение имён обязано попадать в реальный сайдкар, а не только
    /// в собственные ожидания.
    func testNamingMatchesActualSidecar() throws {
        let s = try requireSidecar()
        for layer in 0 ..< Swift.min(3, s.nLayer) {
            for name in ["r_proj", "k_proj", "v_proj", "o_proj"] {
                let key = "blocks.\(layer).tmix.\(name).weight"
                guard let world = RwkvqNaming.worldKey(forX070: key) else {
                    return XCTFail("нет отображения для \(key)")
                }
                XCTAssertTrue(s.contains(world),
                              "\(key) → \(world), но такого тензора в сайдкаре нет")
            }
            for name in ["key", "value"] {
                let world = RwkvqNaming.worldKey(forX070: "blocks.\(layer).cmix.\(name).weight")!
                XCTAssertTrue(s.contains(world), "нет \(world) в сайдкаре")
            }
        }
    }
}
