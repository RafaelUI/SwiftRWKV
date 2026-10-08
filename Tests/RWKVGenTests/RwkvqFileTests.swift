import XCTest
import MLX
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Чтение .rwkvq НАПРЯМУЮ и деквантизация всех раскладок нынешних пресетов.
//
//  Эталон — нормативный codec rwkv-quant (numpy), а не ядро rwkv-metal:
//  порт сверяется с ФОРМАТОМ, а не с соседним портом.
//
//  Фикстуры (вне git):
//      cd ~/Develop/rwkv-metal && .venv/bin/python tools/gen_rwkvq_fixtures.py
//  кладёт в .testdata/ rwkvq2_<preset>_dequant.safetensors (деквант codec,
//  fp32, по два тензора на каждое сочетание раскладки, битности и формы;
//  emb и head — строки [0:1024]) и rwkvq2_<preset>.json с путём исходного
//  .rwkvq. Сайдкары тех же файлов — для сверки интерлива:
//      python -m rwkv_quant.formats.export_mlx <файл>.rwkvq \
//          ~/Develop/SwiftRWKV/.testdata/rwkvq2_<preset>.rwkvq_mlx
// ───────────────────────────────────────────────────────────────────────

final class RwkvqFileTests: XCTestCase {

    static let presets = ["reduction", "compression"]
    static let testdata = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Develop/SwiftRWKV/.testdata")

    /// Путь исходного .rwkvq — из json эталона: эталон и файл обязаны быть
    /// одной парой, и выбирать файл по другому признаку значило бы сверять
    /// один файл с эталоном другого.
    static func sourcePath(_ preset: String) throws -> String {
        let js = testdata.appendingPathComponent("rwkvq2_\(preset).json")
        try XCTSkipIf(!FileManager.default.fileExists(atPath: js.path),
                      "нет rwkvq2_\(preset).json — тест пропущен (см. шапку файла)")
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: js)) as? [String: Any]
        let src = try XCTUnwrap(root?["source"] as? String)
        try XCTSkipIf(!FileManager.default.fileExists(atPath: src),
                      "нет исходного файла \(src) — тест пропущен")
        return src
    }

    static var cache: [String: RwkvqSidecar] = [:]
    static func base(_ preset: String) throws -> RwkvqSidecar {
        if let s = cache[preset] { return s }
        let s = try RwkvqSidecar(rwkvq: try sourcePath(preset))
        cache[preset] = s
        return s
    }

    static func reference(_ preset: String) throws -> [String: MLXArray] {
        let p = testdata.appendingPathComponent("rwkvq2_\(preset)_dequant.safetensors")
        try XCTSkipIf(!FileManager.default.fileExists(atPath: p.path),
                      "нет эталона деквантизации \(p.lastPathComponent) — тест пропущен")
        return try loadArrays(url: p)
    }

    /// "emb.weight[0:1024]" → ("emb.weight", 1024); прочие — (ключ, nil).
    static func split(_ refKey: String) -> (String, Int?) {
        guard let open = refKey.firstIndex(of: "["), refKey.hasSuffix("]"),
              let colon = refKey.firstIndex(of: ":") else { return (refKey, nil) }
        let rows = Int(refKey[refKey.index(after: colon) ..< refKey.index(before: refKey.endIndex)])
        return (String(refKey[..<open]), rows)
    }

    static func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a .!= b).asType(.int32).sum().item(Int.self)
    }

    /// Маска [OUT, IN] вырожденных блоков sb6: half(qs·d) == 0. У них codec
    /// держит scale на зажиме 1e-8, а ядро (как и ядро rwkv-metal) — ноль.
    static func sb6DegenerateMask(_ s: RwkvqSidecar, _ key: String) -> MLXArray {
        let info = s.tensors[key]!
        let OUT = info.outFeatures, NB = info.inFeatures / 32
        let qs = s.arrays["\(key)::qsqm"]!.reshaped([OUT, NB, 2])[0..., 0..., 0].asType(.float32)
        let d = repeated(s.arrays["\(key)::ddm"]!.reshaped([OUT, NB / info.superBlock, 2])[0..., 0..., 0]
            .asType(.float32), count: info.superBlock, axis: 1)
        let zero = (qs * d).asType(.float16) .== MLXArray(Float(0)).asType(.float16)
        return repeated(zero, count: 32, axis: 1)
    }

    // ── Контейнер и манифест ─────────────────────────────────────────

    func testReadsManifestOfBothPresets() throws {
        let expect: [String: [RwkvqKind: Int]] = [
            "reduction": [.sym: 73, .asym: 72, .rtn: 24, .dense: 233],
            "compression": [.sb6: 73, .asym: 72, .rtn: 24, .dense: 233],
        ]
        for preset in Self.presets {
            let s = try Self.base(preset)
            XCTAssertEqual(s.nLayer, 12); XCTAssertEqual(s.nEmbd, 768)
            XCTAssertEqual(s.headSize, 64); XCTAssertEqual(s.vocabSize, 65536)
            XCTAssertEqual(s.naming, "world")
            var counts: [RwkvqKind: Int] = [:]
            for info in s.tensors.values { counts[info.kind, default: 0] += 1 }
            XCTAssertEqual(counts, expect[preset], "\(preset): состав раскладок")
            // плотная цель: o_proj слоя 0 в обоих пресетах лежит как есть
            XCTAssertEqual(s.tensors["blocks.0.att.output.weight"]?.kind, .dense, preset)
            XCTAssertEqual(s.arrays["blocks.0.att.output.weight::dense"]?.dtype, .bfloat16)
            XCTAssertEqual(s.tensors["blocks.1.att.output.weight"]?.isQuantizedBase, true)
        }
        let r = try Self.base("reduction")
        XCTAssertEqual(r.tensors["emb.weight"]?.bits, 8)
        XCTAssertEqual(r.tensors["blocks.0.ffn.key.weight"]?.bits, 6)
        XCTAssertEqual(r.arrays["blocks.0.ffn.key.weight::qblk"]?.shape, [3072, 768 / 32 * 24])
        let c = try Self.base("compression")
        XCTAssertEqual(c.tensors["emb.weight"]?.xbits, 2)
        XCTAssertEqual(c.tensors["head.weight"]?.xbits, 0)
    }

    /// Ориентация low-rank веток — из манифеста. В нынешних файлах они уже
    /// лежат [out, in] и поворота не требуют; таблица имён остаётся только
    /// для манифестов без поля.
    func testOrientationComesFromManifest() throws {
        for preset in Self.presets {
            let s = try Self.base(preset)
            for name in ["w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"] {
                let info = try XCTUnwrap(s.tensors["blocks.3.att.\(name)"])
                XCTAssertFalse(info.transposed, "\(preset) \(name): файл хранит [out, in]")
            }
            XCTAssertEqual(s.tensors["blocks.3.att.w1"]?.shape, [64, 768])
            XCTAssertEqual(s.tensors["blocks.3.att.w2"]?.shape, [768, 64])
        }
        XCTAssertTrue(RwkvqNaming.isRawLoRAWorld("blocks.11.att.g2"))
        XCTAssertTrue(RwkvqNaming.isRawLoRAWorld("blocks.0.att.w1"))
        XCTAssertFalse(RwkvqNaming.isRawLoRAWorld("blocks.0.att.w0"))
        XCTAssertFalse(RwkvqNaming.isRawLoRAWorld("blocks.0.att.key.weight"))
        XCTAssertFalse(RwkvqNaming.isRawLoRAWorld("blocks.0.ffn.w1"))
    }

    func testRejectsForeignContainers() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rwkvq-file-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // pickle-контейнер прежней эры (zip от torch.save)
        let zip = dir.appendingPathComponent("old.rwkvq")
        try Data([0x50, 0x4B, 0x03, 0x04] + [UInt8](repeating: 0, count: 64)).write(to: zip)
        XCTAssertThrowsError(try RwkvqSidecar(rwkvq: zip.path)) {
            XCTAssertTrue("\($0)".contains("pickle"), "\($0)")
        }
        // safetensors без манифеста rwkvq
        let plain = dir.appendingPathComponent("plain.safetensors")
        try save(arrays: ["x": MLXArray([Float(1), 2, 3])], url: plain)
        XCTAssertThrowsError(try RwkvqSidecar(rwkvq: plain.path)) {
            XCTAssertTrue("\($0)".contains("не .rwkvq"), "\($0)")
        }
        // обрезанный файл: заголовок обещает больше, чем есть
        let src = try Self.sourcePath("compression")
        let cut = dir.appendingPathComponent("cut.rwkvq")
        let fh = try FileHandle(forReadingFrom: URL(fileURLWithPath: src))
        try fh.read(upToCount: 1 << 20)!.write(to: cut)
        try fh.close()
        XCTAssertThrowsError(try RwkvqSidecar(rwkvq: cut.path))
        // нет файла
        XCTAssertThrowsError(try RwkvqSidecar(rwkvq: dir.appendingPathComponent("no.rwkvq").path))
    }

    // ── Главное: деквант против codec ────────────────────────────────

    /// fp32 обязан совпасть с codec ТОЧНО на каждой раскладке и форме;
    /// fp16 — с fp16(эталона). Единственное допущенное расхождение —
    /// вырожденные блоки sb6 (зажим 1e-8, см. codec.sb6_to_mlx_affine):
    /// оно считается, печатается и ограничено 63·1e-8.
    func testDequantMatchesCodecOnEveryLayout() throws {
        for preset in Self.presets {
            let s = try Self.base(preset)
            let ref = try Self.reference(preset)
            var seen: [String: Int] = [:]
            var clamped = 0
            for (refKey, expectedAll) in ref.sorted(by: { $0.key < $1.key }) {
                let (key, rows) = Self.split(refKey)
                let info = try XCTUnwrap(s.tensors[key], "\(preset): в файле нет \(key)")
                for dtype in [DType.float32, .float16] {
                    var got = try s.dequantize(key, dtype: dtype)
                    XCTAssertEqual(got.dtype, dtype, "\(key): тип выхода")
                    if let rows { got = got[0 ..< rows] }
                    let expected = expectedAll.asType(dtype)
                    XCTAssertEqual(got.shape, expected.shape, "\(preset) \(key): форма")
                    let bad = got .!= expected
                    let n = bad.asType(.int32).sum().item(Int.self)
                    if info.kind == .sb6 {
                        var mask = Self.sb6DegenerateMask(s, key)
                        if let rows { mask = mask[0 ..< rows] }
                        let outside = (bad .&& .!mask).asType(.int32).sum().item(Int.self)
                        XCTAssertEqual(outside, 0,
                            "\(preset) \(key) [\(dtype)]: \(outside) расхождений ВНЕ вырожденных блоков")
                        let worst = MLX.abs(got.asType(.float32) - expectedAll).max().item(Float.self)
                        if n > 0 {
                            XCTAssertLessThanOrEqual(worst, dtype == .float32 ? 6.3e-7 : 1e-3,
                                                     "\(preset) \(key): зажим дал \(worst)")
                        }
                        if dtype == .float32 { clamped += n }
                    } else {
                        XCTAssertEqual(n, 0,
                            "\(preset) \(key) [\(dtype)]: \(n) элементов разошлись с codec")
                    }
                }
                seen["\(info.kind.rawValue)/\(info.bits)", default: 0] += 1
            }
            print("ДЕКВАНТ \(preset): \(seen.sorted { $0.key < $1.key }), "
                  + "в вырожденных блоках sb6 расходится \(clamped) эл.")
            let want = preset == "reduction"
                ? ["asym/6", "dense/16", "rtn/8", "sym/6", "sym/8"]
                : ["asym/6", "dense/16", "rtn/8", "sb6/4", "sb6/5", "sb6/6"]
            for w in want {
                XCTAssertGreaterThanOrEqual(seen[w] ?? 0, 1, "\(preset): эталон не покрыл \(w)")
            }
        }
    }

    /// Ядро sym против цепочки MLX-операций — на ВСЕХ sym-тензорах файла, а
    /// не только на тех восьми, что попали в эталон. bf16 — одно округление
    /// fp32-результата.
    func testSymKernelEqualsReferenceChainOnAllTensors() throws {
        let s = try Self.base("reduction")
        var n = [6: 0, 8: 0]
        for key in s.keys {
            guard let info = s.tensors[key], info.kind == .sym else { continue }
            let chain = rwkvqDequantSymReference(
                qblk: s.arrays["\(key)::qblk"]!, qs: s.arrays["\(key)::qs"]!,
                d: s.arrays["\(key)::d"]!, outFeatures: info.outFeatures,
                inFeatures: info.inFeatures, bits: info.bits)
            XCTAssertEqual(Self.mismatches(try s.dequantize(key, dtype: .float32), chain), 0, key)
            XCTAssertEqual(Self.mismatches(try s.dequantize(key, dtype: .float16),
                                           chain.asType(.float16)), 0, "\(key) fp16")
            XCTAssertEqual(Self.mismatches(try s.dequantize(key, dtype: .bfloat16),
                                           chain.asType(.bfloat16)), 0, "\(key) bf16")
            XCTAssertGreaterThan(MLX.abs(chain).max().item(Float.self), 0, "\(key): нули")
            n[info.bits, default: 0] += 1
        }
        XCTAssertEqual(n[8], 49); XCTAssertEqual(n[6], 24)
    }

    /// Форма вне суперблока 256 идёт цепочкой операций, а не падает и не
    /// считает мусор: синтетический тензор IN=48 (3 блока) сверяется с
    /// расчётом вручную.
    func testSymOffSuperblockShapeUsesChain() throws {
        let OUT = 2, IN = 48
        let codes: [Int8] = (0 ..< OUT * IN).map { Int8(truncatingIfNeeded: $0 * 37 - 90) }
        let qs: [Int8] = [3, -5, 7, 11, -2, 1]
        let d: [Float] = [0.0123, 0.25]
        let qblk = MLXArray(codes.map { UInt8(bitPattern: $0) }, [OUT, IN])
        let qsA = MLXArray(qs.map { UInt8(bitPattern: $0) }, [OUT, 3])
        let dA = MLXArray(d, [OUT, 1]).asType(.float16)
        let got = rwkvqDequantSym(qblk: qblk, qs: qsA, d: dA, outFeatures: OUT,
                                  inFeatures: IN, bits: 8, dtype: .float32)
        let dh = dA.asType(.float32).asArray(Float.self)
        var want = [Float]()
        for r in 0 ..< OUT {
            for c in 0 ..< IN {
                let scale = Float(Float16(Float(qs[r * 3 + c / 16]) * dh[r]))
                want.append(Float(codes[r * IN + c]) * scale)
            }
        }
        XCTAssertEqual(got.asArray(Float.self), want)
    }

    // ── Интерлив: прямое чтение против сайдкара ──────────────────────

    /// Буферы, построенные из .rwkvq при загрузке, обязаны быть РАВНЫ тем,
    /// что кладёт export_mlx: сайдкар — необязательный кеш, а не другой
    /// формат. Заодно это чтение сайдкара нынешних пресетов (sym, dense).
    func testDirectLayoutEqualsSidecar() throws {
        for preset in Self.presets {
            let p = Self.testdata.appendingPathComponent("rwkvq2_\(preset).rwkvq_mlx").path
            try XCTSkipIf(!FileManager.default.fileExists(atPath: p + ".safetensors"),
                          "нет сайдкара rwkvq2_\(preset).rwkvq_mlx — тест пропущен")
            let side = try RwkvqSidecar.open(p)
            let direct = try Self.base(preset)
            XCTAssertEqual(Set(side.tensors.keys), Set(direct.tensors.keys), preset)
            XCTAssertEqual(Set(side.arrays.keys), Set(direct.arrays.keys), "\(preset): состав буферов")
            for (key, a) in direct.tensors {
                let b = try XCTUnwrap(side.tensors[key])
                XCTAssertEqual(a.kind, b.kind, key); XCTAssertEqual(a.shape, b.shape, key)
                XCTAssertEqual(a.bits, b.bits, key); XCTAssertEqual(a.xbits, b.xbits, key)
                XCTAssertEqual(a.groupSize, b.groupSize, key)
                XCTAssertEqual(a.transposed, b.transposed, key)
            }
            var bad: [String] = []
            for (name, a) in direct.arrays {
                guard let b = side.arrays[name] else { continue }
                if a.shape != b.shape || a.dtype != b.dtype || Self.mismatches(a, b) != 0 {
                    bad.append(name)
                }
            }
            XCTAssertEqual(bad, [], "\(preset): буферы расходятся с сайдкаром")
            // и сам деквант через оба входа — один и тот же
            for key in ["blocks.5.att.key.weight", "blocks.5.ffn.value.weight", "blocks.5.att.w1"] {
                XCTAssertEqual(Self.mismatches(try side.dequantize(key), try direct.dequantize(key)), 0)
            }
        }
    }

    /// `open` выбирает вход по тому, что лежит на диске.
    func testOpenPicksTheSource() throws {
        let src = try Self.sourcePath("reduction")
        XCTAssertEqual(try RwkvqSidecar.open(src).tensors.count, try Self.base("reduction").tensors.count)
        XCTAssertThrowsError(try RwkvqSidecar.open("/nonexistent/base"))
    }
}
