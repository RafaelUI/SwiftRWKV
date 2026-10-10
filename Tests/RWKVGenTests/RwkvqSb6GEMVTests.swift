import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  GEMV-ядро раскладки sb6: путь декода COMPRESSION без плотной матрицы
//  и без родного контейнера MLX.
//
//  Эталон — само ядро Metal-бэкенда rwkv-quant (GwQuantLinear._gemv1, k3):
//      cd ~/Develop/rwkv-metal && .venv/bin/python \
//          ~/Develop/SwiftRWKV/Scripts/dump_rwkvq2_gemv_sb6.py
//  -> .testdata/rwkvq2_compression_gemv.safetensors
//  Исходник и порядок суммирования перенесены дословно — требуется
//  РАВЕНСТВО. Вторая опора, от соседнего репозитория не зависящая: точный
//  матмул с весами своего декванта в fp32, на всех sb6-тензорах.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqSb6GEMVTests: XCTestCase {

    static func reference() throws -> [String: MLXArray] {
        let p = RwkvqFileTests.testdata.appendingPathComponent("rwkvq2_compression_gemv.safetensors")
        try XCTSkipIf(!FileManager.default.fileExists(atPath: p.path),
                      "нет эталона GEMV \(p.lastPathComponent) — тест пропущен (см. шапку файла)")
        return try loadArrays(url: p)
    }

    func testKernelIsBitExactAgainstRwkvQuant() throws {
        let s = try RwkvqFileTests.base("compression")
        let ref = try Self.reference()
        var seen: Set<String> = []
        for name in ref.keys.sorted() where name.hasPrefix("x/") {
            let tag = String(name.dropFirst(2))
            let key = tag.replacingOccurrences(of: "/", with: ".")
            let info = try XCTUnwrap(s.tensors[key], key)
            let want = try XCTUnwrap(ref["y/\(tag)"])
            let got = try XCTUnwrap(s.sb6GEMV(key, ref[name]!), "\(key): ядро не взяло форму")
            XCTAssertEqual(got.shape, want.shape, key)
            XCTAssertEqual(got.dtype, .float32)
            XCTAssertEqual(RwkvqFileTests.mismatches(got, want), 0,
                           "\(key): выход ядра разошёлся с rwkv-quant")
            seen.insert("\(info.bits)/\(info.outFeatures)x\(info.inFeatures)")
        }
        XCTAssertGreaterThanOrEqual(seen.count, 8, "эталон покрыл мало сочетаний: \(seen.sorted())")
        XCTAssertEqual(Set(seen.map { $0.prefix(1) }), ["4", "5", "6"], "эталон обязан покрыть все битности")
    }

    /// Против точного матмула с весами декванта (fp32) — на ВСЕХ sb6-тензорах
    /// пресета, включая таблицу эмбеддингов. Допуск — только на порядок
    /// суммирования в fp32. И другая раскладка потоков (та, что таблица
    /// конфигураций ставит на больших моделях) обязана дать тот же ответ.
    func testKernelMatchesExactMatmulOnAllTensors() throws {
        let s = try RwkvqFileTests.base("compression")
        MLXRandom.seed(17)
        var n = 0
        var worst: Float = 0
        for key in s.keys {
            guard let info = s.tensors[key], info.kind == .sb6 else { continue }
            let x = MLXRandom.normal([1, info.inFeatures])
            let got = try XCTUnwrap(s.sb6GEMV(key, x), key)
            let w = try s.dequantize(key, dtype: .float32)
            let want = matmul(x, w.transposed())
            let e = RwkvqModelTests.relErr(got, want)
            worst = max(worst, e)
            XCTAssertLessThan(e, 2e-5, "\(key): GEMV разошёлся с точным матмулом на \(e)")
            if n % 12 == 0 {
                let other = rwkvqSb6GEMV(x, qblk: s.arrays["\(key)::qblk"]!, qsqm: s.arrays["\(key)::qsqm"]!,
                                         ddm: s.arrays["\(key)::ddm"]!, outFeatures: info.outFeatures,
                                         inFeatures: info.inFeatures, xbits: info.xbits, config: (2, 2))
                XCTAssertLessThan(RwkvqModelTests.relErr(other, want), 2e-5, "\(key): раскладка (2, 2)")
            }
            n += 1
        }
        XCTAssertEqual(n, 73)
        print(String(format: "GEMV sb6 против точного матмула: 73 тензора, худшая отн. ошибка %.2e", worst))
    }

    /// Таблица раскладок потоков — как в rwkv-quant.
    func testThreadConfigTableIsTheReferenceOne() {
        XCTAssertTrue(rwkvqSb6GEMVConfig(outFeatures: 65536, inFeatures: 2048, xbits: 0) == (4, 4))
        XCTAssertTrue(rwkvqSb6GEMVConfig(outFeatures: 8192, inFeatures: 2048, xbits: 0) == (2, 2))
        XCTAssertTrue(rwkvqSb6GEMVConfig(outFeatures: 8192, inFeatures: 2048, xbits: 1) == (4, 4))
        XCTAssertTrue(rwkvqSb6GEMVConfig(outFeatures: 2048, inFeatures: 8192, xbits: 0) == (4, 4))
    }

    /// Чего ядро не берёт — возвращает nil, а не считает мусор.
    func testKernelDeclinesWhatItCannotTake() throws {
        let s = try RwkvqFileTests.base("compression")
        let key = "blocks.1.att.key.weight"
        XCTAssertNotNil(s.sb6GEMV(key, MLXArray.zeros([1, 768])))
        XCTAssertNil(s.sb6GEMV(key, MLXArray.zeros([2, 768])), "больше одной строки")
        XCTAssertNil(s.sb6GEMV(key, MLXArray.zeros([1, 512])), "чужая ширина входа")
        XCTAssertNil(s.sb6GEMV("blocks.0.att.w1", MLXArray.zeros([1, 768])), "asym — не sb6")
        XCTAssertNil(s.sb6GEMV("blocks.0.att.output.weight", MLXArray.zeros([1, 768])), "плотная цель")
        XCTAssertNil(try RwkvqFileTests.base("reduction").sb6GEMV(key, MLXArray.zeros([1, 768])), "sym")
        XCTAssertFalse(rwkvqSb6GEMVApplies(outFeatures: 24, inFeatures: 256, xbits: 0, superBlock: 8, rows: 1))
        XCTAssertFalse(rwkvqSb6GEMVApplies(outFeatures: 16, inFeatures: 96, xbits: 0, superBlock: 8, rows: 1))
        XCTAssertFalse(rwkvqSb6GEMVApplies(outFeatures: 16, inFeatures: 256, xbits: 0, superBlock: 4, rows: 1))
    }
}
