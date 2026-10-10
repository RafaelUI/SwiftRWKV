import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Загрузка .rwkvq без всплеска памяти (`X070Backbone.load(fromRwkvq:)`):
//  файл читается по частям, emb в sym при восьми битах остаётся сжатым.
//
//  Что здесь проверяется:
//    • счёт НЕ изменился: модель потоковой загрузки побитно равна модели,
//      собранной прежним порядком (вся база в MLX, потом сборка);
//    • строки сжатой таблицы эмбеддинга побитно равны строкам плотной;
//    • памяти при загрузке занято не больше, чем остаётся после неё, —
//      ради этого всё и делалось, и без теста это свойство теряется молча:
//      любая лишняя копия в загрузчике ничего не ломает, кроме пика.
// ───────────────────────────────────────────────────────────────────────

final class RwkvqLoadTests: XCTestCase {

    static let ids = (0 ..< 40).map { ($0 * 7919 + 13) % 60000 + 100 }

    static func bad(_ a: MLXArray, _ b: MLXArray) -> Int {
        guard a.shape == b.shape else { return -1 }
        return RwkvqFileTests.mismatches(a.asType(.float32), b.asType(.float32))
    }

    /// Параллельный проход, префилл и шаги одной модели — всё, что видит
    /// приложение.
    static func outputs(_ bb: X070Backbone) -> [MLXArray] {
        let arr = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        var out = [bb(arr)]
        var st = RWKVState(cfg: bb.cfg)
        out.append(bb.prefill(Array(ids[0 ..< 21]), state: &st))
        var cold = RWKVState(cfg: bb.cfg)
        for t in 21 ..< 25 {
            out.append(bb.step(ids[t], state: &st))
            out.append(bb.step(ids[t], state: &cold))
        }
        out.append(st.wkv[bb.cfg.nLayer - 1])
        eval(out)
        return out
    }

    func testEmbeddingRowsEqualTheDenseTable() throws {
        let base = try RwkvqFileTests.base("reduction")
        let table = try XCTUnwrap(base.symRowTable("emb.weight"), "emb пресета REDUCTION — sym при восьми битах")
        let dense = try base.dequantize("emb.weight", dtype: .float16)
        XCTAssertEqual(table.rowCount, 65536)
        // края словаря, повторы и двумерная форма
        let flat: [Int32] = [0, 1, 255, 256, 65535, 65534, 32768, 1, 0, 40000, 12345, 7]
        for shape in [[12], [3, 4], [1, 12]] {
            let ids = MLXArray(flat, shape)
            let rows = table.rows(ids)
            XCTAssertEqual(rows.dtype, .float16)
            XCTAssertEqual(rows.shape, shape + [dense.shape[1]])
            XCTAssertEqual(Self.bad(rows, dense.take(ids, axis: 0)), 0, "строки таблицы, форма \(shape)")
        }
        // строки не нулевые и разные: совпадение не вырождено
        let two = table.rows(MLXArray([Int32(5), Int32(6)])).asType(.float32)
        XCTAssertGreaterThan(abs(two[0]).sum().item(Float.self), 0)
        XCTAssertGreaterThan(Self.bad(two[0], two[1]), 0)
        XCTAssertEqual(Self.bad(table.dense(), dense), 0, "таблица целиком")
        // COMPRESSION: emb в sb6, строчной выборки нет — отказ, а не подмена
        XCTAssertNil(try RwkvqFileTests.base("compression").symRowTable("emb.weight"))
        XCTAssertNil(base.symRowTable("blocks.0.att.w1"))
        // sym при шести битах: коды битплоскостями, строками не читаются
        let six = try XCTUnwrap(base.keys.first { base.tensors[$0]!.kind == .sym && base.tensors[$0]!.bits == 6 },
                                "в пресете нет sym при шести битах — проверка отказа пуста")
        XCTAssertNil(base.symRowTable(six))
    }

    /// Синтетическая таблица со ВСЕМИ значениями байта у кодов и масштабов и
    /// с масштабами суперблока обоих знаков: в настоящем emb масштабы блоков
    /// неотрицательны, и чтение их беззнаковыми там не видно (мутация не
    /// ловилась). Эталон — ядро декванта на тех же буферах.
    func testEmbeddingRowsOnEveryByteValue() throws {
        let rows = 16, IN = 512, NB = IN / 16
        let codes = MLXArray((0 ..< rows * IN).map { UInt8(truncatingIfNeeded: $0 &* 37 &+ $0 / IN) }, [rows, IN])
        let qs = MLXArray((0 ..< rows * NB).map { UInt8(truncatingIfNeeded: $0 &* 11 &+ 3) }, [rows, NB])
        let d = MLXArray((0 ..< rows * 2).map { Float($0 % 2 == 0 ? 1 : -1) * (0.001 + Float($0) * 0.0007) },
                         [rows, 2]).asType(.float16)
        XCTAssertGreaterThan((qs .>= 128).asType(.int32).sum().item(Int.self), 100)
        let table = RwkvqSymRowTable(codes: codes, qs: qs, d: d, rowCount: rows, inFeatures: IN)
        let dense = rwkvqDequantSym(qblk: codes, qs: qs, d: d, outFeatures: rows, inFeatures: IN,
                                    bits: 8, dtype: .float16)
        let ids = MLXArray((0 ..< rows).reversed().map { Int32($0) })
        XCTAssertEqual(Self.bad(table.rows(ids), dense.take(ids, axis: 0)), 0)
        XCTAssertGreaterThan(abs(dense.asType(.float32)).sum().item(Float.self), 1)
    }

    func testPartialReadKeepsTheManifestAndOnlyTheAskedBuffers() throws {
        for preset in RwkvqFileTests.presets {
            let full = try RwkvqFileTests.base(preset)
            let path = try RwkvqFileTests.sourcePath(preset)
            let want: Set<String> = ["head.weight", "blocks.3.att.key.weight", "blocks.0.att.w1", "ln_out.weight"]
            let part = try RwkvqSidecar(rwkvq: path, only: { want.contains($0) })
            XCTAssertEqual(part.tensors.count, full.tensors.count, "\(preset): манифест разобран не весь")
            let got = Set(part.arrays.keys.map { String($0[..<$0.range(of: "::")!.lowerBound]) })
            XCTAssertEqual(got, want, "\(preset): подняты буферы не тех тензоров")
            for (name, a) in part.arrays {
                let b = try XCTUnwrap(full.arrays[name], name)
                XCTAssertEqual(a.dtype, b.dtype, name)
                XCTAssertEqual(Self.bad(a, b), 0, "\(preset) \(name)")
            }
            for key in full.keys {
                let a = try XCTUnwrap(part.tensors[key]), b = try XCTUnwrap(full.tensors[key])
                XCTAssertEqual(a.kind, b.kind, key); XCTAssertEqual(a.shape, b.shape, key)
                XCTAssertEqual(a.bits, b.bits, key); XCTAssertEqual(a.xbits, b.xbits, key)
                XCTAssertEqual(a.packed, b.packed, key); XCTAssertEqual(a.transposed, b.transposed, key)
            }
            XCTAssertThrowsError(try part.dequantize("blocks.5.att.key.weight"))
            XCTAssertTrue(try RwkvqSidecar(rwkvq: path, only: { _ in false }).arrays.isEmpty)
        }
    }

    /// Чтение вперёд просится ровно для буферов запрошенных тензоров — и не
    /// просится при выключателе. Само ускорение (холодный файл 1.5B: 2.2–2.9
    /// → 1.2 с) меряется замером; тест стережёт, что подсказка не потеряна.
    func testReadAheadCoversExactlyTheAskedBuffers() throws {
        let path = try RwkvqFileTests.sourcePath("compression")
        let box = try RwkvqContainer(path: path)
        let want: Set<String> = ["head.weight", "blocks.3.att.key.weight", "blocks.0.att.w1"]
        let expected = box.names.filter { name in
            want.contains(String(name[..<name.range(of: "::")!.lowerBound])) && box.shape(name)!.reduce(1, *) > 0
        }.count
        XCTAssertGreaterThan(expected, 8)
        var before = RwkvqContainer.readAheadBuffers
        _ = try RwkvqSidecar(rwkvq: path, only: { want.contains($0) })
        XCTAssertEqual(RwkvqContainer.readAheadBuffers - before, expected)
        before = RwkvqContainer.readAheadBuffers
        RwkvqContainer.readAheadEnabled = false
        defer { RwkvqContainer.readAheadEnabled = true }
        _ = try RwkvqSidecar(rwkvq: path, only: { want.contains($0) })
        XCTAssertEqual(RwkvqContainer.readAheadBuffers, before, "выключатель чтения вперёд не действует")
    }

    /// Сведение r/k/v при чтении даёт те же буферы, что сведение готовой
    /// базы, и повторно не сводится.
    func testFusingAtReadEqualsFusingAfter() throws {
        let path = try RwkvqFileTests.sourcePath("reduction")
        let keys = ["receptance", "key", "value"].map { "blocks.2.att.\($0).weight" }
        let groups = ["blocks.2.att.rkv": keys]
        let atRead = try RwkvqSidecar(rwkvq: path, only: { keys.contains($0) }, fusingSym: groups)
        let after = try RwkvqFileTests.base("reduction").pruned(keeping: Set(keys)).fusingSym(groups)
        let a = try XCTUnwrap(atRead.symFused["blocks.2.att.rkv"])
        let b = try XCTUnwrap(after.symFused["blocks.2.att.rkv"])
        XCTAssertEqual(a.keys, keys)
        XCTAssertEqual(Self.bad(a.qblk, b.qblk), 0); XCTAssertEqual(Self.bad(a.qs, b.qs), 0)
        XCTAssertEqual(Self.bad(a.d, b.d), 0)
        for name in atRead.arrays.keys { XCTAssertEqual(Self.bad(atRead.arrays[name]!, after.arrays[name]!), 0, name) }
        // группа, которой при чтении не хватает члена, не сводится
        let lone = try RwkvqSidecar(rwkvq: path, only: { $0 == keys[0] }, fusingSym: groups)
        XCTAssertTrue(lone.symFused.isEmpty)
        // повторное сведение — не копия
        eval(Array(atRead.arrays.values)); GPU.clearCache()
        let before = GPU.activeMemory
        let again = atRead.fusingSym(groups)
        eval(Array(again.arrays.values))
        XCTAssertEqual(GPU.activeMemory, before, "уже сведённая группа сведена второй раз (копия)")
        XCTAssertEqual(again.symFused.count, 1)
    }

    func testStreamingLoadEqualsWholeFileLoadBitwise() throws {
        for preset in RwkvqFileTests.presets {
            let path = try RwkvqFileTests.sourcePath(preset)
            let whole = try X070Backbone.load(fromSidecar: try RwkvqSidecar(rwkvq: path))
            let ref = Self.outputs(whole)
            XCTAssertNil(whole.rwkvqEmbTable)
            for native in [false, true] where !(native && preset == "reduction") {
                let wholeN = native ? try X070Backbone.load(fromSidecar: try RwkvqSidecar(rwkvq: path),
                                                            useNativeKernel: true) : whole
                let refN = native ? Self.outputs(wholeN) : ref
                let bb = try X070Backbone.load(fromRwkvq: path, useNativeKernel: native)
                for (i, (a, b)) in zip(Self.outputs(bb), refN).enumerated() {
                    XCTAssertEqual(Self.bad(a, b), 0, "\(preset), native \(native): выход \(i)")
                }
                XCTAssertEqual(bb.rwkvqBackedKeys, wholeN.rwkvqBackedKeys)
                XCTAssertEqual(bb.rwkvqDenseTargets, wholeN.rwkvqDenseTargets)
                XCTAssertEqual(bb.allWeightKeys, wholeN.allWeightKeys, "\(preset): состав весов")
            }
            // r/k/v сведены и при потоковой загрузке: фьюз декода работает
            let bb = try X070Backbone.load(fromRwkvq: path)
            var st = RWKVState(cfg: bb.cfg)
            let fusedBefore = bb.rwkvqSymFusedCalls
            _ = bb.step(5, state: &st); st.eval()
            XCTAssertEqual(bb.rwkvqSymFusedCalls - fusedBefore, preset == "reduction" ? bb.cfg.nLayer : 0)
        }
    }

    /// Где таблица остаётся сжатой, а где плотной — и что обе дают одно.
    func testEmbeddingStaysPackedOnlyWhereItCan() throws {
        let path = try RwkvqFileTests.sourcePath("reduction")
        let bb = try X070Backbone.load(fromRwkvq: path)
        XCTAssertNotNil(bb.rwkvqEmbTable, "REDUCTION, fp16: emb обязан остаться сжатым")
        XCTAssertFalse(bb.hasWeight("emb.weight"), "плотная таблица заведена рядом со сжатой")
        XCTAssertTrue(bb.hasAnyWeight("emb.weight"))
        XCTAssertTrue(bb.allWeightKeys.contains("emb.weight"))

        let dense = try X070Backbone.load(fromRwkvq: path, denseEmbedding: true)
        XCTAssertNil(dense.rwkvqEmbTable, "denseEmbedding не прочитан")
        XCTAssertTrue(dense.hasWeight("emb.weight"))
        XCTAssertEqual(Self.bad(bb.weight("emb.weight"), dense.weight("emb.weight")), 0)
        XCTAssertEqual(bb.weight("emb.weight").dtype, .float16)
        for (i, (a, b)) in zip(Self.outputs(bb), Self.outputs(dense)).enumerated() {
            XCTAssertEqual(Self.bad(a, b), 0, "сжатый emb против плотного: выход \(i)")
        }
        // не fp16 — плотная: тождество строк доказано только для fp16
        let bf = try X070Backbone.load(fromRwkvq: path, paramDType: .bfloat16)
        XCTAssertNil(bf.rwkvqEmbTable, "bf16-модель получила fp16-строки таблицы")
        XCTAssertEqual(bf.weight("emb.weight").dtype, .bfloat16)
        // COMPRESSION: sb6, строчной выборки нет
        XCTAssertNil(try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("compression")).rwkvqEmbTable)

        // обучаемая подмена таблицы имеет приоритет над сжатой
        let arr = MLXArray(Self.ids.map { Int32($0) }, [1, Self.ids.count])
        let plain = bb(arr); eval(plain)
        bb.wOverride = ["emb.weight": bb.weight("emb.weight") * 0.5]
        let overridden = bb(arr); eval(overridden)
        bb.wOverride = nil
        XCTAssertGreaterThan(Self.bad(plain, overridden), 0, "wOverride таблицы не подействовал")
    }

    /// Главное свойство: пик памяти MLX за загрузку — это память модели.
    /// Прежний порядок давал 1.3–1.4 от рабочей.
    func testLoadHasNoMemorySpike() throws {
        for preset in RwkvqFileTests.presets {
            let path = try RwkvqFileTests.sourcePath(preset)
            _ = try X070Backbone.load(fromRwkvq: path)      // ядра собраны, файл в кеше страниц
            GPU.clearCache()
            let before = GPU.activeMemory
            GPU.resetPeakMemory()
            let bb = try X070Backbone.load(fromRwkvq: path)
            let peak = GPU.peakMemory - before, live = GPU.activeMemory - before
            print(String(format: "ЗАГРУЗКА %@: рабочая %.1f МБ, пик %.1f МБ, отношение %.3f",
                         preset, Double(live) / 1e6, Double(peak) / 1e6, Double(peak) / Double(live)))
            XCTAssertGreaterThan(live, 50_000_000)
            XCTAssertLessThan(Double(peak), Double(live) * Self.spikeLimit, "\(preset): всплеск памяти при загрузке")
            _ = bb
        }
    }
    /// Замерено 1.007 на обоих пресетах 0.1B; сведение r/k/v копией даёт 1.11.
    static let spikeLimit = 1.03
}
