//
//  PEFTStateFileTests.swift
//  Состояние в формате RWKV-PEFT (.pth): круг записи-чтения, чтение файлов,
//  записанных самим torch.save, отказы. Фикстуры A/B/C сгенерированы
//  torch 2.14 (swr_bench_keep/w1023/peft/make_fixtures.py):
//    A — fp32, обычный dict, слои — срезы одного тензора (общее хранилище, смещения);
//    B — bf16, OrderedDict с _metadata, слой 0 — nn.Parameter, слой 1 — транспонированный
//        вид (нестрогая раскладка), лишний ключ blocks.0.att.time_faaaa;
//    C — только чужой ключ emb.weight.
//  Значения: v[l,h,i,j] = (l*1000 + h*100 + i*10 + j) / 7 - 20, L=2, H=2, S=4.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

final class PEFTStateFileTests: XCTestCase {

    func tmp(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("peft-\(UUID().uuidString)-\(name)")
    }

    func fixture(_ b64: String) throws -> URL {
        let u = tmp("fixture.pth")
        try Data(base64Encoded: b64)!.write(to: u)
        return u
    }

    /// Ожидаемые значения фикстур в fp32, [L, H, S, S]: torch считал их в double и округлил один раз.
    func expected() -> MLXArray {
        var v = [Float]()
        for l in 0 ..< 2 { for h in 0 ..< 2 { for i in 0 ..< 4 { for j in 0 ..< 4 {
            v.append(Float(Double(l * 1000 + h * 100 + i * 10 + j) / 7 - 20))   // как в Python: в double, затем fp32
        } } } }
        return MLXArray(v, [2, 2, 4, 4])
    }

    func randomState(L: Int = 3, H: Int = 2, S: Int = 4, shifts: Bool = false) -> RWKVBatchState {
        MLXRandom.seed(11)
        let D = H * S
        return RWKVBatchState(wkv: MLXRandom.normal([L, 1, H, S, S]),
                              tmixShift: shifts ? MLXRandom.normal([L, 1, 1, D]) : MLXArray.zeros([L, 1, 1, D]),
                              cmixShift: MLXArray.zeros([L, 1, 1, D]))
    }

    func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
    }

    // ── Круг записи-чтения ──────────────────────────────────────────────

    func testRoundTripIsBitwise() throws {
        let s = randomState()
        let u = tmp("rt.pth")
        try StateFile.exportPEFT(s, to: u)
        let (back, ignored) = try StateFile.importPEFT(u)
        XCTAssertTrue(bitwise(back.wkv, s.wkv.asType(.float32)))
        XCTAssertTrue(bitwise(back.tmixShift, MLXArray.zeros([3, 1, 1, 8])))
        XCTAssertTrue(bitwise(back.cmixShift, MLXArray.zeros([3, 1, 1, 8])))
        XCTAssertEqual(ignored, [])
    }

    /// bf16-состояние пишется fp32 точно (bf16 ⊂ fp32).
    func testBF16StateWrittenExactly() throws {
        var s = randomState()
        s = RWKVBatchState(wkv: s.wkv.asType(.bfloat16), tmixShift: s.tmixShift, cmixShift: s.cmixShift)
        let u = tmp("bf.pth")
        try StateFile.exportPEFT(s, to: u)
        let back = try StateFile.importPEFT(u).state
        XCTAssertTrue(bitwise(back.wkv.asType(.bfloat16), s.wkv))
    }

    /// Сдвиги формат не хранит: ненулевые — отказ; dropShifts — запись без них.
    func testNonZeroShiftsRefusedUnlessDropped() throws {
        let s = randomState(shifts: true)
        let u = tmp("sh.pth")
        XCTAssertThrowsError(try StateFile.exportPEFT(s, to: u)) {
            XCTAssertEqual($0 as? StateFile.PEFTFailure, .nonZeroShifts)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: u.path))
        try StateFile.exportPEFT(s, to: u, dropShifts: true)
        let back = try StateFile.importPEFT(u).state
        XCTAssertTrue(bitwise(back.wkv, s.wkv))
        XCTAssertTrue(bitwise(back.tmixShift, MLXArray.zeros([3, 1, 1, 8])))
    }

    /// Ненулевой сдвиг cmix — тоже отказ (проверяются оба).
    func testNonZeroCmixShiftRefused() throws {
        let s0 = randomState()
        let s = RWKVBatchState(wkv: s0.wkv, tmixShift: s0.tmixShift, cmixShift: s0.cmixShift + 1e-30)
        XCTAssertThrowsError(try StateFile.exportPEFT(s, to: tmp("c.pth"))) {
            XCTAssertEqual($0 as? StateFile.PEFTFailure, .nonZeroShifts)
        }
    }

    /// Файл записан как torch: ключи blocks.{i}.att.time_state по порядку,
    /// хранилища FloatStorage, zip без сжатия, данные выровнены на 64.
    func testFileLayout() throws {
        let u = tmp("lay.pth")
        try StateFile.exportPEFT(randomState(), to: u)
        let z = try TorchPth.StoredZipReader([UInt8](try Data(contentsOf: u)))
        XCTAssertEqual(z.names, ["archive/data.pkl", "archive/byteorder", "archive/data/0", "archive/data/1",
                                 "archive/data/2", "archive/version"])
        for n in z.names { XCTAssertEqual(z.entries[n]!.method, 0); XCTAssertEqual(z.entries[n]!.data % 64, 0, n) }
        let r = try TorchPth.read(u)
        XCTAssertEqual(r.tensors.map(\.0), (0 ..< 3).map { "blocks.\($0).att.time_state" })
        XCTAssertEqual(r.tensors.map { $0.1.dtype }, [.float32, .float32, .float32])
        XCTAssertEqual(r.tensors.map { $0.1.shape }, [[2, 4, 4], [2, 4, 4], [2, 4, 4]])
    }

    /// Подключённое импортированное состояние считает то же, что исходное.
    func testImportedStateRunsLikeOriginal() throws {
        let cfg = TinyBackbone.config()
        let bb = X070Backbone(weights: TinyBackbone.weights(cfg), cfg: cfg, computeDType: .float32)
        MLXRandom.seed(3)
        let s = RWKVBatchState(wkv: MLXRandom.normal([cfg.nLayer, 1, cfg.nHead, 64, 64]) * 0.05,
                               tmixShift: MLXArray.zeros([cfg.nLayer, 1, 1, cfg.nEmbd]),
                               cmixShift: MLXArray.zeros([cfg.nLayer, 1, 1, cfg.nEmbd]))
        let u = tmp("run.pth")
        try StateFile.exportPEFT(s, to: u)
        let back = try StateFile.importPEFT(u, base: bb).state
        let ids = [5, 9, 1, 33, 7]
        let a = bb.foldPrompt(ids, from: s), b = bb.foldPrompt(ids, from: back)
        XCTAssertTrue(bitwise(a.wkv, b.wkv))
        XCTAssertTrue(bitwise(a.tmixShift, b.tmixShift))
        XCTAssertTrue(bitwise(a.cmixShift, b.cmixShift))
    }

    // ── Файлы torch.save ────────────────────────────────────────────────

    func testReadsTorchFP32SharedStorageWithOffsets() throws {
        let (s, ignored) = try StateFile.importPEFT(try fixture(Self.fixtureA))
        XCTAssertTrue(bitwise(s.wkv, expected().expandedDimensions(axis: 1)))
        XCTAssertEqual(ignored, [])
        XCTAssertEqual(s.tmixShift.shape, [2, 1, 1, 8])
    }

    func testReadsTorchBF16ParameterStridedMetadata() throws {
        let (s, ignored) = try StateFile.importPEFT(try fixture(Self.fixtureB))
        let want = expected().asType(.bfloat16).asType(.float32).expandedDimensions(axis: 1)
        XCTAssertTrue(bitwise(s.wkv, want))
        XCTAssertEqual(ignored, ["blocks.0.att.time_faaaa"])
        // Тип хранилища — bf16, а не fp16 с теми же байтами.
        let r = try TorchPth.read(try fixture(Self.fixtureB))
        XCTAssertEqual(r.tensors.map { $0.1.dtype }, [.bfloat16, .bfloat16, .bfloat16])
    }

    func testNoTimeStateRefused() throws {
        XCTAssertThrowsError(try StateFile.importPEFT(try fixture(Self.fixtureC))) {
            XCTAssertEqual($0 as? StateFile.PEFTFailure, .noTimeState)
        }
    }

    // ── Отказы ──────────────────────────────────────────────────────────

    func testArchitectureMismatchRefused() throws {
        let cfg = TinyBackbone.config()
        let bb = X070Backbone(weights: TinyBackbone.weights(cfg), cfg: cfg)
        XCTAssertThrowsError(try StateFile.importPEFT(try fixture(Self.fixtureA), base: bb)) {
            guard case .otherArchitecture = $0 as? StateFile.Failure else { return XCTFail("\($0)") }
        }
    }

    func testLayerGapRefused() throws {
        let t = MLXArray.zeros([2, 4, 4]).asArray(Float.self).withUnsafeBytes { Data($0) }
        let u = tmp("gap.pth")
        try TorchPth.write([("blocks.0.att.time_state", [2, 4, 4], t), ("blocks.2.att.time_state", [2, 4, 4], t)], to: u)
        XCTAssertThrowsError(try StateFile.importPEFT(u)) {
            guard case .shape = $0 as? StateFile.PEFTFailure else { return XCTFail("\($0)") }
        }
    }

    func testShapeMismatchBetweenLayersRefused() throws {
        let a = MLXArray.zeros([2, 4, 4]).asArray(Float.self).withUnsafeBytes { Data($0) }
        let b = MLXArray.zeros([4, 2, 2]).asArray(Float.self).withUnsafeBytes { Data($0) }
        let u = tmp("shp.pth")
        try TorchPth.write([("blocks.0.att.time_state", [2, 4, 4], a), ("blocks.1.att.time_state", [4, 2, 2], b)], to: u)
        XCTAssertThrowsError(try StateFile.importPEFT(u)) {
            guard case .shape = $0 as? StateFile.PEFTFailure else { return XCTFail("\($0)") }
        }
    }

    /// Глобал вне разрешённых не исполняется и не принимается.
    func testForeignGlobalRefused() throws {
        var p = TorchPth.Pickle()
        p.global("os", "system"); p.op(0x28); p.str("echo hi"); p.op(0x74); p.op(0x52); p.op(0x2E)
        var z = TorchPth.StoredZip()
        z.add("archive/data.pkl", p.out)
        let u = tmp("evil.pth")
        try z.finish().write(to: u)
        XCTAssertThrowsError(try StateFile.importPEFT(u)) {
            guard case .unsupported = $0 as? StateFile.PEFTFailure else { return XCTFail("\($0)") }
        }
    }

    func testCorruptedDataRefused() throws {
        let u = tmp("bad.pth")
        try StateFile.exportPEFT(randomState(), to: u)
        var d = [UInt8](try Data(contentsOf: u))
        let z = try TorchPth.StoredZipReader(d)
        d[z.entries["archive/data/1"]!.data + 5] ^= 0x01
        try Data(d).write(to: u)
        XCTAssertThrowsError(try StateFile.importPEFT(u)) {
            guard case .notATorchFile = $0 as? StateFile.PEFTFailure else { return XCTFail("\($0)") }
        }
    }

    func testNotAZipRefused() throws {
        let u = tmp("nz.pth")
        try Data(repeating: 7, count: 100).write(to: u)
        XCTAssertThrowsError(try StateFile.importPEFT(u)) {
            guard case .notATorchFile = $0 as? StateFile.PEFTFailure else { return XCTFail("\($0)") }
        }
    }

    func testCRCMatchesZlib() {
        XCTAssertEqual(TorchPth.crc32(Array("123456789".utf8)), 0xCBF4_3926)
    }
    static let fixtureA = "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAKFgXAAAAYmxvY2tzLjAuYXR0LnRpbWVfc3RhdGVxAWN0b3JjaC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3YyCnECKChYBwAAAHN0b3JhZ2VxA2N0b3JjaApGbG9hdFN0b3JhZ2UKcQRYAQAAADBxBVgDAAAAY3B1cQZLQHRxB1FLAEsCSwRLBIdxCEsQSwRLAYdxCYljY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcQopUnELdHEMUnENWBcAAABibG9ja3MuMS5hdHQudGltZV9zdGF0ZXEOaAIoKGgDaARoBWgGS0B0cQ9RSyBLAksESwSHcRBLEEsESwGHcRGJaAopUnESdHETUnEUdS5QSwcIAh1ZnRIBAAASAQAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAXACkAYXJjaGl2ZS8uZm9ybWF0X3ZlcnNpb25GQiUAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjFQSwcIt+/cgwEAAAABAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAaADcAYXJjaGl2ZS8uc3RvcmFnZV9hbGlnbm1lbnRGQjMAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaNjRQSwcIP3dx6QIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAARAD8AYXJjaGl2ZS9ieXRlb3JkZXJGQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpsaXR0bGVQSwcIhT3jGQYAAAAGAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOAD4AYXJjaGl2ZS9kYXRhLzBGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAoMFu257B27adwUmSnMFJkpTBt22TwSVJksGSJJHBkiSJwQAAiMFu24bB27aFwbdte8GSJHnBbtt2wUmSdMFu27bAJUmywNu2rcCSJKnAkiSJwEmShMAAAIDAbtt2wG7bNsDbti3ASZIkwLdtG8Bu27a/SZKkvyVJkr8AAIC/27b1QgAA9kIlSfZCSZL2QkmS+EJu2/hCkiT5Qrdt+UK3bftC27b7QgAA/EIlSfxCJUn+QkmS/kJu2/5CkiT/QpIkCUMlSQlDt20JQ0mSCUNJkgpD27YKQ27bCkMAAAtDAAAMQ5IkDEMlSQxDt20MQ7dtDUNJkg1D27YNQ27bDUNQSwcIKZ8/pAABAAAAAQAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPAEMAYXJjaGl2ZS92ZXJzaW9uRkI/AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjMKUEsHCNGeZ1UCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgAyAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZEZCLgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMDg0MzQ5NTA3MDIyNjQxMjM2NzcwNTg4NjQwNTE0MTU0ODQ0MzQ3MlBLBwhuo2VBKAAAACgAAABQSwECAAAAAAgIAAAAAAAAAh1ZnRIBAAASAQAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAAC379yDAQAAAAEAAAAXAAAAAAAAAAAAAAAAAGIBAABhcmNoaXZlLy5mb3JtYXRfdmVyc2lvblBLAQIAAAAACAgAAAAAAAA/d3HpAgAAAAIAAAAaAAAAAAAAAAAAAAAAANEBAABhcmNoaXZlLy5zdG9yYWdlX2FsaWdubWVudFBLAQIAAAAACAgAAAAAAACFPeMZBgAAAAYAAAARAAAAAAAAAAAAAAAAAFICAABhcmNoaXZlL2J5dGVvcmRlclBLAQIAAAAACAgAAAAAAAApnz+kAAEAAAABAAAOAAAAAAAAAAAAAAAAANYCAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAAFAEAABhcmNoaXZlL3ZlcnNpb25QSwECAAAAAAgIAAAAAAAAbqNlQSgAAAAoAAAAHgAAAAAAAAAAAAAAAADSBAAAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAABwAAAAAAAAAHAAAAAAAAAM8BAAAAAAAAeAUAAAAAAABQSwYHAAAAAEcHAAAAAAAAAQAAAFBLBQYAAAAABwAHAM8BAAB4BQAAAAA="
    static let fixtureB = "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEAKVJxAShYFwAAAGJsb2Nrcy4wLmF0dC50aW1lX3N0YXRlcQJjdG9yY2guX3V0aWxzCl9yZWJ1aWxkX3BhcmFtZXRlcgpxA2N0b3JjaC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3YyCnEEKChYBwAAAHN0b3JhZ2VxBWN0b3JjaApCRmxvYXQxNlN0b3JhZ2UKcQZYAQAAADBxB1gDAAAAY3B1cQhLIHRxCVFLAEsCSwRLBIdxCksQSwRLAYdxC4loAClScQx0cQ1ScQ6IaAApUnEPh3EQUnERWBcAAABibG9ja3MuMS5hdHQudGltZV9zdGF0ZXESaAQoKGgFaAZYAQAAADFxE2gISyB0cRRRSwBLAksESwSHcRVLEEsBSwSHcRaJaAApUnEXdHEYUnEZWBcAAABibG9ja3MuMC5hdHQudGltZV9mYWFhYXEaaAQoKGgFaAZYAQAAADJxG2gISwh0cRxRSwBLAksEhnEdSwRLAYZxHoloAClScR90cSBScSF1fXEiWAkAAABfbWV0YWRhdGFxI2gAKVJxJChYAAAAAHElfXEmWAcAAAB2ZXJzaW9ucSdLAXNYCAAAAGJsb2Nrcy4wcSh9cSloJ0sCc3VzYi5QSwcIm5sZMPABAADwAQAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAXAAsAYXJjaGl2ZS8uZm9ybWF0X3ZlcnNpb25GQgcAWlpaWlpaWjFQSwcIt+/cgwEAAAABAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAaADcAYXJjaGl2ZS8uc3RvcmFnZV9hbGlnbm1lbnRGQjMAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaNjRQSwcIP3dx6QIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAARAD8AYXJjaGl2ZS9ieXRlb3JkZXJGQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpsaXR0bGVQSwcIhT3jGQYAAAAGAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOAD4AYXJjaGl2ZS9kYXRhLzBGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWqDBn8GewZ3BlcGTwZLBkcGJwYjBh8GGwXvBecF3wXXBt8CywK7AqcCJwIXAgMB3wDfALsAlwBvAt7+lv5K/gL9QSwcIXoUjxkAAAABAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOAAQAYXJjaGl2ZS9kYXRhLzFGQgAA9kL5QvtC/kL2QvlC/EL/QvZC+UL8Qv9C90L5QvxC/0IJQwtDDEMNQwlDC0MMQw5DCUMLQwxDDkMKQwtDDEMOQ1BLBwhBA/wOQAAAAEAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4ABABhcmNoaXZlL2RhdGEvMkZCAACAP4A/gD+AP4A/gD+AP4A/UEsHCCMPkTgQAAAAEAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADwAzAGFyY2hpdmUvdmVyc2lvbkZCLwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjMKUEsHCNGeZ1UCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgAyAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZEZCLgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMDgwNjQwMTMyMDkzNDIwMTExNzcwNzcwMDQwNzE2Nzk1ODQzNjQ4NlBLBwjwvYJIKAAAACgAAABQSwECAAAAAAgIAAAAAAAAm5sZMPABAADwAQAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAAC379yDAQAAAAEAAAAXAAAAAAAAAAAAAAAAAEACAABhcmNoaXZlLy5mb3JtYXRfdmVyc2lvblBLAQIAAAAACAgAAAAAAAA/d3HpAgAAAAIAAAAaAAAAAAAAAAAAAAAAAJECAABhcmNoaXZlLy5zdG9yYWdlX2FsaWdubWVudFBLAQIAAAAACAgAAAAAAACFPeMZBgAAAAYAAAARAAAAAAAAAAAAAAAAABIDAABhcmNoaXZlL2J5dGVvcmRlclBLAQIAAAAACAgAAAAAAABehSPGQAAAAEAAAAAOAAAAAAAAAAAAAAAAAJYDAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAABBA/wOQAAAAEAAAAAOAAAAAAAAAAAAAAAAAFAEAABhcmNoaXZlL2RhdGEvMVBLAQIAAAAACAgAAAAAAAAjD5E4EAAAABAAAAAOAAAAAAAAAAAAAAAAANAEAABhcmNoaXZlL2RhdGEvMlBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAACAFAABhcmNoaXZlL3ZlcnNpb25QSwECAAAAAAgIAAAAAAAA8L2CSCgAAAAoAAAAHgAAAAAAAAAAAAAAAACSBQAAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAACQAAAAAAAAAJAAAAAAAAAEcCAAAAAAAAOAYAAAAAAABQSwYHAAAAAH8IAAAAAAAAAQAAAFBLBQYAAAAACQAJAEcCAAA4BgAAAAA="
    static let fixtureC = "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAWAoAAABlbWIud2VpZ2h0cQFjdG9yY2guX3V0aWxzCl9yZWJ1aWxkX3RlbnNvcl92MgpxAigoWAcAAABzdG9yYWdlcQNjdG9yY2gKRmxvYXRTdG9yYWdlCnEEWAEAAAAwcQVYAwAAAGNwdXEGSwZ0cQdRSwBLA0sChnEISwJLAYZxCYljY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcQopUnELdHEMUnENcy5QSwcI0rUBxq8AAACvAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAXAAwAYXJjaGl2ZS8uZm9ybWF0X3ZlcnNpb25GQggAWlpaWlpaWloxUEsHCLfv3IMBAAAAAQAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAGgA3AGFyY2hpdmUvLnN0b3JhZ2VfYWxpZ25tZW50RkIzAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjY0UEsHCD93cekCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAEQA/AGFyY2hpdmUvYnl0ZW9yZGVyRkI7AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpabGl0dGxlUEsHCIU94xkGAAAABgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA+AGFyY2hpdmUvZGF0YS8wRkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAIA/AACAPwAAgD8AAIA/AACAPwAAgD9QSwcIJRLdKRgAAAAYAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPACsAYXJjaGl2ZS92ZXJzaW9uRkInAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjMKUEsHCNGeZ1UCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgAyAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZEZCLgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMDg0MzQ5NTA3MDIyNjQxMjM2NzcwNjYzMTUzMDYwMjA2NzE1ODAyMVBLBwj1t4/jKAAAACgAAABQSwECAAAAAAgIAAAAAAAA0rUBxq8AAACvAAAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAAC379yDAQAAAAEAAAAXAAAAAAAAAAAAAAAAAP8AAABhcmNoaXZlLy5mb3JtYXRfdmVyc2lvblBLAQIAAAAACAgAAAAAAAA/d3HpAgAAAAIAAAAaAAAAAAAAAAAAAAAAAFEBAABhcmNoaXZlLy5zdG9yYWdlX2FsaWdubWVudFBLAQIAAAAACAgAAAAAAACFPeMZBgAAAAYAAAARAAAAAAAAAAAAAAAAANIBAABhcmNoaXZlL2J5dGVvcmRlclBLAQIAAAAACAgAAAAAAAAlEt0pGAAAABgAAAAOAAAAAAAAAAAAAAAAAFYCAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAAOgCAABhcmNoaXZlL3ZlcnNpb25QSwECAAAAAAgIAAAAAAAA9beP4ygAAAAoAAAAHgAAAAAAAAAAAAAAAABSAwAAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAABwAAAAAAAAAHAAAAAAAAAM8BAAAAAAAA+AMAAAAAAABQSwYHAAAAAMcFAAAAAAAAAQAAAFBLBQYAAAAABwAHAM8BAAD4AwAAAAA="
}
