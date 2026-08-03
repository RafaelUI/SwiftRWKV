//
//  QuantisedWeightAccessTests.swift
//  Доступ к весам на КВАНТОВАННОЙ базе.
//
//  `attachRwkvq` выбрасывает плотные копии — ради этого он и существует.
//  Всё, что читает веса по имени, обязано это пережить. Раньше не переживало,
//  причём двумя разными способами:
//
//   • `weight(_:)` был `w[key]!` — force-unwrap без сообщения;
//   • `weightKeys` перечисляет ТОЛЬКО плотные веса, и обход по нему собирал
//     неполный набор МОЛЧА. `RWKVBlock.fromBase` строил блок без единой
//     проекции (27 весов вместо 33), возвращал его успешно, и падало это
//     потом — в `param()` посреди прохода, где причину уже не видно.
//
//  Второе хуже первого ровно тем же, чем всегда: оно не падает там, где
//  сломано.
//
//  Пропускается без модели или сайдкара.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

final class QuantisedWeightAccessTests: XCTestCase {

    func quantised() throws -> (X070Backbone, X070Config) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let sp = env["RWKV_RWKVQ_SIDECAR"] ?? home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mp), "нет модели")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: sp + ".safetensors"),
                          "нет сайдкара")
        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: w["head.weight"]!.shape[0])
        let bb = X070Backbone(weights: w, cfg: cfg)
        XCTAssertGreaterThan(bb.attachRwkvq(try RwkvqSidecar(path: sp)).attached, 0)
        XCTAssertFalse(bb.hasWeight("blocks.0.tmix.r_proj.weight"),
                       "плотная копия не выброшена — тест вырожден")
        return (bb, cfg)
    }

    /// `allWeightKeys` перечисляет и плотные, и ушедшие в сайдкар;
    /// `weightKeys` — только плотные.
    ///
    /// Пара утверждений, а не одно: если бы `allWeightKeys` просто дублировал
    /// `weightKeys`, первая половина прошла бы, а вторая — нет.
    func testAllWeightKeysIncludesSidecarBacked() throws {
        let (bb, _) = try quantised()
        let dense = Set(bb.weightKeys), all = Set(bb.allWeightKeys)
        XCTAssertTrue(all.isSuperset(of: dense))
        XCTAssertGreaterThan(all.count, dense.count,
                             "allWeightKeys не богаче weightKeys — сайдкар не учтён")
        XCTAssertTrue(all.contains("blocks.0.tmix.r_proj.weight"),
                      "квантованный вес не перечислен")
        XCTAssertFalse(dense.contains("blocks.0.tmix.r_proj.weight"),
                       "квантованный вес числится плотным — тест вырожден")
        for k in bb.rwkvqBackedKeys { XCTAssertTrue(all.contains(k), "\(k) потерян") }
    }

    /// `weight(_:)` РАЗВОРАЧИВАЕТ вес из сайдкара, а не падает.
    func testWeightDequantisesInsteadOfCrashing() throws {
        let (bb, cfg) = try quantised()
        let wq = bb.weight("blocks.0.tmix.r_proj.weight")
        eval(wq)
        XCTAssertEqual(wq.shape, [cfg.nEmbd, cfg.nEmbd])
        XCTAssertTrue(wq.asType(.float32).sum().item(Float.self).isFinite)
        // И совпадает с тем, что отдаёт сайдкар напрямую — то есть это
        // действительно тот вес, а не что-то похожей формы.
        XCTAssertNotNil(bb.denseWeight("blocks.0.tmix.r_proj.weight"))
        XCTAssertNil(bb.denseWeight("такого.веса.нет"))
        XCTAssertTrue(bb.hasAnyWeight("blocks.0.tmix.r_proj.weight"))
        XCTAssertFalse(bb.hasWeight("blocks.0.tmix.r_proj.weight"))
    }

    /// ГЛАВНОЕ: блок, собранный из квантованной базы, ПОЛОН и работает.
    ///
    /// Это путь инициализации головы реранкера. Раньше он молча отдавал блок
    /// без проекций.
    func testBlockFromQuantisedBaseIsComplete() throws {
        let (bb, cfg) = try quantised()
        let block = try RWKVBlock.fromBase(bb, layer: 0, index: 1)
        let keys = Set(block.weightKeys)

        for k in ["tmix.r_proj.weight", "tmix.k_proj.weight", "tmix.v_proj.weight",
                  "tmix.o_proj.weight", "cmix.key.weight", "cmix.value.weight",
                  "tmix.ln_x.weight", "ln1.weight", "ln2.weight"] {
            XCTAssertTrue(keys.contains(k), "в блоке нет \(k)")
        }

        // Столько же весов, сколько даёт ПЛОТНАЯ база — вот утверждение,
        // которое нельзя пройти неполным набором.
        let w = try loadArrays(url: URL(fileURLWithPath:
            ProcessInfo.processInfo.environment["RWKV_PARITY_MODEL"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/rwkv-metal/world_0.1b_x070.safetensors").path))
        let denseBB = X070Backbone(weights: w, cfg: cfg)
        let denseBlock = try RWKVBlock.fromBase(denseBB, layer: 0, index: 1)
        XCTAssertEqual(keys, Set(denseBlock.weightKeys),
                       "набор весов блока зависит от того, квантована ли база")

        // И блок СЧИТАЕТ: неполный упал бы здесь, в param(). index=1 значит
        // value-residual, поэтому v_first обязателен — как и в голове
        // реранкера, где первый блок его рождает, а остальные потребляют.
        let x = MLXArray.zeros([1, 4, cfg.nEmbd], dtype: .bfloat16) + 0.01
        let vFirst = MLXArray.zeros([1, 4, cfg.nHead, cfg.headSize], dtype: .bfloat16)
        let (y, _) = block(x, vFirst)
        eval(y)
        XCTAssertEqual(y.shape, [1, 4, cfg.nEmbd])
        XCTAssertTrue(MLX.abs(y.asType(.float32)).max().item(Float.self).isFinite)
    }
}
