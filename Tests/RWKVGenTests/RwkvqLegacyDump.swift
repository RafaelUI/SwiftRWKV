//
//  RwkvqLegacyDump.swift
//  Характеризационный дамп ПРЕЖНЕГО пути квантованной базы (baseDType == nil):
//  июльский сайдкар sb6 поверх плотной 0.1B, все способы подключения.
//
//  Зачем: правка, меняющая умолчания базы (тип декванта и матмула, правило
//  scale родного контейнера), обязана оставить прежний путь побитно прежним.
//  Допуском это не проверяется — нужен снимок до и после:
//
//      RWKV_RWKVQ_DUMP=/tmp/after.safetensors  swift test --filter RwkvqLegacyDump
//      git stash                                # вернуться к коду до правки
//      # убрать строки с пометкой LEGACY-PIN (до правки поля baseDType нет)
//      RWKV_RWKVQ_DUMP=/tmp/before.safetensors swift test --filter RwkvqLegacyDump
//      git stash pop
//      # сравнить: расхождение обязано быть ровно нулевым
//
//  Без RWKV_RWKVQ_DUMP тест пропускается.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

final class RwkvqLegacyDump: XCTestCase {

    func testDumpLegacyPaths() throws {
        guard let path = ProcessInfo.processInfo.environment["RWKV_RWKVQ_DUMP"] else {
            throw XCTSkip("RWKV_RWKVQ_DUMP не задан — дамп снимается вручную")
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let modelPath = home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/world_0.1b_x070.safetensors").path
        let sidecarPath = home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx").path
        let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
        let cfg = X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536)
        let sidecar = try RwkvqSidecar(path: sidecarPath)
        let ids = MLXArray((0 ..< 48).map { Int32(($0 * 7919 + 13) % 60000 + 100) }, [1, 48])

        var out: [String: MLXArray] = [:]
        for (tag, native, emb) in [("dequant", false, false), ("native", true, false),
                                   ("emb", false, true)] {
            let bb = X070Backbone(weights: weights, cfg: cfg)
            var o = X070Backbone.RwkvqAttachOptions()
            o.useNativeKernel = native
            o.quantizeEmbedding = emb
            o.baseDType = nil   // LEGACY-PIN
            _ = bb.attachRwkvq(sidecar, options: o)
            out["\(tag)_logits"] = bb(ids)
            let (h, st) = bb.bodyWithState(ids[0..., 0 ..< 32])
            let (h2, st2) = bb.bodyWithState(ids[0..., 32 ..< 48], state: st)
            out["\(tag)_split_a"] = h
            out["\(tag)_split_b"] = h2
            out["\(tag)_wkv"] = st2.wkv
            out["\(tag)_tmix"] = st2.tmixShift
            var rs = RWKVState(st, row: 0)
            var last = MLXArray(Float(0))
            for t in 32 ..< 40 {
                last = bb.step(ids[0, t].item(Int.self), state: &rs)
            }
            out["\(tag)_step_logits"] = last
            // до правки вес, оставшийся только в родном контейнере, не читался вовсе
            if !native { out["\(tag)_dense_k"] = bb.weight("blocks.3.tmix.k_proj.weight") }
            eval(Array(out.values))
        }
        try save(arrays: out.mapValues { $0.asType(.float32) },
                 url: URL(fileURLWithPath: path))
        print("дамп записан: \(path), тензоров \(out.count)")
    }
}
