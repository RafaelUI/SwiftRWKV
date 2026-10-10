//
//  RwkvqLoadDump.swift
//  Отпечаток модели, загруженной из .rwkvq НЫНЕШНИМ умолчанием
//  (`X070Backbone.load(fromRwkvq:)`), на обоих пресетах: параллельный проход
//  с состоянием, префилл и рекуррентные шаги.
//
//  Зачем: правки загрузки (порядок сборки, чтение буферов, emb выборкой
//  строк) и правки префилла, которые не должны менять счёт, обязаны оставить
//  выход ПОБИТНО прежним. Допуском это не проверяется — нужен снимок до и
//  после:
//
//      RWKV_LOAD_DUMP=/tmp/after.safetensors  swift test --filter RwkvqLoadDump
//      git stash            # код до правки
//      RWKV_LOAD_DUMP=/tmp/before.safetensors swift test --filter RwkvqLoadDump
//      git stash pop
//      # сравнить: расхождение обязано быть ровно нулевым
//
//  Без RWKV_LOAD_DUMP тест пропускается.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

final class RwkvqLoadDump: XCTestCase {

    func testDumpLoadedModel() throws {
        guard let path = ProcessInfo.processInfo.environment["RWKV_LOAD_DUMP"] else {
            throw XCTSkip("RWKV_LOAD_DUMP не задан — дамп снимается вручную")
        }
        let ids = (0 ..< 80).map { ($0 * 7919 + 13) % 60000 + 100 }
        var out: [String: MLXArray] = [:]
        for preset in RwkvqFileTests.presets {
            for native in [false, true] {
                let tag = "\(preset)_\(native ? "native" : "k3")"
                let bb = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath(preset),
                                               useNativeKernel: native)
                let arr = MLXArray(ids[0 ..< 48].map { Int32($0) }, [1, 48])
                out["\(tag)_logits"] = bb(arr)
                let (h, st) = bb.bodyWithState(arr[0..., 0 ..< 37])
                out["\(tag)_hidden"] = h
                out["\(tag)_wkv"] = st.wkv
                out["\(tag)_tmix"] = st.tmixShift
                out["\(tag)_cmix"] = st.cmixShift
                // префилл некратной CHUNK длины, затем шаги
                var rs = RWKVState(cfg: bb.cfg)
                out["\(tag)_prefill"] = bb.prefill(Array(ids[0 ..< 53]), state: &rs)
                for (i, w) in rs.wkv.enumerated() where i % 5 == 0 { out["\(tag)_pre_wkv\(i)"] = w }
                out["\(tag)_pre_tmix"] = stacked(rs.tmixPrev.map { $0.asType(.float32) })
                out["\(tag)_pre_cmix"] = stacked(rs.cmixPrev.map { $0.asType(.float32) })
                var last = MLXArray(Float(0))
                for t in 53 ..< 61 { last = bb.step(ids[t], state: &rs) }
                out["\(tag)_step_logits"] = last
                out["\(tag)_step_wkv"] = rs.wkv[bb.cfg.nLayer - 1]
                // первые шаги с пустого состояния: emb и слой 0 без префилла
                var cold = RWKVState(cfg: bb.cfg)
                for t in 0 ..< 3 { last = bb.step(ids[t], state: &cold) }
                out["\(tag)_cold_logits"] = last
                eval(Array(out.values))
            }
        }
        try save(arrays: out.mapValues { $0.asType(.float32) }, url: URL(fileURLWithPath: path))
        print("дамп записан: \(path), тензоров \(out.count)")
    }
}
