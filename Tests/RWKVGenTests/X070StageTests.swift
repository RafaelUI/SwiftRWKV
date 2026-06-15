import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

/// Послойная локализация расхождения x070: сравнивает промежуточные этапы
/// блока 0 (after_ln0, blk0_tmix, after_blk0) против Python-эталона (fp32).
final class X070StageTests: XCTestCase {

    private func res(_ n: String) throws -> URL {
        guard let u = Bundle.module.url(forResource: n, withExtension: "safetensors")
        else { throw XCTSkip("\(n) не найден") }
        return u
    }
    private func diff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32)).max(); d.eval()
        return d.item(Float.self)
    }

    func testStages() throws {
        let weights = try loadArrays(url: try res("world_0.1b_x070"))
        let stages = try loadArrays(url: try res("x070_stages"))

        let cfg = X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536)
        let model = X070Backbone(weights: weights, cfg: cfg, computeDType: .bfloat16)

        let ids = MLXArray([Int32(1),2,3,4,5,6,7,8], [1, 8])
        let got = model.debugStages(ids)

        for key in ["after_ln0", "blk0_tmix", "after_blk0"] {
            guard let ref = stages[key], let sw = got[key] else { continue }
            let d = diff(sw.reshaped(ref.shape), ref)
            print("[x070 stage] \(key): Δ=\(d)")
        }
    }

    func testPerLayer() throws {
        let weights = try loadArrays(url: try res("world_0.1b_x070"))
        let ref = try loadArrays(url: try res("x070_perlayer"))
        let cfg = X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536)
        let model = X070Backbone(weights: weights, cfg: cfg, computeDType: .bfloat16)
        let ids = MLXArray([Int32(1),2,3,4,5,6,7,8], [1, 8])
        let got = model.debugPerLayer(ids)
        for i in 0 ..< 12 {
            let key = "after_blk\(i)"
            guard let rf = ref[key], let sw = got[key] else { continue }
            let dMax = diff(sw.reshaped(rf.shape), rf)
            let dMean = abs(sw.reshaped(rf.shape).asType(.float32) - rf.asType(.float32)).mean()
            dMean.eval()
            print("[x070 layer] \(key): maxΔ=\(dMax) meanΔ=\(dMean.item(Float.self))")
        }
    }

    func testTmixSubstages() throws {
        let weights = try loadArrays(url: try res("world_0.1b_x070"))
        let ref = try loadArrays(url: try res("x070_tmix"))
        let cfg = X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536)
        let model = X070Backbone(weights: weights, cfg: cfg, computeDType: .bfloat16)
        let ids = MLXArray([Int32(1),2,3,4,5,6,7,8], [1, 8])
        let got = model.debugTmix(ids)
        for key in ["r","k","v","g","a","w","kk","k2","wkv","out_lnx","res"] {
            guard let rf = ref[key], let sw = got[key] else { continue }
            let d = diff(sw.reshaped(rf.shape), rf)
            print("[x070 tmix] \(key): Δ=\(d)")
        }
    }
}
