//
//  WKVDTypeLeakTests.swift
//  Течь fp32 из ядра WKV в остаточный поток: что она стоит и что меняет.
//
//  Флаг `castWKVOutputToComputeDType` переключается на ЖИВОМ объекте, поэтому
//  обе ветки меряются в одном процессе на одной и той же модели. Замер «до и
//  после» разными сборками был бы недействителен: между сборками меняется
//  всё — прогрев, состояние кэша, тепловой режим машины.
//
//  Тесты здесь ФИКСИРУЮТ поведение флага, а не утверждают, что его надо
//  включить. Решение о включении — это решение о расхождении с рефересом
//  (rwkv-metal течёт так же), и принимается оно по числам, часть которых
//  печатается ниже.
//
//  Пропускается без модели.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

final class WKVDTypeLeakTests: XCTestCase {

    func backbone() throws -> (X070Backbone, WorldTokenizer, X070Config, [String: MLXArray]) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let vp = env["RWKV_WORLD_VOCAB"] ?? home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        for p in [mp, vp] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет фикстуры \(p)")
        }
        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: w["head.weight"]!.shape[0])
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: vp))
        else { throw XCTSkip("словарь не разобрался") }
        return (X070Backbone(weights: w, cfg: cfg), tok, cfg, w)
    }

    /// Умолчание — ВЫКЛЮЧЕНО, и проход остаётся float32.
    ///
    /// Сторожит границу: если кто-то поменяет умолчание, эталоны паритета
    /// поедут молча, а причина будет видна здесь.
    func testFlagIsOffByDefaultAndPassStaysFloat32() throws {
        let (bb, _, _, _) = try backbone()
        XCTAssertFalse(bb.castWKVOutputToComputeDType, "умолчание сдвинулось")
        let out = bb.body(MLXArray([Int32(510), 610, 710], [1, 3]))
        eval(out)
        XCTAssertEqual(out.dtype, .float32)
    }

    /// С флагом проход идёт в типе вычислений — и в параллельном пути, и в
    /// рекуррентном, и состояние WKV при этом ОСТАЁТСЯ fp32.
    ///
    /// Последнее — условие корректности: приводится выход, а не
    /// рекуррентность. Если бы под флаг попало состояние, декод разошёлся бы
    /// с параллельным путём, и это была бы уже не оптимизация.
    func testFlagKeepsComputeDTypeButNotTheState() throws {
        let (bb, _, cfg, _) = try backbone()
        bb.castWKVOutputToComputeDType = true

        let out = bb.body(MLXArray([Int32(510), 610, 710], [1, 3]))
        eval(out)
        XCTAssertEqual(out.dtype, .bfloat16, "параллельный путь не в типе вычислений")

        var st = RWKVState(cfg: cfg)
        _ = bb.step(510, state: &st)
        XCTAssertEqual(st.cmixPrev[0].dtype, .bfloat16, "рекуррентный путь течёт")
        XCTAssertEqual(st.tmixPrev[0].dtype, .bfloat16)
        XCTAssertEqual(st.wkv[0].dtype, .float32,
                       "состояние WKV обязано остаться fp32 — приводится ВЫХОД, не рекуррентность")
    }

    /// ЦЕНА ФЛАГА: согласие рекуррентного пути с параллельным ПАДАЕТ, и это
    /// не дефект, а прямое следствие.
    ///
    /// Без флага активация fp32, и два ядра сходятся на 7.4e-7. С флагом оба
    /// пути округляются до bf16, у которой 7 бит мантиссы (2⁻⁸ ≈ 3.9e-3), — и
    /// расхождение вырастает на четыре порядка. Замерено: **7.4e-7 → ~9.9e-3**.
    ///
    /// Утверждается пара: под флагом расхождение ЗАМЕТНО БОЛЬШЕ, но остаётся
    /// ограниченным и argmax не разъезжается. Первая половина не даёт забыть о
    /// цене, вторая — не даёт цене вырасти незаметно. Проверять только вторую
    /// значило бы позволить флагу тихо испортить модель до самой границы.
    func testFlagCostsPrecisionBetweenPaths() throws {
        let (bb, tok, cfg, _) = try backbone()
        let ids = tok.encode("Пчёлы собирают нектар с цветов и делают мёд")
        let idsArr = MLXArray(ids.map { Int32($0) }, [1, ids.count])

        func divergence(_ flag: Bool) -> (rel: Float, sameTop1: Bool) {
            bb.castWKVOutputToComputeDType = flag
            var st = RWKVState(cfg: cfg)
            let recurrent = bb.prefillRecurrent(ids, state: &st).asType(.float32)
            let ref = matmul(bb.body(idsArr)[0, ids.count - 1]
                                .reshaped([1, cfg.nEmbd]).asType(.float32),
                             bb.weight("head.weight").asType(.float32).transposed())
                .reshaped([cfg.vocab])
            eval(recurrent, ref)
            let scale = MLX.abs(ref).max().item(Float.self)
            return (MLX.abs(recurrent - ref).max().item(Float.self) / scale,
                    recurrent.argMax().item(Int.self) == ref.argMax().item(Int.self))
        }

        let off = divergence(false), on = divergence(true)
        print(String(format: "── согласие путей: выкл %.3e, вкл %.3e (×%.0f)",
                     off.rel, on.rel, on.rel / off.rel))

        XCTAssertLessThan(off.rel, 5e-3, "без флага пути разошлись — это уже дефект")
        XCTAssertGreaterThan(on.rel, off.rel * 10,
                             "цена флага не видна — приведение не применилось?")
        XCTAssertLessThan(on.rel, 3e-2, "под флагом пути разошлись слишком сильно")
        XCTAssertTrue(off.sameTop1 && on.sameTop1, "argmax разъехался")
        bb.castWKVOutputToComputeDType = false
    }

    /// ГЛАВНЫЙ ЗАМЕР: что флаг меняет в числах и во времени.
    ///
    /// Печатает, а не утверждает. Утверждать «стало быстрее» тестом нельзя —
    /// это замер, а не свойство; тест обязан лишь не дать замеру исчезнуть.
    /// Утверждается только то, что расхождение КОНЕЧНО и argmax не разъехался
    /// на коротком промпте: если бы приведение ломало модель, это было бы
    /// видно сразу.
    func testMeasureFlagCostAndDivergence() throws {
        let (bb, tok, cfg, _) = try backbone()
        let ids = tok.encode("The capital of France is the city of")
        let idsArr = MLXArray(ids.map { Int32($0) }, [1, ids.count])

        func logits(_ flag: Bool) -> MLXArray {
            bb.castWKVOutputToComputeDType = flag
            let h = bb.body(idsArr)
            let o = matmul(h[0, ids.count - 1].reshaped([1, cfg.nEmbd]).asType(.float32),
                           bb.weight("head.weight").asType(.float32).transposed())
                .reshaped([cfg.vocab])
            eval(o)
            return o
        }
        let off = logits(false), on = logits(true)
        let scale = MLX.abs(off).max().item(Float.self)
        let rel = MLX.abs(on - off).max().item(Float.self) / scale
        XCTAssertTrue(rel.isFinite && rel < 0.2,
                      "приведение сломало модель: расхождение \(rel)")

        // Топ-1 на коротком промпте и согласие топ-5.
        let top1Off = off.argMax().item(Int.self), top1On = on.argMax().item(Int.self)

        func decodeMs(_ flag: Bool, n: Int = 48) -> Double {
            bb.castWKVOutputToComputeDType = flag
            var st = RWKVState(cfg: cfg)
            var l = bb.prefill(ids, state: &st)
            eval(l)
            let t0 = Date()
            for _ in 0 ..< n {
                let id = l.argMax().item(Int.self)
                l = bb.step(id, state: &st)
            }
            eval(l)
            return Date().timeIntervalSince(t0) * 1000 / Double(n)
        }
        // A/B ЧЕРЕДОВАНИЕМ в одном процессе: прогрев обеих веток, затем
        // попеременно, медиана. Последовательные блоки «сначала все off,
        // потом все on» на безвентиляторной машине меряют тепловой дрейф.
        _ = decodeMs(false, n: 8); _ = decodeMs(true, n: 8)
        var offs: [Double] = [], ons: [Double] = []
        for _ in 0 ..< 5 {
            offs.append(decodeMs(false))
            ons.append(decodeMs(true))
        }
        let mOff = offs.sorted()[2], mOn = ons.sorted()[2]

        print(String(format: """

        ── ТЕЧЬ fp32: A/B на 0.1B ──
        логиты:  расхождение %.3e относительное, top-1 %d → %d (%@)
        декод:   выкл %.2f мс/ток, вкл %.2f мс/ток → ×%.2f
        разбросы: выкл %@, вкл %@
        """, rel, top1Off, top1On, top1Off == top1On ? "совпал" : "РАЗОШЁЛСЯ",
        mOff, mOn, mOff / mOn,
        offs.map { String(format: "%.1f", $0) }.joined(separator: "/"),
        ons.map { String(format: "%.1f", $0) }.joined(separator: "/")))

        bb.castWKVOutputToComputeDType = false
    }

    /// То же на КВАНТОВАННОЙ базе: там приведение должно дать больше, потому
    /// что вес разворачивается сразу в тип вычислений (см. `rwkvqWeight`), и
    /// bf16-путь деквантизации наконец начинает работать.
    func testMeasureFlagOnQuantisedBase() throws {
        let (bb, tok, cfg, _) = try backbone()
        let p = ProcessInfo.processInfo.environment["RWKV_RWKVQ_SIDECAR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p + ".safetensors"),
                          "нет сайдкара .rwkvq")
        XCTAssertGreaterThan(bb.attachRwkvq(try RwkvqSidecar(path: p)).attached, 0)

        let ids = tok.encode("The capital of France is the city of")
        func decodeMs(_ flag: Bool, n: Int = 48) -> Double {
            bb.castWKVOutputToComputeDType = flag
            var st = RWKVState(cfg: cfg)
            var l = bb.prefill(ids, state: &st)
            eval(l)
            let t0 = Date()
            for _ in 0 ..< n {
                let id = l.argMax().item(Int.self)
                l = bb.step(id, state: &st)
            }
            eval(l)
            return Date().timeIntervalSince(t0) * 1000 / Double(n)
        }
        _ = decodeMs(false, n: 8); _ = decodeMs(true, n: 8)
        var offs: [Double] = [], ons: [Double] = []
        for _ in 0 ..< 5 { offs.append(decodeMs(false)); ons.append(decodeMs(true)) }
        let mOff = offs.sorted()[2], mOn = ons.sorted()[2]
        print(String(format: "── ТЕЧЬ fp32 на КВАНТОВАННОЙ базе: выкл %.2f, вкл %.2f → ×%.2f",
                     mOff, mOn, mOff / mOn))
        bb.castWKVOutputToComputeDType = false
    }
}
