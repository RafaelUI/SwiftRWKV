import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Инициализация x070 для предобучения.
//
//  Проверяются СВОЙСТВА, а не значения: точные числа зависят от seed, а
//  свойства обязаны выполняться всегда. Самое диагностичное — стартовый
//  лосс ≈ ln(vocab): у корректно инициализированной языковой модели первый
//  прогноз равномерен, поэтому cross-entropy равна логарифму словаря. Всё,
//  что ломает масштабы (слишком широкие проекции, ненейтральная динамика,
//  забытый o_proj=0), сдвигает это число сразу и заметно.
// ───────────────────────────────────────────────────────────────────────

final class X070InitTests: XCTestCase {

    static let cfg = X070Config(nLayer: 4, nEmbd: 128, headSize: 64, vocab: 256)

    func make(seed: UInt64 = 0) -> X070Backbone {
        X070Init.makeBackbone(cfg: Self.cfg, init: X070InitConfig(seed: seed),
                              computeDType: .float32)
    }

    // ── Главный диагностический тест ─────────────────────────────────

    /// Стартовый лосс должен лежать ЧУТЬ ВЫШЕ ln(vocab).
    ///
    /// Почему не ровно ln(vocab): голова инициализируется ортогонально с
    /// gain = 0.5·√(V/C), то есть стартовые логиты не нулевые, а имеют
    /// ненулевую дисперсию σ². Для случайных целей E[CE] ≈ ln(V) + σ²/2,
    /// поэтому ждать точного равенства неверно — оно было бы верно только
    /// для нулевой головы.
    ///
    /// Обе границы содержательны. Снизу: лосс ниже ln(V) на случайных данных
    /// означал бы, что модель уже «что-то знает» — то есть утечку. Сверху:
    /// разъехавшиеся масштабы (двойной множитель в gain головы, забытый
    /// o_proj=0, широкий k_proj) поднимают лосс сразу и заметно.
    func testInitialLossIsSlightlyAboveLogVocab() {
        let bb = make()
        let ids = TinyBackbone.ids(2, 32, vocab: Self.cfg.vocab, seed: 5)
        let x = ids[0..., 0 ..< 31], y = ids[0..., 1 ... 31]
        let loss = LoRAFinetune.languageModelLoss(bb, (x: x, y: y))
        eval(loss)
        let value = loss.item(Float.self)
        let lnV = log(Float(Self.cfg.vocab))

        XCTAssertTrue(value.isFinite, "стартовый лосс не конечен: \(value)")
        XCTAssertGreaterThan(value, lnV - 0.05,
                             "лосс \(value) НИЖЕ ln(vocab)=\(lnV) на случайных "
                             + "данных — похоже на утечку цели")
        // Верхняя граница вычислена, а не подобрана. Тело на старте —
        // тождественное, поэтому logits = h·headᵀ, где ‖h‖² ≈ C, а
        // head = g·U с ортонормальными столбцами. Тогда дисперсия логита
        //     σ² = g²·C/V = (0.25·V/C)·C/V = 0.25   (не зависит от V и C —
        // в этом и смысл gain = 0.5·√(V/C)), откуда E[CE] ≈ ln V + σ²/2
        // = ln V + 0.125. Допуск 0.2 оставляет запас на выборочный шум, но
        // ловит удвоение множителя в gain (оно даёт σ² = 0.5 и +0.25).
        XCTAssertLessThan(value, lnV + 0.2,
                          "лосс \(value) выше ln(vocab)+0.2 (=\(lnV + 0.2)) — "
                          + "масштабы инициализации разъехались")
    }

    /// Усиление головы — ТОЧНО, а не через лосс.
    ///
    /// Косвенная проверка через лосс отличает правильный gain от удвоенного
    /// всего на ~0.12 натуральных единиц, что слишком близко к выборочному
    /// шуму. Здесь свойство проверяется структурно: headᵀ·head = gain²·I при
    /// V > C, где gain = 0.5·√(V/C) применяется БЕЗ добавочного множителя
    /// √(V/C) из ortho_init. Именно на этой разнице легко ошибиться:
    /// официал зовёт для головы orthogonal_, а не ortho_init.
    func testHeadGainIsExact() {
        for (V, C) in [(256, 128), (512, 128), (128, 128)] {
            let cfg = X070Config(nLayer: 1, nEmbd: C, headSize: 64, vocab: V)
            let w = X070Init.weights(cfg: cfg)
            let head = w["head.weight"]!.asType(.float32)     // [V, C]
            let gram = matmul(head.transposed(), head)        // [C, C]
            let expected = Float(V > C ? 0.25 * Double(V) / Double(C) : 0.25)
            let eye = MLXArray.eye(C) * expected
            eval(gram, eye)
            let err = MLX.abs(gram - eye).max().item(Float.self)
            XCTAssertLessThan(err, 1e-3,
                              "V=\(V) C=\(C): headᵀ·head ≠ \(expected)·I "
                              + "(max|Δ| = \(err)) — неверное усиление головы")
        }
    }

    /// Forward конечен и не вырожден на всех глубинах и размерах.
    func testForwardIsFiniteAcrossShapes() {
        for (nLayer, nEmbd, vocab) in [(1, 64, 128), (4, 128, 256), (6, 192, 512)] {
            let cfg = X070Config(nLayer: nLayer, nEmbd: nEmbd, headSize: 64, vocab: vocab)
            let bb = X070Init.makeBackbone(cfg: cfg, computeDType: .float32)
            let out = bb(TinyBackbone.ids(1, 16, vocab: vocab))
            eval(out)
            let maxAbs = MLX.abs(out).max().item(Float.self)
            XCTAssertTrue(maxAbs.isFinite,
                          "L=\(nLayer) D=\(nEmbd): нефинитный выход")
            XCTAssertGreaterThan(maxAbs, 0,
                                 "L=\(nLayer) D=\(nEmbd): выход тождественно нулевой")
        }
    }

    // ── Нейтральность динамики на нулевом шаге ───────────────────────

    /// LoRA-A = нули ⇒ decay/iclr/gate/v-residual на старте определяются
    /// ТОЛЬКО смещениями w0/a0/v0. Это то, что делает старт предсказуемым.
    func testLowRankAMatricesAreZero() {
        let bb = make()
        for layer in 0 ..< Self.cfg.nLayer {
            for name in ["w", "a", "g"] + (layer > 0 ? ["v"] : []) {
                let key = "blocks.\(layer).tmix.\(name)_lora_A.weight"
                let a = bb.w[key]!
                eval(a)
                XCTAssertEqual(MLX.abs(a).max().item(Float.self), 0,
                               "\(key) не нулевая — динамика не нейтральна на старте")
            }
        }
    }

    /// o_proj и cmix.value = нули ⇒ на нулевом шаге блоки ничего не вносят
    /// в остаточный поток, сеть стартует как identity. Без этого глубокая
    /// модель разваливается на первых шагах.
    func testResidualOutputsStartAtZero() {
        let bb = make()
        for layer in 0 ..< Self.cfg.nLayer {
            for key in ["blocks.\(layer).tmix.o_proj.weight",
                        "blocks.\(layer).cmix.value.weight"] {
                let v = bb.w[key]!
                eval(v)
                XCTAssertEqual(MLX.abs(v).max().item(Float.self), 0,
                               "\(key) не нулевая — блок вносит вклад на нулевом шаге")
            }
        }
    }

    /// Прямое следствие предыдущего: скрытое состояние на выходе body()
    /// равно ln_out(ln0(emb)), т.е. блоки — тождественны.
    func testBodyIsIdentityAtInit() {
        let bb = make()
        let ids = TinyBackbone.ids(1, 16, vocab: Self.cfg.vocab)
        let got = bb.body(ids)

        // эталон: та же цепочка без блоков
        let emb = bb.w["emb.weight"]!.take(ids, axis: 0)
        func ln(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
            let mean = x.mean(axis: -1, keepDims: true)
            let varc = (x - mean).square().mean(axis: -1, keepDims: true)
            return (x - mean) / sqrt(varc + 1e-5) * w + b
        }
        let expected = ln(ln(emb, bb.w["ln0.weight"]!, bb.w["ln0.bias"]!),
                          bb.w["ln_out.weight"]!, bb.w["ln_out.bias"]!)
        eval(got, expected)
        let rel = MLX.abs(got - expected).max().item(Float.self)
                / (MLX.abs(expected).max().item(Float.self) + 1e-9)
        XCTAssertLessThan(rel, 1e-4, "блоки не тождественны на нулевом шаге")
    }

    // ── Кривые ───────────────────────────────────────────────────────

    /// w0 задаёт РАЗБРОС горизонтов памяти по каналам. Плоская
    /// инициализация означала бы одинаковое затухание везде — то, ради чего
    /// поканальный decay и существует, было бы потеряно.
    func testDecayBiasIsSpreadAcrossChannels() {
        let bb = make()
        for layer in 0 ..< Self.cfg.nLayer {
            let w0 = bb.w["blocks.\(layer).tmix.w_lora_B.bias"]!.asType(.float32)
            eval(w0)
            let lo = w0.min().item(Float.self), hi = w0.max().item(Float.self)
            XCTAssertGreaterThan(hi - lo, 3.0,
                                 "слой \(layer): w0 почти плоский (разброс \(hi - lo))")
            XCTAssertTrue(lo.isFinite && hi.isFinite)
        }
    }

    /// С глубиной память становится ДОЛЬШЕ, и выражается это ПОНИЖЕНИЕМ w0.
    ///
    /// Цепочка знаков легко читается наоборот, поэтому по шагам:
    ///   www = −6 + 6·(n/(C−1))^expo,  expo = 1 + ratio01^0.3 растёт с глубиной;
    ///   больший показатель ⇒ кривая дольше держится у −6 ⇒ среднее НИЖЕ;
    ///   w = exp(−0.606531·sigmoid(w0 + …)) ⇒ меньший w0 ⇒ sigmoid ближе к 0
    ///   ⇒ показатель ближе к 0 ⇒ w ближе к 1 ⇒ затухание СЛАБЕЕ.
    /// Итог: ниже w0 = дольше память, и наверху сети её больше.
    func testDecayBiasDecreasesWithDepthMeaningLongerMemory() {
        let bb = make()
        var means: [Float] = []
        for layer in 0 ..< Self.cfg.nLayer {
            let w0 = bb.w["blocks.\(layer).tmix.w_lora_B.bias"]!.asType(.float32)
            eval(w0)
            means.append(w0.mean().item(Float.self))
        }
        XCTAssertLessThan(means.last!, means.first!,
                          "средний w0 не убывает с глубиной: \(means)")

        // и то же напрямую через эффективное затухание w = exp(−0.606531·σ(w0))
        func decay(_ m: Float) -> Float { exp(-0.606531 * (1 / (1 + exp(-m)))) }
        XCTAssertGreaterThan(decay(means.last!), decay(means.first!),
                             "затухание наверху должно быть слабее (w ближе к 1)")
    }

    /// ln_x.weight = ((layer+1)/L)^0.7 — не единица. Это компенсация роста
    /// дисперсии с глубиной, и его легко потерять при переносе.
    func testLnXWeightScalesWithLayer() {
        let bb = make()
        let L = Self.cfg.nLayer
        for layer in 0 ..< L {
            let expected = Float(pow(Double(layer + 1) / Double(L), 0.7))
            let got = bb.w["blocks.\(layer).tmix.ln_x.weight"]!.asType(.float32)
            eval(got)
            XCTAssertEqual(got.mean().item(Float.self), expected, accuracy: 1e-5,
                           "ln_x.weight слоя \(layer) должен быть \(expected)")
            XCTAssertEqual(got.max().item(Float.self) - got.min().item(Float.self), 0,
                           "ln_x.weight должен быть постоянным по каналам")
        }
    }

    /// k_proj в 10 раз уже r/v_proj: ключ входит в рекуррентность напрямую,
    /// широкий старт раскачивает состояние.
    func testKeyProjectionIsDamped() {
        let bb = make()
        for layer in 0 ..< Self.cfg.nLayer {
            let tp = "blocks.\(layer).tmix."
            let k = bb.w[tp + "k_proj.weight"]!.asType(.float32)
            let r = bb.w[tp + "r_proj.weight"]!.asType(.float32)
            eval(k, r)
            let kMax = MLX.abs(k).max().item(Float.self)
            let rMax = MLX.abs(r).max().item(Float.self)
            XCTAssertLessThan(kMax, rMax * 0.25,
                              "слой \(layer): k_proj не демпфирован (\(kMax) vs \(rMax))")
        }
    }

    // ── Ортогональность ──────────────────────────────────────────────

    /// QR-инициализация обязана давать действительно ортогональные строки
    /// (или столбцы для «широкой» матрицы), масштабированные на gain.
    func testOrthogonalInitIsOrthogonal() {
        for shape in [[64, 32], [32, 64], [128, 128]] {
            let gain: Float = 0.1
            let q = X070Init.orthogonal(shape, gain: gain)
            eval(q)
            let rows = shape[0], cols = shape[1]
            // берём меньшую сторону: G = Q·Qᵀ или Qᵀ·Q должна быть gain²·I
            let g = rows <= cols ? matmul(q, q.transposed())
                                 : matmul(q.transposed(), q)
            let n = Swift.min(rows, cols)
            // gain применяется КАК ЕСТЬ (как nn.init.orthogonal_), без
            // множителя √(rows/cols) — тот живёт отдельно, в orthoInit.
            let expectedScale = gain * gain
            let eye = MLXArray.eye(n) * expectedScale
            eval(g, eye)
            let err = MLX.abs(g - eye).max().item(Float.self)
            XCTAssertLessThan(err, 1e-4,
                              "shape \(shape): не ортогональна (max|G−gain²I| = \(err))")
        }
    }

    // ── Воспроизводимость и ранги ────────────────────────────────────

    func testDeterministicGivenSeed() {
        let a = X070Init.weights(cfg: Self.cfg, init: X070InitConfig(seed: 7))
        let b = X070Init.weights(cfg: Self.cfg, init: X070InitConfig(seed: 7))
        let c = X070Init.weights(cfg: Self.cfg, init: X070InitConfig(seed: 8))
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        var anyDiffAcrossSeeds = false
        for (k, va) in a {
            let vb = b[k]!, vc = c[k]!
            eval(va, vb, vc)
            XCTAssertEqual(MLX.abs(va - vb).max().item(Float.self), 0,
                           "\(k): один seed дал разные веса")
            if MLX.abs(va - vc).max().item(Float.self) > 0 { anyDiffAcrossSeeds = true }
        }
        XCTAssertTrue(anyDiffAcrossSeeds, "разные seed дали одинаковые веса")
    }

    /// Слой 0 не имеет v_lora (value-residual появляется с первого слоя).
    func testLayerZeroHasNoValueResidual() {
        let bb = make()
        XCTAssertNil(bb.w["blocks.0.tmix.v_lora_A.weight"])
        XCTAssertNil(bb.w["blocks.0.tmix.v_lora_B.bias"])
        XCTAssertNotNil(bb.w["blocks.1.tmix.v_lora_A.weight"])
        XCTAssertNotNil(bb.w["blocks.1.tmix.v_lora_B.bias"])
    }

    /// Пресеты рангов: формулы разные, но округление до кратного 32 их местами
    /// СХЛОПЫВАЕТ. При D=768 оба дают w=64 и g=128 — там их не отличить, и
    /// проверять надо на размерности, где они реально расходятся.
    func testRankPresets() {
        // D=768: формулы разные, результат одинаковый — фиксируем это явно,
        // чтобы не принять совпадение за ошибку.
        let m768 = X070InitConfig.LoRARanks.rwkvMetal.ranks(nEmbd: 768)
        let o768 = X070InitConfig.LoRARanks.official.ranks(nEmbd: 768)
        XCTAssertEqual(m768.w, o768.w, "при D=768 округление схлопывает w")
        XCTAssertEqual(m768.g, o768.g, "при D=768 округление схлопывает g")

        // D=4096 (√D = 64): расходятся по w, a и g, но НЕ по v.
        let m = X070InitConfig.LoRARanks.rwkvMetal.ranks(nEmbd: 4096)
        let o = X070InitConfig.LoRARanks.official.ranks(nEmbd: 4096)
        XCTAssertEqual(m.w, 128)    // 1.8·64 = 115.2 → round(3.6) = 4 → 128
        XCTAssertEqual(o.w, 160)    // 2.5·64 = 160.0 → round(5.0) = 5 → 160
        XCTAssertEqual(m.a, 128)
        XCTAssertEqual(o.a, 160)
        XCTAssertEqual(m.g, 480)    // 0.6·4096^0.8 = 465.6 → round(14.55) = 15 → 480
        XCTAssertEqual(o.g, 320)    // 5·64 = 320 → round(10) = 10 → 320
        // v совпадает: 1.3·64 = 83.2 и 1.7·64 = 108.8 округляются оба к 3·32.
        XCTAssertEqual(m.v, 96)
        XCTAssertEqual(o.v, 96)

        // нижняя граница 32 соблюдается на маленьких моделях
        let tiny = X070InitConfig.LoRARanks.rwkvMetal.ranks(nEmbd: 64)
        XCTAssertGreaterThanOrEqual(tiny.w, 32)

        let e = X070InitConfig.LoRARanks.explicit(w: 64, a: 64, v: 32, g: 96)
            .ranks(nEmbd: 768)
        XCTAssertEqual(e.w, 64); XCTAssertEqual(e.g, 96)
    }

    // ── Обучаемость ──────────────────────────────────────────────────

    /// Инициализированная с нуля модель обучается: на повторяющемся батче
    /// лосс уверенно уходит ниже ln(vocab). Проверяет всю связку
    /// инициализация → градиенты через ядро → AdamW.
    func testInitializedModelTrains() {
        let cfg = X070Config(nLayer: 2, nEmbd: 128, headSize: 64, vocab: 128)
        let bb = X070Init.makeBackbone(cfg: cfg, computeDType: .float32)
        bb.trainLayers = Set(0 ..< cfg.nLayer)

        let ids = TinyBackbone.ids(1, 17, vocab: cfg.vocab, seed: 7)
        let batch: LoRABatch = (x: ids[0..., 0 ..< 16], y: ids[0..., 1 ... 16])
        let set = BackboneWeightsTrainableSet(
            bb, keys: BackboneWeightsTrainableSet.topLayerKeys(bb, from: 0))

        var losses: [Float] = []
        Trainer<LoRABatch>(
            trainable: set,
            objective: { LoRAFinetune.languageModelLoss(bb, $0) },
            nextBatch: { batch },
            config: TrainingConfig(lr: 1e-3, schedule: .cosine, maxSteps: 40,
                                   cacheLimitGB: 0, logEvery: 1)
        ).run { losses.append($0.loss) }

        XCTAssertTrue(losses.allSatisfy { $0.isFinite }, "NaN/inf при обучении с нуля")
        // первый залогированный лосс — уже ПОСЛЕ первого шага, поэтому просто
        // проверяем, что старт в разумной окрестности ln(vocab), а не точно в нём
        XCTAssertEqual(losses.first!, log(Float(cfg.vocab)), accuracy: 0.4)
        XCTAssertLessThan(losses.last!, losses.first! - 0.5,
                          "модель не учится: \(losses.first!) → \(losses.last!)")
    }
}
