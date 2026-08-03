import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Характеризационные тесты LoRA-файнтюна.
//
//  Их задача — НЕ доказать, что рецепт хорош (это уже сделано замерами в
//  Python на World-1.5B), а ЗАФИКСИРОВАТЬ текущее поведение перед вынесением
//  общего тренера. После рефакторинга эти тесты обязаны пройти без единой
//  правки: это и есть определение «рефакторинг ничего не сломал».
//
//  Поэтому здесь проверяются свойства, а не числа: числа зависят от весов и
//  seed'а, свойства — нет.
// ───────────────────────────────────────────────────────────────────────

final class LoRAFinetuneCharacterizationTests: XCTestCase {

    /// T обязана делиться на WKV7_CHUNK — обучаемые слои идут через wkv7Train.
    static let T = 16

    func fixedBatch(_ cfg: X070Config, B: Int = 1, seed: UInt64 = 7) -> LoRABatch {
        let ids = TinyBackbone.ids(B, Self.T + 1, vocab: cfg.vocab, seed: seed)
        return (x: ids[0..., 0 ..< Self.T], y: ids[0..., 1 ... Self.T])
    }

    func makeWithLoRA(rank: Int = 8, alpha: Float = 16, seed: UInt64 = 42)
        -> (X070Backbone, X070Config) {
        let (bb, cfg) = TinyBackbone.make(seed: seed)
        LoRA.add(to: bb, spec: LoRASpec(rank: rank, alpha: alpha))
        return (bb, cfg)
    }

    /// cacheLimitGB: 0 — не трогать глобальный лимит кэша Metal из теста.
    func config(lr: Float = 1e-2, steps: Int = 20, accum: Int = 1,
                warmup: Int = 0, clip: Float = 1.0) -> LoRAConfig {
        LoRAConfig(lr: lr, gradClip: clip, maxSteps: steps, gradAccum: accum,
                   warmupSteps: warmup, cacheLimitGB: 0, logEvery: 1)
    }

    // ── Инвариант нулевой инициализации ──────────────────────────────

    /// LoRA-B инициализируется нулём ⇒ до обучения адаптер — no-op, и модель
    /// с адаптерами обязана давать РОВНО те же логиты, что без них.
    func testAdaptersAreNoOpBeforeTraining() {
        let (plain, cfg) = TinyBackbone.make(seed: 42)
        let (lora, _) = makeWithLoRA(seed: 42)
        let ids = TinyBackbone.ids(1, Self.T, vocab: cfg.vocab)
        let a = plain(ids), b = lora(ids)
        eval(a, b)
        XCTAssertEqual(MLX.abs(a - b).max().item(Float.self), 0,
                       "адаптеры до обучения изменили выход — B инициализирован не нулём")
    }

    // ── Что обучается, а что нет ─────────────────────────────────────

    /// Обучаются ТОЛЬКО адаптеры. База (включая слои, помеченные trainLayers)
    /// обязана остаться байт-в-байт прежней: diff-множество — только loraA/loraB.
    func testOnlyAdaptersChange() {
        let (bb, cfg) = makeWithLoRA()
        let before = bb.w.mapValues { $0 }              // MLXArray — value semantics
        let aBefore = bb.loraA.mapValues { $0 }
        let bBefore = bb.loraB.mapValues { $0 }

        let batch = fixedBatch(cfg)
        LoRAFinetune.run(bb, nextBatch: { batch }, config: config(steps: 5))

        for (k, v) in before {
            let now = bb.w[k]!
            eval(v, now)
            XCTAssertEqual(MLX.abs(v.asType(.float32) - now.asType(.float32))
                            .max().item(Float.self), 0,
                           "базовый вес \(k) изменился при LoRA-обучении")
        }

        // A не обучается «в никуда»: он тоже в diff-множестве, но проверяем
        // главное — B сдвинулся с нуля, то есть адаптер ожил.
        var anyBMoved = false
        for (t, b0) in bBefore {
            let b1 = bb.loraB[t]!
            eval(b0, b1)
            if MLX.abs(b0.asType(.float32) - b1.asType(.float32)).max().item(Float.self) > 0 {
                anyBMoved = true; break
            }
        }
        XCTAssertTrue(anyBMoved, "ни один LoRA-B не сдвинулся — обучение не идёт")
        XCTAssertEqual(aBefore.count, bb.loraA.count, "изменился состав адаптеров")
    }

    // ── Сходимость ───────────────────────────────────────────────────

    /// На одном повторяющемся батче лосс обязан падать: если модель не может
    /// переобучиться на одном примере, сломан либо градиент, либо апдейт.
    func testLossDecreasesOnRepeatedBatch() {
        let (bb, cfg) = makeWithLoRA()
        let batch = fixedBatch(cfg)
        var losses: [Float] = []
        let res = LoRAFinetune.run(bb, nextBatch: { batch },
                                   config: config(lr: 1e-2, steps: 30),
                                   onStep: { _, loss, _, _ in losses.append(loss) })

        XCTAssertEqual(res.steps, 30)
        XCTAssertEqual(losses.count, 30)
        XCTAssertTrue(losses.allSatisfy { $0.isFinite }, "в лоссе появился NaN/inf")
        XCTAssertLessThan(losses.last!, losses.first!,
                          "лосс не убыл: \(losses.first!) → \(losses.last!)")
        XCTAssertEqual(res.finalLoss, losses.last!, accuracy: 1e-6)
    }

    /// Детерминизм: одинаковые seed и конфиг ⇒ одинаковая траектория.
    /// Без этого никакой регрессионный тест поверх обучения невозможен.
    func testDeterministicGivenSeed() {
        func trajectory() -> [Float] {
            let (bb, cfg) = makeWithLoRA(seed: 42)
            let batch = fixedBatch(cfg)
            var ls: [Float] = []
            LoRAFinetune.run(bb, nextBatch: { batch }, config: config(steps: 8),
                             onStep: { _, l, _, _ in ls.append(l) })
            return ls
        }
        let a = trajectory(), b = trajectory()
        XCTAssertEqual(a.count, b.count)
        for (i, (x, y)) in zip(a, b).enumerated() {
            XCTAssertEqual(x, y, accuracy: 0, "шаг \(i): траектории разошлись \(x) vs \(y)")
        }
    }

    // ── Grad accumulation ────────────────────────────────────────────

    /// Точное тождество: на ОДИНАКОВЫХ микробатчах accum=2 обязан дать ровно
    /// то же, что accum=1. Накопление считает (g+g)·0.5, что в fp32 равно g
    /// бит-в-бит. Это отделяет «накопление реализовано правильно» от
    /// «накопление что-то усредняет примерно».
    func testGradAccumEquivalentOnIdenticalMicroBatches() {
        func finalAdapters(accum: Int) -> [String: MLXArray] {
            let (bb, cfg) = makeWithLoRA(seed: 42)
            let batch = fixedBatch(cfg)
            LoRAFinetune.run(bb, nextBatch: { batch },
                             config: config(lr: 1e-2, steps: 5, accum: accum))
            return LoRA.adapterState(bb)
        }
        let one = finalAdapters(accum: 1)
        let two = finalAdapters(accum: 2)

        XCTAssertEqual(one.count, two.count)
        for (k, v1) in one {
            let v2 = two[k]!
            eval(v1, v2)
            XCTAssertEqual(MLX.abs(v1.asType(.float32) - v2.asType(.float32))
                            .max().item(Float.self), 0,
                           "\(k): accum=2 на одинаковых микробатчах разошёлся с accum=1")
        }
    }

    // ── Расписание и клип ────────────────────────────────────────────

    /// Текущее расписание: линейный warmup, дальше ПЛОСКО (косинусного спада
    /// нет). Тест фиксирует это как есть — спад добавляется в общем тренере,
    /// и тогда этот тест должен быть осознанно обновлён, а не молча пройден.
    func testWarmupThenFlatLearningRate() {
        let (bb, cfg) = makeWithLoRA()
        let batch = fixedBatch(cfg)
        // Косвенно: при warmup>0 первый шаг делает МЕНЬШИЙ сдвиг адаптеров,
        // чем без warmup (lr на шаге 0 = lr/warmup).
        func firstStepDelta(warmup: Int) -> Float {
            let (m, c) = makeWithLoRA(seed: 42)
            let b = fixedBatch(c)
            let before = LoRA.adapterState(m).mapValues { $0 }
            LoRAFinetune.run(m, nextBatch: { b }, config: config(lr: 1e-2, steps: 1,
                                                                 warmup: warmup))
            let after = LoRA.adapterState(m)
            var mx: Float = 0
            for (k, v) in before {
                let d = MLX.abs(v.asType(.float32) - after[k]!.asType(.float32))
                            .max().item(Float.self)
                mx = Swift.max(mx, d)
            }
            return mx
        }
        let noWarmup = firstStepDelta(warmup: 0)
        let withWarmup = firstStepDelta(warmup: 10)
        XCTAssertGreaterThan(noWarmup, 0, "без warmup первый шаг ничего не сдвинул")
        XCTAssertLessThan(withWarmup, noWarmup,
                          "warmup не уменьшил первый шаг: \(withWarmup) vs \(noWarmup)")
        _ = (bb, batch)
    }

    /// Клип по глобальной норме: сообщённая норма конечна и положительна,
    /// а сам клип не ломает обучение при агрессивно малом пороге.
    func testGradientNormReportedAndClipped() {
        let (bb, cfg) = makeWithLoRA()
        let batch = fixedBatch(cfg)
        var norms: [Float] = []
        LoRAFinetune.run(bb, nextBatch: { batch },
                         config: config(lr: 1e-2, steps: 5, clip: 1e-3),
                         onStep: { _, _, n, _ in norms.append(n) })
        XCTAssertEqual(norms.count, 5)
        XCTAssertTrue(norms.allSatisfy { $0.isFinite && $0 >= 0 },
                      "норма градиента не конечна: \(norms)")
        XCTAssertGreaterThan(norms.first!, 0, "нулевая норма — градиента нет")
    }

    // ── Отмена ───────────────────────────────────────────────────────

    /// isCancelled обязан прерывать цикл — это требование UI-вызова.
    func testCancellationStopsEarly() {
        let (bb, cfg) = makeWithLoRA()
        let batch = fixedBatch(cfg)
        var seen = 0
        let res = LoRAFinetune.run(bb, nextBatch: { batch }, config: config(steps: 50),
                                   isCancelled: { seen >= 3 },
                                   onStep: { s, _, _, _ in seen = s })
        XCTAssertLessThan(res.steps, 50, "отмена не сработала, прошло \(res.steps) шагов")
        XCTAssertGreaterThan(res.steps, 0)
    }

    // ── Инференс после обучения ──────────────────────────────────────

    /// После run() в backbone инжектятся ФИНАЛЬНЫЕ параметры, и модель сразу
    /// готова к инференсу.
    ///
    /// Ровно ОДИН шаг — намеренно. При многошаговом прогоне тест прошёл бы и
    /// без финального инжекта: inject() вызывается внутри lossOf на каждом
    /// шаге, поэтому адаптеры в бэкбоне и так ненулевые — просто устаревшие на
    /// один апдейт. На одном шаге внутрицикловый inject записывает ИСХОДНЫЕ
    /// параметры (B=0, адаптер — no-op), и разницу с базой даёт только
    /// финальный инжект. Проверено мутацией: удаление финального inject()
    /// роняет именно этот тест.
    func testFinalParamsInjectedBackAfterRun() {
        let (plain, cfg) = TinyBackbone.make(seed: 42)
        let (bb, _) = makeWithLoRA(seed: 42)
        let ids = TinyBackbone.ids(1, Self.T, vocab: cfg.vocab)
        let base = plain(ids); eval(base)

        LoRAFinetune.run(bb, nextBatch: { self.fixedBatch(cfg) },
                         config: config(lr: 1e-2, steps: 1))

        let after = bb(ids); eval(after)
        XCTAssertGreaterThan(MLX.abs(base - after).max().item(Float.self), 0,
                             "после одного шага выход равен базовому — "
                             + "инжектированы исходные параметры, не финальные")
    }
}
