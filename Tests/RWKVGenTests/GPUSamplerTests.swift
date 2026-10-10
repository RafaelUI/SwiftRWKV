import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  GPUSampler: тот же алгоритм выбора, что у CPU-сэмплера `Sampler`, но без
//  чтения логитов на CPU — ради конвейерного декода с температурой.
//
//  Главный тест — подача ОДНОГО И ТОГО ЖЕ равномерного числа u обоим
//  алгоритмам: поток случайных чисел у них разный по построению, а всё
//  остальное (штрафы, температура, порядок, усечение top-k / top-p,
//  перенормировка, выбор по накопленной массе) обязано совпасть.
// ───────────────────────────────────────────────────────────────────────

final class GPUSamplerTests: XCTestCase {

    /// Обратная функция распределения CPU-сэмплера при заданном u — копия
    /// `Sampler.pick` / `inverseCDF` (там u тянется из генератора внутри).
    static func cpuPick(_ logits: MLXArray, _ config: SamplingConfig, history: PenaltyState, u: Float) -> Int {
        let adjusted = history.adjust(logits.asType(.float32), config: config)
        if config.temperature <= 0 { return adjusted.argMax().item(Int.self) }
        let scaled = adjusted / config.temperature
        let order = MLX.argSort(-scaled)
        let probs = MLX.softmax(MLX.take(scaled, order), axis: -1)
        let p = probs.asArray(Float.self), ids = order.asArray(Int32.self)
        let limit = config.topK > 0 ? min(config.topK, p.count) : p.count
        var kept = 0
        var mass: Float = 0
        while kept < limit {
            mass += p[kept]; kept += 1
            if mass >= config.topP { break }
        }
        let target = u * mass
        var acc: Float = 0
        for i in 0 ..< kept {
            acc += p[i]
            if target < acc { return Int(ids[i]) }
        }
        return Int(ids[kept - 1])
    }

    static func logits(_ seed: UInt64, vocab: Int = 2000, sharp: Float = 3) -> MLXArray {
        MLXRandom.seed(seed)
        let l = MLXRandom.normal([vocab]) * sharp
        eval(l)
        return l
    }

    func testSameUniformGivesTheSameTokenAsTheCPUSampler() {
        let configs = [
            SamplingConfig(temperature: 1.0),
            SamplingConfig(temperature: 0.7, topK: 40),
            SamplingConfig(temperature: 1.3, topP: 0.9),
            SamplingConfig(temperature: 0.8, topK: 50, topP: 0.3),
            SamplingConfig(temperature: 1.0, topK: 1),
            SamplingConfig(temperature: 1.0, topP: 0.01),
            SamplingConfig(temperature: 0.9, topK: 100, topP: 0.95, repetitionPenalty: 1.3,
                           presencePenalty: 0.2, frequencyPenalty: 0.1, penaltyDecay: 0.9),
        ]
        var checked = 0, mismatched = 0
        for (ci, config) in configs.enumerated() {
            var history = PenaltyState()
            for id in [5, 17, 5, 900, 1203, 17, 17] { history.observe(id, decay: config.penaltyDecay) }
            let sampler = GPUSampler(config: config, vocab: 2000, history: history)
            for seed in 0 ..< 6 {
                let l = Self.logits(UInt64(100 + 10 * ci + seed))
                for u in [Float(0), 0.013, 0.2, 0.37, 0.5, 0.61, 0.83, 0.97, 0.9999] {
                    // Самый хвост неусечённого распределения (вероятности
                    // порядка 1e-7 у двух тысяч токенов) зависит от порядка
                    // накопления суммы в float: на CPU он последовательный, на
                    // GPU — свой. Это не алгоритм, а последний разряд, и там u
                    // у границы единицы не сверяется.
                    if u > 0.99, config.topK == 0, config.topP == 1 { continue }
                    let got = sampler.pick(l, uniform: MLXArray([u])).item(Int.self)
                    let want = Self.cpuPick(l, config, history: history, u: u)
                    checked += 1
                    if got != want { mismatched += 1; print("РАСХОЖДЕНИЕ cfg \(ci) seed \(seed) u \(u): gpu \(got) cpu \(want)") }
                }
            }
        }
        XCTAssertEqual(mismatched, 0, "из \(checked) выборов разошлись \(mismatched)")
        XCTAssertEqual(checked, 7 * 6 * 9 - 6)
    }

    /// Штрафы на GPU против `PenaltyState.adjust` при одной и той же истории,
    /// которую оба сэмплера ведут сами: затухание и отсечение мелких весов.
    func testPenaltiesFollowTheCPUHistory() {
        let config = SamplingConfig(temperature: 1, repetitionPenalty: 1.25, presencePenalty: 0.3,
                                    frequencyPenalty: 0.15, penaltyDecay: 0.5)
        var cpu = PenaltyState()
        var gpu = GPUSampler(config: config, vocab: 500)
        let l = Self.logits(7, vocab: 500)
        // 30 наблюдений при затухании 0.5: ранние веса уходят ниже 1e-6 и отсекаются
        for i in 0 ..< 30 {
            let id = (i * 37 + 11) % 500
            cpu.observe(id, decay: config.penaltyDecay)
            gpu.observe(MLXArray([Int32(id)]))
        }
        let want = cpu.adjust(l, config: config), got = gpu.adjusted(l)
        let err = MLX.abs(got - want).max().item(Float.self)
        XCTAssertLessThan(err, 1e-5, "логиты после штрафов разошлись на \(err)")
        XCTAssertGreaterThan(MLX.abs(want - l).max().item(Float.self), 0.1, "штрафы ничего не изменили")
        // отсечение: токены, чей вес ушёл ниже 1e-6, логит не трогают
        let untouchedCPU = (want .== l).asType(.int32).sum().item(Int.self)
        let untouchedGPU = (got .== l).asType(.int32).sum().item(Int.self)
        XCTAssertEqual(untouchedGPU, untouchedCPU, "набор оштрафованных токенов разный")
        // история из промпта переносится из CPU-состояния как есть
        let seeded = GPUSampler(config: config, vocab: 500, history: cpu)
        XCTAssertLessThan(MLX.abs(seeded.adjusted(l) - want).max().item(Float.self), 1e-5)
        // без штрафов история не ведётся и логиты не меняются
        var plain = GPUSampler(config: SamplingConfig(temperature: 1), vocab: 500)
        plain.observe(MLXArray([Int32(3)]))
        XCTAssertEqual((plain.adjusted(l) .!= l).asType(.int32).sum().item(Int.self), 0)
    }

    /// Seed воспроизводит последовательность; другой seed даёт другую;
    /// жадный выбор ключ генератора не трогает.
    func testSeedReproducesAndGreedyDoesNotTouchTheKey() {
        let l = Self.logits(3)
        func draw(_ seed: UInt64, _ n: Int) -> [Int] {
            var s = GPUSampler(config: SamplingConfig(temperature: 1.0, seed: seed), vocab: 2000)
            return (0 ..< n).map { _ in s.next(l).item(Int.self) }
        }
        let a = draw(42, 40)
        XCTAssertEqual(a, draw(42, 40), "один seed дал разные последовательности")
        XCTAssertNotEqual(a, draw(43, 40), "разные seed дали одну последовательность")
        XCTAssertGreaterThan(Set(a).count, 5, "сэмплер с температурой выдаёт один и тот же токен")

        var greedy = GPUSampler(vocab: 2000)
        let before = greedy.arrays[0]
        let t = greedy.next(l)
        XCTAssertEqual(t.item(Int.self), l.argMax().item(Int.self))
        XCTAssertEqual((greedy.arrays[0] .!= before).asType(.int32).sum().item(Int.self), 0, "жадный выбор сдвинул ключ")
        var hot = GPUSampler(config: SamplingConfig(temperature: 1.0, seed: 1), vocab: 2000)
        let k0 = hot.arrays[0]
        _ = hot.next(l)
        XCTAssertGreaterThan((hot.arrays[0] .!= k0).asType(.int32).sum().item(Int.self), 0, "ключ не расщепляется")
    }

    /// Частоты выбора сходятся к распределению: три токена с вероятностями
    /// 0.5 / 0.3 / 0.2 при температуре 1; при top-k = 2 третий не выпадает
    /// никогда, а первые два делят массу 5 : 3.
    func testFrequenciesConvergeToTheDistribution() {
        let l = MLXArray([Float(log(0.5)), Float(log(0.3)), Float(log(0.2))])
        func freq(_ config: SamplingConfig, _ n: Int) -> [Double] {
            var s = GPUSampler(config: config, vocab: 3)
            var tokens: [MLXArray] = []
            for _ in 0 ..< n { tokens.append(s.next(l)) }
            let ids = concatenated(tokens, axis: 0).asArray(Int32.self)
            return (0 ..< 3).map { k in Double(ids.filter { $0 == Int32(k) }.count) / Double(n) }
        }
        let f = freq(SamplingConfig(temperature: 1, seed: 11), 2000)
        XCTAssertEqual(f[0], 0.5, accuracy: 0.04); XCTAssertEqual(f[1], 0.3, accuracy: 0.04)
        XCTAssertEqual(f[2], 0.2, accuracy: 0.04)
        let k2 = freq(SamplingConfig(temperature: 1, topK: 2, seed: 12), 2000)
        XCTAssertEqual(k2[2], 0, "top-k = 2 пропустил третий токен")
        XCTAssertEqual(k2[0], 0.625, accuracy: 0.04)
        // top-p = 0.6: масса 0.5 не набрала 0.6, второй кандидат остаётся, третий — нет
        let p6 = freq(SamplingConfig(temperature: 1, topP: 0.6, seed: 13), 1000)
        XCTAssertEqual(p6[2], 0)
        XCTAssertGreaterThan(p6[1], 0.3)
    }
}
