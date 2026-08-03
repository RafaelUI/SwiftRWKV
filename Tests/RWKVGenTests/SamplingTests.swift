//
//  SamplingTests.swift
//  Сэмплер на СИНТЕТИЧЕСКИХ логитах: модель не нужна, распределение известно
//  точно, и потому утверждать можно не «похоже», а «ровно столько».
//
//  Тесты здесь построены на одном принципе: включение параметра обязано
//  что-то ЗАПРЕТИТЬ и при этом что-то ОСТАВИТЬ. Проверка только запрета
//  проходится реализацией, которая всегда возвращает argmax; проверка только
//  «оставляет» проходится реализацией, которая игнорирует параметр. Ловит
//  дефект пара, а не любая из половин.
//
import XCTest
import MLX
@testable import RWKVGen

final class SamplingTests: XCTestCase {

    // Распределение с известными вероятностями: softmax(log p) == p.
    // Значит по логитам ln(p) softmax вернёт ровно p, и любую отсечку можно
    // посчитать на бумаге, а не подогнать под реализацию.
    static let probs: [Float] = [0.5, 0.3, 0.15, 0.04, 0.01]
    static func logits(_ p: [Float] = probs) -> MLXArray {
        MLXArray(p.map { log($0) })
    }

    /// Эмпирическая частота каждого id за n выборок.
    ///
    /// n везде задан явно и держится маленьким СОЗНАТЕЛЬНО: один `pick` — это
    /// пять диспетчеризаций MLX с синхронизацией, около миллисекунды даже на
    /// словаре из пяти элементов (замерено: 240 000 выборок = 238 с). Набор,
    /// который стоит четыре минуты, перестают гонять. Допуски ниже посчитаны
    /// под конкретное n: при n = 2000 стандартное отклонение частоты около
    /// 0.011, и допуск 0.04 — это примерно 3.5 сигмы.
    func frequencies(_ cfg: SamplingConfig, _ l: MLXArray, n: Int,
                     vocab: Int = 5) -> [Float] {
        var s = Sampler(config: cfg)
        var count = [Int](repeating: 0, count: vocab)
        for _ in 0 ..< n { count[s.pick(l)] += 1 }
        return count.map { Float($0) / Float(n) }
    }

    // ─────────────────────── генератор ───────────────────────

    /// SplitMix64 воспроизводим и НЕ вырожден.
    ///
    /// Второе утверждение важнее первого: генератор, всегда возвращающий 0,
    /// воспроизводим идеально.
    func testSplitMixIsReproducibleAndNotDegenerate() {
        var a = SplitMix64(seed: 42), b = SplitMix64(seed: 42), c = SplitMix64(seed: 43)
        var seqA: [Float] = [], seqB: [Float] = [], seqC: [Float] = []
        for _ in 0 ..< 64 { seqA.append(a.uniform()); seqB.append(b.uniform()); seqC.append(c.uniform()) }
        XCTAssertEqual(seqA, seqB, "один сид — одна последовательность")
        XCTAssertNotEqual(seqA, seqC, "разные сиды обязаны расходиться")
        XCTAssertEqual(Set(seqA).count, 64, "повторов быть не должно")
        for v in seqA { XCTAssertTrue(v >= 0 && v < 1, "uniform вышел за [0,1): \(v)") }
        // Среднее 64 равномерных ≈ 0.5; выродившийся генератор промахнётся.
        let mean = seqA.reduce(0, +) / 64
        XCTAssertEqual(mean, 0.5, accuracy: 0.1, "среднее равномерного далеко от 0.5")
    }

    // ─────────────────────── сид ───────────────────────

    /// Тот же сид — та же выдача; другой сид — другая.
    func testSeedDeterminesOutputAndDifferentSeedsDiverge() {
        let l = Self.logits()
        func draw(_ seed: UInt64) -> [Int] {
            var s = Sampler(config: SamplingConfig(temperature: 1, seed: seed))
            return (0 ..< 40).map { _ in s.pick(l) }
        }
        XCTAssertEqual(draw(7), draw(7))
        XCTAssertNotEqual(draw(7), draw(8), "сид не влияет — RNG не задействован")
    }

    /// `reset()` возвращает сэмплер в исходное состояние — и генератор, и
    /// историю штрафов.
    func testResetRestoresBothStreamAndHistory() {
        let l = Self.logits()
        var s = Sampler(config: SamplingConfig(temperature: 1, repetitionPenalty: 2, seed: 3))
        let first = (0 ..< 20).map { _ in s.next(l) }
        XCTAssertFalse(s.penalty.occurrence.isEmpty)
        s.reset()
        XCTAssertTrue(s.penalty.occurrence.isEmpty, "история не сброшена")
        XCTAssertEqual((0 ..< 20).map { _ in s.next(l) }, first)
    }

    // ─────────────────────── температура ───────────────────────

    /// temperature <= 0 — объявленная ветка argmax.
    func testZeroTemperatureIsArgmax() {
        var s = Sampler(config: .greedy)
        XCTAssertEqual(s.pick(Self.logits()), 0)
        XCTAssertEqual(s.pick(MLXArray([1.0, 9.0, 3.0] as [Float])), 1)
    }

    /// Малая, но НЕ нулевая температура приходит к argmax по общему пути.
    ///
    /// Это не то же самое, что предыдущий тест: там проверяется ветка, здесь
    /// — арифметика отбора, которой ветка не касается.
    func testTinyTemperatureReachesArgmaxThroughTheGeneralPath() {
        let l = Self.logits()
        for seed in UInt64(0) ..< 16 {
            var s = Sampler(config: SamplingConfig(temperature: 0.01, seed: seed))
            XCTAssertEqual(s.pick(l), 0, "малая температура ушла от моды (сид \(seed))")
        }
    }

    /// Высокая температура сглаживает распределение, низкая — заостряет.
    func testTemperatureMovesMassTowardsUniform() {
        let l = Self.logits()
        let cold = frequencies(SamplingConfig(temperature: 0.5, seed: 1), l, n: 800)
        let hot = frequencies(SamplingConfig(temperature: 4.0, seed: 1), l, n: 800)
        XCTAssertGreaterThan(cold[0], hot[0], "тёплое распределение должно быть площе")
        XCTAssertLessThan(cold[4], hot[4], "хвост при высокой температуре обязан ожить")
        // При T=1 частоты обязаны сойтись к самим вероятностям.
        let one = frequencies(SamplingConfig(temperature: 1, seed: 1), l, n: 2000)
        for i in 0 ..< 5 {
            XCTAssertEqual(one[i], Self.probs[i], accuracy: 0.04,
                           "T=1: частота id \(i) разошлась с вероятностью")
        }
    }

    // ─────────────────────── top-k ───────────────────────

    /// topK = 1 приходит к argmax — и приходит ЧЕРЕЗ отбор, а не мимо него.
    func testTopKOneEqualsGreedyWithoutAShortcut() {
        let l = Self.logits()
        for seed in UInt64(0) ..< 16 {
            var s = Sampler(config: SamplingConfig(temperature: 1, topK: 1, seed: seed))
            XCTAssertEqual((0 ..< 8).map { _ in s.pick(l) }, [Int](repeating: 0, count: 8),
                           "topK=1 выдал не моду (сид \(seed))")
        }
    }

    /// topK = 3 запрещает хвост И оставляет ровно три.
    func testTopKKeepsExactlyKCandidates() {
        let f = frequencies(SamplingConfig(temperature: 1, topK: 3, seed: 5),
                            Self.logits(), n: 2000)
        XCTAssertEqual(f[3], 0, "id 3 вне top-3, но выпал")
        XCTAssertEqual(f[4], 0, "id 4 вне top-3, но выпал")
        for i in 0 ..< 3 {
            XCTAssertGreaterThan(f[i], 0.01, "id \(i) внутри top-3, но не выпал ни разу")
        }
        // Перенормировка: сумма оставленных 0.95, значит частоты обязаны
        // быть p/0.95, а не p.
        let mass: Float = 0.5 + 0.3 + 0.15
        for i in 0 ..< 3 {
            XCTAssertEqual(f[i], Self.probs[i] / mass, accuracy: 0.04,
                           "top-k не перенормировал остаток")
        }
    }

    /// topK больше словаря — то же, что без topK.
    ///
    /// Сравниваются два ПОЛНЫХ прогона с одним сидом, поэтому равенство
    /// точное и малого n достаточно.
    func testTopKBeyondVocabularyIsNoOp() {
        let l = Self.logits()
        let capped = frequencies(SamplingConfig(temperature: 1, topK: 999, seed: 11), l, n: 300)
        let plain = frequencies(SamplingConfig(temperature: 1, seed: 11), l, n: 300)
        XCTAssertEqual(capped, plain)
    }

    // ─────────────────────── top-p ───────────────────────

    /// topP отсекает ровно тот хвост, который посчитан на бумаге.
    ///
    /// p = [0.5, 0.3, 0.15, 0.04, 0.01]; при topP = 0.75 накопленная масса
    /// после двух кандидатов — 0.8 >= 0.75, значит остаются ровно первые два.
    func testTopPCutsTheTailComputedOnPaper() {
        let f = frequencies(SamplingConfig(temperature: 1, topP: 0.75, seed: 2),
                            Self.logits(), n: 2000)
        for i in 2 ..< 5 { XCTAssertEqual(f[i], 0, "id \(i) обязан быть отсечён") }
        XCTAssertEqual(f[0], 0.5 / 0.8, accuracy: 0.04)
        XCTAssertEqual(f[1], 0.3 / 0.8, accuracy: 0.04)
    }

    /// topP меньше вероятности моды всё равно оставляет моду.
    ///
    /// Граничный случай, в котором наивная реализация оставляет пустое
    /// множество и падает или возвращает мусор.
    func testTopPBelowTheModeStillKeepsTheMode() {
        let f = frequencies(SamplingConfig(temperature: 1, topP: 0.001, seed: 4),
                            Self.logits(), n: 200)
        XCTAssertEqual(f[0], 1.0, accuracy: 1e-6)
    }

    /// Граница topP ВКЛЮЧАЮЩАЯ: набравшись ровно до topP, отбор
    /// останавливается, а не берёт ещё одного.
    ///
    /// Отличить `>=` от `>` можно только там, где накопленная масса РОВНО
    /// равна порогу, а на произвольных вероятностях это событие нулевой
    /// меры — и потому все остальные тесты обе версии проходят. Четыре
    /// одинаковых логита дают ровно по 0.25 (в float это точные значения),
    /// так что после двух кандидатов масса ровно 0.5.
    ///
    /// Тест появился после мутационной проверки: замена `>=` на `>` не была
    /// поймана ничем.
    func testTopPBoundaryIsInclusive() {
        let flat = MLXArray([Float](repeating: 0, count: 4))
        let f = frequencies(SamplingConfig(temperature: 1, topP: 0.5, seed: 3), flat,
                            n: 600, vocab: 4)
        XCTAssertEqual(f.filter { $0 > 0 }.count, 2,
                       "при массе ровно 0.5 обязано остаться два кандидата, а не \(f)")
    }

    /// topP = 1 ничего не отсекает.
    func testTopPOneIsNoOp() {
        let l = Self.logits()
        XCTAssertEqual(frequencies(SamplingConfig(temperature: 1, topP: 1, seed: 9), l, n: 300),
                       frequencies(SamplingConfig(temperature: 1, seed: 9), l, n: 300))
    }

    /// topK и topP вместе — пересечение, а не выбор одного из них.
    func testTopKAndTopPIntersect() {
        // topP=0.95 оставил бы 3 кандидата, topK=2 оставляет 2. Вместе — 2.
        let f = frequencies(SamplingConfig(temperature: 1, topK: 2, topP: 0.95, seed: 6),
                            Self.logits(), n: 1500)
        XCTAssertEqual(f[2], 0, "topK не применился поверх topP")
        XCTAssertEqual(f[0], 0.5 / 0.8, accuracy: 0.04)
    }

    // ─────────────────────── штрафы ───────────────────────

    /// Мультипликативный штраф снижает логит НЕЗАВИСИМО ОТ ЗНАКА.
    ///
    /// Это та проверка, ради которой стоит писать тест: наивное «поделить на
    /// штраф» на отрицательном логите его ПОВЫШАЕТ, то есть работает наоборот
    /// на большей части словаря — а средняя выдача при этом выглядит
    /// правдоподобно.
    func testRepetitionPenaltyLowersBothSigns() {
        let raw: [Float] = [2.0, -2.0, 0.5, -0.5]
        let l = MLXArray(raw)
        var st = PenaltyState()
        for i in 0 ..< 4 { st.observe(i, decay: 1) }
        let cfg = SamplingConfig(temperature: 1, repetitionPenalty: 2)
        let out = st.adjust(l, config: cfg).asArray(Float.self)
        for i in 0 ..< 4 {
            XCTAssertLessThan(out[i], raw[i],
                              "логит \(raw[i]) не понизился: стал \(out[i])")
        }
        XCTAssertEqual(out[0], 1.0, accuracy: 1e-5, "положительный обязан делиться")
        XCTAssertEqual(out[1], -4.0, accuracy: 1e-5, "отрицательный обязан умножаться")
    }

    /// Штраф не трогает то, чего в истории нет.
    func testPenaltyLeavesUnseenTokensAlone() {
        let raw: [Float] = [2.0, 2.0, 2.0]
        var st = PenaltyState()
        st.observe(1, decay: 1)
        let out = st.adjust(MLXArray(raw),
                            config: SamplingConfig(repetitionPenalty: 2,
                                                   presencePenalty: 1)).asArray(Float.self)
        XCTAssertEqual(out[0], 2.0, accuracy: 1e-6)
        XCTAssertEqual(out[2], 2.0, accuracy: 1e-6)
        XCTAssertLessThan(out[1], 2.0)
    }

    /// Частотный штраф накапливается, presence — нет.
    func testFrequencyAccumulatesAndPresenceDoesNot() {
        let raw: [Float] = [0, 0]
        var st = PenaltyState()
        st.observe(0, decay: 1)
        let once = st.adjust(MLXArray(raw),
                             config: SamplingConfig(presencePenalty: 0.3,
                                                    frequencyPenalty: 0.2)).asArray(Float.self)
        st.observe(0, decay: 1)
        let twice = st.adjust(MLXArray(raw),
                              config: SamplingConfig(presencePenalty: 0.3,
                                                     frequencyPenalty: 0.2)).asArray(Float.self)
        XCTAssertEqual(once[0], -0.5, accuracy: 1e-6)   // 0.3 + 1*0.2
        XCTAssertEqual(twice[0], -0.7, accuracy: 1e-6)  // 0.3 + 2*0.2
    }

    /// Затухание ослабляет старое и не трогает свежее.
    func testPenaltyDecayWeakensOlderTokensOnly() {
        var st = PenaltyState()
        st.observe(0, decay: 0.5)   // occurrence[0] = 1
        st.observe(1, decay: 0.5)   // occurrence[0] = 0.5, occurrence[1] = 1
        XCTAssertEqual(st.occurrence[0]!, 0.5, accuracy: 1e-6)
        XCTAssertEqual(st.occurrence[1]!, 1.0, accuracy: 1e-6)
        st.observe(2, decay: 0.5)   // 0.25, 0.5, 1
        XCTAssertEqual(st.occurrence[0]!, 0.25, accuracy: 1e-6)
        XCTAssertEqual(st.occurrence[2]!, 1.0, accuracy: 1e-6)
    }

    /// Штраф РЕАЛЬНО снижает вероятность уже выданного токена в выдаче,
    /// а не только в логитах.
    ///
    /// Проверка на уровне частот, потому что между логитом и выдачей стоят
    /// softmax, отсечения и перенормировка, и любая из них может штраф съесть.
    /// История держится неподвижной (`pick`, а не `next`), иначе сравнивались
    /// бы два разных распределения на каждом шаге.
    func testPenaltyReducesFrequencyOfTheSeenToken() {
        let l = Self.logits()
        let free = frequencies(SamplingConfig(temperature: 1, seed: 1), l, n: 2000)

        var s = Sampler(config: SamplingConfig(temperature: 1, repetitionPenalty: 3, seed: 1))
        s.observe(0)                       // штрафуем ТОЛЬКО моду
        var count = [Int](repeating: 0, count: 5)
        for _ in 0 ..< 2000 { count[s.pick(l)] += 1 }
        let f = count.map { Float($0) / 2000 }

        XCTAssertLessThan(f[0], free[0] - 0.05,
                          "штраф не снизил частоту моды: \(f[0]) против \(free[0])")
        XCTAssertGreaterThan(f[1], free[1], "масса не перешла к остальным")
    }

    /// Аддитивный штраф уменьшает число повторов в НАСТОЯЩЕМ цикле генерации.
    ///
    /// Именно аддитивный, и это не придирка к формулировке. См. следующий
    /// тест.
    func testAdditivePenaltyReducesRepeatsInAGenerationLoop() {
        let l = Self.logits()
        func repeats(_ cfg: SamplingConfig) -> Int {
            var s = Sampler(config: cfg)
            let out = (0 ..< 300).map { _ in s.next(l) }
            return zip(out, out.dropFirst()).filter { $0 == $1 }.count
        }
        let free = repeats(SamplingConfig(temperature: 1, seed: 1))
        let penalised = repeats(SamplingConfig(temperature: 1, frequencyPenalty: 0.7,
                                               penaltyDecay: 0.6, seed: 1))
        XCTAssertLessThan(penalised, free,
                          "штраф не уменьшил число повторов: \(penalised) против \(free)")
    }

    /// ЗАФИКСИРОВАННОЕ ПОВЕДЕНИЕ, а не желаемое: мультипликативный штраф в
    /// стиле CTRL при отрицательных логитах и полном покрытии истории
    /// РАЗГОНЯЕТ распределение, а не сглаживает его.
    ///
    /// Логиты после softmax-нормировки почти всегда отрицательны (это ln p),
    /// а история за десяток шагов покрывает весь маленький словарь. Умножение
    /// ВСЕХ логитов на 3 — это то же самое, что деление температуры на 3:
    /// разрывы между кандидатами утраиваются, мода становится острее, повторов
    /// СТАНОВИТСЯ БОЛЬШЕ. Измерено здесь: 164 повтора против 128 без штрафа.
    ///
    /// Это свойство самой формулы CTRL, а не дефект реализации, и оно есть в
    /// каждой библиотеке, где этот штраф предлагается. Тест стоит, чтобы
    /// свойство нашли один раз, а не каждый раз заново; правильный ответ для
    /// подавления повторов — `frequencyPenalty`, он аддитивный и от масштаба
    /// логитов не зависит.
    func testCtrlPenaltySharpensWhenEveryCandidateIsPenalised() {
        let l = Self.logits()
        func repeats(_ cfg: SamplingConfig) -> Int {
            var s = Sampler(config: cfg)
            let out = (0 ..< 300).map { _ in s.next(l) }
            return zip(out, out.dropFirst()).filter { $0 == $1 }.count
        }
        let free = repeats(SamplingConfig(temperature: 1, seed: 1))
        let ctrl = repeats(SamplingConfig(temperature: 1, repetitionPenalty: 3,
                                          penaltyDecay: 0.5, seed: 1))
        XCTAssertGreaterThan(ctrl, free,
                             "поведение CTRL-штрафа изменилось — перечитать комментарий")
    }

    /// Жадный декод НЕ обращается к генератору.
    ///
    /// Контракт: жадная генерация воспроизводима без сида, и подмешивание
    /// жадных шагов не сдвигает поток случайных чисел.
    func testGreedyDoesNotConsumeRandomness() {
        let l = Self.logits()
        var a = Sampler(config: SamplingConfig(temperature: 0, seed: 1))
        var b = Sampler(config: SamplingConfig(temperature: 0, seed: 999))
        XCTAssertEqual((0 ..< 10).map { _ in a.pick(l) }, (0 ..< 10).map { _ in b.pick(l) })
    }

    // ─────────────────────── границы конфигурации ───────────────────────

    /// Ровный вход не вырождает отбор: все кандидаты доступны.
    func testUniformLogitsGiveAllCandidates() {
        let f = frequencies(SamplingConfig(temperature: 1, seed: 12),
                            MLXArray([Float](repeating: 0, count: 5)), n: 2000)
        for i in 0 ..< 5 { XCTAssertEqual(f[i], 0.2, accuracy: 0.04) }
    }

    /// Один кандидат в словаре — вырожденный, но допустимый случай.
    func testSingleTokenVocabulary() {
        var s = Sampler(config: SamplingConfig(temperature: 1, topP: 0.5, seed: 1))
        XCTAssertEqual(s.pick(MLXArray([Float(0)])), 0)
    }
}
