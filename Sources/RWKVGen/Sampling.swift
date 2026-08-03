import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Сэмплирование из логитов: температура, top-k, top-p, штрафы за повтор.
//
//  Три решения, каждое стоит объяснить.
//
//  1. СВОЙ генератор случайных чисел, а не MLXRandom. MLXRandom.seed —
//     ГЛОБАЛЬНОЕ состояние, общее с инициализацией весов и с обучением.
//     Сэмплер, который его дёргает, делает выдачу зависимой от того, что
//     происходило в процессе до него, и наоборот — сдвигает обучение, если
//     кто-то параллельно генерирует. Воспроизводимость по сиду при этом
//     недоказуема. SplitMix64 занимает восемь строк и полностью локален.
//
//  2. Отбор идёт на CPU, сортировка — в MLX. Сортировать 65536 значений в
//     Swift — это миллисекунды на каждый токен; на GPU это одно ядро. Но
//     инверсия CDF по отсортированному массиву — цикл с ранним выходом, в
//     MLX он выразим только через маски по всему словарю. Поэтому: сортировка
//     там, обход — здесь. Перенос двух массивов по 65536 (256 КБ + 256 КБ)
//     на унифицированной памяти — это memcpy, десятки микросекунд против
//     десятков миллисекунд на шаг декода.
//
//  3. НЕТ короткого замыкания для topK == 1 и для очень малых температур.
//     Соблазн написать «если вырожденный случай — сразу argmax» велик, но
//     тогда тест «topK = 1 совпадает с жадным» проверяет ветку, которую сам
//     же и задаёт, то есть сравнивает путь сам с собой. Вырожденные
//     конфигурации обязаны проходить общий путь и приходить к argmax по
//     арифметике, а не по условию. Единственное исключение —
//     temperature <= 0: деление на ноль не арифметика, и эта ветка объявлена
//     явно.
// ───────────────────────────────────────────────────────────────────────

/// Детерминированный генератор псевдослучайных чисел (SplitMix64).
///
/// Локальный по построению: два экземпляра с одним сидом дают одну
/// последовательность независимо от того, что происходит вокруг.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { self.state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Равномерное число в [0, 1). 53 значащих бита, приведённые к Float.
    public mutating func uniform() -> Float {
        Float(Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0))
    }
}

/// Параметры сэмплирования.
///
/// Умолчания — жадный декод: `temperature = 0`. Это сознательно. Пакет,
/// который по умолчанию выдаёт случайный текст, невозможно отлаживать, а
/// жадный декод воспроизводим без всяких сидов.
public struct SamplingConfig: Sendable, Equatable {
    /// Делитель логитов. `<= 0` — жадный выбор (argmax), без обращения к RNG.
    public var temperature: Float
    /// Оставить не более K кандидатов с наибольшей вероятностью. 0 — выключено.
    public var topK: Int
    /// Оставить наименьший префикс, чья суммарная вероятность >= topP.
    /// 1 — выключено. Первый кандидат остаётся всегда, даже если topP меньше
    /// вероятности моды.
    public var topP: Float
    /// Мультипликативный штраф в стиле CTRL: логит уже выданного токена
    /// делится на штраф, если положителен, и умножается, если отрицателен.
    /// В обе стороны это СНИЖЕНИЕ. 1 — выключено.
    public var repetitionPenalty: Float
    /// Аддитивный штраф за сам факт появления токена (стиль ChatRWKV).
    public var presencePenalty: Float
    /// Аддитивный штраф, пропорциональный накопленной частоте токена.
    public var frequencyPenalty: Float
    /// Затухание счётчиков частот на каждом шаге. 1 — без затухания.
    public var penaltyDecay: Float
    /// Сид генератора. Один и тот же сид при одном и том же входе даёт
    /// побайтово одну и ту же выдачу.
    public var seed: UInt64

    public init(temperature: Float = 0,
                topK: Int = 0,
                topP: Float = 1,
                repetitionPenalty: Float = 1,
                presencePenalty: Float = 0,
                frequencyPenalty: Float = 0,
                penaltyDecay: Float = 1,
                seed: UInt64 = 0) {
        precondition(topK >= 0, "topK не может быть отрицательным")
        precondition(topP > 0 && topP <= 1, "topP обязан лежать в (0, 1]")
        precondition(repetitionPenalty > 0, "repetitionPenalty обязан быть > 0")
        precondition(penaltyDecay >= 0 && penaltyDecay <= 1,
                     "penaltyDecay обязан лежать в [0, 1]")
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.penaltyDecay = penaltyDecay
        self.seed = seed
    }

    /// Жадный декод: argmax, без RNG.
    public static let greedy = SamplingConfig()

    /// Идёт ли выбор по argmax без обращения к генератору.
    public var isGreedy: Bool { temperature <= 0 }

    /// Штрафуется ли хоть что-нибудь.
    public var penalizes: Bool {
        repetitionPenalty != 1 || presencePenalty != 0 || frequencyPenalty != 0
    }
}

/// Накопленная история токенов для штрафов за повтор.
///
/// Отдельный тип, а не поле сэмплера, потому что историю иногда нужно
/// заполнить промптом до первого шага генерации — а иногда наоборот нельзя.
public struct PenaltyState: Sendable {
    /// Накопленный вес каждого встреченного токена. Не целое: есть затухание.
    public private(set) var occurrence: [Int: Float] = [:]

    public init() {}

    /// Учесть токен: сперва затухание всех счётчиков, потом +1 текущему.
    ///
    /// Порядок важен и совпадает с ChatRWKV: только что выданный токен
    /// получает полный вес, а всё, что было раньше, уже ослаблено.
    public mutating func observe(_ id: Int, decay: Float) {
        if decay != 1 {
            if decay == 0 {
                occurrence.removeAll(keepingCapacity: true)
            } else {
                for (k, v) in occurrence {
                    let d = v * decay
                    if d < 1e-6 {
                        occurrence.removeValue(forKey: k)
                    } else {
                        occurrence[k] = d
                    }
                }
            }
        }
        occurrence[id, default: 0] += 1
    }

    public mutating func reset() { occurrence.removeAll(keepingCapacity: true) }

    /// Применить штрафы к логитам. Возвращает новый массив; вход не меняется.
    public func adjust(_ logits: MLXArray, config: SamplingConfig) -> MLXArray {
        guard config.penalizes, !occurrence.isEmpty else { return logits }
        let ids = occurrence.keys.sorted()
        let idx = MLXArray(ids.map { Int32($0) })
        var out = logits

        if config.repetitionPenalty != 1 {
            // CTRL: положительный логит делим, отрицательный умножаем. Обе
            // ветки — понижение; без ветвления штраф ПОВЫШАЛ бы отрицательные
            // логиты, то есть работал бы наоборот на большей части словаря.
            let seen = out[idx]
            let lowered = MLX.where(seen .> 0,
                                    seen / config.repetitionPenalty,
                                    seen * config.repetitionPenalty)
            out = out.at[idx].add(lowered - seen)
        }

        if config.presencePenalty != 0 || config.frequencyPenalty != 0 {
            let counts = MLXArray(ids.map { occurrence[$0]! })
            let sub = counts * config.frequencyPenalty + config.presencePenalty
            out = out.at[idx].subtract(sub)
        }
        return out
    }
}

/// Сэмплер: конфигурация + генератор + история штрафов.
///
/// Состояние сэмплера отделено от состояния модели (`RWKVState`) намеренно:
/// одну и ту же ветку модели можно продолжать разными сэмплерами, а один и
/// тот же сэмплер не может обслуживать две ветки — он несёт историю.
public struct Sampler: Sendable {
    public var config: SamplingConfig
    public private(set) var penalty = PenaltyState()
    private var rng: SplitMix64

    public init(config: SamplingConfig = .greedy) {
        self.config = config
        self.rng = SplitMix64(seed: config.seed)
    }

    /// Сбросить генератор и историю к исходному состоянию.
    public mutating func reset() {
        rng = SplitMix64(seed: config.seed)
        penalty.reset()
    }

    /// Учесть токен в истории штрафов, ничего не выбирая.
    /// Нужно, чтобы засчитать промпт перед генерацией.
    public mutating func observe(_ id: Int) {
        penalty.observe(id, decay: config.penaltyDecay)
    }

    /// Выбрать следующий токен по логитам `[vocab]` и учесть его в истории.
    public mutating func next(_ logits: MLXArray) -> Int {
        let id = pick(logits)
        observe(id)
        return id
    }

    /// Выбрать токен, НЕ трогая историю. Для тестов и для вызывающих,
    /// которые ведут историю сами.
    public mutating func pick(_ logits: MLXArray) -> Int {
        precondition(logits.ndim == 1, "сэмплер ждёт логиты формы [vocab]")
        let adjusted = penalty.adjust(logits.asType(.float32), config: config)

        // Жадная ветка объявлена явно: temperature <= 0 — это не «очень
        // маленькая температура», а отсутствие деления. RNG при этом НЕ
        // дёргается, и это часть контракта: жадная генерация не сдвигает
        // поток случайных чисел.
        if config.temperature <= 0 {
            return adjusted.argMax().item(Int.self)
        }

        let scaled = adjusted / config.temperature
        // argSort по -logits — это порядок по убыванию. Отдельного
        // «сортировать по убыванию» в MLX нет, а разворот среза стоит дороже
        // смены знака.
        let order = MLX.argSort(-scaled)
        let probs = MLX.softmax(MLX.take(scaled, order), axis: -1)
        MLX.eval(order, probs)

        return inverseCDF(probs.asArray(Float.self), order.asArray(Int32.self))
    }

    /// Обход отсортированного по убыванию распределения: отсечение по topK и
    /// topP, затем инверсия CDF.
    ///
    /// Одним проходом, потому что оба отсечения — это префиксы одного и того
    /// же порядка, а не независимые фильтры.
    private mutating func inverseCDF(_ p: [Float], _ ids: [Int32]) -> Int {
        let limit = config.topK > 0 ? min(config.topK, p.count) : p.count

        // Первый кандидат остаётся всегда: иначе при topP меньшем, чем
        // вероятность моды, не осталось бы ничего.
        var kept = 0
        var mass: Float = 0
        while kept < limit {
            mass += p[kept]
            kept += 1
            if mass >= config.topP { break }
        }

        // u масштабируется на уцелевшую массу — это и есть перенормировка
        // усечённого распределения. Ровно один вызов RNG на токен, независимо
        // от того, сколько кандидатов уцелело: иначе поток случайных чисел
        // зависел бы от формы распределения и сид перестал бы что-либо
        // гарантировать между конфигурациями.
        let u = rng.uniform() * mass
        var acc: Float = 0
        for i in 0 ..< kept {
            acc += p[i]
            if u < acc { return Int(ids[i]) }
        }
        // Досюда доходит только накопленная ошибка сложения float.
        return Int(ids[kept - 1])
    }
}
