import Foundation
import MLX
import MLXRandom

// ───────────────────────────────────────────────────────────────────────
//  Сэмплер НА GPU: температура, top-k, top-p и штрафы за повтор без единого
//  чтения на CPU.
//
//  Зачем. Конвейерный декод (`DecodePipeline`) строит шаг t+1 от токена t,
//  пока тот ещё считается, — значит, токен обязан быть массивом на GPU, а не
//  числом, прочитанным на CPU. `Sampler` читает логиты на CPU (сортировка,
//  обратная функция распределения, свой генератор случайных чисел) и для
//  конвейера не годится; с ним конвейер работал только для жадного выбора.
//
//  АЛГОРИТМ ТОТ ЖЕ, что у `Sampler`, шаг в шаг:
//    штрафы (CTRL-повтор, presence, frequency) → деление на температуру →
//    сортировка по убыванию → softmax → усечение top-k и top-p как сужение
//    одного порядка (первый кандидат остаётся всегда) → одно равномерное
//    число u, масштабированное на уцелевшую массу → первый кандидат, чья
//    накопленная масса превысила u.
//  Это проверяется подачей ОДНОГО И ТОГО ЖЕ u обоим сэмплерам.
//
//  ДРУГОЙ ПОТОК СЛУЧАЙНЫХ ЧИСЕЛ, и это единственное отличие в поведении:
//  `Sampler` тянет u из SplitMix64, здесь u даёт генератор MLX по ключу,
//  который расщепляется на каждый токен. Один и тот же seed воспроизводит
//  текст в пределах одного сэмплера, но текст GPU-сэмплера и текст
//  CPU-сэмплера при равном seed разные.
//
//  Состояние (история токенов для штрафов и ключ генератора) — массивы на
//  GPU, обновляются лениво вместе с шагом модели.
// ───────────────────────────────────────────────────────────────────────

public struct GPUSampler {
    public let config: SamplingConfig
    /// Накопленный вес каждого токена словаря, [vocab] fp32. nil — штрафы
    /// не заданы, история не ведётся вовсе.
    private var occurrence: MLXArray?
    private var key: MLXArray
    /// 0 ..< vocab, int32 — заводится ОДИН раз: нужен top-k и учёту истории
    /// на каждом токене, а собирать 65536 чисел на CPU и везти их на GPU на
    /// каждый токен стоило дороже самого выбора.
    private let indices: MLXArray

    /// - history: история, с которой начать (например, промпт при
    ///   `penalizePrompt`), в том виде, в каком её ведёт CPU-сэмплер.
    public init(config: SamplingConfig = .greedy, vocab: Int, history: PenaltyState = PenaltyState()) {
        self.config = config
        self.key = MLXRandom.key(config.seed)
        self.indices = MLXArray(Int32(0) ..< Int32(vocab))
        eval(indices)
        if config.penalizes {
            var dense = [Float](repeating: 0, count: vocab)
            for (id, weight) in history.occurrence where id >= 0 && id < vocab { dense[id] = weight }
            occurrence = MLXArray(dense)
        } else {
            occurrence = nil
        }
    }

    /// Массивы состояния — для постановки в очередь вычисления вместе с шагом.
    var arrays: [MLXArray] { [key] + (occurrence.map { [$0] } ?? []) }

    /// Логиты после штрафов, [vocab] fp32. Формулы — как в `PenaltyState.adjust`.
    func adjusted(_ logits: MLXArray) -> MLXArray {
        let l = logits.asType(.float32)
        guard let occ = occurrence else { return l }
        let seen = occ .> 0
        var out = l
        if config.repetitionPenalty != 1 {
            // CTRL: положительный логит делим, отрицательный умножаем — обе
            // ветки понижение
            let lowered = MLX.where(out .> 0, out / config.repetitionPenalty,
                                    out * config.repetitionPenalty)
            out = MLX.where(seen, lowered, out)
        }
        if config.presencePenalty != 0 || config.frequencyPenalty != 0 {
            let sub = occ * config.frequencyPenalty + config.presencePenalty
            out = out - MLX.where(seen, sub, MLXArray(Float(0)))
        }
        return out
    }

    /// Выбор токена при ЗАДАННОМ равномерном числе u ∈ [0, 1) (массив [1]).
    /// Вынесено отдельно ради теста: один и тот же u обязан дать тот же
    /// токен, что и обратная функция распределения CPU-сэмплера.
    func pick(_ logits: MLXArray, uniform u: MLXArray) -> MLXArray {
        precondition(logits.ndim == 1, "сэмплер ждёт логиты формы [vocab]")
        let l = adjusted(logits)
        if config.isGreedy {
            return argMax(l, axis: -1).reshaped([1]).asType(.int32)
        }
        let V = l.shape[0]
        let scaled = l / config.temperature
        // argSort по -logits — порядок по убыванию
        let order = MLX.argSort(-scaled)
        let p = MLX.softmax(MLX.take(scaled, order), axis: -1)
        let cum = p.cumsum(axis: 0)
        // Кандидат i остаётся, если до него масса ещё не набрала topP (первый
        // остаётся всегда: до него масса нулевая) и он в пределах top-k.
        var keep = (cum - p) .< config.topP
        if config.topK > 0, config.topK < V {
            keep = keep .&& (indices .< Int32(config.topK))
        }
        let kept = keep.asType(.int32).sum()
        let mass = (p * keep.asType(.float32)).sum()
        // u масштабируется на уцелевшую массу — перенормировка усечённого
        // распределения; выбирается первый кандидат с накопленной массой > u
        let scaledU = u.reshaped([]) * mass
        let index = minimum(((cum .<= scaledU) .&& keep).asType(.int32).sum(), kept - 1)
        return MLX.take(order, index).reshaped([1]).asType(.int32)
    }

    /// Выбрать токен ([1] int32, на GPU) и учесть его в истории. Ровно одно
    /// расщепление ключа на токен, независимо от конфигурации и от формы
    /// распределения, — и жадный выбор ключ не трогает вовсе.
    public mutating func next(_ logits: MLXArray) -> MLXArray {
        let token: MLXArray
        if config.isGreedy {
            token = pick(logits, uniform: MLXArray([Float(0)]))
        } else {
            let (carry, mine) = MLXRandom.split(key: key)
            key = carry
            token = pick(logits, uniform: MLXRandom.uniform(low: Float(0), high: Float(1), [1], key: mine))
        }
        observe(token)
        return token
    }

    /// Учесть токен в истории: сперва затухание всех весов, потом +1 текущему
    /// (порядок — как у `PenaltyState.observe`, включая отсечение весов ниже
    /// 1e-6).
    public mutating func observe(_ token: MLXArray) {
        guard var occ = occurrence else { return }
        if config.penaltyDecay != 1 {
            if config.penaltyDecay == 0 {
                occ = MLXArray.zeros(like: occ)
            } else {
                let d = occ * config.penaltyDecay
                occ = MLX.where(d .< 1e-6, MLXArray(Float(0)), d)
            }
        }
        let hit = indices .== token.reshaped([]).asType(.int32)
        occurrence = occ + hit.asType(.float32)
    }
}
