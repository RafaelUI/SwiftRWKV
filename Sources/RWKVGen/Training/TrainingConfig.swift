import Foundation

// ───────────────────────────────────────────────────────────────────────
//  Конфигурация обучения — общая для всех задач (LM-претрейн, LoRA/QLoRA,
//  верхние N слоёв, голова поверх состояния).
//
//  Здесь ровно то, что НЕ зависит от задачи: оптимизатор, расписание, память.
//  Что дифференцируется — в TrainableSet, чем меряется лосс — в замыкании
//  objective, откуда берутся батчи — в nextBatch. Эти три вещи ортогональны,
//  и держать их флагами в конфиге означало бы полсотни полей, из которых
//  половина комбинаций невалидна.
// ───────────────────────────────────────────────────────────────────────

/// Спад learning rate ПОСЛЕ warmup.
public enum LRSchedule: String, Sendable {
    case constant   // без спада — исторический дефолт LoRA-пути
    case cosine
    case linear
}

public struct TrainingConfig: Sendable {

    // ── Оптимизатор (AdamW, decoupled weight decay) ──────────────────
    public var lr: Float
    public var lrMin: Float
    public var schedule: LRSchedule
    public var warmupSteps: Int
    public var gradClip: Float          // global-norm; <=0 ⇒ без клипа
    public var weightDecay: Float
    public var beta1: Float
    public var beta2: Float
    public var adamEps: Float

    // ── Длительность ─────────────────────────────────────────────────
    public var maxSteps: Int
    public var gradAccum: Int           // эфф. батч = batch × gradAccum

    // ── Память ───────────────────────────────────────────────────────
    /// Лимит буферного кэша Metal. <=0 ⇒ не трогать глобальную настройку.
    public var cacheLimitGB: Double

    // ── Логирование ──────────────────────────────────────────────────
    public var logEvery: Int

    public init(lr: Float = 1e-4,
                lrMin: Float = 0,
                schedule: LRSchedule = .constant,
                warmupSteps: Int = 0,
                gradClip: Float = 1.0,
                weightDecay: Float = 0.0,
                beta1: Float = 0.9,
                beta2: Float = 0.95,
                adamEps: Float = 1e-8,
                maxSteps: Int = 1000,
                gradAccum: Int = 1,
                cacheLimitGB: Double = 1.5,
                logEvery: Int = 10) {
        self.lr = lr; self.lrMin = lrMin; self.schedule = schedule
        self.warmupSteps = warmupSteps; self.gradClip = gradClip
        self.weightDecay = weightDecay
        self.beta1 = beta1; self.beta2 = beta2; self.adamEps = adamEps
        self.maxSteps = maxSteps; self.gradAccum = gradAccum
        self.cacheLimitGB = cacheLimitGB; self.logEvery = logEvery
    }

    /// LR на шаге `step` (0-based).
    ///
    /// Линейный warmup, затем спад от lr к lrMin по выбранному закону.
    /// `.constant` даёт lrMin + (lr − lrMin)·1 = lr, то есть плоскую линию
    /// НЕЗАВИСИМО от lrMin — это исторический дефолт LoRA-пути, и он
    /// воспроизводится бит-в-бит.
    public func learningRate(at step: Int) -> Float {
        if warmupSteps > 0 && step < warmupSteps {
            return lr * Float(step + 1) / Float(warmupSteps)
        }
        guard schedule != .constant else { return lr }

        let denom = Swift.max(1, maxSteps - warmupSteps)
        let progress = Swift.min(Float(step - warmupSteps) / Float(denom), 1.0)
        let decay: Float
        switch schedule {
        case .cosine:   decay = 0.5 * (1.0 + cos(Float.pi * progress))
        case .linear:   decay = 1.0 - progress
        case .constant: decay = 1.0
        }
        return lrMin + (lr - lrMin) * decay
    }
}

/// Что произошло на шаге — структурно, а не форматированной строкой:
/// иначе эти величины невозможно построить в график.
public struct TrainingStep: Sendable {
    public let step: Int            // 1-based
    public let loss: Float
    public let gradNorm: Float
    public let learningRate: Float
    public let peakMemoryMB: Double
}

public struct TrainingResult: Sendable {
    public let finalLoss: Float
    public let steps: Int
}
