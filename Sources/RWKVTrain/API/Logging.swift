import Foundation
import os

/// Уровень важности лог-сообщения.
public enum RWKVLogLevel: Int, Sendable, Comparable {
    case trace = 0
    case debug
    case info
    case warning
    case error

    public static func < (lhs: RWKVLogLevel, rhs: RWKVLogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Именованные метрики, которые фреймворк публикует в ходе обучения/инференса.
///
/// Разработчик может маршрутизировать их в свою аналитику/графики, не парся
/// текстовые логи. Значения — `Double` для единообразия (loss, accuracy,
/// токены/сек, пик памяти в МБ и т. п.).
public enum RWKVMetric: String, Sendable {
    case loss
    case valAccuracy = "val_accuracy"
    case tokensPerSecond = "tokens_per_second"
    case peakMemoryMB = "peak_memory_mb"
    case extractionProgress = "extraction_progress"
    case epoch
    case step
}

/// Протокол логирования и публикации метрик.
///
/// Заменяет россыпь `print(...)` и колбэков прежнего кода единым каналом.
/// Реализации должны быть потокобезопасны (`Sendable`): обучение идёт в фоне.
public protocol RWKVLogger: Sendable {
    /// Текстовое сообщение. `message` — автозамыкание: не вычисляется,
    /// если уровень отфильтрован.
    func log(_ level: RWKVLogLevel, _ message: @autoclosure () -> String)

    /// Числовая метрика на заданном шаге обучения (для графиков/аналитики).
    func metric(_ metric: RWKVMetric, value: Double, step: Int)
}

public extension RWKVLogger {
    func metric(_ metric: RWKVMetric, value: Double, step: Int) {}
    func info(_ message: @autoclosure () -> String)  { log(.info, message()) }
    func debug(_ message: @autoclosure () -> String) { log(.debug, message()) }
    func warn(_ message: @autoclosure () -> String)  { log(.warning, message()) }
    func error(_ message: @autoclosure () -> String) { log(.error, message()) }
}

/// Логгер, который ничего не делает (по умолчанию). Нулевая стоимость.
public struct NoopLogger: RWKVLogger {
    public init() {}
    public func log(_ level: RWKVLogLevel, _ message: @autoclosure () -> String) {}
    public func metric(_ metric: RWKVMetric, value: Double, step: Int) {}
}

/// Логгер поверх `os.Logger` — безопасный системный канал (виден в Console.app
/// и Instruments). Метрики печатаются на уровне `debug`.
public struct OSLogLogger: RWKVLogger {
    private let logger: Logger
    private let minLevel: RWKVLogLevel

    public init(subsystem: String = "com.swiftrwkv", category: String = "RWKVTrain",
                minLevel: RWKVLogLevel = .info) {
        self.logger = Logger(subsystem: subsystem, category: category)
        self.minLevel = minLevel
    }

    public func log(_ level: RWKVLogLevel, _ message: @autoclosure () -> String) {
        guard level >= minLevel else { return }
        let text = message()
        switch level {
        case .trace, .debug: logger.debug("\(text, privacy: .public)")
        case .info:          logger.info("\(text, privacy: .public)")
        case .warning:       logger.warning("\(text, privacy: .public)")
        case .error:         logger.error("\(text, privacy: .public)")
        }
    }

    public func metric(_ metric: RWKVMetric, value: Double, step: Int) {
        guard RWKVLogLevel.debug >= minLevel else { return }
        logger.debug("[metric] \(metric.rawValue, privacy: .public)=\(value) step=\(step)")
    }
}
