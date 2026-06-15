import Foundation

/// Ошибки публичного API `RWKVTrain`.
///
/// Все операции, которые могут не выполниться из-за внешних данных
/// (отсутствующий файл, повреждённый конфиг, несовместимые размеры),
/// бросают `RWKVError` вместо аварийного `precondition`/`fatalError`.
public enum RWKVError: Error, LocalizedError, Sendable {

    /// В папке модели отсутствует обязательный файл.
    /// - Parameters:
    ///   - file: ожидаемое имя файла (например, `config.json`).
    ///   - directory: путь к папке модели.
    case missingFile(file: String, directory: String)

    /// Не удалось декодировать `config.json`.
    case invalidConfig(reason: String)

    /// Не удалось загрузить веса `.safetensors`.
    case weightsLoadFailed(reason: String)

    /// В весах модели нет требуемого тензора (например, `head.weight`).
    case missingWeight(key: String)

    /// Не удалось загрузить/разобрать токенизатор.
    case tokenizerLoadFailed(reason: String)

    /// Геометрия не сходится (например, `ctxLen` не делится на `chunk`,
    /// или `headSize` не совпадает с ядром).
    case invalidGeometry(reason: String)

    /// Передан пустой или некорректный датасет.
    case invalidDataset(reason: String)

    /// Операция была отменена вызывающей стороной.
    case cancelled

    public var errorDescription: String? {
        switch self {
        case let .missingFile(file, directory):
            return "Missing required file '\(file)' in model directory '\(directory)'."
        case let .invalidConfig(reason):
            return "Invalid model config: \(reason)"
        case let .weightsLoadFailed(reason):
            return "Failed to load model weights: \(reason)"
        case let .missingWeight(key):
            return "Model weights are missing required tensor '\(key)'."
        case let .tokenizerLoadFailed(reason):
            return "Failed to load tokenizer: \(reason)"
        case let .invalidGeometry(reason):
            return "Invalid model geometry: \(reason)"
        case let .invalidDataset(reason):
            return "Invalid dataset: \(reason)"
        case .cancelled:
            return "Operation was cancelled."
        }
    }
}
