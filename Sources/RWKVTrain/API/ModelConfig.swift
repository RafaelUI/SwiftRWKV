import Foundation
import RWKVKernel

/// Тип задачи, под которую используется модель.
///
/// Формат данных и пайплайн определяются именно задачей:
/// - `.classification` — один ярлык на текст (обучаются верхние слои + голова);
/// - `.featureExtraction` — извлечение вектор-признаков (голова не нужна).
public enum RWKVTask: String, Codable, Sendable {
    case classification
    case featureExtraction = "feature_extraction"
}

/// Способ свёртки последовательности `[B,T,D]` в вектор `[B,D]`.
public enum RWKVPooling: String, Codable, Sendable {
    /// Усреднение по реальным (непаддинговым) токенам.
    case mean
    /// Вектор последнего реального токена (классика RWKV).
    case last
}

/// Самодостаточный конфиг модели. Сериализуется в `config.json` рядом с
/// весами `model.safetensors` и токенизатором `tokenizer.json`.
///
/// Поля архитектуры обязательны; поля задачи/обучения опциональны
/// (заполняются после файнтюна).
public struct ModelConfig: Codable, Sendable {

    // ── Архитектура ──
    public var arch: String          // напр. "rwkv7"
    public var nLayer: Int
    public var nEmbd: Int
    public var headSize: Int
    public var vocab: Int
    public var contextSize: Int      // максимальная длина контекста (ctxLen)

    // ── Метаданные (опционально) ──
    public var language: String?
    public var tokenizer: String?    // имя токенизатора, напр. "ru16k"

    // ── Задача / обучение (заполняется после файнтюна) ──
    public var task: RWKVTask?
    public var pooling: RWKVPooling?
    public var numClasses: Int?
    public var classes: [String]?
    public var freeze: Int?          // число замороженных нижних слоёв
    public var parent: String?       // id родительской (базовой) модели
    public var valAcc: Float?
    public var createdAt: Double?

    public init(
        arch: String = "rwkv7",
        nLayer: Int,
        nEmbd: Int,
        headSize: Int = 64,
        vocab: Int,
        contextSize: Int = 128,
        language: String? = nil,
        tokenizer: String? = nil,
        task: RWKVTask? = nil,
        pooling: RWKVPooling? = nil,
        numClasses: Int? = nil,
        classes: [String]? = nil,
        freeze: Int? = nil,
        parent: String? = nil,
        valAcc: Float? = nil,
        createdAt: Double? = nil
    ) {
        self.arch = arch
        self.nLayer = nLayer
        self.nEmbd = nEmbd
        self.headSize = headSize
        self.vocab = vocab
        self.contextSize = contextSize
        self.language = language
        self.tokenizer = tokenizer
        self.task = task
        self.pooling = pooling
        self.numClasses = numClasses
        self.classes = classes
        self.freeze = freeze
        self.parent = parent
        self.valAcc = valAcc
        self.createdAt = createdAt
    }

    /// Число голов внимания.
    public var nHead: Int { nEmbd / headSize }

    /// Проверка геометрии. Бросает `RWKVError.invalidGeometry`, если что-то
    /// не сходится (вызывается при загрузке модели и перед обучением).
    public func validate() throws {
        guard nEmbd > 0, headSize > 0, nLayer > 0, vocab > 0 else {
            throw RWKVError.invalidGeometry(reason: "nEmbd/headSize/nLayer/vocab must be > 0")
        }
        guard nEmbd % headSize == 0 else {
            throw RWKVError.invalidGeometry(
                reason: "nEmbd (\(nEmbd)) must be divisible by headSize (\(headSize))")
        }
        guard headSize == WKV7_HEAD_SIZE else {
            throw RWKVError.invalidGeometry(
                reason: "headSize (\(headSize)) must equal kernel HEAD_SIZE (\(WKV7_HEAD_SIZE))")
        }
        guard contextSize % WKV7_CHUNK == 0 else {
            throw RWKVError.invalidGeometry(
                reason: "contextSize (\(contextSize)) must be divisible by CHUNK (\(WKV7_CHUNK))")
        }
    }

    /// Внутреннее представление конфига для backbone-ядра.
    var rwkvConfig: RWKVConfig {
        var c = RWKVConfig()
        c.nLayer = nLayer
        c.nEmbd = nEmbd
        c.headSize = headSize
        c.vocab = vocab
        return c
    }
}
