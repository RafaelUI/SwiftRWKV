import Foundation
import MLX

/// Загруженная модель: самодостаточный пакет из папки на диске.
///
/// Папка модели должна содержать три файла:
/// - `config.json`        — ``ModelConfig``;
/// - `model.safetensors`  — веса backbone (+ голова, если модель обучена);
/// - `tokenizer.json`     — BPE-токенизатор (HF-формат).
///
/// Такой формат позволяет хранить произвольное число моделей и читать их
/// единообразно по пути к папке.
public final class RWKVModel {

    /// Путь к папке модели.
    public let directory: URL

    /// Конфиг модели (прочитан из `config.json`).
    public let config: ModelConfig

    /// Токенизатор модели.
    public let tokenizer: BPETokenizer

    /// Сырые веса (имя тензора → массив). Включают голову, если модель обучена.
    let weights: [String: MLXArray]

    /// Backbone (RWKV-7) поверх весов. Создаётся лениво.
    private(set) lazy var backbone: RWKVBackbone =
        RWKVBackbone(weights: weights, cfg: config.rwkvConfig)

    /// Загружает модель из папки. Бросает ``RWKVError`` при отсутствии файлов,
    /// повреждённом конфиге или несовместимой геометрии.
    public init(directory: URL) throws {
        self.directory = directory

        let configURL = directory.appendingPathComponent("config.json")
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")

        let fm = FileManager.default
        guard fm.fileExists(atPath: configURL.path) else {
            throw RWKVError.missingFile(file: "config.json", directory: directory.path)
        }
        guard fm.fileExists(atPath: weightsURL.path) else {
            throw RWKVError.missingFile(file: "model.safetensors", directory: directory.path)
        }
        guard fm.fileExists(atPath: tokenizerURL.path) else {
            throw RWKVError.missingFile(file: "tokenizer.json", directory: directory.path)
        }

        // config.json
        do {
            let data = try Data(contentsOf: configURL)
            self.config = try JSONDecoder().decode(ModelConfig.self, from: data)
        } catch let e as RWKVError {
            throw e
        } catch {
            throw RWKVError.invalidConfig(reason: "\(error)")
        }
        try config.validate()

        // weights
        do {
            self.weights = try loadArrays(url: weightsURL)
        } catch {
            throw RWKVError.weightsLoadFailed(reason: "\(error)")
        }

        // tokenizer
        guard let tok = BPETokenizer(tokenizerJSONURL: tokenizerURL) else {
            throw RWKVError.tokenizerLoadFailed(reason: "cannot parse tokenizer.json")
        }
        self.tokenizer = tok
    }

    /// Есть ли в модели обученная голова классификатора.
    public var hasClassifierHead: Bool { weights["head.weight"] != nil }
}
