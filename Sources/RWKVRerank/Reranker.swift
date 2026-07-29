import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Замороженная база + обучаемая голова.
//
//  Порт rwkv_metal/reranker/model.py (класс Reranker).
// ───────────────────────────────────────────────────────────────────────

public final class Reranker {

    public let base: X070Backbone
    public let head: RerankerHead

    /// - freezeBase: обнуляет `trainLayers`. База реранкера заморожена по
    ///   построению — на её замороженности стоит весь кэш состояний, ради
    ///   которого голова и обучается за секунды. LoRA-путь поверх базы
    ///   потребовал бы онлайн-кодирования, которого нет.
    public init(base: X070Backbone, cfg: RerankerConfig = RerankerConfig(),
                freezeBase: Bool = true, headDType: DType = .float32,
                seed: UInt64 = 0) throws {
        self.base = base
        self.head = try RerankerHead(base: base, cfg: cfg, dtype: headDType, seed: seed)
        if freezeBase { base.trainLayers = [] }
    }

    // ── Состояние базы ──

    /// Свернуть последовательность в состояние базы.
    ///
    /// `detach: true` (умолчание) обрывает граф. База заморожена, и без этого
    /// MLX всё равно тащил бы backward через весь длинный проход ради ничего.
    public func encode(_ idx: MLXArray, mask: MLXArray? = nil,
                       endIdx: MLXArray? = nil, state: RWKVBatchState? = nil,
                       detach: Bool = true) -> RWKVBatchState {
        let st = base.states(idx, state: state, mask: mask, endIdx: endIdx)
        return detach ? st.stopGradient() : st
    }

    /// Свернуть состояние базы до того, что реально читает голова.
    public func select(_ state: RWKVBatchState) -> MLXArray { head.select(state) }

    /// Скоры по уже отобранному состоянию `[B, nUnique, H, S, S]` → `[B]`.
    public func scoreStates(_ selected: MLXArray) -> MLXArray { head(selected) }

    /// Токены пары → скоры `[B]`.
    public func callAsFunction(_ idx: MLXArray, mask: MLXArray? = nil,
                               endIdx: MLXArray? = nil,
                               state: RWKVBatchState? = nil) -> MLXArray {
        head(select(encode(idx, mask: mask, endIdx: endIdx, state: state)))
    }

    // ── Чекпоинт ──
    //
    // Конфигурация пишется в metadata не для красоты. Голова из одного блока
    // над слоем 5 и голова из одного блока над слоем 11 имеют ОДИНАКОВЫЕ
    // формы ВСЕХ тензоров: перепутав их, загрузка пройдёт молча, а модель
    // будет читать не тот слой и выдавать уверенную бессмыслицу. Здесь это
    // ловится на загрузке.
    //
    // Формат намеренно тот же, что у Python-реализации
    // ("rwkv-metal-reranker-head-v1"), и имена параметров совпадают
    // с tree_flatten питоновской головы. Обмен чекпоинтами в обе стороны
    // ОДНАКО не проверен: token-shift здесь хранится как [D], а в Python
    // как [1,1,D]. На арифметику это не влияет (broadcast), на строгую
    // проверку форм — может.

    public static let checkpointFormat = "rwkv-metal-reranker-head-v1"

    func metadata(_ extra: [String: String]? = nil) -> [String: String] {
        var md = [
            "format": Self.checkpointFormat,
            "layer_idx": head.layerIdx.map(String.init).joined(separator: ","),
            "shared_state": head.cfg.sharedState ? "1" : "0",
            "n_probe": String(head.cfg.nProbe),
            "head_hidden": head.cfg.headHidden.map(String.init) ?? "",
            "base_n_layer": String(base.cfg.nLayer),
            "base_n_embd": String(base.cfg.nEmbd),
            "base_n_head": String(base.cfg.nHead),
        ]
        if let extra { for (k, v) in extra { md[k] = v } }
        return md
    }

    /// Сохранить ТОЛЬКО голову: база по определению не изменилась.
    ///
    /// `extra` — произвольные строки в metadata. Сюда стоит класть контракт
    /// подачи текста (шаблон, обрезки, терминатор, инструкция): модель о нём
    /// не знает, а расходится он так же молча, как и слои.
    public func saveHead(to url: URL, extra: [String: String]? = nil) throws {
        var arrays: [String: MLXArray] = [:]
        for (name, p) in zip(head.parameterNames, head.parameters) {
            arrays[name] = p
        }
        eval(Array(arrays.values))
        try save(arrays: arrays, metadata: metadata(extra), url: url)
    }

    /// Метаданные чекпоинта без загрузки весов в модель.
    public static func readHeadMetadata(_ url: URL) throws -> [String: String] {
        try loadArraysAndMetadata(url: url).1
    }

    @discardableResult
    public func loadHead(from url: URL, strict: Bool = true) throws -> Reranker {
        let (weights, md) = try loadArraysAndMetadata(url: url)
        if strict, (md["format"] ?? "").hasPrefix("rwkv-metal-reranker-head") {
            let want = metadata()
            for key in ["layer_idx", "shared_state", "n_probe",
                        "base_n_layer", "base_n_embd", "base_n_head"] {
                if let inFile = md[key], inFile != want[key] {
                    throw RerankerError.checkpointMismatch(key, inFile, want[key] ?? "")
                }
            }
        }
        let names = head.parameterNames
        let missing = names.filter { weights[$0] == nil }
        precondition(missing.isEmpty,
                     "в чекпоинте нет параметров: \(missing.prefix(5).joined(separator: ", "))")
        head.setParameters(names.map { weights[$0]! })
        eval(head.parameters)
        return self
    }

    /// Собрать реранкер по конфигурации, записанной в самом чекпоинте.
    ///
    /// Рекомендуемый способ загрузки: не нужно помнить, какие слои читала
    /// голова и сколько у неё зондов, — а помнить пришлось бы, потому что
    /// формы тензоров от этого не зависят.
    public static func fromHead(base: X070Backbone, url: URL,
                                freezeBase: Bool = true,
                                headDType: DType = .float32) throws -> Reranker {
        let md = try readHeadMetadata(url)
        guard (md["format"] ?? "").hasPrefix("rwkv-metal-reranker-head") else {
            throw RerankerError.notARerankerCheckpoint(url.path)
        }
        let hidden = md["head_hidden"] ?? ""
        let cfg = RerankerConfig(
            layerIdx: (md["layer_idx"] ?? "").split(separator: ",").map { Int($0)! },
            sharedState: (md["shared_state"] ?? "0") == "1",
            nProbe: Int(md["n_probe"] ?? "1") ?? 1,
            headHidden: hidden.isEmpty ? nil : Int(hidden))
        let model = try Reranker(base: base, cfg: cfg, freezeBase: freezeBase,
                                 headDType: headDType)
        try model.loadHead(from: url)
        return model
    }
}
