import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  ЧТО обучается при дообучении эмбеддингов.
//
//  Голова обязана остаться обучаемой при ЛЮБОЙ судьбе базы. Это не удобство,
//  а условие осмысленности: голова инициализирована тождеством (fc2 = 0), и
//  если её не обучать, модель буквально остаётся предобученной базой —
//  контрастный лосс не имеет ни одного параметра, через который мог бы на
//  что-то повлиять. Замороженная база + замороженная голова = ноль обучения
//  при формально работающем цикле.
//
//  Отсюда и композиция: голова живёт ОТДЕЛЬНЫМ множеством и подмешивается
//  к любому базовому — или существует одна, если база заморожена.
// ───────────────────────────────────────────────────────────────────────

/// Обучаемая голова эмбеддинга.
///
/// Подстановка идёт прямо в объект головы (`setParameters`), как
/// `LoRATrainableSet` подставляет адаптеры в backbone: вызов происходит
/// внутри grad-замыкания, поэтому цепь к fp32-мастеру сохраняется, и
/// `EmbeddingModel.embed` читает уже трассируемые тензоры.
///
/// dtype не понижается. У LoRA исторически инжектится bf16-копия (адаптеры
/// так и живут в модели), здесь понижать нечего: голова — четыре небольших
/// тензора, и её вклад идёт после базы, поверх уже огрублённого bf16-выхода;
/// лишнее округление тут стоило бы точности задаром.
public final class EmbeddingHeadTrainableSet: TrainableSet {

    private let head: EmbeddingHead

    public init(_ head: EmbeddingHead) {
        self.head = head
    }

    public var parameterNames: [String] { EmbeddingHead.parameterNames }

    public func initialParameters() -> [MLXArray] {
        let ps = head.parameters.map { $0.asType(.float32) }
        eval(ps)
        return ps
    }

    public func inject(_ ps: [MLXArray]) {
        head.setParameters(ps)
    }

    public func commit(_ ps: [MLXArray]) {
        head.setParameters(ps)
        eval(head.parameters)
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Готовые рецепты
// ───────────────────────────────────────────────────────────────────────

/// Что делать с базой под головой.
public enum BaseTrainingMode: Sendable, Equatable {
    /// База заморожена целиком — обучается только голова.
    ///
    /// Дёшево и безопасно, но потолок низкий: голова видит лишь пулинг
    /// последнего слоя, изменить сами представления она не может.
    case frozen
    /// Обучаются верхние N слоёв базы (плюс ln_out) и голова.
    case topLayers(Int)
    /// Обучается вся база и голова — рецепт по умолчанию для малых моделей
    /// (совпадает с sft_curriculum у EmbeddingRWKV: --freeze_rwkv 0).
    case full
    /// Обучаются навешенные LoRA/QLoRA-адаптеры и голова. Путь для моделей,
    /// которые уже не влезают в full-FT на Apple Silicon.
    case lora
}

public enum EmbeddingTrainable {

    /// Собрать множество обучаемых параметров под выбранный режим.
    ///
    /// Побочный эффект намеренный: выставляется `backbone.trainLayers` —
    /// набор слоёв, идущих через ДИФФЕРЕНЦИРУЕМОЕ WKV-ядро. Без него
    /// градиент до весов базы просто не дотечёт, а лосс при этом будет
    /// исправно убывать за счёт одной головы. Связывать это с выбором
    /// режима в одном месте безопаснее, чем оставлять вызывающему.
    public static func make(model: EmbeddingModel,
                            mode: BaseTrainingMode) -> TrainableSet {
        let bb = model.backbone
        let headSet = EmbeddingHeadTrainableSet(model.head)

        switch mode {
        case .frozen:
            bb.trainLayers = []
            return CompositeTrainableSet([headSet])

        case .topLayers(let n):
            let from = Swift.max(0, bb.cfg.nLayer - n)
            bb.trainLayers = Set(from ..< bb.cfg.nLayer)
            let keys = BackboneWeightsTrainableSet.topLayerKeys(bb, from: from)
            return CompositeTrainableSet([BackboneWeightsTrainableSet(bb, keys: keys), headSet])

        case .full:
            bb.trainLayers = Set(0 ..< bb.cfg.nLayer)
            // emb.weight и head.weight в множество НЕ входят: голова языковой
            // модели в эмбеддинг-задаче не участвует вовсе (используется
            // body(), не логиты), а таблица эмбеддингов при контрастном
            // дообучении на десятках тысяч строк получает градиент лишь по
            // считанным строкам словаря и от этого только расползается.
            let keys = bb.bodyWeightKeys
            return CompositeTrainableSet([BackboneWeightsTrainableSet(bb, keys: keys), headSet])

        case .lora:
            precondition(bb.hasLoRAAdapters,
                         "режим .lora без адаптеров — сначала LoRA.add(...)")
            bb.trainLayers = Set(0 ..< bb.cfg.nLayer)
            return CompositeTrainableSet([LoRATrainableSet(bb), headSet])
        }
    }
}
