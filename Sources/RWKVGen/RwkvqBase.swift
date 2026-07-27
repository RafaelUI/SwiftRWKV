import Foundation
import MLX
import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Подключение квантованной базы .rwkvq к X070Backbone.
//
//  Порт rwkv_metal/lora/add_rwkvq.py.
//
//  Зачем отдельно от стокового кванта (LoRA.quantizeBaseModel): различаются
//  не параметры, а происхождение чисел. Стоковый mlx.nn.quantize пересчитывает
//  scale/bias по min/max блока — это своя, никем не откалиброванная схема.
//  .rwkvq приходит из пайплайна rwkv-quant с измеренной деградацией ppl
//  (REDUCTION: +0.12% на 1.5B). Смешивать их значило бы потерять то
//  единственное, что делает квантованную базу пригодной для QLoRA, — знание,
//  насколько она отличается от исходной.
// ───────────────────────────────────────────────────────────────────────

extension X070Backbone {

    /// Что именно брать из сайдкара.
    public struct RwkvqAttachOptions: Sendable {
        /// Проекции tmix. Пустой список ⇒ не трогать.
        public var tmixTargets: [String]
        /// Квантовать ли cmix (key/value) — самые крупные матрицы блока.
        public var quantizeCmix: Bool
        /// Квантовать ли выходную голову.
        public var quantizeHead: Bool
        /// Квантовать ли таблицу эмбеддингов.
        ///
        /// По умолчанию НЕТ, и это не осторожность, а арифметика: формат sb6
        /// не поддерживает выборку отдельных строк (коды упакованы блоками
        /// вдоль входной оси), поэтому эмбеддинг пришлось бы разворачивать
        /// целиком на каждом проходе — ~200 МБ транзиента для словаря 65536
        /// ради нескольких сотен нужных строк. Держать таблицу в bf16 дешевле
        /// и по памяти, и по времени.
        public var quantizeEmbedding: Bool
        /// Слои; nil ⇒ все.
        public var layers: Range<Int>?
        /// Выбрасывать ли плотную bf16-копию после подключения. Ради этого
        /// всё и затевается: если её оставить, база займёт и то, и другое.
        public var dropDenseWeights: Bool

        public init(tmixTargets: [String] = ["r_proj", "k_proj", "v_proj", "o_proj"],
                    quantizeCmix: Bool = true,
                    quantizeHead: Bool = true,
                    quantizeEmbedding: Bool = false,
                    layers: Range<Int>? = nil,
                    dropDenseWeights: Bool = true) {
            self.tmixTargets = tmixTargets
            self.quantizeCmix = quantizeCmix
            self.quantizeHead = quantizeHead
            self.quantizeEmbedding = quantizeEmbedding
            self.layers = layers
            self.dropDenseWeights = dropDenseWeights
        }
    }

    public struct RwkvqAttachInfo: Sendable {
        /// Сколько весов теперь читается из сайдкара.
        public let attached: Int
        /// Имена, которых в сайдкаре не нашлось (остались плотными).
        public let missing: [String]
        /// Сколько байт плотных весов освобождено.
        public let freedDenseBytes: Int
        /// Сколько байт занимают упакованные буферы.
        public let packedBytes: Int
    }

    /// Подключить квантованную базу: перечисленные веса начинают читаться из
    /// сайдкара, плотные копии освобождаются.
    ///
    /// Веса, которых в сайдкаре нет (в т.ч. низкоранговые w/a/v_lora —
    /// экспортёр их намеренно не включает), остаются как были.
    @discardableResult
    public func attachRwkvq(_ sidecar: RwkvqSidecar,
                            options: RwkvqAttachOptions = RwkvqAttachOptions())
        -> RwkvqAttachInfo {

        let range = options.layers ?? (0 ..< cfg.nLayer)
        var wanted: [String] = []
        for layer in range {
            for name in options.tmixTargets {
                wanted.append("blocks.\(layer).tmix.\(name).weight")
            }
            if options.quantizeCmix {
                wanted.append("blocks.\(layer).cmix.key.weight")
                wanted.append("blocks.\(layer).cmix.value.weight")
            }
        }
        if options.quantizeHead { wanted.append("head.weight") }
        if options.quantizeEmbedding { wanted.append("emb.weight") }

        var missing: [String] = []
        var freed = 0
        for key in wanted {
            guard let world = RwkvqNaming.worldKey(forX070: key),
                  sidecar.contains(world) else {
                missing.append(key)
                continue
            }
            // Форма из манифеста обязана совпасть с тем, что ждёт модель:
            // подсунутый сайдкар от другой геометрии иначе проявился бы
            // мусором на выходе, а не ошибкой.
            if let dense = w[key] {
                let info = sidecar.tensors[world]!
                guard dense.shape == info.shape else {
                    missing.append(key)
                    continue
                }
                freed += dense.size * dense.dtype.size
            }
            rwkvqKeys[key] = world
            if options.dropDenseWeights { w[key] = nil }
        }
        rwkvqSidecar = sidecar

        return RwkvqAttachInfo(attached: rwkvqKeys.count, missing: missing,
                               freedDenseBytes: freed,
                               packedBytes: sidecar.packedBytes)
    }

    /// Отключить квантованную базу. Плотные веса при этом НЕ возвращаются —
    /// если они были выброшены, модель после этого неработоспособна;
    /// метод существует для тестов и для пересборки конфигурации до forward.
    public func detachRwkvq() {
        rwkvqSidecar = nil
        rwkvqKeys.removeAll()
    }

    /// Читается ли этот вес из квантованной базы.
    public func isRwkvqBacked(_ key: String) -> Bool { rwkvqKeys[key] != nil }

    /// Ключи, читаемые из .rwkvq.
    public var rwkvqBackedKeys: [String] { rwkvqKeys.keys.sorted() }
}
