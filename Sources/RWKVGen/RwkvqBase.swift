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
        /// Переложить sb6 в родной контейнер MLX и считать через
        /// `quantizedMM` вместо «развернуть матрицу и умножить».
        ///
        /// Числа от этого не меняются (см. RwkvqNative.swift), меняется
        /// трафик: нынешний путь читает сжатое, ПИШЕТ плотный транзиент
        /// и читает его обратно. На 2.9B это 13.6 ГБ на токен против
        /// 5.9 у плотного bf16 — отсюда и 2.21x проигрыша.
        /// Перекладка одноразовая, при подключении.
        public var useNativeKernel: Bool
        /// Тип квантованной базы — см. `X070Backbone.rwkvqBaseDType`.
        /// Умолчание fp16 (норма rwkv-quant с 02.10); `nil` — путь порта до
        /// 08.10 (деквант в тип входа, без приведения), под ним сняты
        /// июльские эталоны.
        public var baseDType: DType?

        public init(tmixTargets: [String] = ["r_proj", "k_proj", "v_proj", "o_proj"],
                    quantizeCmix: Bool = true,
                    quantizeHead: Bool = true,
                    quantizeEmbedding: Bool = false,
                    layers: Range<Int>? = nil,
                    dropDenseWeights: Bool = true,
                    useNativeKernel: Bool = false,
                    baseDType: DType? = .float16) {
            self.useNativeKernel = useNativeKernel
            self.baseDType = baseDType
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
        /// Цели, которые в файле лежат плотными и подключены плотной
        /// замороженной базой (см. `X070Backbone.rwkvqDenseTargets`).
        public let denseTargets: [String]
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
        var denseTargets: [String] = []
        var freed = 0
        rwkvqBaseDType = options.baseDType
        for key in wanted {
            guard let world = RwkvqNaming.worldKey(forX070: key),
                  let info = sidecar.tensors[world] else {
                missing.append(key)
                continue
            }
            // Плотная цель — не «не умеем», а «файл так решил»: вес берётся
            // из файла как есть (bf16) и остаётся плотной замороженной базой.
            if info.kind == .dense {
                guard info.shape.count == 2,
                      w[key].map({ $0.shape == info.shape }) ?? true,
                      let raw = try? sidecar.dequantize(world, dtype: .bfloat16) else {
                    missing.append(key)
                    continue
                }
                w[key] = info.transposed ? raw.transposed() : raw
                denseTargetCast[key] = nil
                rwkvqDenseTargets.insert(key)
                denseTargets.append(key)
                continue
            }
            // Квантованная база — sb6 или sym. Полный манифест содержит и
            // asym/rtn, и подключить их сюда нельзя. Молча привязаться и
            // упасть потом, при первом forward, было бы хуже, чем честно
            // назвать ключ ненайденным.
            guard info.isQuantizedBase else {
                missing.append(key)
                continue
            }
            // Форма из манифеста обязана совпасть с тем, что ждёт модель:
            // подсунутый сайдкар от другой геометрии иначе проявился бы
            // мусором на выходе, а не ошибкой.
            if let dense = w[key] {
                guard dense.shape == info.shape else {
                    missing.append(key)
                    continue
                }
                freed += dense.size * dense.dtype.size
            }
            rwkvqKeys[key] = world
            // Родной контейнер есть только у sb6: у sym блок 16, а
            // quantizedMM принимает группы 32, 64 и 128. Прежний путь
            // (baseDType == nil) перекладывает по-прежнему — scale всегда
            // в fp16, — чтобы его числа не сдвинулись.
            if options.useNativeKernel, info.isSb6,
               let native = try? sidecar.nativeAffine(
                   world, exactScales: options.baseDType != nil) {
                rwkvqNative[key] = native
            }
            if options.dropDenseWeights { w[key] = nil }
        }

        // Сайдкар удерживается ТОЛЬКО если он ещё нужен. С родным
        // контейнером исходные буферы K3 не читает уже никто, но ссылка
        // на них держала бы память: замерено — 6.97 ГБ против 3.11 на
        // 2.9B, то есть выигрыш формата съедался целиком тем, что обе
        // раскладки лежат рядом. Условие строгое: хотя бы один ключ без
        // родной перекладки — и сайдкар остаётся, иначе `baseProj`
        // упадёт на ключе, для которого нет ни плотного веса, ни
        // native, ни сайдкара.
        let fullyNative = options.useNativeKernel
            && !rwkvqKeys.isEmpty
            && rwkvqKeys.keys.allSatisfy { rwkvqNative[$0] != nil }
        // Удерживается не вся база, а только буферы ключей, которые ещё
        // читаются деквантом: развёрнутые при сборке ветки, emb и
        // переложенные в родной контейнер sb6 держать незачем.
        let needed = Set(rwkvqKeys.filter { rwkvqNative[$0.key] == nil }.values)
        // r/k/v каждого слоя сводятся в один буфер ради декода одним
        // запуском ядра (только sym одной формы и битности; остальные
        // группы остаются как были). Копии байт при этом нет.
        var groups: [String: [String]] = [:]
        for i in 0 ..< cfg.nLayer {
            let keys = ["receptance", "key", "value"].map { "blocks.\(i).att.\($0).weight" }
            if keys.allSatisfy(needed.contains) { groups["blocks.\(i).att.rkv"] = keys }
        }
        rwkvqSidecar = fullyNative ? nil
            : sidecar.pruned(keeping: needed).fusingSym(options.baseDType == nil ? [:] : groups)

        return RwkvqAttachInfo(attached: rwkvqKeys.count, missing: missing,
                               freedDenseBytes: freed,
                               packedBytes: sidecar.packedBytes,
                               denseTargets: denseTargets.sorted())
    }

    /// Отключить квантованную базу. Плотные веса при этом НЕ возвращаются —
    /// если они были выброшены, модель после этого неработоспособна;
    /// метод существует для тестов и для пересборки конфигурации до forward.
    public func detachRwkvq() {
        rwkvqSidecar = nil
        rwkvqKeys.removeAll()
        rwkvqNative.removeAll()
        rwkvqDenseTargets.removeAll()
        denseTargetCast.removeAll()
        rwkvqBaseDType = nil
    }

    /// Читается ли этот вес из квантованной базы.
    public func isRwkvqBacked(_ key: String) -> Bool { rwkvqKeys[key] != nil }

    /// Держится ли ещё сырой сайдкар (K3-буферы) в памяти после attach.
    ///
    /// true означает, что ХОТЯ БЫ один ключ не переложился в родной MLX-
    /// контейнер (см. attachRwkvq: `fullyNative` условие) — тогда сайдкар
    /// не освобождается и его буферы держатся ОДНОВРЕМЕННО с уже
    /// переложенными native-буферами. На 2.9B это разница 6.97 ГБ против
    /// 3.11 (см. RwkvqBase.swift) — стоит проверять после подключения
    /// квантованной базы, а не удивляться памяти постфактум.
    public var isRawSidecarRetained: Bool { rwkvqSidecar != nil }

    /// Переложен ли КОНКРЕТНЫЙ ключ в родной MLX-контейнер (в отличие от
    /// `isRwkvqBacked`, которое верно и для того, что читается через
    /// `sc.dequantize` -- т.е. ещё не переложено).
    public func hasNativeRepack(_ key: String) -> Bool { rwkvqNative[key] != nil }

    /// Ключи, читаемые из .rwkvq.
    public var rwkvqBackedKeys: [String] { rwkvqKeys.keys.sorted() }
}
