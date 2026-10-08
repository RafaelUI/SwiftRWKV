import Foundation
import MLX
import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Загрузка X070Backbone из ОДНОЙ квантованной базы, без стороннего
//  dense-файла: `.rwkvq` напрямую либо сайдкар `.rwkvq_mlx`.
//
//  Порт rwkv_metal/lora/add_rwkvq.py::load_rwkvq_model(rank=0) и
//  model/convert.py::load_pretrained_rwkvq.
//
//  Тело модели (нормы, token-shift, внутренние low-rank ветки
//  decay/iclr/value/gate) собирается плотным через
//  RwkvqSidecar.buildDenseWeights; проекции tmix, cmix и голова остаются
//  квантованными в памяти и подключаются attachRwkvq — ради этого всё и
//  затевалось. Цель, лежащая в файле плотной (o_proj слоя 0 у нынешних
//  пресетов), подключается плотной замороженной базой.
//
//  ТИПЫ ПО УМОЛЧАНИЮ — fp16 и для плотных параметров, и для квантованной
//  базы: норма rwkv-quant с 02.10. Плотная модель из .safetensors/.pth
//  остаётся в bf16; это умолчание — только про квантованную базу.
// ───────────────────────────────────────────────────────────────────────

extension X070Backbone {

    /// Собрать бэкбон из `.rwkvq` (либо из сайдкара — путь без суффикса).
    ///
    /// - paramDType: тип плотных параметров (emb, нормы, миксы, low-rank
    ///   ветки). fp16 по умолчанию; `.bfloat16` — прежнее.
    /// - baseDType: тип квантованной базы, см. `rwkvqBaseDType`.
    /// - useNativeKernel: sb6 перекладывается в родной контейнер MLX
    ///   (`quantizedMM`) и не разворачивается в плотный транзиент. У sym
    ///   родного контейнера нет (блок 16): она деквантуется на каждый проход.
    public static func load(fromRwkvq path: String,
                            paramDType: DType = .float16,
                            baseDType: DType? = .float16,
                            useNativeKernel: Bool = true) throws -> X070Backbone {
        try load(fromSidecar: try RwkvqSidecar.open(path), computeDType: paramDType,
                 baseDType: baseDType, useNativeKernel: useNativeKernel)
    }

    /// Собрать бэкбон из уже открытой квантованной базы.
    public static func load(fromSidecar sidecar: RwkvqSidecar,
                            computeDType: DType = .float16,
                            baseDType: DType? = .float16,
                            useNativeKernel: Bool = true) throws -> X070Backbone {
        // Каждая цель обязана лежать в файле в известной раскладке —
        // проверяется ДО сборки. Иначе «нет такого тензора» и «не умеем эту
        // раскладку» выглядели бы одинаково: строкой в `missing`, которую
        // никто не читает, и модель считала бы без этих весов до первого
        // обращения.
        var wantedWorld = ["head.weight"]
        for i in 0 ..< sidecar.nLayer {
            for n in ["receptance", "key", "value", "output"] {
                wantedWorld.append("blocks.\(i).att.\(n).weight")
            }
            wantedWorld.append("blocks.\(i).ffn.key.weight")
            wantedWorld.append("blocks.\(i).ffn.value.weight")
        }
        var refused: [String: Int] = [:]
        var first: String? = nil
        for key in wantedWorld {
            let kind = sidecar.tensors[key]?.kind
            if let kind, kind == .sb6 || kind == .sym || kind == .dense { continue }
            refused[kind?.rawValue ?? "НЕТ В ФАЙЛЕ", default: 0] += 1
            if first == nil { first = key }
        }
        if let first {
            let byKind = refused.sorted { $0.key < $1.key }
                .map { "\($0.key) — \($0.value)" }.joined(separator: ", ")
            throw RwkvqError.unsupportedKind(
                tensor: first,
                kind: "\(refused.values.reduce(0, +)) из \(wantedWorld.count) целей "
                    + "не подключаются квантованной базой. По раскладкам: \(byKind)")
        }

        var weights = try sidecar.buildDenseWeights(computeDType: computeDType)

        // emb.weight — единственное исключение: физически он квантован, но
        // разворачивается ОДИН раз здесь, а не на каждом forward. Строчной
        // выборки раскладки не дают (коды упакованы блоками вдоль входной
        // оси), так что оставить его сжатым означало бы разворачивать таблицу
        // 65536×D целиком на КАЖДЫЙ gather.
        weights["emb.weight"] = try sidecar.dequantize("emb.weight", dtype: computeDType)

        let cfg = X070Config(nLayer: sidecar.nLayer, nEmbd: sidecar.nEmbd,
                             headSize: sidecar.headSize, vocab: sidecar.vocabSize)
        let bb = X070Backbone(weights: weights, cfg: cfg, computeDType: computeDType)

        let info = bb.attachRwkvq(sidecar, options: RwkvqAttachOptions(
            quantizeCmix: true, quantizeHead: true, quantizeEmbedding: false,
            useNativeKernel: useNativeKernel, baseDType: baseDType))
        guard info.missing.isEmpty else {
            throw RwkvqError.unsupportedKind(
                tensor: info.missing[0],
                kind: "не подключились \(info.missing.count) целей: форма в файле "
                    + "не сошлась с моделью")
        }

        // Конец сборки: плотные параметры вычисляются СЕЙЧАС, а кеш MLX
        // отпускается. Без eval они остаются ленивым графом, который держит
        // буферы исходной базы целиком (emb, ветки, K3 уже переложенных
        // sb6) до первого прохода: на 1.5B COMPRESSION это +3.5 ГБ footprint
        // после загрузки против 1.7 ГБ после первого декода.
        eval(Array(bb.w.values))
        GPU.clearCache()

        // sb6, не переложившийся в родной контейнер, держит K3-буферы рядом
        // с native-буферами остальных. Печатаем ИМЕНА: «что-то не так»
        // бесполезно для диагностики. sym сюда не относится — у неё родного
        // контейнера нет по построению.
        if useNativeKernel {
            let notNative = bb.rwkvqBackedKeys.filter { key in
                guard let world = RwkvqNaming.worldKey(forX070: key) else { return false }
                return sidecar.tensors[world]?.kind == .sb6 && !bb.hasNativeRepack(key)
            }
            if !notNative.isEmpty {
                FileHandle.standardError.write((
                    "[X070Backbone.load] ВНИМАНИЕ: sb6 не переложился в родной "
                    + "контейнер и держится K3-буферами: \(notNative)\n"
                ).data(using: .utf8)!)
            }
        }
        return bb
    }
}
