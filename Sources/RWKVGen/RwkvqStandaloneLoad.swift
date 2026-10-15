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
    /// - useNativeKernel: как держать sb6 (пресет COMPRESSION). На sym не
    ///   влияет: у неё родного контейнера нет (блок 16).
    ///   • `false` (умолчание) — K3-интерлив, как в файле: декод ядром GEMV,
    ///     параллельный проход деквантом и fp16-матмулом (так же считает
    ///     Metal-бэкенд rwkv-quant). Загрузка без перекладки.
    ///   • `true` — родной контейнер MLX (`quantizedMM`) на обоих путях; так
    ///     считает rwkv-metal, и с ним параллельный проход совпадает до
    ///     ~1e-8 KL. Цена — перекладка при загрузке.
    ///   Замер на 1.5B (K3 против родного): загрузка 0.4 против 3.7 с, пик
    ///   загрузки 1.50 против 2.39 ГБ, память 1.22 против 1.31 ГБ, префилл
    ///   512 токенов 824 против 908 мс; KL декода к точному счёту на 0.1B
    ///   2.0e-6 против 5.3e-6. Держать обе раскладки значило бы удвоить
    ///   память модели, поэтому база собирается под одну.
    /// - denseEmbedding: держать таблицу эмбеддинга плотной, даже когда её
    ///   можно оставить сжатой (см. `rwkvqEmbTable`). Нужно тем, кто обучает
    ///   саму таблицу: сжатой таблицы среди весов (`weightKeys`) нет.
    public static func load(fromRwkvq path: String,
                            paramDType: DType = .float16,
                            baseDType: DType? = .float16,
                            useNativeKernel: Bool = false,
                            denseEmbedding: Bool = false) throws -> X070Backbone {
        let p = (path as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        let isFile = FileManager.default.fileExists(atPath: p, isDirectory: &isDir) && !isDir.boolValue
        if !isFile, FileManager.default.fileExists(atPath: p + ".safetensors") {
            // прежний сайдкар: пара файлов, читается целиком
            return try load(fromSidecar: try RwkvqSidecar(path: p), computeDType: paramDType,
                            baseDType: baseDType, useNativeKernel: useNativeKernel)
        }
        return try loadStreaming(p, computeDType: paramDType, baseDType: baseDType,
                                 useNativeKernel: useNativeKernel, denseEmbedding: denseEmbedding)
    }

    /// Имена целей квантованной базы в файле (world): проекции tmix, cmix и
    /// голова.
    static func rwkvqTargetWorldKeys(nLayer: Int) -> [String] {
        var keys = ["head.weight"]
        for i in 0 ..< nLayer {
            for n in ["receptance", "key", "value", "output"] {
                keys.append("blocks.\(i).att.\(n).weight")
            }
            keys.append("blocks.\(i).ffn.key.weight")
            keys.append("blocks.\(i).ffn.value.weight")
        }
        return keys
    }

    /// Каждая цель обязана лежать в файле в известной раскладке —
    /// проверяется ДО сборки. Иначе «нет такого тензора» и «не умеем эту
    /// раскладку» выглядели бы одинаково: строкой в `missing`, которую
    /// никто не читает, и модель считала бы без этих весов до первого
    /// обращения.
    static func checkRwkvqTargets(_ tensors: [String: RwkvqTensorInfo], _ wantedWorld: [String]) throws {
        var refused: [String: Int] = [:]
        var first: String? = nil
        for key in wantedWorld {
            let kind = tensors[key]?.kind
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
    }

    /// Загрузка `.rwkvq` БЕЗ ВСПЛЕСКА ПАМЯТИ: в памяти MLX в каждый момент
    /// лежит не больше, чем останется после загрузки (плюс один тензор на
    /// перекладке).
    ///
    /// Прежний порядок — «вся база в MLX, потом сборка» — держал рядом то,
    /// что нужно один раз, и то, во что оно разворачивается: коды веток и
    /// emb вместе с их плотными копиями, r/k/v вместе со сведённым буфером.
    /// На 1.5B REDUCTION пик MLX был 2.25 ГБ при рабочих 1.61, и footprint
    /// процесса оставался на пике: освобождённое Metal системе не отдаёт.
    ///
    /// Теперь файл читается ТРИЖДЫ ПО ЧАСТЯМ (он отображён в память; чтение
    /// части — копия её буферов в MLX):
    ///   1. манифест — проверка целей, геометрия;
    ///   2. всё, что разворачивается в плотное (нормы, миксы, низкоранговые
    ///      ветки, emb): поднять, развернуть, отпустить исходные буферы —
    ///      пока база ещё не загружена;
    ///   3. квантованная база — каждый тензор сразу в окончательную
    ///      раскладку, от крупных к мелким, r/k/v — сразу в сведённый буфер.
    /// emb в sym при восьми битах не разворачивается вовсе (`rwkvqEmbTable`).
    static func loadStreaming(_ p: String, computeDType: DType, baseDType: DType?,
                              useNativeKernel: Bool, denseEmbedding: Bool) throws -> X070Backbone {
        let manifest = try RwkvqSidecar(rwkvq: p, only: { _ in false })
        let wantedWorld = rwkvqTargetWorldKeys(nLayer: manifest.nLayer)
        try checkRwkvqTargets(manifest.tensors, wantedWorld)
        let wanted = Set(wantedWorld)

        var weights: [String: MLXArray]
        var table: RwkvqSymRowTable? = nil
        do {
            let small = try RwkvqSidecar(rwkvq: p, only: { !wanted.contains($0) })
            weights = try small.buildDenseWeights(computeDType: computeDType)
            if !denseEmbedding, computeDType == .float16, let t = small.symRowTable("emb.weight") {
                table = t
            } else {
                weights["emb.weight"] = try small.dequantize("emb.weight", dtype: computeDType)
            }
            eval(Array(weights.values))
        }
        // исходные буферы плотных параметров отпущены — вернуть их из кеша
        // MLX до того, как начнёт расти база
        GPU.clearCache()

        let cfg = X070Config(nLayer: manifest.nLayer, nEmbd: manifest.nEmbd,
                             headSize: manifest.headSize, vocab: manifest.vocabSize)
        let bb = X070Backbone(weights: weights, cfg: cfg, computeDType: computeDType)
        weights = [:]
        bb.rwkvqEmbTable = table

        // r/k/v слоя сводятся при чтении (в норме; прежний путь базы не сводит)
        var groups: [String: [String]] = [:]
        if baseDType != nil {
            for i in 0 ..< cfg.nLayer {
                groups["blocks.\(i).att.rkv"] =
                    ["receptance", "key", "value"].map { "blocks.\(i).att.\($0).weight" }
            }
        }
        let base = try RwkvqSidecar(rwkvq: p, only: { wanted.contains($0) }, fusingSym: groups)
        try bb.finishRwkvqLoad(base, baseDType: baseDType, useNativeKernel: useNativeKernel)
        return bb
    }

    /// Собрать бэкбон из уже открытой квантованной базы.
    public static func load(fromSidecar sidecar: RwkvqSidecar,
                            computeDType: DType = .float16,
                            baseDType: DType? = .float16,
                            useNativeKernel: Bool = false) throws -> X070Backbone {
        // Полный экспорт несёт плотные параметры (нормы, миксы token-shift —
        // раскладка dense); неполный — одни сжатые проекции. Без этой проверки
        // неполный отказывал на первом же недостающем тензоре с
        // missingBuffer(…, "manifest"), не называя причины.
        guard sidecar.tensors.values.contains(where: { $0.kind == .dense }) else {
            throw RwkvqError.partialExport(tensors: sidecar.tensors.count)
        }
        try checkRwkvqTargets(sidecar.tensors, rwkvqTargetWorldKeys(nLayer: sidecar.nLayer))

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

        try bb.finishRwkvqLoad(sidecar, baseDType: baseDType, useNativeKernel: useNativeKernel)
        return bb
    }

    /// Общий конец загрузки: подключить базу, вычислить плотные параметры,
    /// отпустить кеш MLX.
    func finishRwkvqLoad(_ sidecar: RwkvqSidecar, baseDType: DType?, useNativeKernel: Bool) throws {
        let bb = self
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

        // короткий параллельный проход — ядрами GEMV; порог по раскладке базы
        if baseDType != nil, !useNativeKernel,
           let kind = sidecar.tensors["blocks.0.att.key.weight"]?.kind, kind == .sym || kind == .sb6 {
            bb.shortPassKernelRows = X070Backbone.defaultShortPassRows(nEmbd: bb.cfg.nEmbd, sym: kind == .sym)
        }

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
    }
}
