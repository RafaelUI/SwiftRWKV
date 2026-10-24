import Foundation
import MLX
import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Обучение с подкачкой весов с диска — по выбору, не по умолчанию (23.10).
//
//  Обратный проход по блокам (`SequentialBody`) держит в работе один блок, но
//  веса всей модели при этом лежат в памяти: у 1.5B в bf16 это 3 ГБ при бюджете
//  приложения на айфоне ~3 ГБ. С подкачкой бэкбон собирается из плотного
//  .safetensors ЛЕНИВО (MLX читает тензор из файла только при вычислении), и
//  после каждого блока прямого и обратного прохода его веса заменяются свежими
//  ленивыми ссылками на тот же файл: прочитанные данные освобождаются, при
//  следующем обращении читаются снова. То же — таблица эмбеддингов после
//  основания и голова после лосса. В памяти живут веса одного-двух блоков.
//
//  Числа те же до бита: читается тот же файл, приводится к тому же типу.
//  Цена — чтение файла на каждом проходе (на маке файл обычно в кэше ОС; на
//  телефоне — с накопителя).
//
//  Освобождение делают пути по блокам (`SequentialBody.forward/backward`,
//  обратный проход языковой модели): обучение состояния, адаптеров, верхних
//  слоёв и эмбеддера по блокам. Проход одним графом (`bb(ids)`, обучение без
//  `sequentialBackward`) веса прочитает и оставит в памяти до
//  `releaseAllStreamedWeights()`.
//
//  Уже собранная модель (в первую очередь из `.rwkvq`, 24.10) — через
//  `enableWeightStreaming(cache:)`: веса блоков, таблица эмбеддингов и голова
//  (у квантованной базы — её буферы) один раз выгружаются в файл-кэш
//  safetensors как есть, и дальше модель читает их оттуда так же лениво. Байты
//  те же — числа те же до бита.
// ───────────────────────────────────────────────────────────────────────

/// Откуда брать веса заново: файл, тип вычислений, имена по частям модели.
struct WeightStream {
    let url: URL
    /// К какому типу приводить прочитанное; nil — как в файле (выгрузка).
    let dtype: DType?
    let byLayer: [[String]]
    let stem: [String]
    let head: [String]
    /// Имена в файле — с префиксом части модели (выгрузка): `w/` — словарь весов,
    /// `sc/` — буферы квантованной базы, `emb/` — сжатая таблица эмбеддингов.
    /// Иначе — ключи словаря весов как есть (плотный чекпоинт).
    let prefixed: Bool

    /// Свежие ленивые ссылки на веса `keys` (данные не читаются до вычисления).
    func fresh(_ keys: [String]) -> [String: MLXArray] {
        let all: [String: MLXArray]
        do { all = try loadArrays(url: url) } catch {
            preconditionFailure("подкачка весов: файл \(url.path) не читается: \(error)")
        }
        var out: [String: MLXArray] = [:]
        for k in keys {
            guard let v = all[k] else { preconditionFailure("подкачка весов: в \(url.lastPathComponent) нет \(k)") }
            out[k] = dtype.map { v.asType($0) } ?? v
        }
        return out
    }
}

extension X070Backbone {

    /// Бэкбон из плотного .safetensors с подкачкой весов с диска (см. шапку файла).
    /// Имена и формы — как у `X070Backbone(weights:cfg:)` из того же файла; числа
    /// всех путей с ним совпадают побитно.
    public static func loadStreaming(safetensors url: URL, computeDType: DType = .bfloat16) throws -> X070Backbone {
        let w = try loadArrays(url: url)                 // ленивые: данные не читаются
        let nLayer = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        guard nLayer > 0, let lnOut = w["ln_out.weight"], let kk = w["blocks.0.tmix.k_k"], let head = w["head.weight"]
        else { throw WeightStreamingError.notAModel(url.lastPathComponent) }
        let cfg = X070Config(nLayer: nLayer, nEmbd: lnOut.shape[0], headSize: kk.shape[1], vocab: head.shape[0])
        let bb = X070Backbone(weights: w, cfg: cfg, computeDType: computeDType)
        var byLayer = [[String]](repeating: [], count: nLayer)
        for k in w.keys where k.hasPrefix("blocks.") { byLayer[Int(k.split(separator: ".")[1])!].append(k) }
        bb.weightStream = WeightStream(url: url, dtype: computeDType, byLayer: byLayer.map { $0.sorted() },
                                       stem: ["emb.weight"], head: ["head.weight"], prefixed: false)
        return bb
    }

    /// Подкачка весов для уже собранной модели — например из `.rwkvq`, где плотного
    /// чекпоинта нет. Веса блоков, таблица эмбеддингов и голова (у квантованной базы —
    /// её буферы, сжатыми) выгружаются в `cache` (safetensors, байты как в памяти; файл
    /// перезаписывается), после чего модель держит на них ленивые ссылки и отпускает
    /// прочитанное так же, как `loadStreaming(safetensors:)`. Числа всех путей — те же до
    /// бита. Нормы основания и выхода (малые) остаются в памяти. Отказ: подкачка уже
    /// включена; база переложена в родной контейнер MLX (`useNativeKernel`).
    public func enableWeightStreaming(cache url: URL) throws {
        guard weightStream == nil else { throw WeightStreamingError.alreadyStreaming }
        guard rwkvqNative.isEmpty else { throw WeightStreamingError.nativeKernel }
        var arrays: [String: MLXArray] = [:]
        var byLayer = [[String]](repeating: [], count: cfg.nLayer), stem: [String] = [], head: [String] = []
        // `owner` — имя, по которому решается часть модели: ключ веса или тензора базы
        func put(_ file: String, _ owner: String, _ v: MLXArray) {
            if owner.hasPrefix("blocks."), let l = Int(owner.split(separator: ".")[1]), l < cfg.nLayer {
                byLayer[l].append(file)
            } else if owner.hasPrefix("emb.") { stem.append(file) }
            else if owner.hasPrefix("head.") { head.append(file) }
            else { return }
            arrays[file] = v
        }
        for (k, v) in w { put("w/" + k, k, v) }
        if let sc = rwkvqSidecar {
            for (k, v) in sc.spillBuffers {
                let owner = k.hasPrefix("fused::") ? k.components(separatedBy: "::")[1] : k
                put("sc/" + k, owner, v)
            }
        }
        if let t = rwkvqEmbTable { for (k, v) in t.spillBuffers { put("emb/" + k, "emb.weight", v) } }
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, url: url)
        weightStream = WeightStream(url: url, dtype: nil, byLayer: byLayer.map { $0.sorted() },
                                    stem: stem.sorted(), head: head.sorted(), prefixed: true)
        arrays = [:]
        releaseAllStreamedWeights()
        GPU.clearCache()
    }

    /// Включена ли подкачка весов.
    public var isWeightStreaming: Bool { weightStream != nil }

    /// Отпустить прочитанные веса всех частей модели (после прохода одним графом).
    public func releaseAllStreamedWeights() {
        guard let s = weightStream else { return }
        replace(s.fresh(s.byLayer.flatMap { $0 } + s.stem + s.head))
    }

    /// Веса блока `layer` — свежими ленивыми ссылками (прочитанные освобождаются).
    func releaseStreamed(layer: Int) {
        guard let s = weightStream else { return }
        replace(s.fresh(s.byLayer[layer]))
    }

    /// Таблица эмбеддингов — после основания.
    func releaseStreamedStem() {
        guard let s = weightStream else { return }
        replace(s.fresh(s.stem))
    }

    /// Голова — после лосса.
    func releaseStreamedHead() {
        guard let s = weightStream else { return }
        replace(s.fresh(s.head))
    }

    private func replace(_ fresh: [String: MLXArray]) {
        guard weightStream?.prefixed == true else {
            var nw = w
            for (k, v) in fresh { nw[k] = v }
            w = nw
            return
        }
        var nw: [String: MLXArray] = [:], sc: [String: MLXArray] = [:], emb: [String: MLXArray] = [:]
        for (k, v) in fresh {
            if k.hasPrefix("w/") { nw[String(k.dropFirst(2))] = v }
            else if k.hasPrefix("sc/") { sc[String(k.dropFirst(3))] = v }
            else if k.hasPrefix("emb/") { emb[String(k.dropFirst(4))] = v }
        }
        if !nw.isEmpty {
            var all = w
            for (k, v) in nw { all[k] = v; denseTargetCast[k] = nil }
            w = all
        }
        if !sc.isEmpty, let s = rwkvqSidecar { rwkvqSidecar = s.replacingSpilled(sc) }
        if !emb.isEmpty, let t = rwkvqEmbTable { rwkvqEmbTable = t.replacingSpilled(emb) }
    }
}

public enum WeightStreamingError: Error, CustomStringConvertible {
    case notAModel(String)
    case alreadyStreaming
    case nativeKernel
    public var description: String {
        switch self {
        case .alreadyStreaming: return "подкачка весов уже включена"
        case .nativeKernel:
            return "подкачка весов: база переложена в родной контейнер MLX (useNativeKernel) — не поддержано"
        case .notAModel(let f): return "подкачка весов: \(f) — не плотный чекпоинт RWKV-7 (нет blocks.*, ln_out, head)"
        }
    }
}
