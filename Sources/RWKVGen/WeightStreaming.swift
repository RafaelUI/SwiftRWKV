import Foundation
import MLX

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
// ───────────────────────────────────────────────────────────────────────

/// Откуда брать веса заново: файл, тип вычислений, имена по частям модели.
struct WeightStream {
    let url: URL
    let dtype: DType
    let byLayer: [[String]]
    static let stemKeys = ["emb.weight"]
    static let headKeys = ["head.weight"]

    /// Свежие ленивые ссылки на веса `keys` (данные не читаются до вычисления).
    func fresh(_ keys: [String]) -> [String: MLXArray] {
        let all: [String: MLXArray]
        do { all = try loadArrays(url: url) } catch {
            preconditionFailure("подкачка весов: файл \(url.path) не читается: \(error)")
        }
        var out: [String: MLXArray] = [:]
        for k in keys {
            guard let v = all[k] else { preconditionFailure("подкачка весов: в \(url.lastPathComponent) нет \(k)") }
            out[k] = v.asType(dtype)
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
        bb.weightStream = WeightStream(url: url, dtype: computeDType, byLayer: byLayer.map { $0.sorted() })
        return bb
    }

    /// Включена ли подкачка весов.
    public var isWeightStreaming: Bool { weightStream != nil }

    /// Отпустить прочитанные веса всех частей модели (после прохода одним графом).
    public func releaseAllStreamedWeights() {
        guard let s = weightStream else { return }
        replace(s.fresh(s.byLayer.flatMap { $0 } + WeightStream.stemKeys + WeightStream.headKeys))
    }

    /// Веса блока `layer` — свежими ленивыми ссылками (прочитанные освобождаются).
    func releaseStreamed(layer: Int) {
        guard let s = weightStream else { return }
        replace(s.fresh(s.byLayer[layer]))
    }

    /// Таблица эмбеддингов — после основания.
    func releaseStreamedStem() {
        guard let s = weightStream else { return }
        replace(s.fresh(WeightStream.stemKeys))
    }

    /// Голова — после лосса.
    func releaseStreamedHead() {
        guard let s = weightStream else { return }
        replace(s.fresh(WeightStream.headKeys))
    }

    private func replace(_ fresh: [String: MLXArray]) {
        var nw = w
        for (k, v) in fresh { nw[k] = v }
        w = nw
    }
}

public enum WeightStreamingError: Error, CustomStringConvertible {
    case notAModel(String)
    public var description: String {
        switch self {
        case .notAModel(let f): return "подкачка весов: \(f) — не плотный чекпоинт RWKV-7 (нет blocks.*, ln_out, head)"
        }
    }
}
