import Foundation
import MLX
import RWKVGen
import RWKVRerank

// ───────────────────────────────────────────────────────────────────────
//  Swift-половина замера кодирования. Питоновская — Scripts/bench_encode.py.
//
//  Обе читают ОДИН файл входа (Scripts/bench_make_input.py) и делают ровно
//  одно и то же. Это единственный способ сравнивать реализации, а не
//  постановки: разные срезы данных дают разные длины документов, разное
//  число уникальных префиксов и разное число пачек, и любая разница во
//  времени объясняется тогда чем угодно.
//
//  Что засекается по отдельности и почему:
//
//    load        — веса и словарь. В разрыв не входит и вынесено, чтобы
//                  не входило.
//    warmup      — первые пачки ОТБРАСЫВАЮТСЯ. На Metal первый запуск
//                  каждой новой формы тянет за собой сборку конвейера;
//                  без прогрева этот разовый расход размазывается по
//                  замеру и притворяется пропускной способностью.
//    tokenize    — чистый CPU, GPU не трогается вовсе.
//    gpuPrefix   — batchIds + проход базы + eval. Синхронизация
//                  ОБЯЗАТЕЛЬНА: без неё засекается постановка в очередь,
//                  а не работа, и Swift «выигрывает» на пустом месте.
//    gpuTail     — то же для хвостов поверх состояния префикса.
//    select      — отбор слоёв и вынос в CPU-буфер.
//
//  Ряд по пачкам печатается целиком: разовый расход виден только по форме
//  кривой, суммой его от медленного кода не отличить.
// ───────────────────────────────────────────────────────────────────────

struct BenchInput: Decodable {
    struct Prefix: Decodable { var doc: String; var queries: [String] }
    var instruct: String
    var max_doc_tokens: Int
    var max_query_tokens: Int
    var terminator: Int?
    var doc_batch: Int
    var query_batch: Int
    var prefixes: [Prefix]
}

/// RSS процесса, ГБ — тем же способом, каким его видит система.
func benchRssGB() -> Double {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/bin/ps")
    p.arguments = ["-o", "rss=", "-p", String(ProcessInfo.processInfo.processIdentifier)]
    let pipe = Pipe(); p.standardOutput = pipe
    guard (try? p.run()) != nil else { return .nan }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    let s = String(data: data, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    return Double(s).map { $0 * 1024 / 1e9 } ?? .nan
}

/// Счётчики свопа системы. Замер, снятый во время свопа, — не «чуть хуже»,
/// а недействительный: страницы ходят на диск, и меряется он, а не код.
func swapCounters() -> (Int, Int) {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/vm_stat")
    let pipe = Pipe(); p.standardOutput = pipe
    guard (try? p.run()) != nil else { return (0, 0) }
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                     encoding: .utf8) ?? ""
    p.waitUntilExit()
    var ins = 0, outs = 0
    for line in out.split(separator: "\n") {
        let v = Int(line.split(separator: ":").last?
            .trimmingCharacters(in: CharacterSet(charactersIn: " .")) ?? "") ?? 0
        if line.contains("Swapins") { ins = v }
        if line.contains("Swapouts") { outs = v }
    }
    return (ins, outs)
}

func runBench(inputPath: String, modelPath: String, vocabPath: String,
              warmup: Int, layers: [Int], bucket: Int, minT: Int,
              cacheLimitGB: Double, outPath: String?) throws {

    // Потолок буферного кэша Metal. БЕЗ него замер недействителен: формы
    // батчей плавают, переиспользовать буферы нечего, кэш растёт линейно по
    // числу пачек и уводит машину в своп — замерено, 10+ ГБ на восьмистах
    // префиксах. Штатный `encodePairs` его ставит, и замер обязан ставить
    // тот же, иначе меряется не тот код, который работает в продакшене.
    if cacheLimitGB > 0 {
        MLX.GPU.set(cacheLimit: Int(cacheLimitGB * 1e9))
    }
    let swap0 = swapCounters()

    let raw = try Data(contentsOf: URL(fileURLWithPath: expand(inputPath)))
    let inp = try JSONDecoder().decode(BenchInput.self, from: raw)

    var t = Date()
    let weights = try loadArrays(url: URL(fileURLWithPath: expand(modelPath)))
    let nLayer = weights.keys.compactMap { k -> Int? in
        guard k.hasPrefix("blocks.") else { return nil }
        return Int(k.split(separator: ".")[1])
    }.max().map { $0 + 1 } ?? 0
    let cfg = X070Config(nLayer: nLayer,
                         nEmbd: weights["ln_out.weight"]!.shape[0],
                         headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                         vocab: weights["head.weight"]!.shape[0])
    let backbone = X070Backbone(weights: weights, cfg: cfg)
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: expand(vocabPath))) else {
        print("не удалось прочитать словарь"); exit(2)
    }
    let model = try Reranker(base: backbone,
                             cfg: RerankerConfig(layerIdx: layers))
    let loadS = Date().timeIntervalSince(t)

    let encCfg = RerankEncodeConfig(
        maxDocTokens: inp.max_doc_tokens, maxQueryTokens: inp.max_query_tokens,
        terminator: inp.terminator, docBatch: inp.doc_batch,
        queryBatch: inp.query_batch, lengthBucket: bucket,
        lengthBucketMinTokens: minT)

    var acc = ["tokenize": 0.0, "gpu_prefix": 0.0, "gpu_tail": 0.0,
               "select": 0.0]
    var series: [[String: Any]] = []
    var nPrefix = 0, nTail = 0
    var shapesPrefix = Set<[Int]>(), shapesTail = Set<[Int]>()

    let wall0 = Date()
    var bi = 0
    var start = 0
    while start < inp.prefixes.count {
        let end = Swift.min(start + inp.doc_batch, inp.prefixes.count)
        let chunk = Array(inp.prefixes[start ..< end])
        let warm = bi < warmup
        var b = ["tokenize": 0.0, "gpu_prefix": 0.0, "gpu_tail": 0.0,
                 "select": 0.0]

        t = Date()
        let seqs = chunk.map {
            RerankEncoder.prefixIds(tok, encCfg, instruct: inp.instruct,
                                    document: $0.doc)
        }
        b["tokenize"]! += Date().timeIntervalSince(t)

        t = Date()
        let (idx, mask, endIdx) = RerankEncoder.batchIds(seqs, bucket: bucket,
                                                        minT: minT)
        let st = model.encode(idx, mask: mask, endIdx: endIdx)
        st.evaluated()
        b["gpu_prefix"]! += Date().timeIntervalSince(t)
        shapesPrefix.insert(idx.shape)

        var jobs: [(local: Int, query: String)] = []
        for (local, p) in chunk.enumerated() {
            for q in p.queries { jobs.append((local, q)) }
        }

        var qs = 0
        while qs < jobs.count {
            let qe = Swift.min(qs + inp.query_batch, jobs.count)
            let part = Array(jobs[qs ..< qe])

            t = Date()
            let qseqs = part.map {
                RerankEncoder.suffixIds(tok, encCfg,
                                        document: chunk[$0.local].doc,
                                        query: $0.query)
            }
            b["tokenize"]! += Date().timeIntervalSince(t)

            t = Date()
            let sub = st.gather(part.map { $0.local })
            let (qidx, qmask, qend) = RerankEncoder.batchIds(qseqs, bucket: bucket,
                                                            minT: minT)
            let pairState = model.encode(qidx, mask: qmask, endIdx: qend,
                                         state: sub)
            let sel = model.select(pairState)
            eval(sel)
            b["gpu_tail"]! += Date().timeIntervalSince(t)
            shapesTail.insert(qidx.shape)

            t = Date()
            // Тот же вынос в CPU-буфер, что делает питоновская сторона:
            // fp32 → fp16 → байты. Без него замер кончался бы на GPU, а
            // кэш пишется всё-таки на хост.
            _ = sel.asType(.float32).asType(.float16).asData(disambiguate: true)
            b["select"]! += Date().timeIntervalSince(t)

            qs = qe
        }

        if !warm {
            for (k, v) in b { acc[k]! += v }
            nPrefix += chunk.count
            nTail += jobs.count
        }
        var row: [String: Any] = ["batch": bi, "warmup": warm,
                                  "prefixes": chunk.count, "tails": jobs.count,
                                  "rss_gb": (benchRssGB() * 100).rounded() / 100]
        for (k, v) in b { row[k] = (v * 1000 * 100).rounded() / 100 }
        series.append(row)
        bi += 1
        start = end
    }
    let wall = Date().timeIntervalSince(wall0)
    let swap1 = swapCounters()
    let swappedOut = swap1.1 - swap0.1

    let res: [String: Any] = [
        "swapouts": swappedOut, "swapins": swap1.0 - swap0.0,
        "cache_limit_gb": cacheLimitGB,
        "valid": swappedOut == 0,
        "side": "swift", "load_s": loadS, "wall_s": wall,
        "measured_s": acc.values.reduce(0, +),
        "prefixes": nPrefix, "tails": nTail,
        "warmup_batches": warmup, "length_bucket": bucket,
        "length_bucket_min_t": minT,
        "ms_per_prefix": acc["gpu_prefix"]! / Double(Swift.max(1, nPrefix)) * 1000,
        "ms_per_tail": acc["gpu_tail"]! / Double(Swift.max(1, nTail)) * 1000,
        "distinct_prefix_shapes": shapesPrefix.count,
        "distinct_tail_shapes": shapesTail.count,
        "rss_gb": benchRssGB(), "footprint_gb": processFootprintGB(),
        "phases_s": acc, "series": series,
    ]
    var head = res
    head.removeValue(forKey: "series")
    if let d = try? JSONSerialization.data(withJSONObject: head,
                                           options: [.prettyPrinted, .sortedKeys]),
       let s = String(data: d, encoding: .utf8) { print(s) }
    if swappedOut > 0 {
        print("ЗАМЕР НЕДЕЙСТВИТЕЛЕН: \(swappedOut) страниц ушло в своп. "
              + "Числа выше меряют диск, а не код. Уменьши вход или "
              + "потолок кэша.")
    }
    if let outPath {
        let url = URL(fileURLWithPath: expand(outPath))
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let d = try? JSONSerialization.data(withJSONObject: res,
                                               options: [.prettyPrinted, .sortedKeys]) {
            try? d.write(to: url)
        }
    }
}
