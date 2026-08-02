import Foundation
import MLX
import RWKVGen
import RWKVRerank

// ───────────────────────────────────────────────────────────────────────
//  Прогон реранкера от начала до конца: данные → кандидаты → кэш состояний
//  → обучение → отчёт.
//
//  Отдельная исполняемая цель, а не тест. Тест обязан быть быстрым и
//  воспроизводимым; прогон — это минуты кодирования на реальной модели и
//  числа, которые сравниваются с питоновскими замерами вручную.
//
//  Отчёт пишется в JSON ПОСЛЕ КАЖДОЙ стадии, а не в конце: кодирование
//  занимает минуты, и прогон, который убили посередине, обязан оставить
//  след. Ровно этого не хватало curriculum'у эмбеддингов.
//
//  Пример:
//    swift run -c release rerank-run \
//      --model ~/Develop/rwkv-metal/world_0.1b_x070.safetensors \
//      --vocab .testdata/rwkv_vocab_v20230424.txt \
//      --data  ~/Develop/reranker-triples-multi/train.jsonl \
//      --queries 400 --candidates 8 --eval-queries 80 \
//      --layers 5 --epochs 8 --lr 2e-4 \
//      --out runs/rerank_0.1b
// ───────────────────────────────────────────────────────────────────────

struct Args {
    var model = ""
    var vocab = ""
    var data = ""
    var queries = 400
    var candidates = 8
    var evalQueries = 80
    var layers = [5]
    var nProbe = 1
    var epochs = 8
    var lr: Float = 2e-4
    var batchSize = 32
    var maxDocTokens = 384
    var maxQueryTokens = 96
    var docBatch = 8
    var queryBatch = 16
    var seed: UInt64 = 0
    /// Сиды для развёртки. Один — обычный прогон; несколько — измерение
    /// РАЗБРОСА, без которого разницу между конфигурациями судить нельзя.
    var seeds: [UInt64] = []
    /// Конфигурации головы через точку с запятой: "5;11;0,5". Кэш при этом
    /// собирается на объединение всех слоёв — одно кодирование на всю
    /// развёртку вместо одного на конфигурацию.
    var sweepLayers: [[Int]] = []
    /// Слои, попадающие в кэш. Пусто ⇒ объединение того, что нужно
    /// конфигурациям. Задаётся явно, когда кэш строится ВПРОК.
    var cacheLayers: [Int] = []
    var language: String? = nil
    var out = "runs/rerank"
    var cachePath: String? = nil
    /// Проверка ВЫДАЧИ: загрузить готовую голову и отранжировать ею
    /// отложенные запросы, минуя кэш и обучение вовсе.
    var serveCheck: String? = nil
    /// Оценка готовой головы на ГОТОВОМ кэше, без обучения.
    var evalHead: String? = nil
    /// Замер кодирования по фазам на общем с Python файле входа.
    var bench: String? = nil
    var benchWarmup = 3
    var benchBucket = 0
    var benchOut: String? = nil
    var benchCacheLimit = 2.0
    var benchMinT = 0
}

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let key = it.next() {
        func val() -> String { it.next() ?? "" }
        switch key {
        case "--model": a.model = val()
        case "--vocab": a.vocab = val()
        case "--data": a.data = val()
        case "--queries": a.queries = Int(val()) ?? a.queries
        case "--candidates": a.candidates = Int(val()) ?? a.candidates
        case "--eval-queries": a.evalQueries = Int(val()) ?? a.evalQueries
        case "--layers": a.layers = val().split(separator: ",").compactMap { Int($0) }
        case "--probes": a.nProbe = Int(val()) ?? a.nProbe
        case "--epochs": a.epochs = Int(val()) ?? a.epochs
        case "--lr": a.lr = Float(val()) ?? a.lr
        case "--batch-size": a.batchSize = Int(val()) ?? a.batchSize
        case "--max-doc-tokens": a.maxDocTokens = Int(val()) ?? a.maxDocTokens
        case "--max-query-tokens": a.maxQueryTokens = Int(val()) ?? a.maxQueryTokens
        case "--doc-batch": a.docBatch = Int(val()) ?? a.docBatch
        case "--query-batch": a.queryBatch = Int(val()) ?? a.queryBatch
        case "--seed": a.seed = UInt64(val()) ?? a.seed
        case "--seeds": a.seeds = val().split(separator: ",").compactMap { UInt64($0) }
        case "--sweep-layers":
            a.sweepLayers = val().split(separator: ";").map {
                $0.split(separator: ",").compactMap { Int($0) }
            }.filter { !$0.isEmpty }
        case "--cache-layers":
            a.cacheLayers = val().split(separator: ",").compactMap { Int($0) }
        case "--language": a.language = val()
        case "--out": a.out = val()
        case "--cache": a.cachePath = val()
        case "--serve-check": a.serveCheck = val()
        case "--eval-head": a.evalHead = val()
        case "--bench": a.bench = val()
        case "--bench-warmup": a.benchWarmup = Int(val()) ?? a.benchWarmup
        case "--bench-bucket": a.benchBucket = Int(val()) ?? a.benchBucket
        case "--bench-out": a.benchOut = val()
        case "--bench-cache-limit": a.benchCacheLimit = Double(val()) ?? a.benchCacheLimit
        case "--bench-min-t": a.benchMinT = Int(val()) ?? a.benchMinT
        default: FileHandle.standardError.write("неизвестный ключ \(key)\n".data(using: .utf8)!)
        }
    }
    return a
}

func expand(_ p: String) -> String { (p as NSString).expandingTildeInPath }

// ── Отчёт ───────────────────────────────────────────────────────────────

final class Report {
    private var root: [String: Any] = [:]
    private let url: URL

    /// Имя файла содержит РЕЖИМ и время запуска.
    ///
    /// Раньше отчёт назывался `report.json` и второй запуск в ту же папку
    /// затирал первый. Ломалось это на самом полезном сценарии: собрать кэш
    /// развёрткой, потом переиспользовать его одиночным прогоном — и
    /// потерять числа развёртки, ради которых всё и делалось.
    ///
    /// Рядом кладётся символическая ссылка `report-latest.json`: «последний»
    /// нужен часто, а искать его по времени в имени неудобно.
    init(dir: String, mode: String) throws {
        let base = URL(fileURLWithPath: expand(dir))
        try FileManager.default.createDirectory(at: base,
                                                withIntermediateDirectories: true)
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        var candidate = base.appendingPathComponent(
            "report-\(mode)-\(fmt.string(from: Date())).json")
        // Два запуска в одну секунду — редкость, но затирать по-прежнему
        // нельзя: ровно от этого весь этот код.
        var n = 1
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = base.appendingPathComponent(
                "report-\(mode)-\(fmt.string(from: Date()))-\(n).json")
            n += 1
        }
        url = candidate

        // Ссылка ОТНОСИТЕЛЬНАЯ — по имени файла, без пути. Вариант с
        // `withDestinationURL:` разрешает имя относительно текущего каталога
        // процесса, а не каталога ссылки, и получается ссылка в никуда:
        // выглядит правильно, ведёт не туда. Замечено на первом же прогоне.
        let link = base.appendingPathComponent("report-latest.json")
        try? FileManager.default.removeItem(at: link)
        try? FileManager.default.createSymbolicLink(
            atPath: link.path, withDestinationPath: url.lastPathComponent)
    }

    func set(_ key: String, _ value: Any) {
        root[key] = value
        flush()
    }

    private func flush() {
        guard let data = try? JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys]) else { return }
        try? data.write(to: url)
    }

    var path: String { url.path }
}

func metricsDict(_ m: RankingMetrics) -> [String: Any] {
    ["mrr": m.mrr, "recall@1": m.recallAt1, "recall@3": m.recallAt3,
     "recall@5": m.recallAt5, "ndcg@10": m.ndcgAt10,
     "pairwise_vs_hard_negative": m.pairwiseVsHardNegative,
     "n_hard": m.nHardPairs,
     "pairwise_vs_sampled_negative": m.pairwiseVsSampledNegative,
     "n_sampled": m.nSampledPairs,
     "n": m.n, "n_candidates": m.nCandidates,
     "random_floor": RankingMetrics.randomFloor(nCandidates: m.nCandidates)]
}

func hhmmss(_ s: Double) -> String {
    String(format: "%d:%02d", Int(s) / 60, Int(s) % 60)
}

/// Physical footprint процесса, ГБ.
///
/// Не `GPU.peakMemory`: счётчики MLX не знают ни про mmap-страницы кэша, ни
/// про веса, ни про то, ушла ли машина в своп.
///
/// Именно footprint, а не `resident_size`: буферы Metal живут в
/// IOAccelerator и в RSS попадают неполностью. На прогоне, где footprint
/// дошёл до 13 ГБ, `ps -o rss` показывал 0.2 ГБ — то есть смотреть надо
/// сюда, иначе рост буферного кэша просто не виден.
func processFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                       / MemoryLayout<natural_t>.size)
    let ok = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return ok == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : 0
}

// ── Прогон ──────────────────────────────────────────────────────────────

let args = parseArgs()
guard !args.model.isEmpty, !args.vocab.isEmpty,
      !args.data.isEmpty || args.bench != nil else {
    print("нужны --model, --vocab и --data (см. комментарий в начале файла)")
    exit(2)
}

// Замер идёт ДО загрузки данных и построения отчёта: ему нужен только файл
// входа, общий с питоновской стороной. Всё, что делается здесь лишнего,
// попало бы в замер или в память.
if let benchInput = args.bench {
    try runBench(inputPath: benchInput, modelPath: args.model,
                 vocabPath: args.vocab, warmup: args.benchWarmup,
                 layers: args.layers, bucket: args.benchBucket,
                 minT: args.benchMinT, cacheLimitGB: args.benchCacheLimit,
                 outPath: args.benchOut)
    exit(0)
}

let t0 = Date()
// Режим определяется по ключам, а не по ходу выполнения: имя файла отчёта
// нужно ДО первой стадии, потому что отчёт пишется после каждой.
let runMode = args.evalHead != nil ? "eval"
    : args.serveCheck != nil ? "serve"
    : (!args.sweepLayers.isEmpty || args.seeds.count > 1 ? "sweep" : "train")
let report = try Report(dir: args.out, mode: runMode)
report.set("args", [
    "model": args.model, "data": args.data, "queries": args.queries,
    "candidates": args.candidates, "eval_queries": args.evalQueries,
    "layers": args.layers, "n_probe": args.nProbe, "epochs": args.epochs,
    "lr": Double(args.lr), "batch_size": args.batchSize,
    "max_doc_tokens": args.maxDocTokens, "seed": Int(args.seed),
    "language": args.language ?? "любой",
])

print("── данные ──")
let rows = try RerankDataset.loadJSONL(path: args.data, limit: args.queries,
                                       seed: args.seed, task: "retrieval",
                                       language: args.language)
let langs = Set(rows.compactMap { $0.language })
print("строк \(rows.count), языков \(langs.isEmpty ? 1 : langs.count), "
      + "майненных негативов на строку: медиана "
      + "\(rows.map { $0.negatives.count }.sorted()[rows.count / 2])")

let (pool, allSamples) = try RerankCandidates.build(rows,
                                                    nCandidates: args.candidates,
                                                    seed: args.seed)
let (trainSamples, evalSamples) = RerankCandidates.splitTrainEval(
    allSamples, nEval: Swift.min(args.evalQueries, allSamples.count / 3),
    seed: args.seed)
print("пул документов \(pool.count), обучение \(trainSamples.count), "
      + "оценка \(evalSamples.count)")
report.set("data", ["rows": rows.count, "pool": pool.count,
                    "train": trainSamples.count, "eval": evalSamples.count,
                    "languages": langs.count])

print("\n── модель ──")
let weights = try loadArrays(url: URL(fileURLWithPath: expand(args.model)))
let nLayer = weights.keys.compactMap { k -> Int? in
    guard k.hasPrefix("blocks.") else { return nil }
    return Int(k.split(separator: ".")[1])
}.max().map { $0 + 1 } ?? 0
let cfg = X070Config(nLayer: nLayer,
                     nEmbd: weights["ln_out.weight"]!.shape[0],
                     headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                     vocab: weights["head.weight"]!.shape[0])
let backbone = X070Backbone(weights: weights, cfg: cfg)
guard let tokenizer = WorldTokenizer(
        vocabURL: URL(fileURLWithPath: expand(args.vocab))) else {
    print("не удалось прочитать словарь \(args.vocab)")
    exit(2)
}
// Конфигурации головы: одна обычным прогоном, несколько — развёрткой.
let sweepConfigs: [[Int]] = args.sweepLayers.isEmpty ? [args.layers]
                                                     : args.sweepLayers
let headConfigs = sweepConfigs.map {
    RerankerConfig(layerIdx: $0, nProbe: args.nProbe)
}
let seedList: [UInt64] = args.seeds.isEmpty ? [args.seed] : args.seeds

// Слои, которые лягут в кэш. Кодирование — минуты и не зависит от числа
// конфигураций; обучение — секунды. Поэтому кэш собирается на ОБЪЕДИНЕНИЕ
// всего, что понадобится, и дальше каждая конфигурация берёт свой срез.
let neededLayers = try Set(sweepConfigs.flatMap {
    try resolveLayerIndices($0, nLayer: cfg.nLayer)
}).sorted()
let cacheSources = args.cacheLayers.isEmpty
    ? neededLayers
    : try RerankEncoder.resolveSources(args.cacheLayers,
                                       head: RerankerHead(base: backbone),
                                       nLayer: cfg.nLayer)
guard Set(cacheSources).isSuperset(of: neededLayers) else {
    print("--cache-layers \(cacheSources) не покрывает нужные конфигурациям "
          + "слои \(neededLayers)")
    exit(2)
}

// Модель для КОДИРОВАНИЯ: состояние базы от конфигурации головы не зависит
// вовсе, голова здесь нужна только чтобы кодировщику было что спросить.
let model = try Reranker(base: backbone, cfg: headConfigs[0], seed: args.seed)
print("L=\(cfg.nLayer) D=\(cfg.nEmbd) V=\(cfg.vocab); обучаемых "
      + String(format: "%.2fM", Double(model.head.parameterCount) / 1e6))
print("конфигураций \(headConfigs.count): "
      + sweepConfigs.map { $0.map(String.init).joined(separator: ",") }
                    .joined(separator: " | ")
      + "; сидов \(seedList.count); в кэш идут слои \(cacheSources)")

// ── Оценка без обучения ─────────────────────────────────────────────────
//
// Нужно там, где обучение мешает: сравнить две сохранённые головы,
// переоценить старую на новом отложенном наборе, проверить чекпоинт после
// переноса. Прогонять ради этого обучение значило бы менять то, что
// собирались измерить.
//
// Кэш берётся ГОТОВЫЙ. Если его нет — это ошибка, а не повод молча собрать
// новый: собранный сейчас кэш отвечал бы на другой вопрос.
if let headPath = args.evalHead {
    print("\n── оценка без обучения ──")
    let headURL = URL(fileURLWithPath: expand(headPath))
    let cacheBase = args.cachePath.map { URL(fileURLWithPath: expand($0) + "_eval") }
        ?? URL(fileURLWithPath: expand(args.out)).appendingPathComponent("cache_eval")
    guard let cache = try? StateCache.load(cacheBase) else {
        print("нет готового кэша \(cacheBase.path) — сначала прогон с обучением")
        exit(2)
    }
    let (m, contract) = try RerankTraining.evaluate(base: backbone,
                                                    head: headURL, cache: cache)
    print("контракт чекпоинта: "
          + contract.sorted { $0.key < $1.key }
                    .map { "\($0.key)=\($0.value)" }.joined(separator: ", "))
    print("кэш: \(cache.nPairs) пар, слои \(cache.sources.map(String.init(describing:)) ?? "неизвестны")")
    print(m.summary)
    report.set("eval_only", ["head": headURL.path, "cache": cacheBase.path,
                             "contract": contract, "metrics": metricsDict(m)])
    report.set("total_seconds", Date().timeIntervalSince(t0))
    print("\nотчёт:  \(report.path)")
    exit(0)
}

// ── Проверка выдачи ─────────────────────────────────────────────────────
//
// Замыкает петлю «обучили → применили». Голова обучалась на СОСТОЯНИЯХ из
// кэша, а применяется к ТЕКСТУ, и между этими двумя мирами лежит вся
// подача: шаблон, обрезки, терминатор, инструкция. Совпадение метрик здесь
// с метриками на кэше — единственное доказательство, что подача не
// разъехалась; ошибок формы при расхождении не возникает нигде.
//
// Кэш и обучение не трогаются вовсе: это путь пользователя, а не прогона.
if let headPath = args.serveCheck {
    print("\n── проверка выдачи ──")
    let headURL = URL(fileURLWithPath: expand(headPath))
    let md = try Reranker.readHeadMetadata(headURL)
    print("контракт из чекпоинта: "
          + md.filter { ["template", "instruct", "max_doc_tokens",
                         "max_query_tokens", "terminator"].contains($0.key) }
               .sorted { $0.key < $1.key }
               .map { "\($0.key)=\($0.value)" }.joined(separator: ", "))

    let inf = try RerankerInference.fromCheckpoint(
        base: backbone, tokenizer: tokenizer, head: headURL)
    print("голова читает слои \(inf.model.head.uniqueSources)")

    // Ранжируем ровно те же отложенные запросы и теми же кандидатами, что
    // видела оценка на кэше. Иначе сравнивать было бы нечего.
    let t = Date()
    var scoresDirect: [[Float]] = []
    var scoresIndexed: [[Float]] = []
    for (i, smp) in evalSamples.enumerated() {
        let cands = smp.docIds.map { pool[$0] }
        scoresDirect.append(try inf.score(query: smp.query, docs: cands,
                                          instruct: smp.instruct))
        let index = try inf.buildIndex(docs: cands, instruct: smp.instruct)
        scoresIndexed.append(try inf.scoreIndexed(query: smp.query, index: index,
                                                  instruct: smp.instruct))
        if (i + 1) % 20 == 0 {
            print("  \(i + 1)/\(evalSamples.count), \(hhmmss(Date().timeIntervalSince(t)))")
            fflush(stdout)
        }
    }
    let serveSeconds = Date().timeIntervalSince(t)

    let labels = evalSamples.map { $0.label }
    let hard = evalSamples.map { $0.hardNegs }
    let mDirect = RerankMetrics.compute(scores: scoresDirect, labels: labels,
                                        hardNegs: hard)
    let mIndexed = RerankMetrics.compute(scores: scoresIndexed, labels: labels,
                                         hardNegs: hard)

    // Расхождение путей на РЕАЛЬНОЙ модели. Тестовое число (2.4e-7) снято на
    // крошечном бэкбоне и на 0.1B не переносится: там двенадцать слоёв и
    // bf16-веса.
    var maxAbs: Float = 0, maxRel: Float = 0
    for (a, b) in zip(scoresDirect, scoresIndexed) {
        for (x, y) in zip(a, b) {
            let d = Swift.abs(x - y)
            maxAbs = Swift.max(maxAbs, d)
            maxRel = Swift.max(maxRel, d / (Swift.abs(x) + 1e-6))
        }
    }
    let sameOrder = zip(scoresDirect, scoresIndexed).allSatisfy { a, b in
        RerankerInference.order(a, ids: Array(0 ..< a.count), topK: nil)
            .map { $0.index }
        == RerankerInference.order(b, ids: Array(0 ..< b.count), topK: nil)
            .map { $0.index }
    }

    print("\n── итог выдачи ──")
    print("прямой путь:")
    print(mDirect.summary)
    print("\nчерез индекс префиксов:")
    print(mIndexed.summary)
    print(String(format: "\nпути расходятся на %.2e абсолютных / %.2e "
                 + "относительных; порядок кандидатов %@",
                 maxAbs, maxRel,
                 sameOrder ? "совпал везде" : "РАЗОШЁЛСЯ"))
    print(String(format: "время: %@ на %d запросов", hhmmss(serveSeconds),
                 evalSamples.count))

    report.set("serving", [
        "head": headURL.path, "seconds": serveSeconds,
        "contract": md,
        "direct": metricsDict(mDirect), "indexed": metricsDict(mIndexed),
        "max_abs_diff": Double(maxAbs), "max_rel_diff": Double(maxRel),
        "same_order": sameOrder,
    ])
    report.set("total_seconds", Date().timeIntervalSince(t0))
    print("\nотчёт:  \(report.path)")
    exit(0)
}

let encCfg = RerankEncodeConfig(maxDocTokens: args.maxDocTokens,
                                maxQueryTokens: args.maxQueryTokens,
                                docBatch: args.docBatch,
                                queryBatch: args.queryBatch)

func buildCache(_ samples: [RerankSample], tag: String) throws -> StateCache {
    let base = args.cachePath.map {
        URL(fileURLWithPath: expand($0) + "_" + tag)
    } ?? URL(fileURLWithPath: expand(args.out)).appendingPathComponent("cache_" + tag)

    // Переиспользование: мало совпадения форм — кэш обязан ПОКРЫВАТЬ все
    // слои развёртки и быть собран тем же контрактом подачи текста. Кэш без
    // состава слоёв (собранный до появления поля) сюда не годится: про его
    // слои ничего не известно, а «столько же слотов» — не то же самое, что
    // «те же слои».
    if let existing = try? StateCache.load(base),
       existing.nSamples == samples.count,
       existing.nCandidates == args.candidates,
       let have = existing.sources, Set(have).isSuperset(of: neededLayers),
       (try? existing.checkCompatible(head: model.head, contract: encCfg.contract)) != nil {
        print("  \(tag): кэш найден готовым (\(existing.nPairs) пар, "
              + String(format: "%.2f", Double(existing.byteCount) / 1e9) + " ГБ)")
        return existing
    }

    let t = Date()
    var lastPrint = Date()
    let cache = try RerankEncoder.encodePairs(
        model, tokenizer: tokenizer, pool: pool, samples: samples,
        config: encCfg, path: base, sources: cacheSources,
        progress: { done, total, pairs, pairsTotal in
            guard Date().timeIntervalSince(lastPrint) > 10 else { return }
            lastPrint = Date()
            let el = Date().timeIntervalSince(t)
            let frac = Double(done) / Double(total)
            // Память печатается КАЖДЫЙ раз, а не в конце: рост буферного
            // кэша Metal видно только по ходу, а к концу прогона машина уже
            // в свопе и все замеры времени — фикция.
            print(String(format: "  \(tag): префиксы %d/%d, пары %d/%d, %@, "
                         + "осталось ~%@, память %.1f ГБ",
                         done, total, pairs, pairsTotal,
                         hhmmss(el), hhmmss(el / Swift.max(1e-9, frac) * (1 - frac)),
                         processFootprintGB()))
            fflush(stdout)
        })
    print(String(format: "  \(tag): %d пар × %d слоёв за %@, кэш %.2f ГБ, "
                 + "память %.1f ГБ",
                 cache.nPairs, cache.nSources, hhmmss(Date().timeIntervalSince(t)),
                 Double(cache.byteCount) / 1e9, processFootprintGB()))
    return cache
}

print("\n── кэш состояний ──")
let encStart = Date()
let trainCache = try buildCache(trainSamples, tag: "train")
let evalCache = try buildCache(evalSamples, tag: "eval")
let encSeconds = Date().timeIntervalSince(encStart)
report.set("encoding", [
    "seconds": encSeconds,
    "train_pairs": trainCache.nPairs, "eval_pairs": evalCache.nPairs,
    "train_bytes": trainCache.byteCount, "eval_bytes": evalCache.byteCount,
    "layers": cacheSources,
    "footprint_gb": processFootprintGB(),
])

// ── Развёртка ───────────────────────────────────────────────────────────
//
// Отдельная ветка, а не флаг внутри общей: у развёртки другой итог. Одиночный
// прогон даёт таблицу «до/после» и обученную голову; развёртка даёт РАЗБРОС,
// и головы не сохраняет вовсе — их тут N×M, и молча положить одну из них
// значило бы выдать случайную за результат.
if headConfigs.count > 1 || seedList.count > 1 {
    print("\n── развёртка ──")
    let scfg = RerankTrainConfig(lr: args.lr, batchSize: args.batchSize,
                                 epochs: args.epochs, logEvery: 0)
    var pointDicts: [[String: Any]] = []

    let sweep = try RerankSweep.run(
        base: backbone, configs: headConfigs,
        trainCache: trainCache, evalCache: evalCache, config: scfg,
        seeds: seedList, contract: encCfg.contract,
        onPoint: { p in
            pointDicts.append([
                "layers": p.layers, "n_probe": p.nProbe,
                "seeds": p.seeds.map { Int($0) },
                "seconds": p.seconds,
                "first_losses": p.firstLosses.map { Double($0) },
                "expected_first_loss": Double(p.expectedFirstLoss),
                "after": p.after.map(metricsDict),
                "mrr_mean": p.mrr.mean, "mrr_std": p.mrr.std,
                "recall@1_mean": p.recallAt1.mean,
                "recall@1_std": p.recallAt1.std,
                "ndcg@10_mean": p.ndcgAt10.mean,
                "ndcg@10_std": p.ndcgAt10.std,
                "hard_mean": p.pairwiseVsHardNegative.mean,
                "hard_std": p.pairwiseVsHardNegative.std,
            ])
            report.set("sweep", pointDicts)
            print("  \(p.label): MRR \(p.mrr.description)")
            fflush(stdout)
        },
        onRun: { ci, i, seed, r in
            // Стартовый лосс печатается для КАЖДОГО прогона: он обязан быть
            // ln(C) у свежей головы, и первое же расхождение означает, что
            // развёртка переиспользует уже обученную.
            print(String(format: "    конфигурация %d, сид %d: лосс старта "
                         + "%.6f (ln C = %.6f)%@ → MRR %.4f",
                         ci, Int(seed), r.firstLoss, r.expectedFirstLoss,
                         abs(r.firstLoss - r.expectedFirstLoss) < 1e-4
                            ? "" : "  ← РАСХОДИТСЯ",
                         r.after?.mrr ?? .nan))
            fflush(stdout)
        })

    print("\n── итог ──")
    print(sweep.summary)
    if seedList.count < 2 {
        print("разброс не измерен: сид один. Развёртка по конфигурациям при "
              + "одном сиде показывает разницу, но не даёт её с чем сравнить.")
    }
    report.set("sweep_seconds", sweep.seconds)
    report.set("total_seconds", Date().timeIntervalSince(t0))
    print("\nголова НЕ сохранена: в развёртке их "
          + "\(headConfigs.count * seedList.count). Перезапусти с одной "
          + "конфигурацией и одним сидом — кэш уже собран и переиспользуется.")
    print("отчёт:  \(report.path)")
    print("всего:  \(hhmmss(Date().timeIntervalSince(t0)))")
    exit(0)
}

print("\n── обучение ──")
let tcfg = RerankTrainConfig(lr: args.lr, batchSize: args.batchSize,
                             epochs: args.epochs, seed: args.seed, logEvery: 20)
let trainStart = Date()
var epochDicts: [[String: Any]] = []

let result = try RerankTraining.train(
    model, trainCache: trainCache, evalCache: evalCache, config: tcfg,
    // Ожидание задаётся ЯВНО, хотя умолчание взяло бы то же самое из кэша.
    // Разница в том, кто кому подчиняется: прогонщик знает, каким текстом
    // он хотел кормить модель, и при переиспользовании чужого кэша обязан
    // об этом узнать, а не молча принять чужие обрезки.
    contract: encCfg.contract,
    onEpoch: { r in
        var d: [String: Any] = ["epoch": r.epoch, "loss": Double(r.loss)]
        if let m = r.metrics {
            d["metrics"] = metricsDict(m)
            print(String(format: "  эпоха %d: лосс %.4f | MRR %.4f | R@1 %.4f | "
                         + "против майненного %.4f",
                         r.epoch, r.loss, m.mrr, m.recallAt1,
                         m.pairwiseVsHardNegative))
        }
        epochDicts.append(d)
        report.set("epochs", epochDicts)
        fflush(stdout)
    },
    onStep: { s in
        print(String(format: "    шаг %d: лосс %.4f | норма %.3f | lr %.2e",
                     s.step, s.loss, s.gradNorm, s.learningRate))
        fflush(stdout)
    })

let trainSeconds = Date().timeIntervalSince(trainStart)
print(String(format: "\nстартовый лосс %.6f, ожидался ln(C) = %.6f — %@",
             result.firstLoss, result.expectedFirstLoss,
             abs(result.firstLoss - result.expectedFirstLoss) < 1e-4
                ? "сходится" : "РАСХОДИТСЯ, проводка сломана"))

report.set("training", [
    "seconds": trainSeconds, "steps": result.steps,
    "best_epoch": result.bestEpoch,
    "first_loss": Double(result.firstLoss),
    "expected_first_loss": Double(result.expectedFirstLoss),
])
if let b = result.before { report.set("before", metricsDict(b)) }
if let a = result.after { report.set("after", metricsDict(a)) }

// ── Итог ────────────────────────────────────────────────────────────────

print("\n── итог ──")
if let b = result.before, let a = result.after {
    func pad(_ s: String, _ n: Int) -> String {
        s + String(repeating: " ", count: Swift.max(0, n - s.count))
    }
    func cell(_ v: Double, _ n: Int) -> String {
        let s = n > 0 ? String(format: "%.4f", v) : "  —   "
        return String(repeating: " ", count: Swift.max(0, 8 - s.count)) + s
    }
    print(pad("", 30) + "      до    после")
    func line(_ name: String, _ x: Double, _ y: Double, n: Int = 1) {
        print(pad(name, 30) + cell(x, n) + " " + cell(y, n))
    }
    line("MRR", b.mrr, a.mrr)
    line("recall@1", b.recallAt1, a.recallAt1)
    line("nDCG@10", b.ndcgAt10, a.ndcgAt10)
    // Прочерк, а не ноль, когда пар этого рода не было вовсе. Ноль здесь
    // читался бы как «модель проиграла все сравнения», хотя сравнений не
    // было ни одного: при nCandidates ≤ 1 + числа майненных негативов
    // добирать из пула просто нечего.
    line("против майненного негатива", b.pairwiseVsHardNegative,
         a.pairwiseVsHardNegative, n: a.nHardPairs)
    line("против добранного из пула", b.pairwiseVsSampledNegative,
         a.pairwiseVsSampledNegative, n: a.nSampledPairs)
    print(String(format: "\nпол случайного угадывания при %d кандидатах: %.4f",
                 a.nCandidates,
                 RankingMetrics.randomFloor(nCandidates: a.nCandidates)))
    print("пар: майненных \(a.nHardPairs), добранных \(a.nSampledPairs)")
}

let headURL = URL(fileURLWithPath: expand(args.out))
    .appendingPathComponent("reranker_head.safetensors")
// Контракт берётся из РЕЗУЛЬТАТА обучения, а не из конфигурации прогонщика.
// Совпадают они ровно тогда, когда кэш собран этим же запуском; при
// переиспользовании готового кэша — не обязаны. Голова и условия её
// обучения обязаны ехать в одном файле, иначе выдача унаследует не то.
try model.saveHead(to: headURL, extra: result.contract.merging(
    ["instruct": defaultRerankInstruct]) { a, _ in a })

report.set("total_seconds", Date().timeIntervalSince(t0))
print("\nголова: \(headURL.path)")
print("отчёт:  \(report.path)")
print("всего:  \(hhmmss(Date().timeIntervalSince(t0)))")
