import Foundation
import MLX
import RWKVGen
import RWKVEmbedding

// ───────────────────────────────────────────────────────────────────────
//  embed-run: эмбеддер на плотной базе либо на базе из .rwkvq — оценка и
//  дообучение одной стадии на LitRetrieval, одним запуском.
//
//  Зачем. Эмбеддинг-путь был проверен только на плотной bf16-модели; этот
//  прогонщик нужен, чтобы тем же кодом и на тех же строках сравнить базы:
//  плотную, REDUCTION и COMPRESSION — качество до и после дообучения, время
//  и память.
//
//      swift run -c release embed-run --model m.rwkvq --vocab v.txt \
//          --data train.jsonl --mode frozen --train 2000 --eval 300 --steps 300
//
//  --mode  frozen     — обучается только голова (база заморожена)
//          lora       — адаптеры LoRA на проекциях tmix и голова; на базе из
//                       .rwkvq это QLoRA: база остаётся сжатой
//          top:N      — верхние N слоёв и голова; на базе из .rwkvq эти N
//                       слоёв разворачиваются в плотные (unpackTopLayers),
//                       низ остаётся сжатым и замороженным
//          full       — вся база и голова (только плотная база)
//          none       — только оценка
//
//  Для замеров: --limit N (не читать JSONL целиком), --log-every 1 (время
//  каждого шага), --guard ГБ (выход, если footprint процесса выше порога).
// ───────────────────────────────────────────────────────────────────────

struct Args {
    var model = "", vocab = "", data = "", mode = "frozen", task = "retrieval"
    var train = 2000, eval = 300, steps = 300, batch = 8, maxTokens = 256, maxChars = 800
    var lr: Float = 2e-5
    var loraRank = 8
    var gradCache = 0
    var sequential = true
    var cacheGB = 0.25
    var seed: UInt64 = 0
    var stateLR: Float = 1.0
    var stateParts = "wkv"
    var stateOut: String? = nil
    var headOut: String? = nil
    var stateIn: String? = nil
    var headIn: String? = nil
    var allowOtherBase = false
    var forwardAB = 0
    var out: String? = nil
    var limit: Int? = nil          // читать не больше стольких строк JSONL (иначе файл целиком в память)
    var logEvery = 50              // печать шага; 1 — время каждого шага (замеры)
    var guardGB = 0.0              // > 0: footprint выше порога — выход (замер под свопом негоден)
    var probe = false              // память по фазам шага (SequentialBody.probe)
    var chunkTokens: Int? = nil    // EmbeddingStage.bodyChunkTokens (не задано — умолчание стадии)
    var traceStride = 1            // EmbeddingStage.traceStride
    var pipelineAB = false         // A/B в одном процессе: чётные шаги — SequentialBody.pipelined
    var unpack: Int? = nil         // развернуть столько верхних слоёв .rwkvq (по умолчанию — N из top:N)
}

func parse() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        func v() -> String { it.next() ?? "" }
        switch k {
        case "--model": a.model = v()
        case "--vocab": a.vocab = v()
        case "--data": a.data = v()
        case "--mode": a.mode = v()
        case "--task": a.task = v()
        case "--train": a.train = Int(v()) ?? a.train
        case "--eval": a.eval = Int(v()) ?? a.eval
        case "--steps": a.steps = Int(v()) ?? a.steps
        case "--batch": a.batch = Int(v()) ?? a.batch
        case "--max-tokens": a.maxTokens = Int(v()) ?? a.maxTokens
        case "--max-chars": a.maxChars = Int(v()) ?? a.maxChars
        case "--lr": a.lr = Float(v()) ?? a.lr
        case "--lora-rank": a.loraRank = Int(v()) ?? a.loraRank
        case "--grad-cache": a.gradCache = Int(v()) ?? a.gradCache
        case "--seq": a.sequential = v() != "0"
        case "--cache-gb": a.cacheGB = Double(v()) ?? a.cacheGB
        case "--seed": a.seed = UInt64(v()) ?? a.seed
        case "--state-lr": a.stateLR = Float(v()) ?? a.stateLR
        case "--state-parts": a.stateParts = v()
        case "--state-out": a.stateOut = v()
        case "--head-out": a.headOut = v()
        case "--state-in": a.stateIn = v()
        case "--head-in": a.headIn = v()
        case "--allow-other-base": a.allowOtherBase = true
        case "--forward-ab": a.forwardAB = Int(v()) ?? 0
        case "--out": a.out = v()
        case "--limit": a.limit = Int(v())
        case "--log-every": a.logEvery = max(1, Int(v()) ?? a.logEvery)
        case "--guard": a.guardGB = Double(v()) ?? a.guardGB
        case "--probe": a.probe = true
        case "--chunk-tokens": a.chunkTokens = Int(v())
        case "--trace-stride": a.traceStride = Int(v()) ?? a.traceStride
        case "--pipeline-ab": a.pipelineAB = true
        case "--unpack": a.unpack = Int(v())
        case "--eval-tokens": EmbeddingMetrics.evaluationBatchTokens = Int(v()) ?? EmbeddingMetrics.evaluationBatchTokens
        case "--pipeline": SequentialBody.pipelined = v() != "0"
        default:
            FileHandle.standardError.write("неизвестный ключ \(k)\n".data(using: .utf8)!)
            exit(2)
        }
    }
    return a
}

func expand(_ p: String) -> String { (p as NSString).expandingTildeInPath }

func footprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let ok = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return ok == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : 0
}

func loadBase(_ path: String) throws -> (X070Backbone, quantised: Bool) {
    let p = expand(path)
    if p.hasSuffix(".rwkvq") { return (try X070Backbone.load(fromRwkvq: p), true) }
    let weights = try loadArrays(url: URL(fileURLWithPath: p))
    let nLayer = weights.keys.compactMap { k -> Int? in
        guard k.hasPrefix("blocks.") else { return nil }
        return Int(k.split(separator: ".")[1])
    }.max().map { $0 + 1 } ?? 0
    let cfg = X070Config(nLayer: nLayer, nEmbd: weights["ln_out.weight"]!.shape[0],
                         headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                         vocab: weights["head.weight"]!.shape[0])
    return (X070Backbone(weights: weights, cfg: cfg), false)
}

let args = parse()
guard !args.model.isEmpty, !args.vocab.isEmpty, !args.data.isEmpty else {
    print("нужны --model, --vocab, --data"); exit(2)
}
guard let task = EmbeddingTask(rawValue: args.task) else { print("задача \(args.task)?"); exit(2) }
guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: expand(args.vocab))) else {
    print("не удалось прочитать словарь"); exit(2)
}

let f0 = footprintGB()
// Пик footprint по фазам (опрос каждые 2 мс): пик всего процесса, который даёт
// `/usr/bin/time -l`, не говорит, где он случился — на загрузке и первом шаге
// (сборка ядер, первое касание весов) или в установившемся обучении.
final class FootprintWatch: @unchecked Sendable {
    private var peak = 0.0
    private let lock = NSLock()
    init(guardGB: Double) {
        let t = Thread { [self] in
            while true {
                let f = footprintGB()
                if guardGB > 0 && f > guardGB {
                    print(String(format: "ПРЕДОХРАНИТЕЛЬ: footprint %.1f ГБ выше порога %.1f — выход", f, guardGB))
                    fflush(stdout); exit(3)
                }
                lock.lock(); if f > peak { peak = f }; lock.unlock()
                usleep(2000)
            }
        }
        t.stackSize = 1 << 16
        t.start()
    }
    /// Пик с прошлого вызова; отсчёт начинается заново.
    func take() -> Double {
        lock.lock(); defer { lock.unlock() }
        let p = peak; peak = 0; return p
    }
}
let watch = FootprintWatch(guardGB: args.guardGB)
var t = Date()
let (base, quantised) = try loadBase(args.model)
print(String(format: "база: %@, L=%d D=%d; загрузка %.1f с; MLX активная %.2f ГБ, footprint +%.2f ГБ",
             quantised ? ".rwkvq" : "плотная", base.cfg.nLayer, base.cfg.nEmbd,
             Date().timeIntervalSince(t), Double(GPU.activeMemory) / 1e9, footprintGB() - f0))

let rows = try EmbeddingDataset.loadJSONL(path: expand(args.data), limit: args.limit, tasks: [task])
guard rows.count >= args.eval + 16 else { print("строк задачи \(task) всего \(rows.count)"); exit(2) }
// отложенные строки — с конца, обучающие — с начала: не пересекаются
let heldOut = Array(rows.suffix(args.eval))
let train = Array(rows.prefix(min(args.train, rows.count - args.eval)))
print("задача \(task.rawValue): обучение \(train.count), оценка \(heldOut.count)")

let mode: BaseTrainingMode?
switch args.mode {
case "none": mode = nil
case "frozen": mode = .frozen
case "full": mode = .full
case "lora":
    LoRA.add(to: base, spec: LoRASpec(rank: args.loraRank, alpha: Float(args.loraRank)))
    mode = .lora
case "state":
    // state-tuning: состояние и голова; шаг состояния --state-lr, шаг головы --lr
    let parts: [StateTrainableSet.Part] = args.stateParts == "all" ? StateTrainableSet.Part.allCases : [.wkv]
    mode = .state(learningRateScale: args.stateLR / args.lr, parts: parts)
default:
    if args.mode.hasPrefix("top:"), let n = Int(args.mode.dropFirst(4)) {
        if quantised {
            let unpacked = base.unpackTopLayers(args.unpack ?? n)
            print(String(format: "развёрнуто в плотные: %d весов верхних %d слоёв; MLX активная %.2f ГБ",
                         unpacked.count, args.unpack ?? n, Double(GPU.activeMemory) / 1e9))
        }
        mode = .topLayers(n)
    }
    else { print("режим \(args.mode)?"); exit(2) }
}
// сид задаёт и порядок данных, и начальные веса головы (fc1): при сиде 0 —
// прежнее умолчание головы, счёт тот же
var model = EmbeddingModel(backbone: base,
                           head: EmbeddingHead(dim: base.cfg.nEmbd, seed: args.seed))
// готовые состояние и голова (перенос на другую базу, оценка без обучения)
if let path = args.headIn { model = try model.loadHead(from: URL(fileURLWithPath: expand(path))) }
if let path = args.stateIn {
    let (state, info) = try StateFile.load(URL(fileURLWithPath: expand(path)), base: base,
                                           allowOtherBase: args.allowOtherBase)
    base.initialState = state
    let here = StateFile.baseFingerprint(base)
    print("состояние: \(info.kind), обучено \(info.trained.joined(separator: ",")), база файла "
          + "\(info.baseFingerprint.prefix(12))… — \(info.baseFingerprint == here ? "та же" : "ДРУГАЯ (\(here.prefix(12))…)")")
}
// поштучные результаты ПОСЛЕДНЕЙ оценки — в отчёт, для парного бутстрэпа
var perRow: [Double] = []
EmbeddingMetrics.perRowObserver = { perRow = $0 }
if let m = mode, let why = EmbeddingTrainable.unsupportedReason(model: model, mode: m) {
    print(why); exit(2)
}
var stage = EmbeddingStage(
    task: task, train: train, heldOut: heldOut,
    config: TrainingConfig(lr: args.lr, lrMin: args.lr / 20, schedule: .cosine,
                           warmupSteps: min(50, args.steps / 5), maxSteps: args.steps),
    batchSize: args.batch, gradCacheChunk: args.gradCache,
    maxTokens: args.maxTokens, maxChars: args.maxChars, seed: args.seed)

stage.sequentialBackward = args.sequential
stage.config.cacheLimitGB = args.cacheGB
stage.config.logEvery = args.logEvery
if let c = args.chunkTokens { stage.bodyChunkTokens = c }
stage.traceStride = args.traceStride
var report: [String: Any] = ["model": args.model, "mode": args.mode, "task": task.rawValue,
                             "train": train.count, "eval": heldOut.count, "steps": args.steps,
                             "batch": args.batch, "max_tokens": args.maxTokens, "quantised": quantised]

if args.forwardAB > 0 {
    // ПРЯМОЙ ПРОХОД: один граф против прохода по блокам. Один и тот же батч
    // (--batch строк по --max-tokens токенов), плечи чередуются в одном
    // процессе, --forward-ab повторов на плечо; первый круг — прогрев.
    // Плечи: одним графом; по блокам с eval после каждого; то же с asyncEval;
    // eval после каждых 2 / 4 / 8 блоков; eval один раз в конце.
    if args.cacheGB > 0 { MLX.Memory.cacheLimit = Int(args.cacheGB * 1e9) }
    let texts = heldOut.prefix(args.batch).map { String($0.positive.prefix(args.maxChars)) }
    let (idx, pool) = encodeBatch(tokenizer: tok, texts: Array(texts), terminator: 0, maxTokens: args.maxTokens)
    let plan = SequentialBody.Plan(names: [], nLayer: base.cfg.nLayer)
    let L = base.cfg.nLayer
    var arms: [(String, () -> MLXArray)] = [("один граф", { model.embed(idx, poolIndex: pool) })]
    func blocks(_ every: Int, async: Bool = false) -> () -> MLXArray {
        return {
            SequentialBody.forwardSettleEvery = every; SequentialBody.pipelined = async
            defer { SequentialBody.forwardSettleEvery = 1; SequentialBody.pipelined = false }
            return model.embedFromTop(SequentialBody.forward(base, ids: idx, plan: plan).top, poolIndex: pool)
        }
    }
    arms.append(("по блокам, eval", blocks(1)))
    arms.append(("по блокам, asyncEval", blocks(1, async: true)))
    for k in [2, 4, 8] where k < L { arms.append(("eval каждые \(k)", blocks(k))) }
    arms.append(("по блокам, eval в конце", blocks(L)))
    var times = [[Double]](repeating: [], count: arms.count), peaks = [[Double]](repeating: [], count: arms.count)
    var reference: MLXArray? = nil
    for round in 0 ... args.forwardAB {
        for (i, arm) in arms.enumerated() {
            GPU.resetPeakMemory()
            let t0 = Date()
            let v = base.withoutGradientPath { arm.1() }
            eval(v)
            if round > 0 { times[i].append(Date().timeIntervalSince(t0)); peaks[i].append(Double(GPU.peakMemory) / 1e9) }
            if reference == nil { reference = v }
            else if round == 0 {
                print(String(format: "  %@: расхождение с одним графом %.1e", arm.0 as NSString,
                             MLX.abs(v.asType(.float32) - reference!.asType(.float32)).max().item(Float.self)))
            }
        }
    }
    func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
    print("ПРЯМОЙ ПРОХОД, \(idx.shape[0]) x \(idx.shape[1]) токенов, кэш \(args.cacheGB) ГБ, \(args.forwardAB) повторов:")
    let baseTime = med(times[0])
    for (i, arm) in arms.enumerated() {
        print(String(format: "  %@ %.3f с (%+.1f %%)  мин %.3f  пик MLX %.2f ГБ", (arm.0 as NSString).padding(toLength: 26, withPad: " ", startingAt: 0) as NSString,
                     med(times[i]), (med(times[i]) / baseTime - 1) * 100, times[i].min()!, med(peaks[i])))
    }
    exit(0)
}
if args.probe {
    // Отметка: активная память MLX после фазы и пик MLX ЗА фазу (сбрасывается).
    // Разность пика и активной памяти до фазы — её временное потребление.
    var stepNo = 0
    var before = Double(GPU.activeMemory) / 1e9
    SequentialBody.probe = { name in
        if name == "batch" { stepNo += 1 }
        let active = Double(GPU.activeMemory) / 1e9, peak = Double(GPU.peakMemory) / 1e9
        if stepNo == 3 {       // третий шаг: установившийся (моменты Adam заведены, ядра собраны)
            print(String(format: "  ПРИБОР %-10@ активная %.3f  пик за фазу %.3f  (+%.3f к началу фазы)  кэш %.3f",
                         name as NSString, active, peak, peak - before, Double(GPU.cacheMemory) / 1e9))
        }
        before = active
        GPU.resetPeakMemory()
    }
}
if let mode {
    // Оценку до и после делает сама стадия (`EmbeddingFinetune.run`) — отдельно
    // её здесь не гоняем, иначе отложенная выборка считалась бы трижды.
    GPU.resetPeakMemory()
    t = Date()
    var last = Date(), abTime: [[Double]] = [[], []], abPeak: [[Double]] = [[], []]
    let results = try EmbeddingFinetune.run(model: model, tokenizer: tok, mode: mode, stages: [stage],
                                            onStep: { _, s in
        if args.pipelineAB {
            // шаг s.step только что кончился в режиме (s.step - 1) % 2; следующий — в другом
            let arm = (s.step - 1) % 2
            if s.step > 2 {
                abTime[arm].append(Date().timeIntervalSince(last))
                abPeak[arm].append(Double(GPU.peakMemory) / 1e9)
            }
            SequentialBody.pipelined = s.step % 2 == 1
            GPU.resetPeakMemory()
            last = Date()
        }
        if s.step == 1 {
            print(String(format: "  пик footprint до конца первого шага (загрузка, оценка, сборка ядер): %.2f ГБ",
                         watch.take()))
        }
        if s.step % args.logEvery == 0 || s.step == 1 {
            print(String(format: "  шаг %d: лосс %.4f, %.2f с от начала стадии", s.step, s.loss,
                         Date().timeIntervalSince(t)))
            fflush(stdout)
        }
    })
    let r = results[0]
    if args.pipelineAB {
        func med(_ a: [Double]) -> Double { a.isEmpty ? .nan : a.sorted()[a.count / 2] }
        print(String(format: "КОНВЕЙЕР A/B (чередование шагов, медианы): eval после блока %.3f с, пик MLX %.2f ГБ | asyncEval %.3f с, пик MLX %.2f ГБ | шагов %d + %d",
                     med(abTime[0]), med(abPeak[0]), med(abTime[1]), med(abPeak[1]), abTime[0].count, abTime[1].count))
    }
    print(String(format: "пик footprint со второго шага: %.2f ГБ", watch.take()))
    print("до обучения: \(r.before?.description ?? "—")")
    print(String(format: "после обучения: %@ | %d шагов за %.1f с (%.2f с/шаг) | пик MLX %.2f ГБ, footprint +%.2f ГБ",
                 r.after?.description ?? "—", r.steps, r.seconds, r.seconds / Double(max(1, r.steps)),
                 Double(GPU.peakMemory) / 1e9, footprintGB() - f0))
    report["before"] = r.before?.headline ?? Double.nan
    report["after"] = r.after?.headline ?? Double.nan
    report["final_loss"] = Double(r.finalLoss); report["train_seconds"] = r.seconds
    report["train_peak_gb"] = Double(GPU.peakMemory) / 1e9
} else {
    if args.cacheGB > 0 { MLX.Memory.cacheLimit = Int(args.cacheGB * 1e9) }
    _ = EmbeddingMetrics.embedAll(model: model, tokenizer: tok, texts: ["прогрев"])   // первое касание весов — не в замер
    GPU.resetPeakMemory()
    _ = watch.take()
    t = Date()
    let before = EmbeddingFinetune.evaluate(model: model, tokenizer: tok, stage: stage)
    let evalSeconds = Date().timeIntervalSince(t)
    print(String(format: "оценка: %@ | %.1f с (%.1f мс на строку) | пик MLX %.2f ГБ, пик footprint оценки %.2f ГБ",
                 before.description, evalSeconds, evalSeconds * 1e3 / Double(heldOut.count),
                 Double(GPU.peakMemory) / 1e9, watch.take()))
    report["before"] = before.headline; report["eval_seconds"] = evalSeconds
    report["eval_peak_gb"] = Double(GPU.peakMemory) / 1e9
}
report["footprint_gb"] = footprintGB() - f0
if case .state = mode, let state = base.initialState {
    report["state_numbers"] = state.wkv.size
    report["state_abs_max"] = Double(MLX.abs(state.wkv).max().item(Float.self))
    if let path = args.stateOut {
        let parts: [StateTrainableSet.Part] = args.stateParts == "all" ? StateTrainableSet.Part.allCases : [.wkv]
        try StateFile.save(state, base: base, to: URL(fileURLWithPath: expand(path)), kind: .tuned,
                           trained: parts, note: (args.model as NSString).lastPathComponent)
    }
}
if let path = args.headOut { try model.saveHead(to: URL(fileURLWithPath: expand(path))) }
report["seed"] = Int(args.seed); report["per_row"] = perRow
if let out = args.out {
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: expand(out)))
}
