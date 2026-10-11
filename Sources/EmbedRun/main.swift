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
//          top:N      — верхние N слоёв и голова (только плотная база: у
//                       квантованной проекции — не веса, обучать нечего)
//          full       — вся база и голова (только плотная база)
//          none       — только оценка
// ───────────────────────────────────────────────────────────────────────

struct Args {
    var model = "", vocab = "", data = "", mode = "frozen", task = "retrieval"
    var train = 2000, eval = 300, steps = 300, batch = 8, maxTokens = 256, maxChars = 800
    var lr: Float = 2e-5
    var loraRank = 8
    var gradCache = 0
    var seed: UInt64 = 0
    var out: String? = nil
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
        case "--seed": a.seed = UInt64(v()) ?? a.seed
        case "--out": a.out = v()
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
var t = Date()
let (base, quantised) = try loadBase(args.model)
print(String(format: "база: %@, L=%d D=%d; загрузка %.1f с; MLX активная %.2f ГБ, footprint +%.2f ГБ",
             quantised ? ".rwkvq" : "плотная", base.cfg.nLayer, base.cfg.nEmbd,
             Date().timeIntervalSince(t), Double(GPU.activeMemory) / 1e9, footprintGB() - f0))

let rows = try EmbeddingDataset.loadJSONL(path: expand(args.data), limit: nil, tasks: [task])
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
default:
    if args.mode.hasPrefix("top:"), let n = Int(args.mode.dropFirst(4)) { mode = .topLayers(n) }
    else { print("режим \(args.mode)?"); exit(2) }
}
let model = EmbeddingModel(backbone: base)
if let m = mode, let why = EmbeddingTrainable.unsupportedReason(model: model, mode: m) {
    print(why); exit(2)
}
let stage = EmbeddingStage(
    task: task, train: train, heldOut: heldOut,
    config: TrainingConfig(lr: args.lr, lrMin: args.lr / 20, schedule: .cosine,
                           warmupSteps: min(50, args.steps / 5), maxSteps: args.steps),
    batchSize: args.batch, gradCacheChunk: args.gradCache,
    maxTokens: args.maxTokens, maxChars: args.maxChars, seed: args.seed)

var report: [String: Any] = ["model": args.model, "mode": args.mode, "task": task.rawValue,
                             "train": train.count, "eval": heldOut.count, "steps": args.steps,
                             "batch": args.batch, "max_tokens": args.maxTokens, "quantised": quantised]

if let mode {
    // Оценку до и после делает сама стадия (`EmbeddingFinetune.run`) — отдельно
    // её здесь не гоняем, иначе отложенная выборка считалась бы трижды.
    GPU.resetPeakMemory()
    t = Date()
    let results = try EmbeddingFinetune.run(model: model, tokenizer: tok, mode: mode, stages: [stage],
                                            onStep: { _, s in
        if s.step % 50 == 0 || s.step == 1 {
            print(String(format: "  шаг %d: лосс %.4f, %.1f с от начала стадии", s.step, s.loss,
                         Date().timeIntervalSince(t)))
            fflush(stdout)
        }
    })
    let r = results[0]
    print("до обучения: \(r.before?.description ?? "—")")
    print(String(format: "после обучения: %@ | %d шагов за %.1f с (%.2f с/шаг) | пик MLX %.2f ГБ, footprint +%.2f ГБ",
                 r.after?.description ?? "—", r.steps, r.seconds, r.seconds / Double(max(1, r.steps)),
                 Double(GPU.peakMemory) / 1e9, footprintGB() - f0))
    report["before"] = r.before?.headline ?? Double.nan
    report["after"] = r.after?.headline ?? Double.nan
    report["final_loss"] = Double(r.finalLoss); report["train_seconds"] = r.seconds
    report["train_peak_gb"] = Double(GPU.peakMemory) / 1e9
} else {
    GPU.resetPeakMemory()
    t = Date()
    let before = EmbeddingFinetune.evaluate(model: model, tokenizer: tok, stage: stage)
    let evalSeconds = Date().timeIntervalSince(t)
    print(String(format: "оценка: %@ | %.1f с (%.1f мс на строку) | пик MLX %.2f ГБ",
                 before.description, evalSeconds, evalSeconds * 1e3 / Double(heldOut.count),
                 Double(GPU.peakMemory) / 1e9))
    report["before"] = before.headline; report["eval_seconds"] = evalSeconds
    report["eval_peak_gb"] = Double(GPU.peakMemory) / 1e9
}
report["footprint_gb"] = footprintGB() - f0
if let out = args.out {
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: expand(out)))
}
