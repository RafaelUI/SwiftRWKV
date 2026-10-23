import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  gen-run: ответы чат-модели на список вопросов при разных способах дать ей
//  системную инструкцию — для проверки «состояние вместо системного промпта».
//
//      gen-run --model M --vocab V --questions q.txt [--system "…"]
//              [--mode inline|fold|none] [--state-in s.safetensors]
//              [--samples N --temp T --top-p P --seed S] [--max-tokens K]
//              [--out o.json]
//
//  Промпт g1: «System: X\n\nUser: q\n\nAssistant:».
//    inline — системная строка в промпте каждого вопроса;
//    fold   — «System: X\n\n» свёрнута в состояние один раз и подключена
//             (обязана дать то же, что inline: продолжение прохода);
//    none   — без системной строки (с --state-in — обученное состояние).
//  Вопросы — по строке в файле; пустые и начинающиеся с # пропускаются.
//
//  Обучение состояния вместо системной строки (--distill):
//      gen-run --model M --vocab V --system "…" --distill
//              --train-questions t.txt --teacher-cache d.json
//              [--teacher-samples 4] [--init zero|fold] [--steps 300 --lr 3e-4
//               --batch 4] --state-out s.safetensors
//  Строки t.txt — «вид: вопрос»: id (ответ обязан совпасть с --keep-regex),
//  skill (ответ упоминает стихи), poem (в ответе не меньше трёх строк), gen
//  (берётся, если не оборван лимитом токенов), fact («fact: регэксп :: вопрос»
//  — берётся только ВЕРНЫЙ ответ учителя); --reject-regex отбрасывает ответы
//  (например, пересказ системной строки). Учитель —
//  та же модель с системной строкой; его ответы фильтруются по виду и все,
//  с отметкой «взят», сохраняются в --teacher-cache (повторный запуск берёт
//  их оттуда и догенерирует только новые вопросы — у плеч одни данные; в
//  обучение идут только вопросы текущего --train-questions). Ученик — та же модель БЕЗ системной строки с обучаемым
//  начальным состоянием (с нуля или со свёрнутой системной строки); лосс
//  только по ответу (цели вопроса < 0).
//
//  Замер шага state-tuning (--state-bench):
//      gen-run --model M --vocab V --state-bench tokens.txt [--steps 30]
//              [--sb-warm 5] [--sb-whole] [--sb-cache ГБ] [--sb-profile]
//              [--sb-cast | --sb-nocast] [--sb-cast-ab N --sb-arms off,wkv,lin] [--out o.json]
//  tokens.txt — T+1 номеров токенов через пробел (одна строка, батч 1; T
//  кратна 16); шаг — `StateTuning.run` с умолчаниями `StateTuneConfig`
//  (обратный проход по блокам; --sb-whole — одним графом). Печатает медиану
//  времени шага после разогрева и ток/с. Типы: умолчание — приведение входа
//  проекций (StateTuneConfig.castProjectionInput), --sb-cast — выход WKV,
//  --sb-nocast — без приведения; --sb-cast-ab N — плечи вперемежку в одном процессе.
// ───────────────────────────────────────────────────────────────────────

struct Args {
    var model = "", vocab = "", questions = "", system = "", mode = "inline"
    var stateIn: String? = nil, out: String? = nil
    var samples = 1, temp: Float = 0, topP: Float = 1, seed: UInt64 = 0, maxTokens = 160
    var think = false
    var distill = false, trainQuestions = "", teacherCache = "", stateOut = ""
    var teacherSamples = 4, steps = 300, lr: Float = 3e-4, batch = 4, initMode = "zero"
    var keepRegex = "neuron|нейрон", parts = "wkv", gradClip: Float? = nil, logEvery = 25
    var rejectRegex: String? = nil
    var stateBench: String? = nil, sbWarm = 5, sbWhole = false, sbCache: Double? = nil, sbProfile = false
    var stream = false
    var sbCastAB = 0, sbCast = false, sbNoCast = false, sbArms = "off,wkv,lin"
}

func parse() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        func v() -> String { it.next() ?? "" }
        switch k {
        case "--model": a.model = v()
        case "--vocab": a.vocab = v()
        case "--questions": a.questions = v()
        case "--system": a.system = v()
        case "--mode": a.mode = v()
        case "--state-in": a.stateIn = v()
        case "--out": a.out = v()
        case "--samples": a.samples = max(1, Int(v()) ?? 1)
        case "--temp": a.temp = Float(v()) ?? 0
        case "--top-p": a.topP = Float(v()) ?? 1
        case "--seed": a.seed = UInt64(v()) ?? 0
        case "--max-tokens": a.maxTokens = Int(v()) ?? a.maxTokens
        case "--think": a.think = true      // не подавлять рассуждение g1
        case "--distill": a.distill = true
        case "--train-questions": a.trainQuestions = v()
        case "--teacher-cache": a.teacherCache = v()
        case "--teacher-samples": a.teacherSamples = max(1, Int(v()) ?? 4)
        case "--state-out": a.stateOut = v()
        case "--steps": a.steps = Int(v()) ?? a.steps
        case "--lr": a.lr = Float(v()) ?? a.lr
        case "--batch": a.batch = max(1, Int(v()) ?? a.batch)
        case "--init": a.initMode = v()
        case "--keep-regex": a.keepRegex = v()
        case "--state-parts": a.parts = v()
        case "--grad-clip": a.gradClip = Float(v())
        case "--reject-regex": a.rejectRegex = v()
        case "--log-every": a.logEvery = max(1, Int(v()) ?? 25)
        case "--state-bench": a.stateBench = v()
        case "--stream": a.stream = true                  // подкачка весов с диска (плотный .safetensors)
        case "--sb-warm": a.sbWarm = max(0, Int(v()) ?? 5)
        case "--sb-whole": a.sbWhole = true
        case "--sb-cache": a.sbCache = Double(v())
        case "--sb-cast": a.sbCast = true          // castWKVOutputToComputeDType = true
        case "--sb-cast-ab": a.sbCastAB = Int(v()) ?? 0   // кругов A/B флага в одном процессе
        case "--sb-arms": a.sbArms = v()                  // плечи A/B: off | wkv | lin (через запятую)
        case "--sb-nocast": a.sbNoCast = true             // без приведения (как до 23.10)
        case "--sb-profile": a.sbProfile = true      // фазы шага (по блокам): прямой, голова+лосс, блоки, Adam
        default:
            FileHandle.standardError.write("неизвестный ключ \(k)\n".data(using: .utf8)!); exit(2)
        }
    }
    return a
}

func expand(_ p: String) -> String { (p as NSString).expandingTildeInPath }

func loadBase(_ path: String) throws -> X070Backbone {
    let p = expand(path)
    if p.hasSuffix(".rwkvq") { return try X070Backbone.load(fromRwkvq: p) }
    if args.stream { return try X070Backbone.loadStreaming(safetensors: URL(fileURLWithPath: p)) }
    let weights = try loadArrays(url: URL(fileURLWithPath: p))
    let nLayer = weights.keys.compactMap { k -> Int? in
        guard k.hasPrefix("blocks.") else { return nil }
        return Int(k.split(separator: ".")[1])
    }.max().map { $0 + 1 } ?? 0
    let cfg = X070Config(nLayer: nLayer, nEmbd: weights["ln_out.weight"]!.shape[0],
                         headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                         vocab: weights["head.weight"]!.shape[0])
    return X070Backbone(weights: weights, cfg: cfg, computeDType: .bfloat16)
}

let args = parse()
guard !args.model.isEmpty, !args.vocab.isEmpty, !args.questions.isEmpty || args.distill || args.stateBench != nil else {
    print("нужны --model, --vocab, --questions"); exit(2)
}
guard ["inline", "fold", "none"].contains(args.mode) else { print("--mode: inline | fold | none"); exit(2) }
guard args.mode == "none" || !args.system.isEmpty else { print("--mode \(args.mode): нужен --system"); exit(2) }
guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: expand(args.vocab))) else {
    print("не удалось прочитать словарь"); exit(2)
}
func lines(_ path: String) throws -> [String] {
    try String(contentsOfFile: expand(path), encoding: .utf8)
        .split(separator: "\n").map(String.init)
        .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty && !$0.hasPrefix("#") }
}
let questions = args.distill || args.stateBench != nil ? [] : try lines(args.questions)

let t0 = Date()
let base = try loadBase(args.model)
print(String(format: "модель: L=%d D=%d, загрузка %.1f с", base.cfg.nLayer, base.cfg.nEmbd, Date().timeIntervalSince(t0)))

let systemPart = args.system.isEmpty ? "" : "System: \(args.system)\n\n"
// g1 без рассуждения: пустой блок <think> (как в шаблоне чата g1)
let assistant = args.think ? "Assistant:" : "Assistant: <think>\n</think>"
func userPart(_ q: String) -> String { "User: \(q)\n\n\(assistant)" }

if args.distill {
    try distill()
    exit(0)
}
if let path = args.stateBench {
    try stateBench(path)
    exit(0)
}

/// Замер шага state-tuning на одной последовательности (см. шапку).
func stateBench(_ path: String) throws {
    let ids = try String(contentsOfFile: expand(path), encoding: .utf8)
        .split(whereSeparator: { $0 == " " || $0 == "\n" }).compactMap { Int32($0) }
    let T = ids.count - 1
    guard T > 0, T % 16 == 0 else { print("--state-bench: нужно T+1 токенов, T кратна 16 (сейчас T = \(T))"); exit(2) }
    let x = MLXArray(Array(ids.dropLast()), [1, T]), y = MLXArray(Array(ids.dropFirst()), [1, T])
    var cfg = StateTuneConfig()
    cfg.maxSteps = args.steps; cfg.logEvery = 1; cfg.sequentialBackward = !args.sbWhole
    // типы на время шага — из cfg (StateTuning.run выставляет флаги бэкбона сам): умолчание
    // библиотеки — приведение входа проекций; --sb-cast — выход WKV; --sb-nocast — как до 23.10
    if args.sbCast { cfg.castWKVOutput = true; cfg.castProjectionInput = false }
    if args.sbNoCast { cfg.castProjectionInput = false }
    if args.sbCastAB > 0 {
        // A/B нескольких способов не пускать fp32 в веса: off — как есть; wkv — приведение выхода
        // WKV (castWKVOutputToComputeDType); lin — приведение входа плотной проекции к типу веса
        // (castDenseProjectionInput). Плечи вперемежку в одном процессе, порядок сдвигается по кругам.
        if let c = args.sbCache { cfg.cacheLimitGB = c }
        let arms = args.sbArms.split(separator: ",").map(String.init)
        var med: [String: [Double]] = [:], loss: [String: [Float]] = [:]
        for round in 0 ..< args.sbCastAB {
            let order = (0 ..< arms.count).map { arms[($0 + round) % arms.count] }
            for arm in order {
                cfg.castWKVOutput = arm == "wkv"
                cfg.castProjectionInput = arm == "lin"
                base.initialState = nil
                var marks = [Date()], ls: [Float] = []
                StateTuning.run(base, nextBatch: { (x: x, y: y) }, config: cfg, onStep: { _, l, _, _ in
                    marks.append(Date()); ls.append(l)
                })
                let ts = zip(marks, marks.dropFirst()).map { $1.timeIntervalSince($0) }.dropFirst(args.sbWarm).sorted()
                med[arm, default: []].append(ts[ts.count / 2]); loss[arm] = ls
                print(String(format: "  круг %d, %@: %.3f с/шаг, лосс %.5f -> %.5f", round, arm as NSString,
                             ts[ts.count / 2], ls.first ?? .nan, ls.last ?? .nan)); fflush(stdout)
            }
        }
        func m(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        print("A/B ТИПОВ, \(args.sbWhole ? "одним графом" : "по блокам"), T \(T): " + arms.map {
            String(format: "%@ %.3f с/шаг (лосс шага 1 %.6f)", $0 as NSString, m(med[$0]!), loss[$0]![0]) }.joined(separator: "; "))
        return
    }
    if let c = args.sbCache { cfg.cacheLimitGB = c }
    var marks = [Date()], losses: [Float] = []
    // --sb-profile: время между отметками SequentialBody.probe (каждая — после eval) на шагах после
    // разогрева; фаза «adam» — от последней отметки шага до конца обновления.
    var phase: [String: [Double]] = [:], last = Date(), stepNo = 0
    if args.sbProfile {
        SequentialBody.probe = { name in
            let now = Date()
            if stepNo >= args.sbWarm {
                let key = name.hasPrefix("block") ? "блоки" : name == "loss" ? "голова+лосс" : name
                phase[key, default: []].append(now.timeIntervalSince(last))
            }
            last = now
            if name == "adam" { stepNo += 1 }
        }
    }
    StateTuning.run(base, nextBatch: { (x: x, y: y) }, config: cfg, onStep: { _, loss, _, _ in
        marks.append(Date()); losses.append(loss)
    })
    SequentialBody.probe = nil
    let times = zip(marks, marks.dropFirst()).map { $1.timeIntervalSince($0) }
    if args.sbProfile && stepNo > args.sbWarm {
        let n = Double(stepNo - args.sbWarm)
        print("фазы шага (среднее за шаг, \(Int(n)) шагов):")
        for (k, v) in phase.sorted(by: { $0.value.reduce(0, +) > $1.value.reduce(0, +) }) {
            print(String(format: "  %-12@ %.3f с  (%d отметок за шаг)", k as NSString, v.reduce(0, +) / n, Int(Double(v.count) / n)))
        }
    }
    let tail = Array(times.dropFirst(min(args.sbWarm, max(0, times.count - 1)))).sorted()
    let med = tail[tail.count / 2]
    let report: [String: Any] = ["arm": args.sbWhole ? "swift_whole" : "swift_seq", "T": T, "B": 1,
                                 "cache": cfg.cacheLimitGB, "median_s": med, "tok_s": Double(T) / med,
                                 "times": times, "loss_first": Double(losses.first ?? .nan),
                                 "loss_last": Double(losses.last ?? .nan), "model": args.model]
    print(String(format: "state-bench %@: T %d, медиана шага %.3f с, %.0f ток/с, лосс %.4f -> %.4f",
                 args.sbWhole ? "одним графом" : "по блокам", T, med, Double(T) / med,
                 losses.first ?? .nan, losses.last ?? .nan))
    if let out = args.out {
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]).write(to: URL(fileURLWithPath: expand(out)))
    }
}

/// Обучение состояния вместо системной строки (см. шапку).
func distill() throws {
    guard !args.system.isEmpty, !args.trainQuestions.isEmpty, !args.teacherCache.isEmpty, !args.stateOut.isEmpty else {
        print("--distill: нужны --system, --train-questions, --teacher-cache, --state-out"); exit(2)
    }
    guard ["zero", "fold"].contains(args.initMode) else { print("--init: zero | fold"); exit(2) }
    let cacheURL = URL(fileURLWithPath: expand(args.teacherCache))
    // кэш учителя: ВСЕ ответы с отметкой kept (старые кэши — только взятые, без
    // отметки); вопросы, которых в кэше нет, догенерируются
    var cache: [[String: String]] = []
    if FileManager.default.fileExists(atPath: cacheURL.path) {
        cache = try JSONSerialization.jsonObject(with: Data(contentsOf: cacheURL)) as! [[String: String]]
        print("ответы учителя — из \(cacheURL.lastPathComponent): \(cache.count)")
    }
    let asked = Set(cache.map { $0["question"]! })
    var wanted = Set<String>()
    let keep = try NSRegularExpression(pattern: args.keepRegex, options: [.caseInsensitive])
    let skillRe = try NSRegularExpression(pattern: "стих|поэ|poem|poet", options: [.caseInsensitive])
    var kept = [String: Int](), total = [String: Int]()
    for line in try lines(args.trainQuestions) {
        guard let colon = line.firstIndex(of: ":") else { print("строка без вида: \(line)"); exit(2) }
        let kind = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
        var q = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
        guard ["id", "skill", "poem", "gen", "fact"].contains(kind) else {
            print("вид \(kind)? (id | skill | poem | gen | fact)"); exit(2)
        }
        var factRe: NSRegularExpression? = nil
        if kind == "fact" {
            let parts = q.components(separatedBy: " :: ")
            guard parts.count == 2 else { print("fact: «регэксп :: вопрос»: \(line)"); exit(2) }
            factRe = try NSRegularExpression(pattern: parts[0], options: [.caseInsensitive])
            q = parts[1]
        }
        wanted.insert(q)
        if asked.contains(q) { continue }
        for s in 0 ..< args.teacherSamples {
            var state = base.startState()
            let cfg = GenerationConfig(
                maxTokens: args.maxTokens, stopStrings: ["\n\nUser:", "\n\nSystem:"], stopTokens: [0],
                sampling: SamplingConfig(temperature: args.temp, topP: args.topP, seed: args.seed &+ UInt64(s)))
            let r = base.generate(prompt: tok.encode(systemPart + userPart(q)), config: cfg,
                                  tokenizer: tok, state: &state)
            let a = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
            total[kind, default: 0] += 1
            let ok: Bool
            switch kind {
            case "id": ok = keep.firstMatch(in: a, range: NSRange(a.startIndex..., in: a)) != nil
            case "skill": ok = skillRe.firstMatch(in: a, range: NSRange(a.startIndex..., in: a)) != nil
            case "fact": ok = r.stopReason != .maxTokens
                && factRe!.firstMatch(in: a, range: NSRange(a.startIndex..., in: a)) != nil
            case "poem": ok = a.split(separator: "\n").count >= 3
            default: ok = r.stopReason != .maxTokens        // оборванные посторонние ответы не учим
            }
            cache.append(["kind": kind, "question": q, "answer": a, "raw": r.text, "kept": ok ? "1" : "0"])
            if ok { kept[kind, default: 0] += 1 }
        }
    }
    if !total.isEmpty {
        print("учитель (новые вопросы): оставлено \(kept) из \(total)")
        try JSONSerialization.data(withJSONObject: cache, options: [.prettyPrinted, .sortedKeys]).write(to: cacheURL)
    }
    var data = cache.filter { ($0["kept"] ?? "1") == "1" && wanted.contains($0["question"]!) }
    if let pattern = args.rejectRegex {
        // например, повтор системной строки: учитель иногда пересказывает промпт
        let re = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        let before = data.count
        data = data.filter { d in
            let a = d["raw"] ?? d["answer"]!
            return re.firstMatch(in: a, range: NSRange(a.startIndex..., in: a)) == nil
        }
        print("--reject-regex «\(pattern)»: отброшено \(before - data.count) из \(before)")
    }
    var byKind = [String: Int]()
    for d in data { byKind[d["kind"]!, default: 0] += 1 }
    print("в обучение: \(data.count) ответов \(byKind.sorted { $0.key < $1.key })")
    guard !data.isEmpty else { print("нечему учить"); exit(1) }

    // последовательности ученика: без системной строки; цели вопроса < 0
    var seqs: [(x: [Int32], y: [Int32])] = []
    var masked = 0, mismatched = 0
    for d in data {
        let prompt = userPart(d["question"]!)
        let p = tok.encode(prompt)
        // сырой текст учителя: стык «</think>» / ответ — как его выдала модель; в
        // конце — «\n\nUser:», на котором учитель остановился: без него ученик не
        // учится заканчивать реплику (первый прогон 20.10 болтал дальше «\n\n»)
        let full = tok.encode(prompt + (d["raw"] ?? " " + d["answer"]!) + "\n\nUser:")
        var cut = p.count
        if Array(full.prefix(p.count)) != p {                  // стык токенизировался иначе
            mismatched += 1
            cut = zip(full, p).prefix { $0 == $1 }.count
        }
        let x = full.dropLast().map { Int32($0) }
        let y = full.dropFirst().enumerated().map { i, t in i + 1 < cut ? Int32(-1) : Int32(t) }
        masked += y.filter { $0 < 0 }.count
        seqs.append((x, y))
    }
    let answerTokens = seqs.reduce(0) { $0 + $1.y.filter { $0 >= 0 }.count }
    print("ученик: \(seqs.count) последовательностей, токенов ответа \(answerTokens), вопроса (вне лосса) \(masked); "
          + "стык вопрос/ответ токенизирован иначе у \(mismatched)")

    var order = Array(seqs.indices), cursor = order.count
    var gen = SplitMix64Shuffle(seed: args.seed)
    func nextBatch() -> LoRABatch {
        var rows: [(x: [Int32], y: [Int32])] = []
        while rows.count < args.batch {
            if cursor == order.count { order.shuffle(using: &gen); cursor = 0 }
            rows.append(seqs[order[cursor]]); cursor += 1
        }
        let T = ((rows.map(\.x.count).max()! + 15) / 16) * 16           // кратно WKV7_CHUNK
        var xs: [Int32] = [], ys: [Int32] = []
        for r in rows {
            xs += r.x + [Int32](repeating: 0, count: T - r.x.count)
            ys += r.y + [Int32](repeating: -1, count: T - r.y.count)
        }
        return (x: MLXArray(xs, [rows.count, T]), y: MLXArray(ys, [rows.count, T]))
    }

    var cfg = StateTuneConfig()
    cfg.lr = args.lr; cfg.lrMin = args.lr / 10; cfg.maxSteps = args.steps; cfg.warmupSteps = min(20, args.steps / 10)
    cfg.train = args.parts == "all" ? StateTrainableSet.Part.allCases : [.wkv]
    cfg.logEvery = args.logEvery
    if let c = args.gradClip { cfg.gradClip = c }        // иначе — умолчание библиотеки (1.0)
    let initial: RWKVBatchState? = args.initMode == "fold" ? base.foldPrompt(tok.encode(systemPart)) : nil
    let t1 = Date()
    let out = StateTuning.run(base, nextBatch: nextBatch, config: cfg, initial: initial,
                              onStep: { step, loss, gradNorm, peakMB in
        if step % args.logEvery == 0 || step == 1 {
            print(String(format: "  шаг %d: лосс %.4f, норма градиента %.3g, пик MLX %.0f МБ, %.0f с", step, loss,
                         gradNorm, peakMB, Date().timeIntervalSince(t1)))
            fflush(stdout)
        }
    })
    let note = "distill \(args.initMode), \(args.steps) шагов, lr \(args.lr), отсечение \(cfg.gradClip)"
    _ = try StateFile.save(out.state, base: base, to: URL(fileURLWithPath: expand(args.stateOut)),
                           kind: .tuned, trained: cfg.train, note: note)
    print(String(format: "состояние сохранено: %@ (%@), %.0f с", args.stateOut, note, Date().timeIntervalSince(t1)))
}

/// Перемешивание с сидом (воспроизводимый порядок обучающих строк).
struct SplitMix64Shuffle: RandomNumberGenerator {
    var s: UInt64
    init(seed: UInt64) { s = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

if let path = args.stateIn {
    let (state, info) = try StateFile.load(URL(fileURLWithPath: expand(path)), base: base)
    base.initialState = state
    print("подключено состояние: \(info.kind), \(info.note)")
}
if args.mode == "fold" {
    precondition(args.stateIn == nil, "--mode fold и --state-in вместе: свёртка стартовала бы с обученного")
    let whole = tok.encode(systemPart + userPart(questions[0]))
    let pieces = tok.encode(systemPart) + tok.encode(userPart(questions[0]))
    print(whole == pieces ? "токенизация кусков = сплошной (первый вопрос)"
                          : "ВНИМАНИЕ: токенизация кусков расходится со сплошной — fold не обязан совпасть с inline")
    base.initialState = base.foldPrompt(tok.encode(systemPart))
}

var rows: [[String: Any]] = []
for (qi, q) in questions.enumerated() {
    let prompt = (args.mode == "inline" ? systemPart : "") + userPart(q)
    var answers: [String] = []
    for s in 0 ..< args.samples {
        var state = base.startState()
        let cfg = GenerationConfig(
            maxTokens: args.maxTokens, stopStrings: ["\n\nUser:", "\n\nSystem:"], stopTokens: [0],
            sampling: SamplingConfig(temperature: args.temp, topP: args.topP, seed: args.seed &+ UInt64(s)))
        let r = base.generate(prompt: tok.encode(prompt), config: cfg, tokenizer: tok, state: &state)
        answers.append(r.text.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    print("\n[\(qi)] \(q)")
    for (s, a) in answers.enumerated() { print("  #\(s): \(a.replacingOccurrences(of: "\n", with: " ⏎ "))") }
    fflush(stdout)
    rows.append(["question": q, "answers": answers])
}
print(String(format: "\nвсего %.1f с", Date().timeIntervalSince(t0)))
if let out = args.out {
    let report: [String: Any] = ["model": args.model, "mode": args.mode, "system": args.system,
                                 "state_in": args.stateIn ?? "", "samples": args.samples,
                                 "temp": Double(args.temp), "top_p": Double(args.topP),
                                 "seed": Int(args.seed), "think": args.think, "rows": rows]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: expand(out)))
}
