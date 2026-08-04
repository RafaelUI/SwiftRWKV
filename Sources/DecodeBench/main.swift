import Foundation
import MLX
import RWKVGen
import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Сколько стоит декодирование на квантованной базе .rwkvq против плотной.
//
//  Зачем это вообще. Числа декода в rwkv-quant (65.6 ток/с на 1.5B) сняты
//  на ДРУГОЙ механике: там Metal-ядро декодит веса внутри GEMV, и плотная
//  матрица не материализуется никогда. Здесь `X070Backbone.baseProj`
//  разворачивает матрицу целиком на КАЖДУЮ проекцию (`rwkvqWeight` не
//  кэширует намеренно — иначе квантованная база превращается в LoRA с
//  лишними шагами). Для QLoRA это правильный размен, для инференса —
//  другой, и до сих пор никем не измеренный. Переносить сюда чужие
//  тик/с нельзя.
//
//  Методика:
//
//    ЧЕРЕДОВАНИЕ В ОДНОМ ПРОЦЕССЕ. Обе базы живут одновременно, раунды
//    идут по очереди. Безвентиляторная машина под нагрузкой уводит «тот
//    же» замер в полтора раза, так что две отдельные сборки или два
//    запуска подряд сравнивают тепловое состояние, а не код.
//
//    ПРОГРЕВ ОТБРАСЫВАЕТСЯ. Первый запуск каждой новой формы на Metal
//    тянет сборку конвейера; без прогрева разовый расход размазывается
//    по замеру и притворяется пропускной способностью.
//
//    eval НА КАЖДОМ ШАГЕ. MLX ленив: без синхронизации засекается
//    постановка графа в очередь, а не работа.
//
//    МЕДИАНА, а не среднее: один тепловой выброс среднее сдвигает,
//    медиану нет.
//
//    Сверка выдачи — ТОЛЬКО с общим входом. Гонять обе базы жадно от
//    одного токена бесполезно: первое же расхождение уводит цепочки
//    навсегда, и совпадение выходит нулевым даже у исправного пути.
//    Поэтому сверка отдельная, teacher-forced: обеим базам скармливается
//    ОДНА последовательность и сравнивается top-1 на каждой позиции.
//    Замер сломанного пути бесполезен, а выглядит он точно так же, как
//    рабочий.
//
//    На больших моделях --only: две базы 2.9B одновременно в 16 ГБ не
//    помещаются. Тогда чередование делается запусками (как ppl-прогоны
//    в rwkv-quant, по конфигу на процесс), а сверка top-1 недоступна.
//
//  Собирается и запускается ТОЛЬКО в release:
//
//      swift run -c release decode-bench \
//        --model ~/Develop/rwkv-metal/world_0.1b_x070.safetensors \
//        --sidecar ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx
// ───────────────────────────────────────────────────────────────────────

struct Args {
    var model = "~/Develop/rwkv-metal/world_0.1b_x070.safetensors"
    var sidecar = "~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx"
    var steps = 64
    var rounds = 5
    var warmup = 8
    var quantizeHead = true
    var quantizeCmix = true
    var only = "both"          // both | dense | rwkvq
    var vocab = "~/Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt"
}

func expand(_ p: String) -> String { (p as NSString).expandingTildeInPath }

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        switch k {
        case "--model": a.model = it.next() ?? a.model
        case "--sidecar": a.sidecar = it.next() ?? a.sidecar
        case "--steps": a.steps = Int(it.next() ?? "") ?? a.steps
        case "--rounds": a.rounds = Int(it.next() ?? "") ?? a.rounds
        case "--warmup": a.warmup = Int(it.next() ?? "") ?? a.warmup
        case "--no-head": a.quantizeHead = false
        case "--no-cmix": a.quantizeCmix = false
        case "--only": a.only = it.next() ?? a.only
        case "--vocab": a.vocab = it.next() ?? a.vocab
        default:
            print("неизвестный аргумент \(k)")
            exit(2)
        }
    }
    return a
}

/// Physical footprint процесса, ГБ. Не `GPU.peakMemory`: счётчики MLX не
/// знают ни про mmap-страницы, ни про веса. Метод тот же, что в
/// RerankRun/main.swift — расхождение методик между замерами одного
/// репозитория дороже дублирования двадцати строк.
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

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s.isEmpty ? 0 : s[s.count / 2]
}

/// N шагов декода по ЗАДАННОЙ последовательности входов; возвращает
/// (секунды, top-1 на каждой позиции).
///
/// Вход фиксирован, а не берётся из собственной выдачи, и это важно для
/// обеих задач сразу. Для замера: жадная цепочка у двух баз расходится,
/// и они начинают считать разные вещи — сравнивать их время тогда можно
/// только с оговорками. Для сверки: teacher-forcing — единственный
/// способ увидеть, СОГЛАСНЫ ли пути, а не насколько быстро они разошлись.
func decode(_ model: X070Backbone, cfg: X070Config,
            ids: [Int]) -> (Double, [Int], MLXArray) {
    var state = RWKVState(cfg: cfg)
    var picked: [Int] = []
    picked.reserveCapacity(ids.count)
    var last = MLXArray.zeros([cfg.vocab])
    let t0 = Date()
    for id in ids {
        let logits = model.step(id, state: &state)
        let next = argMax(logits, axis: -1)
        eval(next)
        state.eval()
        picked.append(next.item(Int.self))
        last = logits
    }
    let dt = Date().timeIntervalSince(t0)
    eval(last)
    return (dt, picked, last)
}

/// Вход для замера и сверки.
///
/// На ВРЕМЯ шага значения токенов не влияют вовсе — читаются те же веса.
/// А вот на сверку влияют решающе: на случайных токенах рекуррентное
/// состояние уходит далеко за распределение, где ошибка накапливается по
/// контексту (это известное свойство: деградация квантования растёт с
/// длиной), и два почти одинаковых пути расходятся сколь угодно сильно.
/// Замерено: на случайном входе относительное расхождение логитов 2.67 —
/// цифра пугающая и бессмысленная. Поэтому берётся настоящий текст,
/// и только если словаря нет — псевдослучайный запас с оговоркой.
func inputIds(_ n: Int, vocab: Int, vocabPath: String) -> ([Int], Bool) {
    let text = """
    Квантование весов языковой модели — это способ хранить каждый вес не \
    шестнадцатью битами, а четырьмя-шестью, разбивая матрицу на блоки и \
    подбирая на блок общий масштаб. The point is not the size on disk: \
    decoding is bound by memory bandwidth, so fewer bits per weight means \
    fewer bytes across the bus and more tokens per second.
    """
    if let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: expand(vocabPath))) {
        var ids = tok.encode(text)
        guard !ids.isEmpty else { return (fallbackIds(n, vocab: vocab), false) }
        while ids.count < n { ids += ids }
        return (Array(ids.prefix(n)), true)
    }
    return (fallbackIds(n, vocab: vocab), false)
}

func fallbackIds(_ n: Int, vocab: Int) -> [Int] {
    var x = 12345
    return (0 ..< n).map { _ in
        x = (x &* 1103515245 &+ 12345) & 0x7fffffff
        return x % vocab
    }
}

// ── Прогон ──────────────────────────────────────────────────────────────

let args = parseArgs()
let fm = FileManager.default
guard fm.fileExists(atPath: expand(args.model)) else {
    print("нет модели: \(args.model)"); exit(2)
}
guard fm.fileExists(atPath: expand(args.sidecar) + ".safetensors") else {
    print("нет сайдкара: \(args.sidecar).safetensors"); exit(2)
}

print("── модель ──")
var weights = try loadArrays(url: URL(fileURLWithPath: expand(args.model)))
let nLayer = weights.keys.compactMap { k -> Int? in
    guard k.hasPrefix("blocks.") else { return nil }
    return Int(k.split(separator: ".")[1])
}.max().map { $0 + 1 } ?? 0
let cfg = X070Config(nLayer: nLayer,
                     nEmbd: weights["ln_out.weight"]!.shape[0],
                     headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                     vocab: weights["head.weight"]!.shape[0])
print("L=\(cfg.nLayer) D=\(cfg.nEmbd) H=\(cfg.headSize) V=\(cfg.vocab)")

let wantDense = args.only != "rwkvq"
let wantQuant = args.only != "dense"

var dense: X070Backbone? = wantDense ? X070Backbone(weights: weights, cfg: cfg) : nil
var quantized: X070Backbone? = wantQuant ? X070Backbone(weights: weights, cfg: cfg) : nil

if let q = quantized {
    let sidecar = try RwkvqSidecar(path: expand(args.sidecar))
    let info = q.attachRwkvq(
        sidecar,
        options: X070Backbone.RwkvqAttachOptions(quantizeCmix: args.quantizeCmix,
                                                 quantizeHead: args.quantizeHead))
    print("сайдкар: подключено \(info.attached) весов, "
          + "не найдено \(info.missing.count), "
          + "сжатых буферов \(String(format: "%.1f", Double(info.packedBytes) / 1e6)) МБ, "
          + "освобождено плотных \(String(format: "%.1f", Double(info.freedDenseBytes) / 1e6)) МБ")
    if !info.missing.isEmpty {
        print("  ВНИМАНИЕ: без сайдкара остались \(info.missing.prefix(3))… — "
              + "замер будет про смесь путей, а не про .rwkvq")
    }
}
// Локальный словарь держит ссылки на плотные веса и сводит на нет
// dropDenseWeights: без этого режим --only rwkvq мерил бы память так,
// будто плотная база никуда не делась.
weights = [:]

let (ids, realText) = inputIds(args.steps, vocab: cfg.vocab, vocabPath: args.vocab)
let (warm, _) = inputIds(args.warmup, vocab: cfg.vocab, vocabPath: args.vocab)
if !realText {
    print("ВНИМАНИЕ: словаря \(args.vocab) нет, вход псевдослучайный — "
          + "время замера верное, сверка выдачи бессмысленна")
}

print("\n── прогрев (\(args.warmup) шагов, отбрасывается) ──")
if let d = dense { _ = decode(d, cfg: cfg, ids: warm) }
if let q = quantized { _ = decode(q, cfg: cfg, ids: warm) }
var lastDenseLogits: MLXArray? = nil
var lastQuantLogits: MLXArray? = nil

print("\n── чередование: \(args.rounds) раундов по \(args.steps) шагов ──")
var msDense: [Double] = []
var msQuant: [Double] = []
var top1Dense: [Int] = []
var top1Quant: [Int] = []
for r in 0 ..< args.rounds {
    var line = String(format: "  раунд %d:", r)
    if let d = dense {
        let (t, out, lg) = decode(d, cfg: cfg, ids: ids)
        msDense.append(t * 1e3 / Double(args.steps))
        top1Dense = out
        lastDenseLogits = lg
        line += String(format: " плотная %6.2f мс/ток (%5.1f ток/с)  ",
                       msDense[r], 1e3 / msDense[r])
    }
    if let q = quantized {
        let (t, out, lg) = decode(q, cfg: cfg, ids: ids)
        msQuant.append(t * 1e3 / Double(args.steps))
        top1Quant = out
        lastQuantLogits = lg
        line += String(format: " .rwkvq %6.2f мс/ток (%5.1f ток/с)",
                       msQuant[r], 1e3 / msQuant[r])
    }
    print(line)
}

print("")
if !msDense.isEmpty {
    let m = median(msDense)
    print(String(format: "плотная: медиана %.2f мс/ток (%.1f ток/с)", m, 1e3 / m))
}
if !msQuant.isEmpty {
    let m = median(msQuant)
    print(String(format: ".rwkvq:  медиана %.2f мс/ток (%.1f ток/с)", m, 1e3 / m))
}
if !msDense.isEmpty && !msQuant.isEmpty {
    print(String(format: ".rwkvq / плотная: %.2fx", median(msQuant) / median(msDense)))
    // Санити-проверка. По top-1 на СЛУЧАЙНОМ входе судить нельзя:
    // распределение там почти вырождено и argmax решает шум, так что
    // низкое совпадение ничего не доказывает. Относительная норма
    // расхождения логитов от входа так не зависит: у исправного
    // квантования это единицы процентов, у перепутанной проводки весов --
    // порядок величины.
    if let a = lastDenseLogits, let b = lastQuantLogits {
        let d = (a.asType(.float32) - b.asType(.float32))
        let rel = sqrt((d * d).sum()) / (sqrt((a.asType(.float32)
                                               * a.asType(.float32)).sum()) + 1e-9)
        eval(rel)
        let agree = zip(top1Dense, top1Quant).filter { $0 == $1 }.count
        print(String(format: "санити: расхождение логитов %.4f отн., "
                     + "top-1 %d/%d (на случайном входе показатель слабый)",
                     rel.item(Float.self), agree, top1Dense.count))
    }
}
print("footprint в конце \(String(format: "%.2f", processFootprintGB())) ГБ")
