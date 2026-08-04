import Foundation
import MLX
import MLXRandom
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
    /// Приведение типа выхода WKV. По умолчанию в модели ВЫКЛЮЧЕНО, а
    /// стоит оно ×2.8 на декоде (docs/Inference.md: 5.58 мс/ток против
    /// 15.74 на 0.1B): без него fp32 из WKV течёт в остаточный поток и
    /// удваивает трафик всего, что ниже. Здесь по умолчанию ВКЛЮЧЕНО —
    /// замер должен показывать скорость модели, а не цену забытого
    /// флага. Отключается явно, чтобы обе ветки были сравнимы.
    var castWKV = true
    var micro = false
    var checkNative = false
    var native = false
    var profile = false
    /// Пауза до и после измеряемых раундов, мс.
    ///
    /// Для трейса через Instruments: измеряемое окно иначе неотличимо
    /// от прогрева и загрузки, а в GPU-таймлайне оно становится
    /// очевидным между двумя провалами в ноль. На сам замер не влияет —
    /// пауза снаружи засекаемого участка.
    var gap = 0
    var nativeRef = "~/Develop/SwiftRWKV/.testdata/mlx_affine_ref.safetensors"
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
        case "--no-cast-wkv": a.castWKV = false
        case "--micro": a.micro = true
        case "--check-native": a.checkNative = true
        case "--native": a.native = true
        case "--profile": a.profile = true
        case "--gap": a.gap = Int(it.next() ?? "") ?? a.gap
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
            ids: [Int], readback: Bool = true) -> (Double, [Int], MLXArray) {
    var state = RWKVState(cfg: cfg)
    var picked: [Int] = []
    picked.reserveCapacity(ids.count)
    var last = MLXArray.zeros([cfg.vocab])
    let t0 = Date()
    for id in ids {
        let logits = model.step(id, state: &state)
        if readback {
            // argMax + item() -- это ЧТЕНИЕ ОБРАТНО В CPU на каждом
            // шаге: полная синхронизация конвейера плюс латентность
            // CPU<->GPU (по трейсу ~2 мс). Сэмплеру токен нужен, так
            // что расход настоящий, но к скорости МОДЕЛИ он не
            // относится, и мерить их вместе -- значит приписывать
            // модели чужое. Отсюда флаг.
            let next = argMax(logits, axis: -1)
            eval(next)
            state.eval()
            picked.append(next.item(Int.self))
        } else {
            state.eval()
        }
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

// ── Микрозамер одной проекции ───────────────────────────────────────────
//
// Отвечает ровно на один вопрос: даст ли переход на родной
// `quantizedMM` то, что обещает арифметика трафика. Прежде чем писать
// репак sb6 в контейнер MLX (а это выверенная битовая работа), надо
// знать, что выигрыш есть.
//
// Три варианта на форму, чередованием:
//   deq+mm   — что делает `baseProj` сейчас: развернуть и умножить
//   dense    — плотный bf16, верхняя граница по скорости сегодня
//   native   — `quantizedMM` по нативно упакованному весу
//
// ВАЖНО про native: вес для него получен `quantized(dense)`, то есть
// ЧИСЛА в нём пересчитаны по min/max блока и НЕ соответствуют нашей
// калибровке. Для скорости это неважно — раскладка, размеры и ядро те
// же самые, — но использовать этот путь как рабочий нельзя, и мерить
// им качество нельзя тем более. Это проба механизма, а не реализация.
func micro(sidecar: RwkvqSidecar, rounds: Int, iters: Int) {
    print("\n── микрозамер проекций (вектор [1, IN], \(rounds)x\(iters)) ──")
    print("native: числа пересчитаны quantized(dense) — проба СКОРОСТИ, не путь")

    // по одному представителю на форму, крупные первыми
    var seen = Set<[Int]>()
    var keys: [String] = []
    for k in sidecar.keys.sorted() {
        let sh = sidecar.tensors[k]!.shape
        if seen.insert(sh).inserted { keys.append(k) }
    }
    keys.sort { (sidecar.tensors[$0]!.shape.reduce(1, *))
                > (sidecar.tensors[$1]!.shape.reduce(1, *)) }

    // без String(format:) с %s: он ждёт C-строку, а Swift-строка туда
    // приводится мусорным указателем — падение с SIGSEGV, не ошибка формата
    func pad(_ s: String, _ n: Int) -> String {
        s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
    }
    print(pad("форма", 16) + pad("deq+mm", 10) + pad("dense", 10)
          + pad("native", 10) + "native даёт")
    for key in keys.prefix(5) {
        let info = sidecar.tensors[key]!
        let (OUT, IN) = (info.outFeatures, info.inFeatures)
        guard let w = try? sidecar.dequantize(key, dtype: .bfloat16) else { continue }
        eval(w)
        // MLX. обязательно: на верхнем уровне main.swift лежит глобальная
        // `quantized` (модель), и она перекрывает функцию MLX
        let (wq, sc, bi) = MLX.quantized(w, groupSize: 32, bits: 6)
        eval(wq, sc)
        let x = MLXRandom.normal([1, IN]).asType(.bfloat16)
        eval(x)

        func time(_ body: () -> MLXArray) -> Double {
            _ = body(); eval(body())                      // прогрев формы
            let t0 = Date()
            for _ in 0 ..< iters { eval(body()) }
            return Date().timeIntervalSince(t0) * 1e3 / Double(iters)
        }

        var a: [Double] = [], b: [Double] = [], c: [Double] = []
        for _ in 0 ..< rounds {
            a.append(time { matmul(x, (try! sidecar.dequantize(key, dtype: .bfloat16))
                                    .transposed()) })
            b.append(time { matmul(x, w.transposed()) })
            c.append(time { quantizedMM(x, wq, scales: sc, biases: bi,
                                        transpose: true, groupSize: 32, bits: 6) })
        }
        let (ma, mb, mc) = (median(a), median(b), median(c))
        func f(_ v: Double) -> String { String(format: "%9.3f ", v) }
        print(pad("\(OUT)x\(IN)", 16) + f(ma) + f(mb) + f(mc)
              + String(format: "  %.2fx к deq+mm, %.2fx к dense", ma / mc, mb / mc))
    }
}

/// Сверка перекладки в родной контейнер с питоновским эталоном.
///
/// Эталон снят с КАНОНИЧЕСКОЙ дисковой раскладки (.rwkvq), а здесь
/// перекладка идёт из K3-интерлива сайдкара. Это два разных источника
/// одних и тех же чисел, поэтому совпадение бит-в-бит — содержательное
/// утверждение, а не тавтология.
func checkNative(sidecar: RwkvqSidecar, refPath: String) -> Int {
    print("── сверка nativeAffine с эталоном ──")
    guard let ref = try? loadArrays(url: URL(fileURLWithPath: expand(refPath))) else {
        print("нет эталона \(refPath) — соберите его:")
        print("  python rwkv-quant/tests/test_mlx_affine_repack.py "
              + "<model.rwkvq> --dump \(refPath)")
        return 2
    }
    let keys = Set(ref.keys.compactMap { k -> String? in
        guard let r = k.range(of: "::wq", options: .backwards) else { return nil }
        return String(k[k.startIndex ..< r.lowerBound])
    }).sorted()

    var bad = 0
    for key in keys {
        guard sidecar.contains(key), let nat = try? sidecar.nativeAffine(key) else {
            print("  !! \(key): нет в сайдкаре"); bad += 1; continue
        }
        let rows = ref["\(key)::wq"]!.shape[0]
        func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
            guard a.shape == b.shape else { return false }
            let eq = (a .== b).all()
            eval(eq)
            return eq.item(Bool.self)
        }
        let okW = same(nat.wq[0 ..< rows], ref["\(key)::wq"]!)
        let okS = same(nat.scales[0 ..< rows], ref["\(key)::scales"]!)
        let okB = same(nat.biases[0 ..< rows], ref["\(key)::biases"]!)
        let okBits = nat.bits == ref["\(key)::bits"]!.item(Int32.self)

        // и главное: то, что из этого прочитает само ядро, против
        // плотного эталона
        let dense = ref["\(key)::dense"]!.asType(.float32)
        let got = dequantized(nat.wq[0 ..< rows], scales: nat.scales[0 ..< rows],
                              biases: nat.biases[0 ..< rows],
                              groupSize: 32, bits: nat.bits).asType(.float32)
        // Допуск — ulp BF16, а не fp16: эталон в фикстуре хранится в
        // bf16 (в нём же считает и модель), а bf16 несёт 8 бит мантиссы,
        // то есть на весах порядка 0.05 шаг сетки уже 2e-4. Первая
        // версия сверяла с порогом 6.3e-7 и показывала расхождение
        // 2.4e-4 на ВСЕХ тензорах при бит-в-бит совпавших wq/scales/
        // biases — то есть ловила формат хранения эталона, а не ошибку.
        // Плюс абсолютный пол на вырожденные блоки (scale=0 против
        // 1e-8, см. RwkvqNative.swift).
        let tol = maximum(abs(dense) * Float(pow(2.0, -8.0)), 6.3e-7)
        let over = (abs(got - dense) - tol).max()
        eval(over)
        let maxd = abs(got - dense).max()
        eval(maxd)
        let okV = over.item(Float.self) <= 0

        let ok = okW && okS && okB && okBits && okV
        if !ok { bad += 1 }
        print("  \(ok ? "ok  " : "FAIL") \(key) bits=\(nat.bits) "
              + "[\(nat.outFeatures)x\(nat.inFeatures)]"
              + (ok ? "  max|Δ| \(String(format: "%.2e", maxd.item(Float.self)))"
                    : "  wq/scales/biases/bits/значения = "
                      + "\(okW)/\(okS)/\(okB)/\(okBits)/\(okV), "
                      + "max|Δ| \(maxd.item(Float.self))"))
    }
    print(bad == 0 ? "\nСВЕРКА ПРОЙДЕНА" : "\nСВЕРКА ПРОВАЛЕНА: \(bad)")
    return bad == 0 ? 0 : 1
}

/// Разложение шага: проекции против всего остального.
///
/// Вопрос, на который отвечает: 30.0 мс/ток при поле по памяти 19.4 --
/// это недобор на проекциях или расход вне их? Гипотеза «упростить
/// математику» имеет смысл только во втором случае; в первом упрощать
/// нечего, потому что проекции упираются в чтение весов, а не в счёт
/// (в rwkv-quant это отдельно закрыто руфлайн-пробой: декодный ALU
/// бесплатен).
///
/// Меряется «только проекции»: те же запуски `quantizedMM` в том же
/// количестве и порядке, но без WKV, нормировок, LoRA-веток,
/// token-shift и сэмплера. Разница с полным шагом и есть всё
/// остальное.
///
/// ЧЕСТНАЯ ОГОВОРКА: у проекций здесь фиктивный вход, поэтому
/// зависимостей между слоями нет и MLX волен ставить их в очередь
/// свободнее, чем в настоящем шаге. То есть «только проекции» -- это
/// НИЖНЯЯ оценка их вклада, а «всё остальное» -- ВЕРХНЯЯ. Для ответа
/// «есть ли вне проекций что оптимизировать» этого достаточно, для
/// точного бюджета -- нет.
func profileStep(_ model: X070Backbone, cfg: X070Config, ids: [Int],
                 rounds: Int) {
    print("\n── разложение шага ──")
    let nat = model.rwkvqNative
    guard !nat.isEmpty else {
        print("нет родного контейнера (нужен --native)")
        return
    }
    // порядок как в шаге: r/k/v/o на слой, затем cmix key/value, затем голова
    var chain: [RwkvqNativeWeight] = []
    for l in 0 ..< cfg.nLayer {
        for n in ["r_proj", "k_proj", "v_proj", "o_proj"] {
            if let w = nat["blocks.\(l).tmix.\(n).weight"] { chain.append(w) }
        }
        for n in ["key", "value"] {
            if let w = nat["blocks.\(l).cmix.\(n).weight"] { chain.append(w) }
        }
    }
    if let h = nat["head.weight"] { chain.append(h) }
    print("проекций в цепочке: \(chain.count)")

    // цена чтения токена обратно в CPU: тот же шаг без argMax/item()
    var noReadMs: [Double] = []
    for _ in 0 ..< rounds {
        let (t, _, _) = decode(model, cfg: cfg, ids: ids, readback: false)
        noReadMs.append(t * 1e3 / Double(ids.count))
    }

    var projMs: [Double] = []
    for _ in 0 ..< rounds {
        let t0 = Date()
        for _ in 0 ..< ids.count {
            // Накопление ОБЯЗАТЕЛЬНО. MLX ленив, и если результат
            // проекции никуда не идёт, граф выбрасывает её целиком:
            // первая версия писала `last = w(x)` в цикле и eval'ила
            // только последнюю -- получалось 1.73 мс на 193 проекции,
            // при том что одно чтение 1855 МБ весов стоит 19.4 мс.
            // Замер измерял ровно одну голову.
            var acc = MLXArray(Float(0))
            for w in chain {
                let x = MLXArray.zeros([1, w.inFeatures], dtype: .float16)
                acc = acc + w(x).sum().asType(.float32)
            }
            eval(acc)
        }
        projMs.append(Date().timeIntervalSince(t0) * 1e3 / Double(ids.count))
    }

    var fullMs: [Double] = []
    for _ in 0 ..< rounds {
        let (t, _, _) = decode(model, cfg: cfg, ids: ids)
        fullMs.append(t * 1e3 / Double(ids.count))
    }

    let p = median(projMs), f = median(fullMs), nr = median(noReadMs)
    print(String(format: "только проекции   %6.2f мс/ток", p))
    print(String(format: "шаг без readback  %6.2f мс/ток", nr))
    print(String(format: "полный шаг        %6.2f мс/ток", f))
    print(String(format: "  из них readback %6.2f мс/ток (%.0f%%)",
                 f - nr, 100 * (f - nr) / f))
    print(String(format: "  вне проекций    %6.2f мс/ток (%.0f%% шага)",
                 nr - p, 100 * (nr - p) / f))
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

if args.checkNative {
    exit(Int32(checkNative(sidecar: try RwkvqSidecar(path: expand(args.sidecar)),
                           refPath: args.nativeRef)))
}

if args.micro {
    // модель не нужна: меряются отдельные проекции по сайдкару
    micro(sidecar: try RwkvqSidecar(path: expand(args.sidecar)),
          rounds: args.rounds, iters: 20)
    exit(0)
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
dense?.castWKVOutputToComputeDType = args.castWKV
quantized?.castWKVOutputToComputeDType = args.castWKV
print("castWKVOutputToComputeDType = \(args.castWKV)"
      + (args.castWKV ? "" : "  (в модели это умолчание, но оно стоит ×2.8)"))

if let q = quantized {
    let sidecar = try RwkvqSidecar(path: expand(args.sidecar))
    let info = q.attachRwkvq(
        sidecar,
        options: X070Backbone.RwkvqAttachOptions(quantizeCmix: args.quantizeCmix,
                                                 quantizeHead: args.quantizeHead,
                                                 useNativeKernel: args.native))
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

if args.gap > 0 {
    print("пауза \(args.gap) мс перед измеряемым окном (маркер для трейса)")
    fflush(stdout)
    Thread.sleep(forTimeInterval: Double(args.gap) / 1000)
}

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

if args.gap > 0 {
    print("пауза \(args.gap) мс после измеряемого окна")
    fflush(stdout)
    Thread.sleep(forTimeInterval: Double(args.gap) / 1000)
}

if args.profile, let q = quantized {
    profileStep(q, cfg: cfg, ids: ids, rounds: args.rounds)
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
// clearCache обязателен перед замером памяти: MLX не возвращает
// освобождённые буферы системе, а держит их в своём пуле, и
// phys_footprint их видит. Без этого «освободили сайдкар» и «не
// освободили» дают одинаковые 6.9 ГБ.
let footBefore = processFootprintGB()
MLX.GPU.clearCache()
print(String(format: "footprint в конце %.2f ГБ (до clearCache %.2f)",
             processFootprintGB(), footBefore))
if let q = quantized, !q.rwkvqNative.isEmpty {
    let bytes = q.rwkvqNative.values.reduce(0) { $0 + $1.bytes }
    print(String(format: "родной контейнер: %d весов, %.1f МБ",
                 q.rwkvqNative.count, Double(bytes) / 1e6))
}
