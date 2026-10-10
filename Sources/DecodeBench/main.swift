import Foundation
import MLX
import MLXRandom
import RWKVGen
import RWKVQuant
import RWKVKernel

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
//        --model ~/Develop/SwiftRWKV/.testdata/world_0.1b_x070.safetensors \
//        --sidecar ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx
// ───────────────────────────────────────────────────────────────────────

struct Args {
    var model = "~/Develop/SwiftRWKV/.testdata/world_0.1b_x070.safetensors"
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
    var fuseParity = false
    var noFuseLerp = false
    var manifest = false
    var nativeRef = "~/Develop/SwiftRWKV/.testdata/mlx_affine_ref.safetensors"
    /// A/B нормы против прежнего пути порта на ОДНОМ .rwkvq, чередованием:
    /// см. `rwkvqNormAB`. Плотная модель и сайдкар в этом режиме не нужны.
    var rwkvq: String? = nil
    /// Прежний путь базы (деквант в тип входа, без приведения) в обычном
    /// режиме замера. По умолчанию ВКЛЮЧЁН: под ним сняты все числа
    /// docs/Inference.md, и замер «как раньше» обязан мерить то же самое.
    var legacyBase = true
    /// В режиме --rwkvq: плечо «норма» подаёт в родное ядро fp16 вместо fp32.
    var nativeIn16 = false
    /// Размен «вход родного ядра sb6: fp32 против fp16» на одном файле:
    /// точность (KL к исходной модели и к точному счёту тех же весов, по
    /// окнам, парный бутстрэп) и скорость декода — см. `nativeInStudy`.
    var nativeInStudy: String? = nil
    /// Окна токенов [N, T] (safetensors, ключ "ids") для --native-in-study.
    var windows = "/tmp/swr_0910/windows.safetensors"
    /// Плотная исходная модель (safetensors, имена x070) — «истина» замера;
    /// считается в fp32. Не задана — меряется только шум счёта.
    var truth: String? = nil
    /// Скорость декода при двух положениях одного переключателя на ОДНОЙ
    /// модели, чередованием: "native-in" (вход родного ядра sb6 fp32 / fp16)
    /// либо "sym-gemv" (декод sym ядром GEMV / деквант + матмул).
    var knob: String? = nil
    var knobPath: String? = nil
    /// Замер «как в приложении» на одном .rwkvq: загрузка, префилл T=512,
    /// декод — скорость и память. Протокол общий с эталоном (Metal-бэкенд
    /// rwkv-quant, /tmp/swr_1010/ref_bench.py): те же окна текста.
    var appBench: String? = nil
    /// --app-bench: собрать sb6 без родного контейнера MLX (K3 + ядро GEMV).
    var noNative = false
    /// Сколько из шага занимают сами проекции: все GEMV одного токена
    /// (sym или sb6) на постоянных входах под одним eval.
    var gemvMicro: String? = nil
    /// Разложение шага: сколько уходит на построение графа (вызов step на
    /// CPU) и сколько на исполнение (eval), собранным и покомандным путём.
    var stepProfile: String? = nil
    /// Загрузка .rwkvq: время, память MLX и footprint процесса.
    var loadProfile: String? = nil
    /// Префилл T=512 по статьям: деквант, матмулы, WKV, остальное.
    var prefillProfile: String? = nil
    /// Короткий параллельный проход: ядра GEMV против «деквант + матмул» по длинам.
    var shortBench: String? = nil
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
        case "--fuse-parity": a.fuseParity = true
        case "--no-fuse-lerp": a.noFuseLerp = true
        case "--manifest": a.manifest = true
        case "--rwkvq": a.rwkvq = it.next()
        case "--norm-base": a.legacyBase = false
        case "--native-in16": a.nativeIn16 = true
        case "--native-in-study": a.nativeInStudy = it.next()
        case "--windows": a.windows = it.next() ?? a.windows
        case "--truth": a.truth = it.next()
        case "--knob-ab": a.knob = it.next(); a.knobPath = it.next()
        case "--app-bench": a.appBench = it.next()
        case "--no-native": a.noNative = true
        case "--gemv-micro": a.gemvMicro = it.next()
        case "--step-profile": a.stepProfile = it.next()
        case "--load-profile": a.loadProfile = it.next()
        case "--no-read-ahead": RwkvqContainer.readAheadEnabled = false
        case "--prefill-profile": a.prefillProfile = it.next()
        case "--short-bench": a.shortBench = it.next()
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

/// Жадный декод БЕЗ чтения токена в CPU на каждом шаге.
///
/// Выбранный токен остаётся `MLXArray` и уходит в следующий шаг как
/// есть; в CPU он забирается раз в `readEvery` шагов. Это и есть
/// конвейеризация: пока CPU разбирает пачку токенов, GPU уже считает
/// следующие. Синхронизация состояния (`state.eval()`) остаётся --
/// без неё MLX копил бы граф на всю длину генерации.
///
/// Считается ЖАДНО, а не teacher-forced: смысл именно в том, что вход
/// следующего шага берётся из выхода предыдущего, не покидая GPU. Для
/// замера времени это неважно (веса читаются те же), а иначе цепочка
/// зависимостей была бы другой, чем в настоящей генерации.
func decodePipelined(_ model: X070Backbone, cfg: X070Config,
                     first: Int, steps: Int, readEvery: Int) -> (Double, [Int]) {
    var state = RWKVState(cfg: cfg)
    var id = MLXArray([Int32(first)])
    var pending: [MLXArray] = []
    var picked: [Int] = []
    let t0 = Date()
    for i in 0 ..< steps {
        let logits = model.step(id, state: &state)
        id = argMax(logits, axis: -1).reshaped([1])
        state.eval()
        pending.append(id)
        if pending.count == readEvery || i == steps - 1 {
            eval(pending)
            picked.append(contentsOf: pending.map { $0.item(Int.self) })
            pending.removeAll(keepingCapacity: true)
        }
    }
    return (Date().timeIntervalSince(t0), picked)
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

    // ЧЕРЕДОВАНИЕ ВАРИАНТОВ ВНУТРИ РАУНДА, а не блоками подряд.
    //
    // Первая версия мерила каждый вариант своим циклом: сначала все
    // раунды «без readback», потом все «полный шаг». Между блоками
    // машина успевала нагреться, и разница блоков приписывалась коду.
    // Так родилась цифра «readback стоит 2.75 мс/ток (9%)», которая в
    // следующем прогоне обернулась −0.17 мс. Это ровно закон 1 из
    // rwkv-quant, и он про то же самое: сравнивать можно только
    // чередованием в одном процессе.
    var noReadMs: [Double] = []
    var projMs: [Double] = []
    var fullMs: [Double] = []
    var pipeMs: [[Double]] = Array(repeating: [], count: 3)
    let everies = [1, 4, 16]

    for _ in 0 ..< rounds {
        let (tn, _, _) = decode(model, cfg: cfg, ids: ids, readback: false)
        noReadMs.append(tn * 1e3 / Double(ids.count))

        let (tf, _, _) = decode(model, cfg: cfg, ids: ids)
        fullMs.append(tf * 1e3 / Double(ids.count))

        for (i, every) in everies.enumerated() {
            let (t, _) = decodePipelined(model, cfg: cfg, first: ids[0],
                                         steps: ids.count, readEvery: every)
            pipeMs[i].append(t * 1e3 / Double(ids.count))
        }

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

    let p = median(projMs), f = median(fullMs), nr = median(noReadMs)
    print(String(format: "только проекции   %6.2f мс/ток", p))
    print(String(format: "шаг без readback  %6.2f мс/ток", nr))
    print(String(format: "полный шаг        %6.2f мс/ток", f))
    print(String(format: "  из них readback %6.2f мс/ток (%.0f%%)",
                 f - nr, 100 * (f - nr) / f))
    print(String(format: "  вне проекций    %6.2f мс/ток (%.0f%% шага)",
                 nr - p, 100 * (nr - p) / f))
    print("\nконвейеризация readback (жадный декод, токен не уходит в CPU):")
    for (i, every) in everies.enumerated() {
        let m = median(pipeMs[i])
        print(String(format: "  читать раз в %2d шагов  %6.2f мс/ток "
                     + "(%.1f ток/с), к полному шагу %+.2f мс",
                     every, m, 1e3 / m, m - f))
    }
    print(String(format: "  для сравнения: без readback вовсе %6.2f мс/ток "
                 + "-- нижняя граница", nr))
}

/// Фьюз лерпов обязан быть тождественным: арифметика поэлементная, от
/// группировки не зависит. Проверяется на ОДНОЙ модели переключением
/// флага на живом объекте -- две сборки сравнивали бы что угодно.
func fuseParity(_ model: X070Backbone, cfg: X070Config, ids: [Int]) -> Int {
    print("── тождественность фьюза лерпов ──")
    model.fuseLerp = false
    let (_, offIds, offLogits) = decode(model, cfg: cfg, ids: ids)
    model.fuseLerp = true
    let (_, onIds, onLogits) = decode(model, cfg: cfg, ids: ids)

    let sameTop1 = offIds == onIds
    let d = (offLogits.asType(.float32) - onLogits.asType(.float32))
    let maxd = abs(d).max()
    eval(maxd)
    let bitExact = maxd.item(Float.self) == 0
    print("  top-1 совпал: \(sameTop1 ? "да" : "НЕТ") (\(offIds.count) позиций)")
    print("  логиты: max|Δ| = \(maxd.item(Float.self))"
          + (bitExact ? "  (бит-в-бит)" : "  -- ФЬЮЗ СДВИНУЛ ЧИСЛА"))
    return (sameTop1 && bitExact) ? 0 : 1
}

// ── Прогон ──────────────────────────────────────────────────────────────

let args = parseArgs()

/// Норма (.rwkvq: параметры fp16, база fp16) против прежнего пути порта
/// (параметры bf16, деквант в тип входа) на одном файле, чередованием в
/// одном процессе. `--only norm|legacy` — для моделей, две копии которых в
/// память не помещаются; тогда чередование делается запусками.
func rwkvqNormAB(_ path: String, _ args: Args) throws {
    func load(_ norm: Bool) throws -> (X070Backbone, Double) {
        let before = processFootprintGB()
        let bb = norm
            ? try X070Backbone.load(fromRwkvq: expand(path), useNativeKernel: true)
            : try X070Backbone.load(fromRwkvq: expand(path), paramDType: .bfloat16,
                                    baseDType: nil, useNativeKernel: true)
        bb.castWKVOutputToComputeDType = args.castWKV
        bb.fuseLerp = !args.noFuseLerp
        if norm, args.nativeIn16 { bb.rwkvqNativeInputDType = .float16 }
        GPU.clearCache()
        let grown = processFootprintGB() - before
        print(String(format: "  после загрузки (%@): footprint +%.2f ГБ; MLX активная %.2f, кеш %.2f, пик %.2f ГБ",
                     norm ? "норма" : "прежний", grown, Double(GPU.activeMemory) / 1e9,
                     Double(GPU.cacheMemory) / 1e9, Double(GPU.peakMemory) / 1e9))
        return (bb, grown)
    }
    let wantNorm = args.only != "legacy", wantLegacy = args.only != "norm"
    let norm = wantNorm ? try load(true) : nil
    let legacy = wantLegacy ? try load(false) : nil
    let cfg = (norm ?? legacy)!.0.cfg
    print("L=\(cfg.nLayer) D=\(cfg.nEmbd) V=\(cfg.vocab); castWKV = \(args.castWKV); "
          + "вход родного ядра в норме: \(args.nativeIn16 ? "fp16" : "fp32")")
    if let n = norm { print(String(format: "footprint загрузки, норма: +%.2f ГБ", n.1)) }
    if let l = legacy { print(String(format: "footprint загрузки, прежний путь: +%.2f ГБ", l.1)) }

    let (ids, _) = inputIds(args.steps, vocab: cfg.vocab, vocabPath: args.vocab)
    let (warm, _) = inputIds(args.warmup, vocab: cfg.vocab, vocabPath: args.vocab)
    if let n = norm { _ = decode(n.0, cfg: cfg, ids: warm) }
    if let l = legacy { _ = decode(l.0, cfg: cfg, ids: warm) }
    var msN: [Double] = [], msL: [Double] = []
    var topN: [Int] = [], topL: [Int] = []
    for r in 0 ..< args.rounds {
        var line = String(format: "  раунд %d:", r)
        // порядок плеч чередуется: кто идёт вторым, тот идёт по нагретому
        let order = r % 2 == 0 ? [true, false] : [false, true]
        for isNorm in order {
            guard let m = isNorm ? norm : legacy else { continue }
            let (t, out, _) = decode(m.0, cfg: cfg, ids: ids)
            let ms = t * 1e3 / Double(args.steps)
            if isNorm { msN.append(ms); topN = out } else { msL.append(ms); topL = out }
            line += String(format: "  %@ %.2f мс/ток", isNorm ? "норма" : "прежний", ms)
        }
        print(line); fflush(stdout)
    }
    if !msN.isEmpty { print(String(format: "норма:        медиана %.2f мс/ток = %.1f ток/с", median(msN), 1000 / median(msN))) }
    if !msL.isEmpty { print(String(format: "прежний путь: медиана %.2f мс/ток = %.1f ток/с", median(msL), 1000 / median(msL))) }
    if !msN.isEmpty, !msL.isEmpty {
        print(String(format: "норма / прежний = %.3f", median(msN) / median(msL)))
        let agree = zip(topN, topL).filter { $0 == $1 }.count
        print("top-1 при общем входе: совпало \(agree) из \(topN.count)")
    }
    print(String(format: "footprint в конце: %.2f ГБ; MLX активная %.2f, кеш %.2f, пик %.2f ГБ",
                 processFootprintGB(), Double(GPU.activeMemory) / 1e9,
                 Double(GPU.cacheMemory) / 1e9, Double(GPU.peakMemory) / 1e9))
}

if let path = args.rwkvq {
    try rwkvqNormAB(path, args)
    exit(0)
}

/// Вход родного ядра sb6 (`quantizedMM`): fp32 (как в rwkv-metal) против
/// fp16 — на ОДНОЙ модели, переключением `rwkvqNativeInputDType`.
///
/// Точность — прибором статьи rwkv-quant: KL(истина ‖ плечо), истина —
/// исходный чекпоинт (веса bf16, счёт fp32), по окнам текста; разница плеч —
/// парным бутстрэпом по окнам. Вторая мера, доступная без исходной модели:
/// KL к точному счёту ТЕХ ЖЕ квантованных весов (fp32-параметры, fp32-вход) —
/// это чистый шум счёта, без ошибки квантования.
func nativeInStudy(_ path: String, _ args: Args) throws {
    let win = try loadArrays(url: URL(fileURLWithPath: expand(args.windows)))["ids"]!
    let N = win.shape[0], T = win.shape[1]
    print("окон \(N) по \(T) токенов; файл \(path)")

    func logProbs(_ bb: X070Backbone, _ i: Int) -> MLXArray {
        let l = bb(win[i ..< (i + 1)])[0].asType(.float32)
        let lp = l - logSumExp(l, axis: -1, keepDims: true)
        eval(lp)
        return lp
    }
    func kl(_ lp: MLXArray, _ lq: MLXArray) -> Double {
        Double((exp(lp) * (lp - lq)).sum(axis: -1).mean().item(Float.self))
    }
    func mean(_ x: [Double]) -> Double { x.reduce(0, +) / Double(x.count) }
    /// Δ% среднего b к среднему a, парный бутстрэп по окнам, 95%.
    func deltaPct(_ a: [Double], _ b: [Double]) -> (Double, Double, Double) {
        var state: UInt64 = 0x9E3779B97F4A7C15
        func next() -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 33) % UInt64(a.count))
        }
        var ds: [Double] = []
        for _ in 0 ..< 4000 {
            var sa = 0.0, sb = 0.0
            for _ in 0 ..< a.count { let i = next(); sa += a[i]; sb += b[i] }
            ds.append((sb / sa - 1) * 100)
        }
        ds.sort()
        return ((mean(b) / mean(a) - 1) * 100, ds[Int(0.025 * 4000)], ds[Int(0.975 * 4000)])
    }

    // 1. истина
    var truth: [MLXArray] = []
    if let tp = args.truth {
        let weights = try loadArrays(url: URL(fileURLWithPath: expand(tp)))
        let nLayer = weights.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer, nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        let t = X070Backbone(weights: weights, cfg: cfg, computeDType: .float32)
        for i in 0 ..< N { truth.append(logProbs(t, i)) }
        print("истина посчитана (\(tp), счёт fp32)")
    }
    GPU.clearCache()

    // 2. точный счёт тех же квантованных весов
    var exact: [MLXArray] = []
    do {
        let e = try X070Backbone.load(fromRwkvq: expand(path), paramDType: .float32)
        for i in 0 ..< N { exact.append(logProbs(e, i)) }
    }
    GPU.clearCache()

    // 3. норма, два плеча на одной модели
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    let nativeKeys = bb.rwkvqBackedKeys.filter { bb.hasNativeRepack($0) }.count
    print("L=\(bb.cfg.nLayer) D=\(bb.cfg.nEmbd); в родном контейнере \(nativeKeys) из \(bb.rwkvqBackedKeys.count) тензоров")
    var t32: [Double] = [], t16: [Double] = [], e32: [Double] = [], e16: [Double] = [], cross: [Double] = []
    var agree = 0, total = 0
    for i in 0 ..< N {
        bb.rwkvqNativeInputDType = .float32
        let a = logProbs(bb, i)
        bb.rwkvqNativeInputDType = .float16
        let b = logProbs(bb, i)
        if !truth.isEmpty { t32.append(kl(truth[i], a)); t16.append(kl(truth[i], b)) }
        e32.append(kl(exact[i], a)); e16.append(kl(exact[i], b)); cross.append(kl(a, b))
        agree += (argMax(a, axis: -1) .== argMax(b, axis: -1)).asType(.int32).sum().item(Int.self)
        total += T
    }
    if !truth.isEmpty {
        let (d, lo, hi) = deltaPct(t32, t16)
        let te = (0 ..< N).map { kl(truth[$0], exact[$0]) }
        print(String(format: "KL(истина‖·): точный счёт %.6f | вход fp32 %.6f | вход fp16 %.6f | fp16 к fp32: %+.3f%% [%+.3f; %+.3f]",
                     mean(te), mean(t32), mean(t16), d, lo, hi))
        let (d2, lo2, hi2) = deltaPct(te, t32)
        print(String(format: "              норма (вход fp32) к точному счёту: %+.3f%% [%+.3f; %+.3f]", d2, lo2, hi2))
    }
    print(String(format: "KL(точный счёт‖·): вход fp32 %.3e | вход fp16 %.3e;  KL(fp32‖fp16) %.3e;  top-1 fp32 == fp16: %d из %d",
                 mean(e32), mean(e16), mean(cross), agree, total))

    // 4. скорость декода, чередование плеч на одной модели
    let (ids, _) = inputIds(args.steps, vocab: bb.cfg.vocab, vocabPath: args.vocab)
    let (warm, _) = inputIds(args.warmup, vocab: bb.cfg.vocab, vocabPath: args.vocab)
    bb.castWKVOutputToComputeDType = false
    for t in [DType.float32, .float16] { bb.rwkvqNativeInputDType = t; _ = decode(bb, cfg: bb.cfg, ids: warm) }
    var m32: [Double] = [], m16: [Double] = []
    for r in 0 ..< args.rounds {
        for t in (r % 2 == 0 ? [DType.float32, .float16] : [.float16, .float32]) {
            bb.rwkvqNativeInputDType = t
            let (dt, _, _) = decode(bb, cfg: bb.cfg, ids: ids)
            if t == .float32 { m32.append(dt * 1e3 / Double(args.steps)) } else { m16.append(dt * 1e3 / Double(args.steps)) }
        }
    }
    print(String(format: "декод, мс/ток (медиана, %d раундов по %d): вход fp32 %.2f | вход fp16 %.2f | fp16 / fp32 = %.3f",
                 args.rounds, args.steps, median(m32), median(m16), median(m16) / median(m32)))
}

/// Скорость декода при двух положениях переключателя, одна модель.
func knobSpeedAB(_ knob: String, _ path: String, _ args: Args) throws {
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    bb.castWKVOutputToComputeDType = false
    let labels: (String, String)
    let set: (Bool) -> Void
    switch knob {
    case "native-in":
        labels = ("вход fp32", "вход fp16")
        set = { bb.rwkvqNativeInputDType = $0 ? .float32 : .float16 }
    case "sym-gemv":
        labels = ("деквант+матмул", "ядро GEMV")
        set = { bb.rwkvqSymDecodeKernel = !$0 }
    case "compiled":
        labels = ("покомандный шаг", "собранный шаг")
        set = { bb.useCompiledStep = !$0 }
    case "step-kernels":
        labels = ("шаг WKV и хвост операциями", "шаг WKV и хвост ядрами")
        set = { bb.useStepKernels = !$0 }
    case "sb6-gemv":
        labels = ("sb6: деквант + матмул", "sb6: ядро GEMV")
        set = { bb.rwkvqSb6DecodeKernel = !$0 }
    case "fuse-rkv":
        labels = ("r/k/v по одной", "фьюз r/k/v")
        set = { bb.rwkvqSymFuseRKV = !$0 }
    default:
        print("неизвестный переключатель \(knob): native-in | sym-gemv | compiled | fuse-rkv"); exit(2)
    }
    print("L=\(bb.cfg.nLayer) D=\(bb.cfg.nEmbd); файл \(path); переключатель \(knob)")
    let (ids, _) = inputIds(args.steps, vocab: bb.cfg.vocab, vocabPath: args.vocab)
    let (warm, _) = inputIds(args.warmup, vocab: bb.cfg.vocab, vocabPath: args.vocab)
    for a in [true, false] { set(a); _ = decode(bb, cfg: bb.cfg, ids: warm) }
    var mA: [Double] = [], mB: [Double] = []
    var topA: [Int] = [], topB: [Int] = []
    for r in 0 ..< args.rounds {
        var line = String(format: "  раунд %d:", r)
        for a in (r % 2 == 0 ? [true, false] : [false, true]) {
            set(a)
            let (dt, out, _) = decode(bb, cfg: bb.cfg, ids: ids)
            let ms = dt * 1e3 / Double(args.steps)
            if a { mA.append(ms); topA = out } else { mB.append(ms); topB = out }
            line += String(format: "  %@ %.2f", a ? labels.0 : labels.1, ms)
        }
        print(line); fflush(stdout)
    }
    let agree = zip(topA, topB).filter { $0 == $1 }.count
    print(String(format: "декод, мс/ток (медиана, %d раундов по %d): %@ %.2f (%.1f ток/с) | %@ %.2f (%.1f ток/с) | отношение %.3f; top-1 совпало %d из %d",
                 args.rounds, args.steps, labels.0, median(mA), 1000 / median(mA),
                 labels.1, median(mB), 1000 / median(mB), median(mB) / median(mA), agree, topA.count))
    print(String(format: "footprint в конце: %.2f ГБ; MLX активная %.2f, пик %.2f ГБ",
                 processFootprintGB(), Double(GPU.activeMemory) / 1e9, Double(GPU.peakMemory) / 1e9))
}

/// Цена своего VJP квантованной базы (обратный проход bf16 при базе fp16)
/// против автограда в типе базы: параллельный прямой проход (T=256) и шаг
/// градиента по адаптерам (T=64, блочный чекпоинт), чередованием.
func wideBackwardAB(_ path: String, _ args: Args) throws {
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    LoRA.add(to: bb, spec: LoRASpec(rank: 16, alpha: 32))
    bb.useBlockCheckpoint = true
    print("L=\(bb.cfg.nLayer) D=\(bb.cfg.nEmbd); файл \(path); адаптеров \(bb.loraTargets.count)")
    let (ids, _) = inputIds(257, vocab: bb.cfg.vocab, vocabPath: args.vocab)
    let all = MLXArray(ids.map { Int32($0) }, [1, 257])
    let xf = all[0..., 0 ..< 256]
    let xg = all[0..., 0 ..< 64], yg = all[0..., 1 ..< 65]
    let keysA = bb.loraTargets
    func forward() -> Double {
        let t0 = Date(); let l = bb(xf); eval(l); return Date().timeIntervalSince(t0) * 1e3
    }
    func gradStep() -> Double {
        let params = keysA.map { bb.loraAdapter($0).0 } + keysA.map { bb.loraAdapter($0).1 }
        let t0 = Date()
        let g = grad({ (ps: [MLXArray]) -> [MLXArray] in
            for (i, k) in keysA.enumerated() { bb.setLoraAdapter(k, ps[i], ps[keysA.count + i]) }
            let l = bb(xg).asType(.float32)
            let lp = l - logSumExp(l, axis: -1, keepDims: true)
            return [-takeAlong(lp, yg.expandedDimensions(axis: -1), axis: -1).mean()]
        }, argumentNumbers: Array(0 ..< params.count))(params)
        eval(g)
        let dt = Date().timeIntervalSince(t0) * 1e3
        for (i, k) in keysA.enumerated() { bb.setLoraAdapter(k, params[i], params[keysA.count + i]) }
        return dt
    }
    for (name, work) in [("прямой проход T=256", forward), ("шаг градиента T=64", gradStep)] {
        // --only wide | through: одно плечо (для моделей, где шаг тяжёл)
        let arms: [Bool] = args.only == "wide" ? [true] : args.only == "through" ? [false] : [true, false]
        for wide in arms { bb.rwkvqBackwardDType = wide ? .bfloat16 : nil; _ = work() }
        print("  \(name): прогрев пройден"); fflush(stdout)
        var mW: [Double] = [], mT: [Double] = []
        for r in 0 ..< args.rounds {
            for wide in (r % 2 == 0 ? arms : arms.reversed()) {
                bb.rwkvqBackwardDType = wide ? .bfloat16 : nil
                let ms = work()
                if wide { mW.append(ms) } else { mT.append(ms) }
                print(String(format: "    %@ %.1f", wide ? "обратный bf16" : "автоград fp16", ms)); fflush(stdout)
            }
        }
        print(String(format: "%@, мс (медиана, %d раундов): автоград fp16 %.1f | обратный bf16 %.1f",
                     name, args.rounds, mT.isEmpty ? .nan : median(mT), mW.isEmpty ? .nan : median(mW)))
    }
    print(String(format: "footprint в конце: %.2f ГБ; MLX пик %.2f ГБ", processFootprintGB(), Double(GPU.peakMemory) / 1e9))
}

/// Загрузка, префилл T=512 и декод одной модели: скорость и память.
func appBench(_ path: String, _ args: Args) throws {
    let win = try loadArrays(url: URL(fileURLWithPath: expand(args.windows)))["ids"]!
    let pre = (win[0].asArray(Int32.self) + win[4].asArray(Int32.self)).map(Int.init)   // en + ru, 512
    let dec = win[8].asArray(Int32.self).prefix(72).map(Int.init)                       // sr
    let f0 = processFootprintGB()
    let t0 = Date()
    let bb = args.native ? try X070Backbone.load(fromRwkvq: expand(path), useNativeKernel: true)
        : args.noNative ? try X070Backbone.load(fromRwkvq: expand(path), useNativeKernel: false)
        : try X070Backbone.load(fromRwkvq: expand(path))
    GPU.clearCache()
    print("sb6: \(bb.rwkvqBackedKeys.contains { bb.hasNativeRepack($0) } ? "родной контейнер MLX" : "K3 (ядро GEMV на шаге)")")
    print(String(format: "загрузка %.1f с; footprint +%.2f ГБ; MLX активная %.2f ГБ, пик %.2f ГБ",
                 Date().timeIntervalSince(t0), processFootprintGB() - f0,
                 Double(GPU.activeMemory) / 1e9, Double(GPU.peakMemory) / 1e9))
    fflush(stdout)
    // префилл: первый вызов — сборка ядер, меряются следующие
    var ts: [Double] = []
    var state = RWKVState(cfg: bb.cfg)
    for i in 0 ..< 4 {
        if i == 1 { GPU.resetPeakMemory() }
        state = RWKVState(cfg: bb.cfg)
        let t = Date()
        let lg = bb.prefill(pre, state: &state)
        eval(lg); state.eval()
        ts.append(Date().timeIntervalSince(t) * 1e3)
    }
    let pm = median(Array(ts.dropFirst()))
    print(String(format: "префилл T=512: первый вызов %.0f мс; далее медиана %.1f мс = %.0f ток/с; MLX пик %.2f ГБ",
                 ts[0], pm, 512e3 / pm, Double(GPU.peakMemory) / 1e9))
    fflush(stdout)
    // декод: продолжение после префилла, teacher-forced, чтение argmax на каждом шаге
    func run(_ ids: ArraySlice<Int>) -> Double {
        let t = Date()
        for id in ids {
            let lg = bb.step(id, state: &state)
            let next = argMax(lg, axis: -1); eval(next); state.eval()
            _ = next.item(Int.self)
        }
        return Date().timeIntervalSince(t) * 1e3 / Double(ids.count)
    }
    _ = run(dec[0 ..< 8])
    GPU.resetPeakMemory()
    let ms = (0 ..< 5).map { _ in run(dec[8 ..< 72]) }
    print(String(format: "декод: медиана %.2f мс/ток = %.1f ток/с (раунды %@); MLX активная %.2f, пик %.2f ГБ; footprint %.2f ГБ",
                 median(ms), 1e3 / median(ms), ms.map { String(format: "%.2f", $0) }.joined(separator: " "),
                 Double(GPU.activeMemory) / 1e9, Double(GPU.peakMemory) / 1e9, processFootprintGB()))
    // конвейер против синхронного жадного декода, чередованием: свободная
    // генерация от одного и того же состояния, 64 токена на раунд
    let start = state
    let startLogits = bb.step(dec[71], state: &state); eval(startLogits); state.eval()
    let after = state
    func greedy(_ pipelined: Bool) -> (Double, [Int]) {
        var st = after
        var out: [Int] = []
        let t = Date()
        if pipelined {
            var pipe = GreedyDecodePipeline(logits: startLogits, state: st)
            for _ in 0 ..< 64 { out.append(pipe.next(bb, state: &st)) }
        } else {
            var lg = startLogits
            for _ in 0 ..< 64 {
                let id = argMax(lg, axis: -1).item(Int.self)
                out.append(id)
                lg = bb.step(id, state: &st)
            }
        }
        st.eval()
        return (Date().timeIntervalSince(t) * 1e3 / 64, out)
    }
    _ = start
    _ = greedy(true); _ = greedy(false)
    var mp: [Double] = [], msync: [Double] = []
    var same = true
    for r in 0 ..< 6 {
        for p in (r % 2 == 0 ? [true, false] : [false, true]) {
            let (t, toks) = greedy(p)
            if p { mp.append(t) } else { msync.append(t) }
            if r == 0, p { same = toks == greedy(false).1 }
        }
    }
    print(String(format: "жадная генерация: синхронно %.2f мс/ток | конвейером %.2f мс/ток (%.1f ток/с) | отношение %.3f; токены %@",
                 median(msync), median(mp), 1e3 / median(mp), median(mp) / median(msync), same ? "совпали" : "РАЗОШЛИСЬ"))

    // с температурой: CPU-сэмплер синхронно против GPU-сэмплера конвейером
    let sampling = SamplingConfig(temperature: 0.8, topP: 0.9, repetitionPenalty: 1.1, penaltyDecay: 0.99, seed: 1)
    func sampled(_ pipelined: Bool) -> Double {
        var st = after
        let t = Date()
        if pipelined {
            var pipe = DecodePipeline(logits: startLogits, state: st,
                                      sampler: GPUSampler(config: sampling, vocab: bb.cfg.vocab))
            for _ in 0 ..< 64 { _ = pipe.next(bb, state: &st) }
        } else {
            var cpu = Sampler(config: sampling)
            var lg = startLogits
            for _ in 0 ..< 64 { lg = bb.step(cpu.next(lg), state: &st) }
        }
        st.eval()
        return Date().timeIntervalSince(t) * 1e3 / 64
    }
    _ = sampled(true); _ = sampled(false)
    var sp: [Double] = [], ssync: [Double] = []
    for r in 0 ..< 6 {
        for p in (r % 2 == 0 ? [true, false] : [false, true]) {
            if p { sp.append(sampled(true)) } else { ssync.append(sampled(false)) }
        }
    }
    print(String(format: "генерация с температурой: CPU-сэмплер синхронно %.2f мс/ток | GPU-сэмплер конвейером %.2f мс/ток (%.1f ток/с) | отношение %.3f",
                 median(ssync), median(sp), 1e3 / median(sp), median(sp) / median(ssync)))
}

/// Все проекции одного токена ядрами GEMV, без остального шага.
func gemvMicro(_ path: String, _ args: Args) throws {
    let sc = try RwkvqSidecar.open(expand(path))
    var jobs: [(String, MLXArray)] = []
    for key in sc.keys {
        guard let info = sc.tensors[key], info.shape.count == 2, key != "emb.weight",
              info.kind == .sym || info.kind == .sb6 else { continue }
        let x = MLXRandom.normal([1, info.inFeatures]); eval(x)
        jobs.append((key, x))
    }
    func once() -> Double {
        let t = Date()
        let outs = jobs.compactMap { sc.symGEMV($0.0, $0.1) ?? sc.sb6GEMV($0.0, $0.1) }
        eval(outs)
        return Date().timeIntervalSince(t) * 1e3
    }
    for _ in 0 ..< 8 { _ = once() }
    let ms = (0 ..< 40).map { _ in once() }
    print(String(format: "проекций %d; все GEMV одного токена под одним eval: медиана %.2f мс (мин %.2f)",
                 jobs.count, median(ms), ms.min()!))
}

/// Построение графа шага против его исполнения.
func stepProfile(_ path: String, _ args: Args) throws {
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    let (ids, _) = inputIds(args.steps + args.warmup, vocab: bb.cfg.vocab, vocabPath: args.vocab)
    for compiled in [true, false, true, false] {
        bb.useCompiledStep = compiled
        var st = RWKVState(cfg: bb.cfg)
        var build: [Double] = [], run: [Double] = []
        for (i, id) in ids.enumerated() {
            let t0 = Date()
            let lg = bb.step(id, state: &st)
            let next = argMax(lg, axis: -1)
            let t1 = Date()
            eval(next); st.eval()
            _ = next.item(Int.self)
            let t2 = Date()
            if i >= args.warmup {
                build.append(t1.timeIntervalSince(t0) * 1e3); run.append(t2.timeIntervalSince(t1) * 1e3)
            }
        }
        print(String(format: "%@ шаг: построение графа %.2f мс, исполнение %.2f мс, всего %.2f мс/ток",
                     compiled ? "собранный  " : "покомандный", median(build), median(run), median(build) + median(run)))
    }
}

/// Загрузка `.rwkvq`: время, память MLX (активная, кеш, пик) и footprint
/// процесса. Пик MLX обязан равняться рабочей памяти (см.
/// `X070Backbone.loadStreaming`); холодный и прогретый кеш страниц
/// меряются отдельными процессами.
func loadProfile(_ path: String, _ args: Args) throws {
    func gb(_ x: Int) -> Double { Double(x) / 1e9 }
    let f0 = processFootprintGB()
    GPU.resetPeakMemory()
    let t = Date()
    let bb = try X070Backbone.load(fromRwkvq: expand(path), useNativeKernel: args.native)
    let dt = Date().timeIntervalSince(t)
    print(String(format: "загрузка %.2f с | MLX активная %.3f, кеш %.3f, пик %.3f ГБ | footprint +%.3f ГБ | emb %@",
                 dt, gb(GPU.activeMemory), gb(GPU.cacheMemory), gb(GPU.peakMemory),
                 processFootprintGB() - f0, bb.rwkvqEmbTable == nil ? "плотный" : "сжатый (строки выборкой)"))
    // освобождённое при загрузке Metal возвращает системе не сразу
    Thread.sleep(forTimeInterval: 1.0)
    print(String(format: "через 1 с: footprint +%.3f ГБ", processFootprintGB() - f0))
}

/// Префилл T=512 по статьям, тем же способом, что эталон
/// (/tmp/swr_1010_s2/ref_prefill_profile.py): каждая статья под своим eval.
func prefillProfile(_ path: String, _ args: Args) throws {
    let T = 512
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    let sc = try RwkvqSidecar.open(expand(path))
    GPU.clearCache()
    let win = try loadArrays(url: URL(fileURLWithPath: expand(args.windows)))["ids"]!
    let pre = (win[0].asArray(Int32.self) + win[4].asArray(Int32.self)).map(Int.init)
    func med(_ f: () -> Void) -> Double {
        f()
        return median((0 ..< 5).map { _ in let t = Date(); f(); return Date().timeIntervalSince(t) * 1e3 })
    }
    var body: [(String, MLXArray)] = []
    var xs: [Int: MLXArray] = [:]
    for key in sc.keys {
        guard let info = sc.tensors[key], info.shape.count == 2, info.isQuantizedBase,
              key != "emb.weight", key != "head.weight", RwkvqNaming.x070Key(forWorld: key) != nil else { continue }
        if xs[info.inFeatures] == nil {
            let x = MLXRandom.normal([T, info.inFeatures]); eval(x); xs[info.inFeatures] = x
        }
        body.append((key, xs[info.inFeatures]!))
    }
    let tDq = med { for (k, _) in body { eval(try! sc.dequantize(k, dtype: .float16)) } }
    let tLin = med {
        for (k, x) in body {
            let w = try! sc.dequantize(k, dtype: .float16)
            eval(matmul(x.asType(.float16), w.transposed()).asType(.float32))
        }
    }
    let tLin32 = med {
        for (k, x) in body {
            let w = try! sc.dequantize(k, dtype: .float32)
            eval(matmul(x, w.transposed()))
        }
    }
    print("слоёв \(bb.cfg.nLayer), D \(bb.cfg.nEmbd); квантованных проекций тела \(body.count)")
    print(String(format: "деквант всех матриц тела: %.1f мс | проекции тела целиком (деквант + fp16-матмул): %.1f мс | матмулы (разность): %.1f мс | то же в fp32: %.1f мс",
                 tDq, tLin, tLin - tDq, tLin32))
    fflush(stdout)
    let H = bb.cfg.nHead, S = bb.cfg.headSize, L = bb.cfg.nLayer
    var r = (0 ..< 6).map { _ in MLXRandom.normal([1, T, H, S]) * 0.1 }
    r[1] = exp(-abs(r[1]))
    eval(r)
    func wkvAll(_ whole: Bool) -> Double {
        med {
            for _ in 0 ..< L {
                let (o, h) = whole ? wkv7SequenceForward(r[0], r[1], r[2], r[3], r[4], r[5])
                                   : wkv7ForwardChunked(r[0], r[1], r[2], r[3], r[4], r[5])
                eval(o, h)
            }
        }
    }
    let tChunk = wkvAll(false), tWkv = wkvAll(true)
    print(String(format: "WKV на весь промпт x %d слоёв: одним запуском %.1f мс | нарезкой по 16 — %.1f мс", L, tWkv, tChunk))
    fflush(stdout)
    // целиком: три плеча чередованием в одном процессе
    func whole(_ seq: Bool, _ compiled: Bool) -> Double {
        wkv7WholeSequenceForward = seq
        bb.useCompiledPrefill = compiled
        defer { wkv7WholeSequenceForward = true; bb.useCompiledPrefill = false }
        var st = RWKVState(cfg: bb.cfg)
        let t = Date()
        let lg = bb.prefill(pre, state: &st); eval(lg); st.eval()
        return Date().timeIntervalSince(t) * 1e3
    }
    let arms: [(String, Bool, Bool, FrozenPassKernels)] = [
        ("нарезка, покомандно, без слитых", false, false, []),
        ("один запуск, без слитых", true, false, []),
        ("+ хвост", true, false, [.tail]),
        ("+ нормировки", true, false, [.layerNorm]),
        ("+ cmix", true, false, [.cmix]),
        ("все слитые (умолчание)", true, false, .all),
        ("все слитые, собранный", true, true, .all),
    ]
    var ts = [[Double]](repeating: [], count: arms.count)
    func arm(_ i: Int) -> Double {
        bb.frozenPassKernels = arms[i].3
        defer { bb.frozenPassKernels = .all }
        return whole(arms[i].1, arms[i].2)
    }
    for i in 0 ..< arms.count { _ = arm(i); _ = arm(i) }
    for round in 0 ..< 7 {
        for i in (round % 2 == 0 ? Array(0 ..< arms.count) : Array((0 ..< arms.count).reversed())) {
            ts[i].append(arm(i))
        }
    }
    print("префилл целиком (голова на 1 позиции), медиана 7:")
    for (a, t) in zip(arms, ts) { print(String(format: "   %@: %.1f мс", a.0, median(t))) }
    print(String(format: "остальное (умолчание - проекции тела - WKV): %.1f мс", median(ts[5]) - tLin - tWkv))
}

/// Короткий префилл: проекции ядрами GEMV (shortPassKernelRows) против
/// «деквант + fp16-матмул», чередованием в одном процессе, по длинам.
func shortBench(_ path: String, _ args: Args) throws {
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    let win = try loadArrays(url: URL(fileURLWithPath: expand(args.windows)))["ids"]!
    let ids = win[0].asArray(Int32.self).map(Int.init)
    func run(_ T: Int, _ kernels: Bool, recurrent: Bool = false) -> Double {
        bb.shortPassKernelRows = kernels ? 1 << 20 : 0
        defer { bb.shortPassKernelRows = 0 }
        var st = RWKVState(cfg: bb.cfg)
        let t = Date()
        let lg = recurrent ? bb.prefillRecurrent(Array(ids[0 ..< T]), state: &st)
                           : bb.prefill(Array(ids[0 ..< T]), state: &st)
        eval(lg); st.eval()
        return Date().timeIntervalSince(t) * 1e3
    }
    for T in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] {
        var a: [Double] = [], b: [Double] = [], c: [Double] = []
        for k in [true, false] { _ = run(T, k); _ = run(T, k) }
        _ = run(T, false, recurrent: true)
        for r in 0 ..< 7 {
            for k in (r % 2 == 0 ? [true, false] : [false, true]) {
                if k { a.append(run(T, true)) } else { b.append(run(T, false)) }
            }
            if T <= 16 { c.append(run(T, false, recurrent: true)) }
        }
        print(String(format: "T=%2d: ядра %.1f мс | матмул %.1f мс | отношение %.2f%@", T, median(a), median(b),
                     median(a) / median(b), c.isEmpty ? "" : String(format: " | шагами %.1f мс", median(c))))
        fflush(stdout)
    }
}

if let path = args.shortBench {
    try shortBench(path, args)
    exit(0)
}

if let path = args.loadProfile {
    try loadProfile(path, args)
    exit(0)
}

if let path = args.prefillProfile {
    try prefillProfile(path, args)
    exit(0)
}

if let path = args.stepProfile {
    try stepProfile(path, args)
    exit(0)
}

if let path = args.gemvMicro {
    try gemvMicro(path, args)
    exit(0)
}

if let path = args.appBench {
    try appBench(path, args)
    exit(0)
}

if let knob = args.knob, let path = args.knobPath {
    if knob == "wide-bwd" { try wideBackwardAB(path, args) } else { try knobSpeedAB(knob, path, args) }
    exit(0)
}

if let path = args.nativeInStudy {
    try nativeInStudy(path, args)
    exit(0)
}
let fm = FileManager.default
guard fm.fileExists(atPath: expand(args.model)) else {
    print("нет модели: \(args.model)"); exit(2)
}
guard fm.fileExists(atPath: expand(args.sidecar) + ".safetensors") else {
    print("нет сайдкара: \(args.sidecar).safetensors"); exit(2)
}

if args.manifest {
    let sc = try RwkvqSidecar(path: expand(args.sidecar))
    var byKind: [String: (n: Int, dims: Set<Int>)] = [:]
    for (_, i) in sc.tensors {
        var e = byKind[i.kind.rawValue] ?? (0, [])
        e.n += 1
        e.dims.insert(i.shape.count)
        byKind[i.kind.rawValue] = e
    }
    print("манифест: \(sc.tensors.count) записей, "
          + "буферов \(String(format: "%.1f", Double(sc.packedBytes) / 1e6)) МБ")
    for (k, v) in byKind.sorted(by: { $0.key < $1.key }) {
        print("  \(k): \(v.n), рангов формы \(v.dims.sorted())")
    }
    print("разобран без ошибок — записи не-sb6 больше не роняют загрузку")
    exit(0)
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
dense?.fuseLerp = !args.noFuseLerp
quantized?.fuseLerp = !args.noFuseLerp
print("fuseLerp = \(!args.noFuseLerp)")
print("castWKVOutputToComputeDType = \(args.castWKV)"
      + (args.castWKV ? "" : "  (в модели это умолчание, но оно стоит ×2.8)"))

if let q = quantized {
    let sidecar = try RwkvqSidecar(path: expand(args.sidecar))
    let info = q.attachRwkvq(
        sidecar,
        options: X070Backbone.RwkvqAttachOptions(quantizeCmix: args.quantizeCmix,
                                                 quantizeHead: args.quantizeHead,
                                                 useNativeKernel: args.native,
                                                 baseDType: args.legacyBase ? nil : .float16))
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

if args.fuseParity, let m = quantized ?? dense {
    let (w, _) = inputIds(args.steps, vocab: cfg.vocab, vocabPath: args.vocab)
    exit(Int32(fuseParity(m, cfg: cfg, ids: w)))
}

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
