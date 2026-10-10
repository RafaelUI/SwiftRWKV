import Foundation
import MLX
import MLXNN
import RWKVKernel
import RWKVQuant      // база .rwkvq (sb6)

// ───────────────────────────────────────────────────────────────────────
//  RWKV-7 "Goose" x070 — точный порт rwkv_metal/model/rwkv7_x070.py для
//  загрузки официальных World-весов. Функциональный forward (без nn.Module-
//  дерева) ради прозрачного паритета logits против Python-эталона.
//
//  Отличия от RWKVBackbone (RWKVTrain, версия rwkv7 from-scratch):
//   1. decay w: sigmoid(w0 + B(tanh(A(xw))))     (w0 = bias w_lora_B)
//   2. iclr a:  sigmoid(a0 + B(A(xa)))           БЕЗ tanh  (a0 = bias a_lora_B)
//   3. gate g:  B(sigmoid(A(xg)))                sigmoid ВНУТРИ, линейно наружу
//   4. ln_x:    GroupNorm по головам (eps=64e-5, pytorch_compatible)
//   5. порядок: WKV → ln_x → +bonus → *g
//   6. token-shift: нулевой паддинг t=0 в каждом блоке, БЕЗ межблочного переноса
//   7. cmix FFN размер D*4
// ───────────────────────────────────────────────────────────────────────

/// Конфиг x070-модели.
public struct X070Config: Sendable {
    public var nLayer: Int
    public var nEmbd: Int
    public var headSize: Int
    public var vocab: Int
    public var nHead: Int { nEmbd / headSize }

    public init(nLayer: Int, nEmbd: Int, headSize: Int = 64, vocab: Int) {
        self.nLayer = nLayer
        self.nEmbd = nEmbd
        self.headSize = headSize
        self.vocab = vocab
    }
}

// ─────────────────── Утилиты ───────────────────

// internal, а не private: ровно эти же три помощника нужны блоку, вынесенному
// в RWKVBlock.swift. Дублировать их там значило бы завести вторую копию
// арифметики, которая обязана совпадать побитово.
func l2norm(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}

func layerNorm(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
               eps: Float = 1e-5) -> MLXArray {
    // fp16 — норма квантованной базы (.rwkvq): там нормировка идёт штатным
    // ядром MLX, тем же, что у nn.LayerNorm в rwkv-metal. Ручная формула ниже
    // в fp16 копит среднее и дисперсию заметно грубее: на 0.1B это +1.1e-6 KL
    // к точному счёту при 3.7–4.7e-6 у самого rwkv-metal, а со штатным ядром
    // порт встаёт с ним вровень (COMPRESSION: KL между ними ~1e-8).
    // Остальные типы считаются по-прежнему: под ручной формулой сняты все
    // эталоны паритета плотной модели, и сдвигать их незачем.
    if x.dtype == .float16 { return MLX.layerNorm(x, weight: weight, bias: bias, eps: eps) }
    let mean = x.mean(axis: -1, keepDims: true)
    let varc = (x - mean).square().mean(axis: -1, keepDims: true)
    let normed = (x - mean) / sqrt(varc + eps)
    return normed * weight + bias
}

// Linear без bias: x[...,in] @ Wᵀ, W хранится [out, in].
func linear(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
    matmul(x, w.transposed())
}
func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    matmul(x, w.transposed()) + b
}

// ─────────────────── Backbone ───────────────────

/// Квантованная (frozen) матрица: упакованные веса + scale/bias по группам.
struct QuantBase {
    var wq: MLXArray
    var scales: MLXArray
    var biases: MLXArray?
    var groupSize: Int
    var bits: Int
}

/// x070-backbone поверх плоского словаря весов (MLX-имена тензоров).
public final class X070Backbone {
    public let cfg: X070Config

    /// Версия всего, от чего зависит рекуррентный шаг: веса, адаптеры,
    /// квантованная база и переключатели пути. Растёт при ЛЮБОЙ записи в
    /// эти свойства (у каждого didSet); по ней собранный шаг узнаёт, что
    /// его пора пересобрать, — см. `step` в X070Generation.swift.
    var stepVersion = 0
    /// Собранный (mx.compile) рекуррентный шаг и версия, под которую он собран.
    var compiledStepFn: (@Sendable ([MLXArray]) -> [MLXArray])? = nil
    var compiledStepVersion = -1
    /// Рекуррентный шаг через `compile`: граф шага трассируется один раз и
    /// дальше исполняется готовым, цепочки поэлементных операций сливаются.
    /// На декоде малых моделей время шага — это число операций, а не
    /// арифметика, поэтому выигрыш тем больше, чем модель меньше.
    /// Числа те же в пределах шума fp16 (слитые цепочки считаются в другом
    /// порядке округлений); `false` — покомандный путь всегда.
    ///
    /// СОБИРАЕТСЯ НЕ ВСЯКАЯ МОДЕЛЬ — только та, в графе шага которой нет ни
    /// одного массива bfloat16 (см. `stepIsCompilable`): модель из .rwkvq в
    /// норме fp16 без неслитых адаптеров, плотная fp32 / fp16. Причина —
    /// дефект MLX (ядро 0.31.1): библиотеку слитого поэлементного ядра он
    /// именует по РОДУ И РАЗМЕРУ типа («f2»), и у float16 с bfloat16 они
    /// совпадают. Собранные в одном процессе одинаковые по строению цепочки
    /// на fp16 и на bf16 получают одно имя, вторая исполняется кодом первой
    /// — биты одного формата читаются как другой, и шаг выдаёт мусор без
    /// единой ошибки (найдено 10.10: плотная bf16-модель после квантованной
    /// fp16 в одном процессе). Пока bf16 в собранные графы не попадает,
    /// столкнуться нечему. Остальные модели идут покомандным путём.
    public var useCompiledStep = true
    /// Префилл (`prefill`) через `compile` — см. `prefillCompiled`.
    ///
    /// ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН, и это итог замера, а не осторожность. Собранный
    /// префилл быстрее на 7–8% (T=512: 0.4B 333 → 305 мс, 1.5B 829 → 771), но
    /// каждая НОВАЯ длина промпта стоит трассировки (~60–80 мс, первая в
    /// процессе — до 0.8 с на сборку слитых ядер) и +0.1 ГБ пика. В чате
    /// длина промпта каждый раз другая — там это чистый убыток. Включать
    /// стоит, когда длины повторяются (реранкинг, фиксированное окно).
    /// Короткие промпты (меньше `compiledPrefillMinTokens`) идут обычным
    /// путём в любом случае.
    public var useCompiledPrefill = false
    public var compiledPrefillMinTokens = 64
    /// Сколько разных длин держит кеш трасс до пересоздания функции.
    public var compiledPrefillMaxLengths = 32
    var compiledPrefillFn: (@Sendable ([MLXArray]) -> [MLXArray])? = nil
    var compiledPrefillVersion = -1
    var compiledPrefillLengths: Set<Int> = []
    /// Сколько префиллов прошло собранным путём (счётчик для тестов).
    public internal(set) var compiledPrefillCalls = 0
    /// Сколько шагов прошло собранным путём (счётчик для тестов и замеров).
    public internal(set) var compiledStepCalls = 0
    private var compilableVersion = -1
    private var compilable = false

    /// Можно ли собирать шаг этой модели: в его графе нет bfloat16.
    /// Плотные цели .rwkvq лежат в bf16, но в шаге читаются уже
    /// приведённой к типу базы копией, вычисленной ДО трассировки.
    var stepIsCompilable: Bool {
        if compilableVersion == stepVersion { return compilable }
        var ok = wOverride == nil && quant.isEmpty
            && rwkvqBaseDType != .bfloat16
            && !loraA.values.contains { $0.dtype == .bfloat16 }
            && !loraB.values.contains { $0.dtype == .bfloat16 }
        if ok {
            for (k, v) in w where v.dtype == .bfloat16 {
                if rwkvqDenseTargets.contains(k), let base = rwkvqBaseDType, base != .bfloat16 {
                    continue
                }
                ok = false
                break
            }
        }
        if ok, let base = rwkvqBaseDType {
            // приведённые копии плотных целей — константами, до трассировки
            eval(rwkvqDenseTargets.map { denseTarget($0, base) })
        }
        compilable = ok
        compilableVersion = stepVersion
        return ok
    }
    var w: [String: MLXArray] { didSet { stepVersion &+= 1 } }

    // ── LoRA / QLoRA / training hooks ──
    // Инертны для обычного инференса (пустые словари + trainLayers пуст) ⇒
    // поведение идентично исходному, parity-тесты остаются зелёными.
    /// Слои, чьё WKV считается через дифференцируемое ядро (wkv7Train).
    public var trainLayers: Set<Int> = []
    /// Подмена весов на обучаемые (fp32) во время valueAndGrad для full-weight
    /// partial-finetune верхних слоёв. nil ⇒ читаются frozen-веса (инференс не меняется).
    public var wOverride: [String: MLXArray]? = nil { didSet { stepVersion &+= 1 } }
    /// Если true — каждый блок (ln1+tmix+resid+ln2+cmix+resid) считается через
    /// gradient checkpoint (recompute в backward). Дифф-входы (x, v_first, LoRA)
    /// протягиваются явными аргументами чекпоинта; frozen-веса захватываются.
    public var useBlockCheckpoint = false

    /// ЭКСПЕРИМЕНТ, по умолчанию ВЫКЛЮЧЕН: приводить выход WKV обратно к типу
    /// вычислений вместо того, чтобы пускать fp32 в остаточный поток.
    ///
    /// Зачем. Ядро WKV считает рекуррентность в fp32 (иначе нельзя) и
    /// возвращает fp32. Дальше этот тип едет через ln_x, bonus, o_proj, все
    /// следующие слои и голову; веса при этом bf16, поэтому каждый матмул
    /// поднимает ВЕСЬ вес до fp32. На 0.1B голова 65536×768 стоит 5.503 мс
    /// против 1.283 на bf16-активации — ×4.29, раз на токен. Течь начинается
    /// в tmix СЛОЯ 0: `tmixPrev[0]` ещё bfloat16, `cmixPrev[0]` уже float32.
    ///
    /// Почему за флагом, а не просто исправлено. Включение сдвигает числа во
    /// ВСЁМ стеке, и каждый эталон паритета с Python придётся перемерить.
    /// Плюс rwkv-metal устроен так же (после `wkv7` приведения тоже нет), то
    /// есть Swift сейчас ВЕРЕН источнику, а с флагом станет от него отличаться
    /// — это решение о расхождении с референсом, и принимать его надо с
    /// числами на руках, а не мимоходом.
    ///
    /// Флаг переключается на живом объекте, так что обе ветки сравниваются в
    /// ОДНОМ процессе — замер «до и после» разными сборками недействителен.
    public var castWKVOutputToComputeDType = false { didSet { stepVersion &+= 1 } }
    /// LoRA-адаптеры по таргету: "blocks.L.tmix.r_proj" и т.п.
    var loraA: [String: MLXArray] = [:] { didSet { stepVersion &+= 1 } }      // [rank, in]
    var loraB: [String: MLXArray] = [:] { didSet { stepVersion &+= 1 } }      // [out, rank]
    var loraScale: [String: Float] = [:] { didSet { stepVersion &+= 1 } }     // alpha / rank
    /// Квантованная замороженная база по имени веса (напр. "...r_proj.weight", "head.weight", "emb.weight").
    var quant: [String: QuantBase] = [:] { didSet { stepVersion &+= 1 } }

    /// Второй бэкенд квантованной базы: родной формат .rwkvq (sb6).
    ///
    /// Отличие от `quant` (стоковый mlx.nn.quantize) — не в интерфейсе, а в
    /// происхождении чисел: .rwkvq приходит из откалиброванного пайплайна
    /// rwkv-quant с известной деградацией ppl, тогда как стоковый квант
    /// пересчитывает scale/bias по min/max блока и этой калибровке не
    /// соответствует. Поэтому это отдельный путь, а не «другие параметры» того же.
    ///
    /// Ключи — x070-имена (blocks.N.tmix.k_proj.weight); в сайдкар они
    /// переводятся таблицей RwkvqNaming.
    var rwkvqSidecar: RwkvqSidecar? = nil { didSet { stepVersion &+= 1 } }
    var rwkvqKeys: [String: String] = [:] { didSet { stepVersion &+= 1 } }     // x070-имя → world-имя в сайдкаре

    /// Тот же `.rwkvq`, переложенный в родной контейнер MLX.
    ///
    /// Отличие от `rwkvqSidecar` не в числах — они те же до последнего
    /// разряда (см. RwkvqNative.swift) — а в том, что плотная матрица не
    /// материализуется вовсе: `quantizedMM` читает сжатое и считает.
    /// Путь через `sc.dequantize` пишет и перечитывает плотный
    /// транзиент, отчего на 2.9B и выходит 2.21x против bf16.
    /// Заполняется по `RwkvqAttachOptions.useNativeKernel`.
    public internal(set) var rwkvqNative: [String: RwkvqNativeWeight] = [:] { didSet { stepVersion &+= 1 } }

    /// Тип квантованной базы: чем округляется деквант и в чём считается
    /// матмул проекции. Вход приводится к этому типу, выход — обратно к типу
    /// входа, так что fp32 из ядра WKV не тащит за собой fp32-копию матрицы.
    ///
    /// `.float16` — норма rwkv-quant с 02.10 и то, что `attachRwkvq` ставит
    /// по умолчанию: у 8-битных кодов bf16-округление съедало четверть шага
    /// сетки (+2–3% KL на REDUCTION). `.bfloat16` — база rwkv-metal до 07.10.
    /// `nil` — путь, каким порт считал до 08.10: деквант в тип ВХОДА без
    /// приведения (после WKV это fp32), вход родного ядра в fp16.
    ///
    /// Переключается на живом объекте: плечи A/B сравниваются в одном
    /// процессе.
    public var rwkvqBaseDType: DType? = nil { didSet { stepVersion &+= 1 } }

    /// Тип входа родного ядра sb6 (`quantizedMM`). `nil` — по правилу:
    ///   • генерация (рекуррентный шаг) — fp16: на 1.5B и 2.9B это −22…24%
    ///     времени декода при +0.08% KL к исходной модели на 1.5B (на 0.1B
    ///     и 0.4B разница незначима), см. docs/Inference.md;
    ///   • параллельный проход (префилл, обучение) — fp32, как
    ///     RwkvqNativeLinear в rwkv-metal: с ним порт на COMPRESSION
    ///     совпадает до ~1e-8 KL, и градиент не идёт сквозь fp16;
    ///   • прежний путь базы (`rwkvqBaseDType == nil`) — fp16 везде.
    /// Явное значение действует на оба пути — для A/B в одном процессе.
    public var rwkvqNativeInputDType: DType? = nil { didSet { stepVersion &+= 1 } }

    /// Тип ОБРАТНОГО прохода через квантованную базу при базе fp16.
    ///
    /// Умолчание bf16: прямой проход в fp16, обратный — своим VJP,
    /// dX = dY · W с W, деквантованной в bf16. Почему не fp16 насквозь: у
    /// fp16 узкая экспонента (минимум нормального 6e-5), и мелкие котангенты
    /// теряют разряды — в rwkv-metal градиенты адаптеров при fp16 насквозь
    /// уходили от плеча без приведения на 5e-2…1.5e-1 (косинус 0.988) против
    /// 1.3e-2 при обратном bf16. `nil` — автоград в типе базы (fp16 насквозь),
    /// `.float32` — точнее и дороже (+360 МБ пика на 1.5B в rwkv-metal).
    /// При базе не fp16 не действует. Прямой проход от этого не зависит.
    /// Порт rwkv_metal/lora/rwkvq_linear.py::_build_wide_bwd_fn.
    public var rwkvqBackwardDType: DType? = .bfloat16
    /// Сколько проекций прошло через свой VJP (счётчик для тестов).
    public internal(set) var rwkvqWideBackwardCalls = 0

    /// x · Wᵀ в типе базы с приведением входа и выхода. `weight(dtype)` —
    /// вес [out, in] в запрошенном типе (деквант либо плотная цель).
    private func baseMatmul(_ x: MLXArray, base: DType,
                            weight: @escaping (DType) -> MLXArray) -> MLXArray {
        guard base == .float16, let bwd = rwkvqBackwardDType else {
            return matmul(x.asType(base), weight(base).transposed()).asType(x.dtype)
        }
        rwkvqWideBackwardCalls += 1
        let fn = CustomFunction {
            Forward { ins in
                [matmul(ins[0].asType(base), weight(base).transposed()).asType(ins[0].dtype)]
            }
            // буферы базы — константы, котангент только по x
            VJP { primals, cotangents in
                [matmul(cotangents[0].asType(bwd), weight(bwd)).asType(primals[0].dtype)]
            }
        }
        return fn([x])[0]
    }

    /// Декод sym ядром GEMV (без плотной матрицы) вместо «деквант + матмул».
    ///
    /// Действует ТОЛЬКО на рекуррентном шаге (`step`, генерация) и только в
    /// норме (`rwkvqBaseDType != nil`). Параллельный проход и всё, через что
    /// идёт градиент, считают прежним путём: ядро непрозрачно для автограда,
    /// и адаптеры ниже слоя молча получили бы нулевые градиенты.
    /// Числа у ядра другие (точный вес, сумма в fp32), см. RwkvqSymGEMV.swift.
    /// Переключается на живом объекте — A/B в одном процессе.
    public var rwkvqSymDecodeKernel = true { didSet { stepVersion &+= 1 } }
    /// Шаг WKV и хвост tmix на рекуррентном шаге — по одному ядру вместо
    /// двух десятков поэлементных операций на слой (см. WKV7Step.swift).
    /// Числа те же в допуске порядка суммирования (не побитно). Действует
    /// только у модели из .rwkvq в норме (`rwkvqBaseDType != nil`): у
    /// плотной модели шаг остаётся прежним — под ним сняты эталоны декода.
    public var useStepKernels = true { didSet { stepVersion &+= 1 } }
    /// Сколько слоёв прошло ядрами шага (счётчик для тестов).
    public internal(set) var stepKernelCalls = 0

    /// Декод sb6 ядром GEMV (без плотной матрицы и без родного контейнера
    /// MLX) — порт ядра k3 из rwkv-quant, см. RwkvqSb6GEMV.swift. Те же
    /// правила, что у sym: только рекуррентный шаг, только норма. Работает,
    /// когда база собрана БЕЗ родного контейнера (`useNativeKernel: false`):
    /// ядро читает K3-интерлив, а при родном контейнере он отпущен.
    public var rwkvqSb6DecodeKernel = true { didSet { stepVersion &+= 1 } }
    /// Сколько проекций прошло ядром GEMV sb6 (счётчик для тестов).
    public internal(set) var rwkvqSb6GEMVCalls = 0

    /// Декод r/k/v слоя ОДНИМ запуском ядра (фьюз, порт SymQuantLinearFused):
    /// бит-в-бит равен трём отдельным запускам, памяти не добавляет
    /// (см. RwkvqSymGEMV.swift). Действует там же, где ядро GEMV.
    public var rwkvqSymFuseRKV = true { didSet { stepVersion &+= 1 } }
    /// Сколько слоёв прошло фьюзом r/k/v (счётчик для тестов).
    public internal(set) var rwkvqSymFusedCalls = 0

    /// r, k, v слоя на рекуррентном шаге одним запуском ядра; nil — если
    /// фьюз неприменим (тогда проекции считаются по одной). Адаптеры LoRA
    /// прибавляются поверх, как в обычной проекции.
    func decodeRKV(_ xr: MLXArray, _ xk: MLXArray, _ xv: MLXArray,
                   layer: Int) -> (MLXArray, MLXArray, MLXArray)? {
        guard rwkvqSymFuseRKV, rwkvqSymDecodeKernel, rwkvqBaseDType != nil,
              let sc = rwkvqSidecar else { return nil }
        let p = "blocks.\(layer).tmix."
        let targets = [p + "r_proj", p + "k_proj", p + "v_proj"]
        guard targets.allSatisfy({ rwkvqKeys[$0 + ".weight"] != nil && quant[$0 + ".weight"] == nil }),
              let y = sc.symGEMVFused("blocks.\(layer).att.rkv",
                                      concatenated([xr, xk, xv], axis: 0)) else { return nil }
        rwkvqSymGEMVCalls += 3
        rwkvqSymFusedCalls += 1
        func finish(_ i: Int, _ x: MLXArray) -> MLXArray {
            let base = y[i ..< (i + 1)].asType(x.dtype)
            guard let a = loraA[targets[i]], let b = loraB[targets[i]] else { return base }
            let z = matmul(x, a.transposed())
            return base + (loraScale[targets[i]] ?? 1.0) * matmul(z, b.transposed())
        }
        return (finish(0, xr), finish(1, xk), finish(2, xv))
    }

    /// Сколько проекций прошло ядром GEMV. Счётчик для тестов и замеров:
    /// «правка не применилась» обязана быть видимой, а не читаться как
    /// «правка ничего не меняет».
    public internal(set) var rwkvqSymGEMVCalls = 0

    /// Цели, которые лежат в файле ПЛОТНЫМИ (нынешние пресеты: o_proj слоя 0,
    /// bits_overrides → 16). Это намерение файла, а не ошибка: вес —
    /// исходный bf16 как есть, замороженная плотная база, LoRA встаёт поверх
    /// так же, как поверх квантованной. Считается в типе базы.
    public internal(set) var rwkvqDenseTargets: Set<String> = [] { didSet { stepVersion &+= 1 } }
    var denseTargetCast: [String: MLXArray] = [:]

    /// Плотная цель в типе базы. Копия кэшируется: вес и так лежит в памяти
    /// плотным, здесь нет сжатия, которое кэш мог бы отменить.
    private func denseTarget(_ wKey: String, _ dtype: DType) -> MLXArray {
        let raw = w[wKey]!
        if raw.dtype == dtype { return raw }
        if let c = denseTargetCast[wKey], c.dtype == dtype { return c }
        let c = raw.asType(dtype)
        denseTargetCast[wKey] = c
        return c
    }

    /// Снять квантованную подложку с веса (после слияния адаптера вес
    /// становится плотным и лежит в словаре весов).
    func releaseRwkvqBacking(_ wKey: String) {
        rwkvqKeys[wKey] = nil
        rwkvqNative[wKey] = nil
        rwkvqDenseTargets.remove(wKey)
        denseTargetCast[wKey] = nil
        if let sc = rwkvqSidecar {
            let needed = Set(rwkvqKeys.filter { rwkvqNative[$0.key] == nil }.values)
            rwkvqSidecar = needed.isEmpty ? nil : sc.pruned(keeping: needed)
        }
    }

    /// Форма (out, in) проекции, где бы ни лежал её вес: плотным, в
    /// стоковом кванте, в сайдкаре .rwkvq или в родном контейнере. Нужна
    /// тем, кто навешивает адаптеры: на базе из .rwkvq плотных копий
    /// проекций нет вовсе, и поиск по словарю плотных весов молча не
    /// находил ни одной цели.
    func projectionShape(_ wKey: String) -> (out: Int, in: Int)? {
        if let v = w[wKey], v.ndim == 2 { return (v.shape[0], v.shape[1]) }
        if let q = quant[wKey] { return (q.scales.shape[0], q.scales.shape[1] * q.groupSize) }
        if let n = rwkvqNative[wKey] { return (n.outFeatures, n.inFeatures) }
        if let world = rwkvqKeys[wKey], let info = rwkvqSidecar?.tensors[world],
           info.shape.count == 2 {
            return (info.outFeatures, info.inFeatures)
        }
        return nil
    }

    /// Считать шесть лерпов token-shift одним broadcast вместо шести пар
    /// «умножить и сложить».
    ///
    /// Арифметика поэлементная и от группировки не зависит, так что
    /// результат обязан быть БИТ-В-БИТ прежним — это проверяется, а не
    /// предполагается (`decode-bench --fuse-parity`).
    ///
    /// Зачем: на декоде 2.9B вне проекций уходит 8.16 мс/ток, и по
    /// трейсу это не арифметика (ALU 30%) и не CPU (простаивает), а
    /// число мелких ядер. Шесть лерпов -- двенадцать операций из ~68 на
    /// слой.
    public var fuseLerp = true { didSet { stepVersion &+= 1 } }

    /// Стек коэффициентов token-shift на слой, [6,1,D]. Строится лениво
    /// и кэшируется: сам стек стоит операций, и делать его на каждом
    /// шаге значило бы менять шило на мыло.
    private var xcoefCache: [Int: MLXArray] = [:]

    /// nil ⇒ считать по-старому. Это не осторожность: при `wOverride`
    /// (партиал-файнтюн) веса подменяются на обучаемые, и кэш стал бы
    /// молча возвращать замороженные.
    func xcoefStack(_ layer: Int) -> MLXArray? {
        guard fuseLerp, wOverride == nil else { return nil }
        if let c = xcoefCache[layer] { return c }
        let p = "blocks.\(layer).tmix."
        let names = ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]
        let c = stacked(names.map { w[p + $0]!.reshaped([1, cfg.nEmbd]) }, axis: 0)
        eval(c)
        xcoefCache[layer] = c
        return c
    }

    // GroupNorm с pytorch_compatible-семантикой (eps=64e-5). Применяется
    // per-token к [N, D] (каждый токен нормализуется независимо по головам).
    private let groupNorm: GroupNorm

    public init(weights: [String: MLXArray], cfg: X070Config,
                computeDType: DType = .bfloat16) {
        self.cfg = cfg
        var conv: [String: MLXArray] = [:]
        for (k, v) in weights { conv[k] = v.asType(computeDType) }
        self.w = conv

        // affine=false: вес/смещение ln_x применяем вручную из весов модели,
        // т.к. они per-layer (blocks.N.tmix.ln_x.{weight,bias}).
        self.groupNorm = GroupNorm(groupCount: cfg.nHead, dimensions: cfg.nEmbd,
                                   eps: 64e-5, affine: false, pytorchCompatible: true)
    }

    // ── Публичная интроспекция для сборки обучаемых множеств ─────────
    //
    // Словарь весов остаётся internal (наружу его отдавать незачем и опасно),
    // но соседним модулям нужно уметь спросить «какие веса тут есть» —
    // иначе они вынуждены угадывать имена строками.

    /// Все имена весов модели.
    public var weightKeys: [String] { Array(w.keys).sorted() }

    /// Веса ТЕЛА модели: всё, кроме таблицы эмбеддингов и LM-головы.
    ///
    /// Это множество имеет смысл там, где логиты не используются вовсе
    /// (эмбеддинги, реранкер): `head.weight` при словаре 65536 — примерно
    /// треть параметров 0.1B-модели, и держать под неё моменты Adam ради
    /// тензора, который не участвует в лоссе, — чистая потеря памяти.
    public var bodyWeightKeys: [String] {
        weightKeys.filter { $0 != "emb.weight" && $0 != "head.weight" }
    }

    /// Навешены ли LoRA-адаптеры.
    public var hasLoRAAdapters: Bool { !loraA.isEmpty }

    /// Имена таргетов с адаптерами.
    public var loraTargets: [String] { loraA.keys.sorted() }

    /// Пара (A, B) адаптера цели — для внешнего цикла обучения и замеров.
    public func loraAdapter(_ target: String) -> (MLXArray, MLXArray) {
        (loraA[target]!, loraB[target]!)
    }

    /// Подставить пару (A, B) адаптера цели (форма и тип — на вызывающем).
    public func setLoraAdapter(_ target: String, _ a: MLXArray, _ b: MLXArray) {
        precondition(loraA[target] != nil, "у цели \(target) нет адаптера")
        loraA[target] = a
        loraB[target] = b
    }

    // Чтение веса с учётом wOverride (обучаемая подмена) → frozen.
    private func wv(_ key: String) -> MLXArray { wOverride?[key] ?? w[key]! }
    private func g(_ key: String) -> MLXArray { wv(key) }

    // Восстановленный вес из .rwkvq, если он оттуда. ТРАНЗИЕНТНЫЙ: не
    // кэшируется намеренно — база обязана жить в памяти сжатой, иначе смысл
    // квантованной базы теряется (кэш плотных весов превращает QLoRA в LoRA
    // с лишними шагами).
    ///
    /// Разворачивается СРАЗУ в тип вычислений, а не в fp32 с приведением
    /// после. Числа от этого не меняются ни на бит — приведение к bf16
    /// происходило и раньше, просто на шаг позже, — а транзиента вдвое
    /// меньше и лишнего прохода по памяти нет вовсе. На векторной проекции
    /// всё упирается именно в память: замерено на 2.9B, деквантизация это
    /// 57–62% времени проекции.
    private func rwkvqWeight(_ wKey: String, _ dtype: DType) -> MLXArray? {
        guard let worldKey = rwkvqKeys[wKey], let sc = rwkvqSidecar else { return nil }
        return try? sc.dequantize(worldKey, dtype: dtype)
    }

    // База проекции: если есть quant — x·Wᵀ с деквантизацией на лету
    // (веса [out,in] ⇒ transpose: true), иначе обычный linear.
    private func baseProj(_ x: MLXArray, _ wKey: String, decode: Bool = false) -> MLXArray {
        if let q = quant[wKey] {
            return quantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                               transpose: true, groupSize: q.groupSize, bits: q.bits)
        }
        // родной контейнер раньше сайдкарного: числа те же, но без
        // плотного транзиента
        if let n = rwkvqNative[wKey] {
            // Норма: вход quantizedMM в fp32, как у RwkvqNativeLinear в
            // rwkv-metal. Прежний путь подавал fp16.
            // Норма: fp16 на генерации (рекуррентный шаг), fp32 на
            // параллельном проходе — через него идёт градиент, а сквозь
            // fp16 мелкие котангенты теряются. Прежний путь подавал fp16.
            // Голова — исключение: тип входа quantizedMM есть и тип его
            // ВЫХОДА, а логиты в fp16 (шаг сетки ~0.01 при |логит| ~ 16)
            // стоят на порядок больше, чем fp16 во всех внутренних
            // проекциях вместе; на декоде голова считается в fp32.
            let half = rwkvqBaseDType == nil || (decode && wKey != "head.weight")
            let inType = rwkvqNativeInputDType ?? (half ? DType.float16 : DType.float32)
            return n(x.asType(inType)).asType(x.dtype)
        }
        guard let base = rwkvqBaseDType else {
            if let dense = rwkvqWeight(wKey, x.dtype) {
                return matmul(x, dense.transposed())
            }
            return matmul(x, wv(wKey).transposed())
        }
        if decode, let world = rwkvqKeys[wKey], let sc = rwkvqSidecar {
            let lead = Array(x.shape.dropLast())
            let x2d = x.reshaped([-1, x.shape[x.ndim - 1]])
            if rwkvqSymDecodeKernel, let y = sc.symGEMV(world, x2d) {
                rwkvqSymGEMVCalls += 1
                return y.reshaped(lead + [y.shape[1]]).asType(x.dtype)
            }
            if rwkvqSb6DecodeKernel, let y = sc.sb6GEMV(world, x2d) {
                rwkvqSb6GEMVCalls += 1
                return y.reshaped(lead + [y.shape[1]]).asType(x.dtype)
            }
        }
        if rwkvqKeys[wKey] != nil, rwkvqSidecar != nil {
            // вес деквантуется заново на каждое обращение (и в обратном
            // проходе тоже): транзиент не хранится
            return baseMatmul(x, base: base) { [unowned self] dt in
                self.rwkvqWeight(wKey, dt)!
            }
        }
        if rwkvqDenseTargets.contains(wKey), wOverride?[wKey] == nil {
            return baseMatmul(x, base: base) { [unowned self] dt in
                self.denseTarget(wKey, dt)
            }
        }
        return matmul(x, wv(wKey).transposed())
    }

    // Проекция с опциональным LoRA: base + scale·(x·Aᵀ)·Bᵀ. target=nil ⇒ только база.
    private func proj(_ x: MLXArray, _ wKey: String, lora target: String?,
                      decode: Bool = false) -> MLXArray {
        let base = baseProj(x, wKey, decode: decode)
        guard let t = target, let a = loraA[t], let b = loraB[t] else { return base }
        let z = matmul(x, a.transposed())                       // [..., rank]
        return base + (loraScale[t] ?? 1.0) * matmul(z, b.transposed())
    }

    // Эмбеддинг: gather строк (с деквантизацией, если emb квантован).
    private func embed(_ ids: MLXArray) -> MLXArray {
        if let q = quant["emb.weight"] {
            let rows = q.wq.take(ids, axis: 0)
            let sc = q.scales.take(ids, axis: 0)
            let bi = q.biases?.take(ids, axis: 0)
            return dequantized(rows, scales: sc, biases: bi,
                               groupSize: q.groupSize, bits: q.bits,
                               dtype: w["emb.weight"]?.dtype ?? .bfloat16)
        }
        if let dense = rwkvqWeight("emb.weight", w["ln0.weight"]?.dtype ?? .bfloat16) {
            // ВНИМАНИЕ: sb6-ядро разворачивает таблицу целиком, а для словаря
            // 65536×768 это ~100 МБ транзиента bf16 на КАЖДЫЙ проход (и 335 МБ
            // на 2.9B). Строчной выборки формат не поддерживает: коды упакованы
            // по блокам вдоль входной оси, и достать одну строку дешевле, чем
            // блок, нельзя. Поэтому emb в сайдкар обычно и не отдают —
            // см. attachRwkvq.
            return dense.take(ids, axis: 0)
        }
        return wv("emb.weight").take(ids, axis: 0)
    }

    // token-shift: prev[t]=x[t-1]. Возвращает xx = prev - x.
    //
    // prev0 — вход на позиции −1, то есть последний токен ПРЕДЫДУЩЕГО куска
    // той же последовательности [B,1,D]. nil ⇒ нули (начало последовательности,
    // прежнее поведение). Без этого продолжение расходится со сплошным проходом
    // ровно на первом токене каждого слоя.
    private func tokenShift(_ x: MLXArray, _ prev0: MLXArray? = nil) -> MLXArray {
        rwkvTokenShift(x, prev0)
    }

    // GroupNorm ln_x (per-token). Канон RWKV: ln_x(x.view(B*T, C)) — каждый
    // токен нормируется независимо по головам. КАУЗАЛЬНО: токен t не видит
    // t+1..T-1, поэтому параллельный путь совпадает с рекуррентным lnXStep.
    //
    // ВАЖНО: подаём [B*T, D], НЕ [B,T,D]. MLX GroupNorm на [B,T,D] трактует
    // batch=B и смешал бы все T внутри головы (cross-token, утечка будущего) —
    // это была причина старого расхождения уже на позиции 0. Регресс ловится
    // рекуррентным parity-тестом (recurrent vs parallel).
    private func lnX(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
        let B = x.shape[0], T = x.shape[1], D = x.shape[2]
        let normed = groupNorm(x.reshaped([B * T, D])).reshaped([B, T, D])
        return normed * weight + bias
    }

    // ─────────── Блок: арифметика вынесена в RWKVBlock.swift ───────────
    //
    // Тело tmix/cmix здесь больше не живёт. Оно ОДНО на весь пакет
    // (rwkvTmixForward / rwkvCmixForward), а бэкбон приносит только контекст:
    // откуда брать веса (плоский словарь с префиксом слоя, с учётом
    // wOverride), как считать проекции (плотно / из .rwkvq / с LoRA) и каким
    // ядром считать WKV (trainLayers).
    //
    // Так голова реранкера получает ТЕ ЖЕ блоки, что и база, не копируя ни
    // строки арифметики. Что поведение бэкбона при этом не изменилось —
    // проверяется характеризационными тестами, а не заявляется.

    /// Контекст блока слоя `layer`. Кэшируется: объект создаётся один раз на
    /// слой, а не на каждый проход.
    private var blockContexts: [Int: BackboneBlockContext] = [:]
    private func context(_ layer: Int) -> BackboneBlockContext {
        if let c = blockContexts[layer] { return c }
        let c = BackboneBlockContext(self, layer)
        blockContexts[layer] = c
        return c
    }

    // Доступ для контекста (он живёт вне класса, но внутри модуля).
    func weightForBlock(_ key: String) -> MLXArray { wv(key) }
    /// Строки таблицы эмбеддингов с учётом квантования — для рекуррентного
    /// декода. Он живёт в отдельном файле и обязан ходить теми же дорогами,
    /// иначе снова разойдётся с параллельным путём.
    func embedForBlock(_ ids: MLXArray) -> MLXArray { embed(ids) }
    func projectForBlock(_ x: MLXArray, _ key: String, lora: String?,
                         decode: Bool = false) -> MLXArray {
        proj(x, key, lora: lora, decode: decode)
    }
    func groupNormRaw(_ x: MLXArray) -> MLXArray { groupNorm(x) }

    /// Значение веса по имени, БЕЗ учёта wOverride: нужно тем, кто копирует
    /// веса наружу (инициализация блоков головы), — там интересен frozen-вес,
    /// а не временная обучаемая подмена.
    ///
    /// На квантованной базе вес РАЗВОРАЧИВАЕТСЯ из сайдкара. Это уместно
    /// именно здесь и было бы неуместно на горячем пути: копирование весов
    /// наружу происходит один раз при сборке, а не на каждом токене.
    ///
    /// Раньше здесь стоял `w[key]!`, и на квантованной базе это был
    /// force-unwrap без сообщения: `attachRwkvq` плотные копии выбрасывает.
    public func weight(_ key: String) -> MLXArray {
        guard let v = denseWeight(key) else {
            preconditionFailure(
                "нет веса \(key). Известные имена — `allWeightKeys`; если база "
                + "квантована, проверьте `isRwkvqBacked` и что сайдкар подключён.")
        }
        return v
    }

    /// Есть ли ПЛОТНАЯ копия веса. Осторожно: на квантованной базе это `false`
    /// для всех выброшенных весов, хотя модель их имеет — см. `hasAnyWeight`.
    public func hasWeight(_ key: String) -> Bool { w[key] != nil }

    /// Есть ли вес ВООБЩЕ — плотный или в сайдкаре.
    public func hasAnyWeight(_ key: String) -> Bool {
        w[key] != nil || isRwkvqBacked(key)
    }

    /// Все имена весов модели, включая ушедшие в сайдкар.
    ///
    /// `weightKeys` перечисляет только плотные, и на квантованной базе это
    /// ЛОЖНО МАЛЫЙ список: обход по нему собирает неполный набор молча. Так
    /// `RWKVBlock.fromBase` строил блок без единой проекции — объект
    /// создавался успешно и падал потом, в `param()` посреди прохода.
    public var allWeightKeys: [String] {
        Array(Set(w.keys).union(rwkvqKeys.keys)).sorted()
    }

    /// Плотное значение веса: своё, либо развёрнутое из сайдкара, либо nil.
    ///
    /// Транзиент НЕ кэшируется — как и везде на квантованной базе. Звать на
    /// горячем пути не нужно: там `projectForBlock`, который разворачивает
    /// сразу в тип вычислений и умеет LoRA.
    public func denseWeight(_ key: String) -> MLXArray? {
        if let v = w[key] { return v }
        let dtype = w["ln0.weight"]?.dtype ?? .bfloat16
        if let v = rwkvqWeight(key, dtype) { return v }
        // Вес, оставшийся ТОЛЬКО в родном контейнере (сайдкар отпущен):
        // разворачивается штатной обратной функцией упаковки. Раньше здесь
        // был nil, и `weight(_:)` падал на базе, собранной с
        // useNativeKernel, — то есть на загрузке по умолчанию.
        if let n = rwkvqNative[key] {
            return dequantized(n.wq, scales: n.scales.asType(.float32),
                               biases: n.biases.asType(.float32),
                               groupSize: n.groupSize, bits: n.bits).asType(dtype)
        }
        return nil
    }

    private func tmix(_ x: MLXArray, _ vFirst: MLXArray?, _ layer: Int,
                      shiftPrev: MLXArray? = nil, hIn: MLXArray? = nil,
                      mask: MLXArray? = nil, wantState: Bool = false)
        -> (MLXArray, MLXArray, MLXArray?) {
        rwkvTmixForward(x, vFirst, context(layer), shiftPrev: shiftPrev,
                        hIn: hIn, mask: mask, wantState: wantState)
    }

    private func cmix(_ x: MLXArray, _ layer: Int, shiftPrev: MLXArray? = nil) -> MLXArray {
        rwkvCmixForward(x, context(layer), shiftPrev: shiftPrev)
    }

    /// Один блок: ln1+tmix+resid+ln2+cmix+resid. (x0, vFirst?) -> (x', vFirstOut).
    func blockForward(_ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int) -> (MLXArray, MLXArray) {
        rwkvBlockForward(x0, vFirst, context(layer))
    }

    /// Тот же блок, но с граничным состоянием. Возвращает
    /// (x', vFirstOut, hOut, tmixShiftOut, cmixShiftOut).
    func blockForwardWithState(
        _ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int,
        hIn: MLXArray?, mask: MLXArray?, tmixPrev: MLXArray?,
        cmixPrev: MLXArray?, endIdx: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        rwkvBlockForwardWithState(x0, vFirst, context(layer), hIn: hIn, mask: mask,
                                  tmixPrev: tmixPrev, cmixPrev: cmixPrev, endIdx: endIdx)
    }

    /// Отсортированные ключи LoRA-таргетов слоя (детерминированный порядок упаковки).
    private func loraKeysForLayer(_ layer: Int) -> [String] {
        loraA.keys.filter { $0.hasPrefix("blocks.\(layer).") }.sorted()
    }

    /// blockForward через gradient checkpoint. LoRA-адаптеры слоя идут ЯВНЫМИ
    /// входами чекпоинта (иначе vjp не даст к ним градиент). v_first и x — тоже
    /// явные входы, чтобы grad тёк сквозь value-residual и остаточный поток.
    func blockCheckpointed(_ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int) -> (MLXArray, MLXArray) {
        let keys = loraKeysForLayer(layer)
        let hasV = vFirst != nil
        var inputs: [MLXArray] = [x0]
        if hasV { inputs.append(vFirst!) }
        for t in keys { inputs.append(loraA[t]!); inputs.append(loraB[t]!) }

        let f: ([MLXArray]) -> [MLXArray] = { ins in
            var i = 0
            let xin = ins[i]; i += 1
            var vin: MLXArray? = nil
            if hasV { vin = ins[i]; i += 1 }
            for t in keys {
                self.loraA[t] = ins[i]; i += 1
                self.loraB[t] = ins[i]; i += 1
            }
            let (xo, vo) = self.blockForward(xin, vin, layer)
            return [xo, vo]
        }
        let out = checkpointed(f)(inputs)
        return (out[0], out[1])
    }

    /// body: всё кроме головы. ids [B,T] → ln_out [B,T,D].
    public func body(_ ids: MLXArray) -> MLXArray {
        let emb = embed(ids)        // [B,T,D]
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil
        for layer in 0 ..< cfg.nLayer {
            let (xo, vf): (MLXArray, MLXArray) = useBlockCheckpoint
                ? blockCheckpointed(x, vFirst, layer)
                : blockForward(x, vFirst, layer)
            x = xo
            vFirst = vf
        }
        return layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
    }

    /// Полный forward в логиты: ids [B,T] → logits [B,T,vocab].
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        proj(body(ids), "head.weight", lora: nil)
    }

    /// body с граничным состоянием: продолжить с `state` и вернуть состояние
    /// на конце. ids [B,T] → (ln_out [B,T,D], состояние после последнего
    /// РЕАЛЬНОГО токена каждой строки).
    ///
    /// Это то, на чём стоит префикс-кэш: длинный документ сворачивается один
    /// раз, а каждый запрос продолжает с готового состояния и стоит O(своей
    /// длины) вместо O(документ + запрос).
    ///
    /// - state:  продолжить с него; nil ⇒ начало последовательности.
    /// - mask:   [B,T], 1 у реального токена, 0 у right-паддинга. Без неё
    ///           пад-токены пройдут через рекуррентность и испортят КОНЕЧНОЕ
    ///           состояние (на скрытые состояния реальных токенов они не
    ///           влияют — модель каузальна).
    /// - endIdx: где снимать token-shift; обычно `lastRealIndex(lengths:)`.
    ///           nil ⇒ последняя позиция, что верно только без паддинга.
    ///
    /// Путь с состоянием намеренно НЕ оборачивается в блочный gradient
    /// checkpoint: он существует ради инференса и обучения НАД замороженной
    /// базой, где пересчитывать активации нечего и незачем.
    public func bodyWithState(
        _ ids: MLXArray, state: RWKVBatchState? = nil,
        mask: MLXArray? = nil, endIdx: MLXArray? = nil
    ) -> (MLXArray, RWKVBatchState) {
        if let state {
            precondition(state.nLayer == cfg.nLayer,
                         "состояние на \(state.nLayer) слоёв, модель на \(cfg.nLayer)")
            precondition(state.batch == ids.shape[0],
                         "batch состояния \(state.batch) != batch ids \(ids.shape[0])")
        }
        let emb = embed(ids)
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil

        var wkvs: [MLXArray] = [], tshifts: [MLXArray] = [], cshifts: [MLXArray] = []
        wkvs.reserveCapacity(cfg.nLayer)
        tshifts.reserveCapacity(cfg.nLayer)
        cshifts.reserveCapacity(cfg.nLayer)

        for layer in 0 ..< cfg.nLayer {
            let (xo, vf, hOut, ts, cs) = blockForwardWithState(
                x, vFirst, layer,
                hIn: state?.layerWKV(layer),
                mask: mask,
                tmixPrev: state?.layerTmixShift(layer),
                cmixPrev: state?.layerCmixShift(layer),
                endIdx: endIdx)
            x = xo
            vFirst = vf
            wkvs.append(hOut); tshifts.append(ts); cshifts.append(cs)
        }

        let lnOut = layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
        return (lnOut, RWKVBatchState.stacked(wkv: wkvs, tmix: tshifts, cmix: cshifts))
    }

    /// Только состояние на конце последовательности, без скрытых состояний.
    /// Обёртка над bodyWithState для случая, когда нужен лишь кэш префикса.
    public func states(_ ids: MLXArray, state: RWKVBatchState? = nil,
                       mask: MLXArray? = nil, endIdx: MLXArray? = nil) -> RWKVBatchState {
        bodyWithState(ids, state: state, mask: mask, endIdx: endIdx).1
    }

    // ─────────── Partial-finetune: разрез сети на слое f ───────────
    // token-shift внутриблочный ⇒ между блоками течёт только (x, vFirst),
    // межблочного xPrev НЕТ. Поэтому граница = (x после блока f-1, vFirst).

    /// Frozen-проход блоков [0..<f]. Возвращает (x, vFirst) на границе.
    /// Без gradient checkpoint — этот участок не обучается.
    public func boundaryState(_ ids: MLXArray, upTo f: Int) -> (MLXArray, MLXArray?) {
        let emb = embed(ids)
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil
        for layer in 0 ..< f {
            let (xo, vf) = blockForward(x, vFirst, layer)
            x = xo
            vFirst = vf
        }
        return (x, vFirst)
    }

    /// Обучаемый хвост: блоки [f..<nLayer] + ln_out. Граничные (x, vFirst)
    /// приходят из boundaryState (или из дискового кэша). Уважает useBlockCheckpoint.
    public func forwardFrom(_ x0: MLXArray, _ vFirst0: MLXArray?, from f: Int) -> MLXArray {
        var x = x0
        var vFirst = vFirst0
        for layer in f ..< cfg.nLayer {
            let (xo, vf): (MLXArray, MLXArray) = useBlockCheckpoint
                ? blockCheckpointed(x, vFirst, layer)
                : blockForward(x, vFirst, layer)
            x = xo
            vFirst = vf
        }
        return layerNorm(x, g("ln_out.weight"), g("ln_out.bias"))
    }

    /// vFirst (= v слоя 0) для пересчёта на train-шаге без хранения на диске.
    /// Слой 0 frozen ⇒ вызывать вне grad-тейпа.
    public func vFirstFrom(_ ids: MLXArray) -> MLXArray {
        let (_, vFirst) = boundaryState(ids, upTo: 1)
        return vFirst!
    }

    // ── Отладка паритета: промежуточные этапы (internal, для тестов) ──
    func debugPerLayer(_ ids: MLXArray) -> [String: MLXArray] {
        let emb = embed(ids)
        var x = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        var vFirst: MLXArray? = nil
        var out: [String: MLXArray] = [:]
        for layer in 0 ..< cfg.nLayer {
            let (h, vf, _) = tmix(layerNorm(x, g("blocks.\(layer).ln1.weight"),
                                            g("blocks.\(layer).ln1.bias")), vFirst, layer)
            vFirst = vf
            x = x + h
            x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                                   g("blocks.\(layer).ln2.bias")), layer)
            out["after_blk\(layer)"] = x
        }
        return out
    }

    func debugTmix(_ ids: MLXArray) -> [String: MLXArray] {
        let emb = embed(ids)
        let afterLn0 = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        let x = layerNorm(afterLn0, g("blocks.0.ln1.weight"), g("blocks.0.ln1.bias"))
        let p = "blocks.0.tmix."
        let B = x.shape[0], T = x.shape[1], D = cfg.nEmbd, H = cfg.nHead, S = cfg.headSize
        let xx = tokenShift(x)
        let xr = x + xx * g(p+"x_r"), xw = x + xx * g(p+"x_w"), xk = x + xx * g(p+"x_k")
        let xv = x + xx * g(p+"x_v"), xa = x + xx * g(p+"x_a"), xg = x + xx * g(p+"x_g")
        let r = linear(xr, g(p+"r_proj.weight")).reshaped([B,T,H,S])
        let k = linear(xk, g(p+"k_proj.weight")).reshaped([B,T,H,S])
        let v = linear(xv, g(p+"v_proj.weight")).reshaped([B,T,H,S])
        let gate = linear(sigmoid(linear(xg, g(p+"g_lora_A.weight"))), g(p+"g_lora_B.weight"))
        let a = sigmoid(linear(linear(xa, g(p+"a_lora_A.weight")),
                               g(p+"a_lora_B.weight"), g(p+"a_lora_B.bias"))).reshaped([B,T,H,S])
        var ww = linear(tanh(linear(xw, g(p+"w_lora_A.weight"))),
                        g(p+"w_lora_B.weight"), g(p+"w_lora_B.bias"))
        ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype).reshaped([B,T,H,S])
        let kk = l2norm(k * g(p+"k_k"))
        let k2 = k * (1.0 + (a - 1.0) * g(p+"k_a"))
        let wkv = wkv7Forward(r, ww, k2, v, -kk, kk * a)
        let outLnx = lnX(wkv.reshaped([B,T,D]), g(p+"ln_x.weight"), g(p+"ln_x.bias")).reshaped([B,T,H,S])
        let bonus = (r * k2 * g(p+"r_k")).sum(axis: -1, keepDims: true) * v
        let outF = (outLnx + bonus).reshaped([B,T,D])
        let res = linear(outF * gate, g(p+"o_proj.weight"))
        return ["r":r,"k":k,"v":v,"g":gate,"a":a,"w":ww,"kk":kk,"k2":k2,
                "wkv":wkv,"out_lnx":outLnx,"res":res]
    }

    func debugStages(_ ids: MLXArray) -> [String: MLXArray] {
        let emb = embed(ids)
        let afterLn0 = layerNorm(emb, g("ln0.weight"), g("ln0.bias"))
        let (h, _, _) = tmix(layerNorm(afterLn0, g("blocks.0.ln1.weight"),
                                       g("blocks.0.ln1.bias")), nil, 0)
        var x2 = afterLn0 + h
        x2 = x2 + cmix(layerNorm(x2, g("blocks.0.ln2.weight"),
                                 g("blocks.0.ln2.bias")), 0)
        return ["after_ln0": afterLn0, "blk0_tmix": h, "after_blk0": x2]
    }
}
