import Foundation
import MLX
import MLXNN
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Блок RWKV-7 (ln1 + tmix + резид. + ln2 + cmix + резид.) как самостоятельная
//  сущность, а не приватный метод бэкбона.
//
//  Зачем. Голова реранкера — это стек РWKV-блоков поверх состояния базы:
//  блоки инициализируются весами выбранных слоёв, но дальше живут своей
//  жизнью, обучаются и сохраняются отдельно от базы. Пока блок был зашит в
//  X070Backbone, у головы было два выхода, и оба плохие: скопировать
//  арифметику tmix/cmix (вторая копия, обязанная совпадать побитово, — это
//  расхождение через полгода) или притвориться бэкбоном (голова из одного
//  блока не бэкбон, и всё, что читает cfg.nLayer, начнёт врать).
//
//  Решение — разделить «арифметику блока» и «откуда берутся веса»:
//
//    RWKVBlockContext   — откуда веса, как считать проекции (плотно /
//                         квантованно / с LoRA), каким ядром считать WKV.
//    rwkvBlockForward   — сама арифметика, ОДНА копия на весь пакет.
//    RWKVBlock          — блок с СОБСТВЕННЫМИ весами (реализация контекста).
//
//  X070Backbone реализует тот же контекст поверх своего плоского словаря
//  весов, поэтому его поведение не изменилось ни на бит — это проверяется
//  характеризационными тестами, а не утверждается.
// ───────────────────────────────────────────────────────────────────────

/// Каким ядром считать WKV на данной длине.
public enum WKVKernelPath: Sendable {
    /// Frozen Metal-ядро: быстро, backward отсутствует.
    case forward
    /// Дифференцируемый checkpoint: нужен T, кратный WKV7_CHUNK.
    case train
    /// Один токен на чистых MLX-операциях. Требует T == 1, автоград бесплатно.
    case step
}

/// Всё, что блоку нужно от окружения. Намеренно узкий протокол: блок не знает
/// ни про словарь весов, ни про квантование, ни про LoRA — только про то, что
/// «вот параметр по имени» и «вот проекция входа».
public protocol RWKVBlockContext: AnyObject {
    var nEmbd: Int { get }
    var nHead: Int { get }
    var headSize: Int { get }

    /// Есть ли у блока value-residual. У ПЕРВОГО блока стека её нет: v_first
    /// именно здесь и рождается. Это свойство позиции в стеке, а не слоя базы,
    /// из которого блок инициализирован.
    var hasValueResidual: Bool { get }

    /// Параметр по имени, относительному для блока: "ln1.weight",
    /// "tmix.x_r", "tmix.w_lora_B.bias" и т.д.
    func param(_ key: String) -> MLXArray

    /// Линейная проекция без bias. Отдельно от `param`, потому что реализация
    /// может быть не matmul'ом: квантованная база разворачивает вес на лету,
    /// LoRA добавляет низкоранговую поправку.
    func project(_ x: MLXArray, _ key: String) -> MLXArray

    /// GroupNorm по головам БЕЗ affine (вес и смещение ln_x применяются
    /// снаружи, они per-layer). Вход [N, D], где N = B·T.
    func groupNormHeads(_ x: MLXArray) -> MLXArray

    /// Каким ядром считать WKV при данной длине.
    func wkvPath(_ T: Int) -> WKVKernelPath
}

// ─────────────────── Арифметика блока (единственная копия) ───────────────────

/// token-shift: prev[t] = x[t-1]. Возвращает xx = prev − x.
///
/// prev0 — вход на позиции −1, то есть последний токен ПРЕДЫДУЩЕГО куска той
/// же последовательности [B,1,D]. nil ⇒ нули (начало последовательности).
/// Без него продолжение расходится со сплошным проходом ровно на первом
/// токене каждого слоя.
func rwkvTokenShift(_ x: MLXArray, _ prev0: MLXArray? = nil) -> MLXArray {
    let B = x.shape[0], T = x.shape[1], D = x.shape[2]
    let head = prev0?.asType(x.dtype) ?? MLXArray.zeros([B, 1, D], dtype: x.dtype)
    let shifted = concatenated([head, x[0..., 0 ..< (T - 1)]], axis: 1)
    return shifted - x
}

/// time-mix. Возвращает (выход, обновлённый v_first, конечное состояние WKV?).
///
/// Опциональные аргументы (nil ⇒ поведение без состояния):
///   shiftPrev — вход на позиции −1 для token-shift, [B,1,D];
///   hIn       — начальная матрица WKV [B,H,S,S];
///   mask      — [B,T], 1 у реального токена, 0 у right-паддинга;
///   wantState — вернуть конечную матрицу WKV третьим элементом.
func rwkvTmixForward(
    _ x: MLXArray, _ vFirst: MLXArray?, _ ctx: RWKVBlockContext,
    shiftPrev: MLXArray? = nil, hIn: MLXArray? = nil,
    mask: MLXArray? = nil, wantState: Bool = false
) -> (MLXArray, MLXArray, MLXArray?) {
    let B = x.shape[0], T = x.shape[1], D = ctx.nEmbd
    let H = ctx.nHead, S = ctx.headSize
    func g(_ key: String) -> MLXArray { ctx.param("tmix." + key) }
    func pr(_ y: MLXArray, _ key: String) -> MLXArray { ctx.project(y, "tmix." + key) }

    let xx = rwkvTokenShift(x, shiftPrev)
    let xr = x + xx * g("x_r")
    let xw = x + xx * g("x_w")
    let xk = x + xx * g("x_k")
    let xv = x + xx * g("x_v")
    let xa = x + xx * g("x_a")
    let xg = x + xx * g("x_g")

    let r = pr(xr, "r_proj.weight").reshaped([B, T, H, S])
    var k = pr(xk, "k_proj.weight").reshaped([B, T, H, S])
    var v = pr(xv, "v_proj.weight").reshaped([B, T, H, S])

    // gate: B(sigmoid(A(xg))) — sigmoid ВНУТРИ, линейно наружу, без bias.
    let gate = linear(sigmoid(linear(xg, g("g_lora_A.weight"))),
                      g("g_lora_B.weight"))

    // value-residual: v0 = bias v_lora_B. У первого блока стека её нет —
    // v_first здесь и рождается.
    var vFirstOut: MLXArray
    if !ctx.hasValueResidual {
        vFirstOut = v
    } else {
        let vv = sigmoid(linear(linear(xv, g("v_lora_A.weight")),
                                g("v_lora_B.weight"), g("v_lora_B.bias")))
                    .reshaped([B, T, H, S])
        v = v + (vFirst! - v) * vv
        vFirstOut = vFirst!
    }

    // iclr a: sigmoid(a0 + B(A(xa))) — БЕЗ tanh.
    let a = sigmoid(linear(linear(xa, g("a_lora_A.weight")),
                           g("a_lora_B.weight"), g("a_lora_B.bias")))
                .reshaped([B, T, H, S])

    // decay w: exp(-0.606531 * sigmoid(w0 + B(tanh(A(xw))))), reductions в fp32.
    var ww = linear(tanh(linear(xw, g("w_lora_A.weight"))),
                    g("w_lora_B.weight"), g("w_lora_B.bias"))
    ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype)
    ww = ww.reshaped([B, T, H, S])

    // kk = l2norm(k * k_k);  k = k*(1+(a-1)*k_a)
    let kk = l2norm(k * g("k_k"))
    k = k * (1.0 + (a - 1.0) * g("k_a"))

    // WKV-7: a_kernel = -kk, b_kernel = kk * a
    var kWkv = k
    var bWkv = kk * a

    // Right-padding: делаем пад-позиции НЕЙТРАЛЬНЫМИ для рекуррентности.
    //   w←1, k←0, b←0  ⇒  h' = 1·h + v·0ᵀ + sa·0ᵀ = h
    // Состояние строки замирает на её последнем реальном токене и не зависит
    // ни от числа пад-токенов, ни от соседей по батчу.
    //
    // a (=-kk) и v НЕ маскируются намеренно: k=0 уже убивает член v·kᵀ,
    // b=0 — член sa·bᵀ, поэтому их значения на паддинге ни на что не влияют.
    //
    // Выход на пад-позициях остаётся мусорным — это нормально: модель
    // каузальна, мусор может попасть только на пад-позиции следующих слоёв
    // и до реальных токенов не доходит.
    if let mask {
        let m = mask.reshaped([B, T, 1, 1]).asType(ww.dtype)
        ww = ww * m + (1.0 - m)
        kWkv = kWkv * m
        bWkv = bWkv * m
    }

    var hOut: MLXArray? = nil
    var out: MLXArray
    switch ctx.wkvPath(T) {
    case .step:
        // Несколько токенов: ядру потребовалось бы 16 шагов вместо T, а
        // обучаемому — ещё и кратность CHUNK, которой у пары зондов нет.
        let (o, h) = wkv7Steps(r, ww, kWkv, v, -kk, bWkv, hIn)
        out = o
        hOut = h
    case .train:
        if wantState || hIn != nil {
            let (o, h) = wkv7TrainWithState(r, ww, kWkv, v, -kk, bWkv, hIn)
            out = o
            hOut = h
        } else {
            out = wkv7Train(r, ww, kWkv, v, -kk, bWkv)
        }
    case .forward:
        if wantState || hIn != nil {
            let (o, h) = wkv7ForwardWithState(r, ww, kWkv, v, -kk, bWkv, hIn)
            out = o
            hOut = h
        } else {
            out = wkv7Forward(r, ww, kWkv, v, -kk, bWkv)
        }
    }

    // Порядок официала: ln_x (GroupNorm) ДО bonus.
    //
    // ln_x per-token: канон RWKV-7 — F.group_norm(x.view(B*T, C), H). Подача
    // [B,T,D] в GroupNorm усреднила бы по ВСЕМ T внутри головы, то есть дала
    // бы утечку будущего; ловится рекуррентным parity-тестом.
    let normed = ctx.groupNormHeads(out.reshaped([B * T, D])).reshaped([B, T, D])
    out = (normed * g("ln_x.weight") + g("ln_x.bias")).reshaped([B, T, H, S])

    // bonus считается по НЕмаскированному k: маска существует только ради
    // рекуррентности, а bonus живёт на позиции и на состояние не влияет.
    let bonus = (r * k * g("r_k")).sum(axis: -1, keepDims: true) * v
    out = (out + bonus).reshaped([B, T, D])

    return (pr(out * gate, "o_proj.weight"), vFirstOut, hOut)
}

/// channel-mix: value(relu(key(xk))²). token-shift свой, нулевой паддинг.
func rwkvCmixForward(_ x: MLXArray, _ ctx: RWKVBlockContext,
                     shiftPrev: MLXArray? = nil) -> MLXArray {
    let xx = rwkvTokenShift(x, shiftPrev)
    let xk = x + xx * ctx.param("cmix.x_k")
    let h = relu(ctx.project(xk, "cmix.key.weight"))
    return ctx.project(h * h, "cmix.value.weight")
}

/// Полный блок без состояния: (x0, vFirst?) -> (x', vFirstOut).
func rwkvBlockForward(_ x0: MLXArray, _ vFirst: MLXArray?,
                      _ ctx: RWKVBlockContext) -> (MLXArray, MLXArray) {
    let (h, vf, _) = rwkvTmixForward(
        layerNorm(x0, ctx.param("ln1.weight"), ctx.param("ln1.bias")),
        vFirst, ctx)
    var x = x0 + h
    x = x + rwkvCmixForward(
        layerNorm(x, ctx.param("ln2.weight"), ctx.param("ln2.bias")), ctx)
    return (x, vf)
}

/// Тот же блок с граничным состоянием. Возвращает
/// (x', vFirstOut, hOut, tmixShiftOut, cmixShiftOut).
///
/// Сдвиги снимаются с позиции endIdx (последний РЕАЛЬНЫЙ токен строки), а не
/// с конца паддинга — иначе продолжение стартовало бы со сдвига, снятого с
/// пад-позиции.
func rwkvBlockForwardWithState(
    _ x0: MLXArray, _ vFirst: MLXArray?, _ ctx: RWKVBlockContext,
    hIn: MLXArray?, mask: MLXArray?, tmixPrev: MLXArray?,
    cmixPrev: MLXArray?, endIdx: MLXArray?
) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
    let x1 = layerNorm(x0, ctx.param("ln1.weight"), ctx.param("ln1.bias"))
    let (h, vf, hOut) = rwkvTmixForward(x1, vFirst, ctx, shiftPrev: tmixPrev,
                                        hIn: hIn, mask: mask, wantState: true)
    var x = x0 + h
    let x2 = layerNorm(x, ctx.param("ln2.weight"), ctx.param("ln2.bias"))
    x = x + rwkvCmixForward(x2, ctx, shiftPrev: cmixPrev)
    return (x, vf, hOut!, gatherLast(x1, endIdx), gatherLast(x2, endIdx))
}

// ─────────────────── Контекст поверх бэкбона ───────────────────

/// Блок слоя `layer` внутри X070Backbone: веса из плоского словаря бэкбона,
/// проекции — его же путём (плотно / .rwkvq / stock-quant / +LoRA), ядро — по
/// `trainLayers`.
///
/// `unowned`, а не `let`: бэкбон держит контексты в кэше, контекст держал бы
/// бэкбон — цикл удержания на ровном месте. Контекст не переживает бэкбон по
/// построению, он только его деталь.
final class BackboneBlockContext: RWKVBlockContext {
    private unowned let backbone: X070Backbone
    let layer: Int

    init(_ backbone: X070Backbone, _ layer: Int) {
        self.backbone = backbone
        self.layer = layer
    }

    var nEmbd: Int { backbone.cfg.nEmbd }
    var nHead: Int { backbone.cfg.nHead }
    var headSize: Int { backbone.cfg.headSize }

    /// В базе v_first рождается в СЛОЕ 0 — здесь это свойство слоя, а не
    /// позиции в стеке (у бэкбона они совпадают).
    var hasValueResidual: Bool { layer > 0 }

    func param(_ key: String) -> MLXArray {
        backbone.weightForBlock("blocks.\(layer).\(key)")
    }

    func project(_ x: MLXArray, _ key: String) -> MLXArray {
        let full = "blocks.\(layer).\(key)"
        // Имя LoRA-таргета — имя веса без ".weight": так они и заведены
        // (blocks.3.tmix.r_proj против blocks.3.tmix.r_proj.weight).
        precondition(full.hasSuffix(".weight"), "проекция \(full) не .weight")
        return backbone.projectForBlock(x, full, lora: String(full.dropLast(7)))
    }

    func groupNormHeads(_ x: MLXArray) -> MLXArray { backbone.groupNormRaw(x) }

    func wkvPath(_ T: Int) -> WKVKernelPath {
        // Намеренно БЕЗ ветки .step на T == 1: одношаговый путь численно
        // эквивалентен ядру, но не равен ему побитово, а через T == 1 идёт
        // рекуррентный декод. Переключить его — значит сдвинуть все
        // существующие замеры паритета ради экономии, которая инференсу
        // базы ничего не даёт (ядро на T=1 и так один launch).
        backbone.trainLayers.contains(layer) ? .train : .forward
    }
}

// ─────────────────── Блок с собственными весами ───────────────────

public enum RWKVBlockError: Error, CustomStringConvertible {
    case missingWeight(String)
    case noValueResidualSource(Int)

    public var description: String {
        switch self {
        case .missingWeight(let k):
            return "в базе нет веса \(k) — блок нечем инициализировать"
        case .noValueResidualSource(let l):
            return """
            блок инициализируется слоем \(l), у которого нет v_lora (слой 0 \
            её не имеет), но в стеке он стоит не первым и value-residual ему \
            нужна. Взять ранг неоткуда: в модели нет ни одного слоя с v_lora.
            """
        }
    }
}

/// Блок RWKV-7 с собственными весами — строительный элемент головы реранкера.
///
/// Не подмодуль базы, а её СИБЛИНГ. Разница не косметическая: заморозка базы
/// (и всё, что её вызывает — навеска LoRA, квантование) не должна молча
/// заморозить голову, а голова не должна попасть в чекпоинт базы.
///
/// Веса хранятся плоским словарём с относительными именами
/// ("ln1.weight", "tmix.x_r", "cmix.key.weight"), как в базе, но без
/// префикса "blocks.N.". Это позволяет копировать веса слоя базы один в один.
public final class RWKVBlock: RWKVBlockContext {

    public let cfg: X070Config
    /// Позиция в СТЕКЕ головы (не индекс слоя базы). От неё зависит только
    /// одно: есть ли у блока value-residual.
    public let index: Int

    /// Собственные веса блока.
    public private(set) var w: [String: MLXArray]

    /// Подмена весов на обучаемые (fp32) во время valueAndGrad. nil ⇒ читаются
    /// собственные. Тот же механизм, что у бэкбона: градиент в mlx-swift
    /// берётся по плоскому массиву тензоров, а не по дереву модуля.
    public var wOverride: [String: MLXArray]? = nil

    /// Считать ли WKV дифференцируемым ядром на длинах от CHUNK и выше.
    /// На коротких входах (штатный режим головы) не влияет: там всегда
    /// развёрнутый шаг, он и так дифференцируем.
    public var trainable = true

    private let groupNorm: GroupNorm

    public init(weights: [String: MLXArray], cfg: X070Config, index: Int) {
        self.cfg = cfg
        self.index = index
        self.w = weights
        self.groupNorm = GroupNorm(groupCount: cfg.nHead, dimensions: cfg.nEmbd,
                                   eps: 64e-5, affine: false, pytorchCompatible: true)
    }

    // ── RWKVBlockContext ──
    public var nEmbd: Int { cfg.nEmbd }
    public var nHead: Int { cfg.nHead }
    public var headSize: Int { cfg.headSize }
    public var hasValueResidual: Bool { index > 0 }

    public func param(_ key: String) -> MLXArray {
        if let o = wOverride?[key] { return o }
        guard let v = w[key] else {
            preconditionFailure("у блока нет веса \(key)")
        }
        return v
    }

    public func project(_ x: MLXArray, _ key: String) -> MLXArray {
        matmul(x, param(key).transposed())
    }

    public func groupNormHeads(_ x: MLXArray) -> MLXArray { groupNorm(x) }

    /// Короче одного чанка — развёрнутый шаг, иначе ядро.
    ///
    /// Граница не подобрана: ядро добивает длину до кратной CHUNK=16
    /// no-op шагами, поэтому ниже CHUNK добивка составляет БОЛЬШУЮ часть
    /// работы (при T=1 — пятнадцать шестнадцатых). Обучаемое ядро на такой
    /// длине не работает вовсе: оно требует кратности и падает, а не
    /// добивает само.
    ///
    /// Штатный режим головы (T = nProbe, обычно 1) попадает сюда всегда.
    public func wkvPath(_ T: Int) -> WKVKernelPath {
        if T < WKV7_CHUNK { return .step }
        return trainable ? .train : .forward
    }

    // ── Проход ──

    /// (x, vFirst?) -> (x', vFirstOut). Состояние WKV стартует с нуля.
    public func callAsFunction(_ x: MLXArray, _ vFirst: MLXArray?) -> (MLXArray, MLXArray) {
        rwkvBlockForward(x, vFirst, self)
    }

    /// Проход поверх ЗАДАННОГО состояния — то, ради чего блок и вынут наружу.
    /// hIn [B,H,S,S] приходит от базы; hOut отдаётся наружу, чтобы блоки
    /// можно было ставить друг на друга.
    public func callAsFunction(_ x: MLXArray, _ vFirst: MLXArray?, hIn: MLXArray?)
        -> (MLXArray, MLXArray, MLXArray) {
        let (h, vf, hOut) = rwkvTmixForward(
            layerNorm(x, param("ln1.weight"), param("ln1.bias")),
            vFirst, self, hIn: hIn, wantState: true)
        var y = x + h
        y = y + rwkvCmixForward(
            layerNorm(y, param("ln2.weight"), param("ln2.bias")), self)
        return (y, vf, hOut!)
    }

    // ── Веса ──

    /// Имена весов в детерминированном порядке.
    public var weightKeys: [String] { Array(w.keys).sorted() }

    /// Заменить веса (после обучения или загрузки чекпоинта).
    public func setWeights(_ ps: [String: MLXArray]) {
        for (k, v) in ps { w[k] = v }
    }

    /// Привести ВСЕ веса блока к одному типу.
    ///
    /// Нужно потому, что инициализация из базы копирует веса как есть, а
    /// официальные чекпоинты лежат в bf16: 8 бит мантиссы против 24 у fp32.
    /// При lr порядка 1e-4 и весах порядка 0.05 часть шага оптимизатора
    /// оказывается меньше кванта представления и теряется на округлении —
    /// лосс при этом убывает (за счёт последнего линейного слоя головы), так
    /// что без этого проблема не видна. База остаётся в своём типе, она
    /// заморожена и её точность здесь ни при чём.
    public func setDType(_ dtype: DType) {
        for (k, v) in w { w[k] = v.asType(dtype) }
    }

    // ── Инициализация из базы ──

    /// Скопировать в блок веса слоя `layer` базы.
    ///
    /// `index` — позиция в стеке головы, она же решает судьбу value-residual.
    /// Несовпадение случаев разбирается так:
    ///
    ///   index == 0, у слоя есть v_lora  → v_lora просто не копируется:
    ///       первый блок стека её не использует.
    ///   index  > 0, у слоя НЕТ v_lora (слой базы 0) → веса синтезируются
    ///       нейтральными: v_lora_B = 0, bias = −10, откуда sigmoid ≈ 4.5e-5
    ///       и v_first практически не подмешивается. Альтернатива —
    ///       случайная инициализация — означала бы, что блок стартует со
    ///       случайной примесью чужого v, и это никак не проявилось бы,
    ///       кроме худшего результата. Ранг берётся у любого слоя базы,
    ///       где v_lora есть.
    public static func fromBase(_ base: X070Backbone, layer: Int, index: Int,
                                dtype: DType = .float32) throws -> RWKVBlock {
        let prefix = "blocks.\(layer)."
        var weights: [String: MLXArray] = [:]
        for key in base.weightKeys where key.hasPrefix(prefix) {
            weights[String(key.dropFirst(prefix.count))] = base.weight(key)
        }

        let needsV = index > 0
        let hasV = weights["tmix.v_lora_B.weight"] != nil
        if needsV && !hasV {
            // Ранг v_lora — у любого слоя, который её имеет.
            guard let donor = (0 ..< base.cfg.nLayer).first(where: {
                base.hasWeight("blocks.\($0).tmix.v_lora_B.weight")
            }) else { throw RWKVBlockError.noValueResidualSource(layer) }
            let a = base.weight("blocks.\(donor).tmix.v_lora_A.weight")
            let b = base.weight("blocks.\(donor).tmix.v_lora_B.weight")
            let bias = base.weight("blocks.\(donor).tmix.v_lora_B.bias")
            weights["tmix.v_lora_A.weight"] = MLXArray.zeros(a.shape, dtype: a.dtype)
            weights["tmix.v_lora_B.weight"] = MLXArray.zeros(b.shape, dtype: b.dtype)
            weights["tmix.v_lora_B.bias"] =
                MLXArray.full(bias.shape, values: MLXArray(Float(-10)), dtype: bias.dtype)
        } else if !needsV {
            for key in ["tmix.v_lora_A.weight", "tmix.v_lora_B.weight",
                        "tmix.v_lora_B.bias"] {
                weights.removeValue(forKey: key)
            }
        }

        let block = RWKVBlock(weights: weights, cfg: base.cfg, index: index)
        block.setDType(dtype)
        return block
    }
}
