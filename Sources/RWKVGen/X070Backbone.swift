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

private func l2norm(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}

private func layerNorm(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
                       eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let varc = (x - mean).square().mean(axis: -1, keepDims: true)
    let normed = (x - mean) / sqrt(varc + eps)
    return normed * weight + bias
}

// Linear без bias: x[...,in] @ Wᵀ, W хранится [out, in].
private func linear(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
    matmul(x, w.transposed())
}
private func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
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
    var w: [String: MLXArray]

    // ── LoRA / QLoRA / training hooks ──
    // Инертны для обычного инференса (пустые словари + trainLayers пуст) ⇒
    // поведение идентично исходному, parity-тесты остаются зелёными.
    /// Слои, чьё WKV считается через дифференцируемое ядро (wkv7Train).
    public var trainLayers: Set<Int> = []
    /// Подмена весов на обучаемые (fp32) во время valueAndGrad для full-weight
    /// partial-finetune верхних слоёв. nil ⇒ читаются frozen-веса (инференс не меняется).
    public var wOverride: [String: MLXArray]? = nil
    /// Если true — каждый блок (ln1+tmix+resid+ln2+cmix+resid) считается через
    /// gradient checkpoint (recompute в backward). Дифф-входы (x, v_first, LoRA)
    /// протягиваются явными аргументами чекпоинта; frozen-веса захватываются.
    public var useBlockCheckpoint = false
    /// LoRA-адаптеры по таргету: "blocks.L.tmix.r_proj" и т.п.
    var loraA: [String: MLXArray] = [:]      // [rank, in]
    var loraB: [String: MLXArray] = [:]      // [out, rank]
    var loraScale: [String: Float] = [:]     // alpha / rank
    /// Квантованная замороженная база по имени веса (напр. "...r_proj.weight", "head.weight", "emb.weight").
    var quant: [String: QuantBase] = [:]

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
    var rwkvqSidecar: RwkvqSidecar? = nil
    var rwkvqKeys: [String: String] = [:]     // x070-имя → world-имя в сайдкаре

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

    // Чтение веса с учётом wOverride (обучаемая подмена) → frozen.
    private func wv(_ key: String) -> MLXArray { wOverride?[key] ?? w[key]! }
    private func g(_ key: String) -> MLXArray { wv(key) }

    // Восстановленный вес из .rwkvq, если он оттуда. ТРАНЗИЕНТНЫЙ: не
    // кэшируется намеренно — база обязана жить в памяти сжатой, иначе смысл
    // квантованной базы теряется (кэш плотных весов превращает QLoRA в LoRA
    // с лишними шагами).
    private func rwkvqWeight(_ wKey: String) -> MLXArray? {
        guard let worldKey = rwkvqKeys[wKey], let sc = rwkvqSidecar else { return nil }
        return try? sc.dequantize(worldKey)
    }

    // База проекции: если есть quant — x·Wᵀ с деквантизацией на лету
    // (веса [out,in] ⇒ transpose: true), иначе обычный linear.
    private func baseProj(_ x: MLXArray, _ wKey: String) -> MLXArray {
        if let q = quant[wKey] {
            return quantizedMM(x, q.wq, scales: q.scales, biases: q.biases,
                               transpose: true, groupSize: q.groupSize, bits: q.bits)
        }
        if let dense = rwkvqWeight(wKey) {
            // Деквант в fp32; приводим к типу вычислений, чтобы matmul не
            // тянул всю цепочку в fp32 и не ломал bf16-профиль памяти.
            return matmul(x, dense.asType(x.dtype).transposed())
        }
        return matmul(x, wv(wKey).transposed())
    }

    // Проекция с опциональным LoRA: base + scale·(x·Aᵀ)·Bᵀ. target=nil ⇒ только база.
    private func proj(_ x: MLXArray, _ wKey: String, lora target: String?) -> MLXArray {
        let base = baseProj(x, wKey)
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
        if let dense = rwkvqWeight("emb.weight") {
            // ВНИМАНИЕ: sb6-ядро разворачивает таблицу целиком, а для словаря
            // 65536×768 это ~200 МБ транзиента на КАЖДЫЙ проход. Строчной
            // выборки формат не поддерживает: коды упакованы по блокам вдоль
            // входной оси, и достать одну строку дешевле, чем блок, нельзя.
            // Поэтому emb в сайдкар обычно и не отдают — см. attachRwkvq.
            return dense.asType(w["ln0.weight"]?.dtype ?? .bfloat16).take(ids, axis: 0)
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
        let B = x.shape[0], T = x.shape[1], D = x.shape[2]
        let head = prev0?.asType(x.dtype) ?? MLXArray.zeros([B, 1, D], dtype: x.dtype)
        let shifted = concatenated([head, x[0..., 0 ..< (T - 1)]], axis: 1)
        return shifted - x
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

    // time-mix блока layer. Возвращает (выход, обновлённый v_first).
    //
    // Состояние и паддинг (оба опциональны, nil ⇒ прежнее поведение):
    //   shiftPrev — вход на позиции −1 для token-shift, [B,1,D];
    //   hIn       — начальная матрица WKV [B,H,S,S];
    //   mask      — [B,T], 1 у реального токена, 0 у right-паддинга;
    //   wantState — вернуть конечную матрицу WKV третьим элементом.
    private func tmix(_ x: MLXArray, _ vFirst: MLXArray?, _ layer: Int,
                      shiftPrev: MLXArray? = nil, hIn: MLXArray? = nil,
                      mask: MLXArray? = nil, wantState: Bool = false)
        -> (MLXArray, MLXArray, MLXArray?) {
        let p = "blocks.\(layer).tmix."
        let B = x.shape[0], T = x.shape[1], D = cfg.nEmbd
        let H = cfg.nHead, S = cfg.headSize

        let xx = tokenShift(x, shiftPrev)
        let xr = x + xx * g(p + "x_r")
        let xw = x + xx * g(p + "x_w")
        let xk = x + xx * g(p + "x_k")
        let xv = x + xx * g(p + "x_v")
        let xa = x + xx * g(p + "x_a")
        let xg = x + xx * g(p + "x_g")

        var r = proj(xr, p + "r_proj.weight", lora: p + "r_proj").reshaped([B, T, H, S])
        var k = proj(xk, p + "k_proj.weight", lora: p + "k_proj").reshaped([B, T, H, S])
        var v = proj(xv, p + "v_proj.weight", lora: p + "v_proj").reshaped([B, T, H, S])

        // gate: B(sigmoid(A(xg))) — sigmoid ВНУТРИ, линейно наружу, без bias.
        let gate = linear(sigmoid(linear(xg, g(p + "g_lora_A.weight"))),
                          g(p + "g_lora_B.weight"))

        // value-residual (слои > 0): v0 = bias v_lora_B
        var vFirstOut: MLXArray
        if layer == 0 {
            vFirstOut = v
        } else {
            let vv = sigmoid(linear(linear(xv, g(p + "v_lora_A.weight")),
                                    g(p + "v_lora_B.weight"), g(p + "v_lora_B.bias")))
                        .reshaped([B, T, H, S])
            v = v + (vFirst! - v) * vv
            vFirstOut = vFirst!
        }

        // iclr a: sigmoid(a0 + B(A(xa))) — БЕЗ tanh.
        let a = sigmoid(linear(linear(xa, g(p + "a_lora_A.weight")),
                               g(p + "a_lora_B.weight"), g(p + "a_lora_B.bias")))
                    .reshaped([B, T, H, S])

        // decay w: exp(-0.606531 * sigmoid(w0 + B(tanh(A(xw))))), reductions в fp32.
        var ww = linear(tanh(linear(xw, g(p + "w_lora_A.weight"))),
                        g(p + "w_lora_B.weight"), g(p + "w_lora_B.bias"))
        ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype)
        ww = ww.reshaped([B, T, H, S])

        // kk = l2norm(k * k_k);  k = k*(1+(a-1)*k_a)
        let kk = l2norm(k * g(p + "k_k"))
        k = k * (1.0 + (a - 1.0) * g(p + "k_a"))

        // WKV-7: a_kernel = -kk, b_kernel = kk * a
        var kWkv = k
        var bWkv = kk * a

        // Right-padding: делаем пад-позиции НЕЙТРАЛЬНЫМИ для рекуррентности.
        //   w←1, k←0, b←0  ⇒  h' = 1·h + v·0ᵀ + sa·0ᵀ = h
        // Состояние строки замирает на её последнем реальном токене и не
        // зависит ни от числа пад-токенов, ни от соседей по батчу.
        //
        // a (=-kk) и v НЕ маскируются намеренно: k=0 уже убивает член v·kᵀ,
        // b=0 — член sa·bᵀ, поэтому их значения на паддинге ни на что не
        // влияют. Лишняя маскировка была бы просто лишней арифметикой.
        //
        // Выход на пад-позициях остаётся мусорным — это нормально: модель
        // каузальна, мусор может попасть только на пад-позиции следующих
        // слоёв и до реальных токенов не доходит.
        if let mask {
            let m = mask.reshaped([B, T, 1, 1]).asType(ww.dtype)
            ww = ww * m + (1.0 - m)
            kWkv = kWkv * m
            bWkv = bWkv * m
        }

        var hOut: MLXArray? = nil
        var out: MLXArray
        if wantState || hIn != nil {
            let (o, h) = trainLayers.contains(layer)
                ? wkv7TrainWithState(r, ww, kWkv, v, -kk, bWkv, hIn)
                : wkv7ForwardWithState(r, ww, kWkv, v, -kk, bWkv, hIn)
            out = o
            hOut = h
        } else {
            out = trainLayers.contains(layer)
                ? wkv7Train(r, ww, kWkv, v, -kk, bWkv)
                : wkv7Forward(r, ww, kWkv, v, -kk, bWkv)    // [B,T,H,S]
        }

        // Порядок официала: ln_x (GroupNorm) ДО bonus.
        out = lnX(out.reshaped([B, T, D]), g(p + "ln_x.weight"), g(p + "ln_x.bias"))
              .reshaped([B, T, H, S])
        // bonus считается по НЕмаскированному k — так же, как в Python: маска
        // существует только ради рекуррентности, а bonus живёт на позиции и на
        // состояние не влияет.
        let bonus = (r * k * g(p + "r_k")).sum(axis: -1, keepDims: true) * v
        out = (out + bonus).reshaped([B, T, D])

        let res = proj(out * gate, p + "o_proj.weight", lora: p + "o_proj")
        return (res, vFirstOut, hOut)
    }

    // channel-mix: value(relu(key(xk))^2). token-shift свой, нулевой паддинг.
    private func cmix(_ x: MLXArray, _ layer: Int, shiftPrev: MLXArray? = nil) -> MLXArray {
        let p = "blocks.\(layer).cmix."
        let xx = tokenShift(x, shiftPrev)
        let xk = x + xx * g(p + "x_k")
        let h = relu(proj(xk, p + "key.weight", lora: p + "key"))
        return proj(h * h, p + "value.weight", lora: p + "value")
    }

    /// Один блок: ln1+tmix+resid+ln2+cmix+resid. (x0, vFirst?) -> (x', vFirstOut).
    func blockForward(_ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int) -> (MLXArray, MLXArray) {
        let (h, vf, _) = tmix(layerNorm(x0, g("blocks.\(layer).ln1.weight"),
                                        g("blocks.\(layer).ln1.bias")), vFirst, layer)
        var x = x0 + h
        x = x + cmix(layerNorm(x, g("blocks.\(layer).ln2.weight"),
                               g("blocks.\(layer).ln2.bias")), layer)
        return (x, vf)
    }

    /// Тот же блок, но с граничным состоянием. Возвращает
    /// (x', vFirstOut, hOut, tmixShiftOut, cmixShiftOut).
    ///
    /// Сдвиги снимаются с позиции endIdx (последний РЕАЛЬНЫЙ токен строки), а
    /// не с конца паддинга — иначе продолжение стартовало бы со сдвига,
    /// снятого с пад-позиции.
    func blockForwardWithState(
        _ x0: MLXArray, _ vFirst: MLXArray?, _ layer: Int,
        hIn: MLXArray?, mask: MLXArray?, tmixPrev: MLXArray?,
        cmixPrev: MLXArray?, endIdx: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let x1 = layerNorm(x0, g("blocks.\(layer).ln1.weight"),
                           g("blocks.\(layer).ln1.bias"))
        let (h, vf, hOut) = tmix(x1, vFirst, layer, shiftPrev: tmixPrev,
                                 hIn: hIn, mask: mask, wantState: true)
        var x = x0 + h
        let x2 = layerNorm(x, g("blocks.\(layer).ln2.weight"),
                           g("blocks.\(layer).ln2.bias"))
        x = x + cmix(x2, layer, shiftPrev: cmixPrev)
        return (x, vf, hOut!, gatherLast(x1, endIdx), gatherLast(x2, endIdx))
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
