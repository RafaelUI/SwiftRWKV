import Foundation
import MLX
import MLXRandom

// ───────────────────────────────────────────────────────────────────────
//  Инициализация весов RWKV-7 x070 «с нуля» — для предобучения.
//
//  Откуда взято
//  ────────────
//  Порт официальной инициализации из BlinkDL/RWKV-LM,
//  RWKV-v7/train_temp/src/model.py (RWKV_Tmix_x070.__init__,
//  RWKV_CMix_x070.__init__, RWKV.generate_init_weight).
//
//  Портировать было НЕОТКУДА в рамках rwkv-metal: тамошний init_weights
//  относится к другой, упрощённой модели (rwkv_metal/model/rwkv7.py — ранги
//  фиксированы 64, у w_lora_B нет bias), а x070-модель веса только ЗАГРУЖАЕТ
//  и не инициализирует. X070Backbone реализует именно x070, поэтому взят
//  официальный источник, а не упрощённый.
//
//  Правила из rwkv-metal при этом воспроизводятся — они оказались огрублением
//  официальных: «k_proj демпфируется» ⇒ у официала k_proj получает разброс
//  0.05/√C против 0.5/√C у r/v_proj, то есть ровно в 10 раз меньше;
//  «LoRA-B → нули, чтобы динамика была нейтральна на старте» ⇒ у официала
//  нулевые LoRA-A, что даёт ту же нейтральность.
//
//  Почему это важно, а не косметика
//  ─────────────────────────────────
//  Кривая w0 (decay) задаёт РАЗБРОС горизонтов памяти по каналам. Плоская
//  инициализация дала бы всем каналам одинаковое затухание, и модель потеряла
//  бы то, ради чего в RWKV вообще есть поканалный decay. Это не «чуть хуже
//  сходится» — это другая модель.
// ───────────────────────────────────────────────────────────────────────

public struct X070InitConfig: Sendable {

    /// Ранги low-rank проекций. По умолчанию — формула rwkv-metal
    /// (`lora_ranks`), а НЕ официальные «suggestion»-константы из RWKV-LM
    /// (2.5/2.5/1.7/5 × √C). Причина: так обученная в Swift модель читается
    /// rwkv-metal без переходников. Официальные значения доступны через
    /// `.official`, если нужна совместимость с чекпоинтами RWKV-LM.
    public enum LoRARanks: Sendable {
        case rwkvMetal
        case official
        case explicit(w: Int, a: Int, v: Int, g: Int)

        func ranks(nEmbd D: Int) -> (w: Int, a: Int, v: Int, g: Int) {
            func f(_ c: Double, _ p: Double) -> Int {
                Swift.max(32, Int((c * pow(Double(D), p) / 32).rounded()) * 32)
            }
            switch self {
            case .rwkvMetal:
                return (f(1.8, 0.5), f(1.8, 0.5), f(1.3, 0.5), f(0.6, 0.8))
            case .official:
                return (f(2.5, 0.5), f(2.5, 0.5), f(1.7, 0.5), f(5.0, 0.5))
            case .explicit(let w, let a, let v, let g):
                return (w, a, v, g)
            }
        }
    }

    public var ranks: LoRARanks
    /// Ширина cmix. 4× — как в rwkv-metal; официальный «suggestion» 3.5×,
    /// округлённый до 32. Влияет только на новую модель, не на загрузку.
    public var ffnMultiplier: Double
    public var seed: UInt64

    public init(ranks: LoRARanks = .rwkvMetal, ffnMultiplier: Double = 4.0,
                seed: UInt64 = 0) {
        self.ranks = ranks
        self.ffnMultiplier = ffnMultiplier
        self.seed = seed
    }
}

public enum X070Init {

    /// Полный словарь весов для X070Backbone — модель, готовая к обучению
    /// с нуля.
    public static func weights(cfg: X070Config,
                               init ic: X070InitConfig = X070InitConfig())
        -> [String: MLXArray] {
        MLXRandom.seed(ic.seed)

        let C = cfg.nEmbd, H = cfg.nHead, N = cfg.headSize
        let V = cfg.vocab, L = cfg.nLayer
        let r = ic.ranks.ranks(nEmbd: C)
        let ffn = Int((Double(C) * ic.ffnMultiplier / 32).rounded(.down)) * 32

        var w: [String: MLXArray] = [:]

        // ── Верхний уровень ──────────────────────────────────────────
        // emb: очень узкий равномерный разброс. ln0 сразу нормирует, поэтому
        // масштаб эмбеддинга почти не важен, а маленький старт не даёт
        // первым шагам дёргать LayerNorm.
        w["emb.weight"] = MLXRandom.uniform(low: -1e-4, high: 1e-4, [V, C])
        w["ln0.weight"] = MLXArray.ones([C])
        w["ln0.bias"] = MLXArray.zeros([C])
        w["ln_out.weight"] = MLXArray.ones([C])
        w["ln_out.bias"] = MLXArray.zeros([C])

        // head: ортогональная, gain = 0.5·√(V/C) при V > C.
        let headGain = V > C ? 0.5 * sqrt(Double(V) / Double(C)) : 0.5
        w["head.weight"] = orthogonal([V, C], gain: Float(headGain))

        // ── Поканалные кривые, общие для всех слоёв ──────────────────
        // linear[n] = n/(C-1) − 0.5           — плавный наклон по каналам
        // zigzag[n] — пила ВНУТРИ головы, возведённая в квадрат со знаком:
        //             разводит каналы внутри одной головы
        let linear = MLXArray((0 ..< C).map { Float($0) / Float(C - 1) - 0.5 })
        let zigzag = MLXArray((0 ..< C).map { n -> Float in
            let z = (Float(n % N) - Float(N - 1) / 2) / (Float(N - 1) / 2)
            return z * abs(z)
        })

        for layer in 0 ..< L {
            let p = "blocks.\(layer)."
            let tp = p + "tmix.", cp = p + "cmix."

            let ratio01 = L > 1 ? Double(layer) / Double(L - 1) : 0.0   // 0→1
            let ratio1a = 1.0 - Double(layer) / Double(L)               // 1→~0

            w[p + "ln1.weight"] = MLXArray.ones([C])
            w[p + "ln1.bias"] = MLXArray.zeros([C])
            w[p + "ln2.weight"] = MLXArray.ones([C])
            w[p + "ln2.bias"] = MLXArray.zeros([C])

            // ── token-shift lerp: 1 − (i/C)^(k·ratio1a) ──
            // Показатели 0.2/0.9/0.7/0.7/0.9/0.2 — из актуального x070.
            // (В более ранних версиях у x_k/x_v были другие показатели плюс
            //  добавка от ratio_0_to_1; здесь их НЕТ.)
            w[tp + "x_r"] = lerpCurve(C: C, k: 0.2 * ratio1a)
            w[tp + "x_w"] = lerpCurve(C: C, k: 0.9 * ratio1a)
            w[tp + "x_k"] = lerpCurve(C: C, k: 0.7 * ratio1a)
            w[tp + "x_v"] = lerpCurve(C: C, k: 0.7 * ratio1a)
            w[tp + "x_a"] = lerpCurve(C: C, k: 0.9 * ratio1a)
            w[tp + "x_g"] = lerpCurve(C: C, k: 0.2 * ratio1a)

            // ── w0: разброс горизонтов памяти по каналам ──
            // www[n] = −6 + 6·(n/(C−1))^(1 + ratio01^0.3)
            // Показатель растёт с глубиной ⇒ верхние слои смещены к более
            // долгой памяти. Плюс пила внутри головы (×2.5) и сдвиг +0.5.
            let expo = 1.0 + pow(ratio01, 0.3)
            let www = MLXArray((0 ..< C).map {
                Float(-6.0 + 6.0 * pow(Double($0) / Double(C - 1), expo))
            })
            w[tp + "w_lora_B.bias"] = www + 0.5 + zigzag * 2.5

            // a0 (iclr), v0 (value-residual): смещения, дающие осмысленный
            // старт до того, как low-rank части чему-либо научатся.
            w[tp + "a_lora_B.bias"] = MLXArray(Float(-0.19)) + zigzag * 0.3 + linear * 0.4
            if layer > 0 {
                w[tp + "v_lora_B.bias"] = MLXArray(Float(0.73)) - linear * 0.4
            }

            // ── low-rank: A = нули, B = ортогональная ×0.1 ──
            // A=0 делает всю динамическую часть (decay/iclr/gate/v-residual)
            // нейтральной на нулевом шаге: произведение A·B равно нулю
            // независимо от B, поэтому в forward остаются только смещения
            // w0/a0/v0. Обучение «включает» её постепенно.
            //
            // Swift хранит A как [rank, C], B как [C, rank] — транспонированно
            // к питоновским w1 [C, rank] / w2 [rank, C].
            for (name, rank) in [("w", r.w), ("a", r.a), ("g", r.g)] {
                w[tp + "\(name)_lora_A.weight"] = MLXArray.zeros([rank, C])
                w[tp + "\(name)_lora_B.weight"] = orthoInit([rank, C], scale: 0.1)
                                                    .transposed()
            }
            if layer > 0 {
                w[tp + "v_lora_A.weight"] = MLXArray.zeros([r.v, C])
                w[tp + "v_lora_B.weight"] = orthoInit([r.v, C], scale: 0.1).transposed()
            }

            // ── поканалные скаляры ──
            w[tp + "k_k"] = (MLXArray(Float(0.71)) - linear * 0.1).reshaped([H, N])
            w[tp + "k_a"] = MLXArray.full([H, N], values: MLXArray(Float(1.02)))
            w[tp + "r_k"] = MLXArray.full([H, N], values: MLXArray(Float(-0.04)))

            // ── проекции ──
            // k_proj в 10 раз уже остальных: ключ входит в рекуррентность
            // напрямую (v·kᵀ), и широкий старт раскачивает состояние.
            let s = Float(0.5 / sqrt(Double(C)))
            w[tp + "r_proj.weight"] = MLXRandom.uniform(low: -s, high: s, [C, C])
            w[tp + "k_proj.weight"] = MLXRandom.uniform(low: -s * 0.1, high: s * 0.1, [C, C])
            w[tp + "v_proj.weight"] = MLXRandom.uniform(low: -s, high: s, [C, C])
            // o_proj = 0 ⇒ на нулевом шаге блок не вносит ничего в остаточный
            // поток: сеть стартует как identity и «отращивает» слои по мере
            // обучения. Это и есть причина, по которой глубокая модель не
            // разваливается на первых шагах.
            w[tp + "o_proj.weight"] = MLXArray.zeros([C, C])

            // ln_x (GroupNorm по головам): вес НЕ единичный, а
            // ((layer+1)/L)^0.7 — компенсация роста дисперсии с глубиной.
            let lnxScale = Float(pow(Double(layer + 1) / Double(L), 0.7))
            w[tp + "ln_x.weight"] = MLXArray.full([C], values: MLXArray(lnxScale))
            w[tp + "ln_x.bias"] = MLXArray.zeros([C])

            // ── cmix ──
            w[cp + "x_k"] = lerpCurve(C: C, k: pow(ratio1a, 4))
            let sf = Float(0.5 / sqrt(Double(C)))
            w[cp + "key.weight"] = MLXRandom.uniform(low: -sf, high: sf, [ffn, C])
            w[cp + "value.weight"] = MLXArray.zeros([C, ffn])   // тоже 0, как o_proj
        }

        eval(Array(w.values))
        return w
    }

    /// Готовый бэкбон со случайной инициализацией.
    public static func makeBackbone(cfg: X070Config,
                                    init ic: X070InitConfig = X070InitConfig(),
                                    computeDType: DType = .bfloat16) -> X070Backbone {
        X070Backbone(weights: weights(cfg: cfg, init: ic), cfg: cfg,
                     computeDType: computeDType)
    }

    // ── Вспомогательное ──────────────────────────────────────────────

    /// 1 − (i/C)^k по каналам. При k → 0 даёт ~0 (нижние слои почти не
    /// смешивают с предыдущим токеном), при k → 1 — линейный подъём.
    private static func lerpCurve(C: Int, k: Double) -> MLXArray {
        MLXArray((0 ..< C).map { Float(1.0 - pow(Double($0) / Double(C), k)) })
    }

    /// Ортогональная инициализация через QR — точный аналог
    /// `nn.init.orthogonal_(x, gain)`: gain применяется КАК ЕСТЬ, без
    /// дополнительных множителей.
    ///
    /// Строится гауссова матрица, берётся Q из QR и корректируются знаки по
    /// диагонали R — без этого Q не распределена равномерно по группе Хаара
    /// и знаки строк оказываются смещены.
    static func orthogonal(_ shape: [Int], gain: Float) -> MLXArray {
        let rows = shape[0], cols = shape[1]
        let flip = rows < cols
        // QR в MLX ждёт rows >= cols; для «широкой» матрицы работаем с
        // транспонированной и переворачиваем результат.
        let m = flip ? cols : rows
        let n = flip ? rows : cols

        let a = MLXRandom.normal([m, n])
        // QR в MLX не реализован на GPU — нужен явный CPU-поток, иначе
        // рантайм падает. Инициализация одноразовая, так что цена приемлема,
        // но для крупного словаря она заметна: head [V, C] при V=65536 — это
        // QR тонкой матрицы 65536×768 на CPU, десятки секунд.
        let (q, rMat) = MLX.qr(a, stream: .cpu)
        // знаки диагонали R → равномерность по Хаару
        let d = rMat.diagonal()
        let signs = MLX.where(d .< 0, MLXArray(Float(-1)), MLXArray(Float(1)))
        var out = q * signs.reshaped([1, n])
        if flip { out = out.transposed() }

        let res = out * gain
        eval(res)
        return res
    }

    /// Хелпер `ortho_init` из RWKV_Tmix_x070 — ОТЛИЧАЕТСЯ от orthogonal_:
    /// при rows > cols масштаб домножается на √(rows/cols).
    ///
    /// Разница существенна и её легко потерять: у `head` официал вызывает
    /// именно `orthogonal_`, причём множитель √(V/C) уже включён в сам gain.
    /// Применить сверху ещё и множитель из `ortho_init` означает возвести его
    /// в квадрат — стартовые логиты станут шире, и лосс на нулевом шаге
    /// уедет выше ln(vocab).
    ///
    /// В low-rank матрицах [rank, C] rank < C, поэтому здесь множитель всегда
    /// равен 1; функция отдельна ради точности переноса, а не ради эффекта.
    static func orthoInit(_ shape: [Int], scale: Float) -> MLXArray {
        let rows = shape[0], cols = shape[1]
        let gain: Float = rows > cols
            ? Float(sqrt(Double(rows) / Double(cols)))
            : 1
        return orthogonal(shape, gain: gain * scale)
    }
}
