import Foundation
import MLX
import MLXNN
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Инкрементальный (рекуррентный) инференс x070 для генерации (B=1).
//
//  RWKV — RNN: состояние полностью описывает контекст, не нужен KV-кэш по
//  длине. Состояние на слой:
//    • wkv      [H,D,D]  — матрица состояния WKV-ядра,
//    • tmixPrev [1,D]    — предыдущий x для token-shift в tmix,
//    • cmixPrev [1,D]    — предыдущий x для token-shift в cmix.
//  Плюс глобальный v_first [1,H,D] (value первого слоя, фиксируется на слое 0).
//
//  Decode по одному токену: чистые MLX-операции (вариант A), без Metal-ядра —
//  на 1 токен матрица [H,D,D] мала и это быстро.
// ───────────────────────────────────────────────────────────────────────

/// Рекуррентное состояние модели (B=1).
public struct RWKVState {
    public var wkv: [MLXArray]        // [nLayer] × [H, D, D]  fp32
    public var tmixPrev: [MLXArray]   // [nLayer] × [1, D]
    public var cmixPrev: [MLXArray]   // [nLayer] × [1, D]
    public var vFirst: MLXArray?      // [1, H, D] (устанавливается на слое 0)

    /// Пустое состояние (нули) для модели заданной геометрии.
    public init(cfg: X070Config, dtype: DType = .bfloat16) {
        let H = cfg.nHead, D = cfg.headSize, E = cfg.nEmbd
        wkv = (0 ..< cfg.nLayer).map { _ in MLXArray.zeros([H, D, D], dtype: .float32) }
        tmixPrev = (0 ..< cfg.nLayer).map { _ in MLXArray.zeros([1, E], dtype: dtype) }
        cmixPrev = (0 ..< cfg.nLayer).map { _ in MLXArray.zeros([1, E], dtype: dtype) }
        vFirst = nil
    }

    /// Состояние из батчевого — строка `row` батча.
    ///
    /// Мост между параллельным путём (`bodyWithState` возвращает
    /// `RWKVBatchState`) и пошаговым декодом. Раскладка разная — там стопка
    /// `[L,B,…]`, здесь массив на слой, — поэтому это разбор, а не приведение
    /// типа.
    ///
    /// **`vFirst` намеренно остаётся nil, и это не потеря.** В x070 `v_first`
    /// не бегущая величина: слой 0 вычисляет её на КАЖДОЙ позиции, слои выше
    /// потребляют на ТОЙ ЖЕ позиции. `step` перезаписывает её на слое 0 раньше,
    /// чем любой слой выше её прочитает, — то есть через границу вызова она не
    /// переносится вообще. Поле в `RWKVState` существует ради передачи между
    /// слоями ОДНОГО шага, а не между шагами. Утверждение проверяется тестом:
    /// обнуление `vFirst` перед `step` не меняет логиты ни на бит.
    public init(_ batch: RWKVBatchState, row: Int = 0) {
        precondition(row >= 0 && row < batch.batch,
                     "строки \(row) нет: в состоянии \(batch.batch)")
        let L = batch.nLayer
        wkv = (0 ..< L).map { batch.layerWKV($0)[row] }               // [H,S,S]
        tmixPrev = (0 ..< L).map { batch.layerTmixShift($0)[row] }    // [1,D]
        cmixPrev = (0 ..< L).map { batch.layerCmixShift($0)[row] }    // [1,D]
        vFirst = nil
    }

    /// Зафиксировать состояние (eval) — полезно между шагами генерации.
    public mutating func eval() {
        MLX.eval(wkv + tmixPrev + cmixPrev + (vFirst.map { [$0] } ?? []))
    }
}

extension X070Backbone {

    // Веса, которые НЕ бывают ни квантованными, ни целями LoRA: множители
    // token-shift, нормировки, внутренние low-rank параметры x070. Их можно
    // читать напрямую.
    private func gg(_ key: String) -> MLXArray {
        guard let v = wOverride?[key] ?? w[key] else {
            preconditionFailure(
                "нет плотного веса \(key). Если база квантована (.rwkvq) или "
                + "на неё навешены адаптеры, проекции обязаны идти через "
                + "projectForBlock, а не через gg — см. tmixStep/cmixStep.")
        }
        return v
    }

    // Проекция ТЕМ ЖЕ путём, что у параллельного прохода: с деквантизацией
    // .rwkvq на лету и с прибавкой LoRA-адаптеров.
    //
    // Раньше здесь стоял `linear_(x, gg(key))`, и это была вторая копия
    // арифметики со всеми последствиями: на квантованной базе декод падал
    // force-unwrap'ом без сообщения, а навешенные адаптеры игнорировал молча
    // — дообученная модель генерировала так, будто её не дообучали. Второе
    // хуже: оно не падает.
    private func pj(_ x: MLXArray, _ wKey: String) -> MLXArray {
        precondition(wKey.hasSuffix(".weight"), "проекция \(wKey) не .weight")
        return projectForBlock(x, wKey, lora: String(wKey.dropLast(7)))
    }

    // ─────────────── Один рекуррентный WKV-шаг (вариант A) ───────────────
    // Вход: r,w,k,v,a,b — [H,D];  h — [H,D,D] (h[head, dv, dk]).
    // Возврат: (out [H,D], h' [H,D,D]). Всё в fp32 (как ядро).
    private func wkvStep(_ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
                         _ a: MLXArray, _ b: MLXArray, _ h: MLXArray) -> (MLXArray, MLXArray) {
        let rf = r.asType(.float32), wf = w.asType(.float32), kf = k.asType(.float32)
        let vf = v.asType(.float32), af = a.asType(.float32), bf = b.asType(.float32)

        // sa[head,dv] = Σ_dk h[head,dv,dk] * a[head,dk]
        let sa = (h * af.expandedDimensions(axis: 1)).sum(axis: -1)          // [H,D]

        // h'[head,dv,dk] = w[dk]*h + v[dv]*k[dk] + sa[dv]*b[dk]
        let wTerm = h * wf.expandedDimensions(axis: 1)                       // [H,D,D] (по dk)
        let vk = vf.expandedDimensions(axis: 2) * kf.expandedDimensions(axis: 1)  // [H,D,D]
        let sab = sa.expandedDimensions(axis: 2) * bf.expandedDimensions(axis: 1) // [H,D,D]
        let hNew = wTerm + vk + sab

        // out[head,dv] = Σ_dk h'[head,dv,dk] * r[head,dk]
        let out = (hNew * rf.expandedDimensions(axis: 1)).sum(axis: -1)      // [H,D]
        return (out, hNew)
    }

    // ─────────────── tmix для одного токена ───────────────
    // x: [1, D] (выход ln1). Возвращает (res [1,D]) и мутирует state.
    private func tmixStep(_ x: MLXArray, _ layer: Int, _ state: inout RWKVState) -> MLXArray {
        let p = "blocks.\(layer).tmix."
        let D = cfg.nEmbd, H = cfg.nHead, S = cfg.headSize

        let prev = state.tmixPrev[layer]      // [1,D]
        let xx = prev - x                     // token-shift: prev - x
        state.tmixPrev[layer] = x             // обновляем shift-state

        // Шесть лерпов одним broadcast: xs = x + xx*[6,1,D] -> [6,1,D].
        // Порядок в стеке -- r,w,k,v,a,g (см. xcoefStack), и он же
        // повторён здесь; перепутать их местами значит посчитать
        // правдоподобную чушь, поэтому разбор идёт явными индексами, а
        // не кортежем.
        let xr, xw, xk, xv, xa, xg: MLXArray
        if let coef = xcoefStack(layer) {
            let xs = x + xx * coef
            xr = xs[0]; xw = xs[1]; xk = xs[2]
            xv = xs[3]; xa = xs[4]; xg = xs[5]
        } else {
            xr = x + xx * gg(p+"x_r"); xw = x + xx * gg(p+"x_w"); xk = x + xx * gg(p+"x_k")
            xv = x + xx * gg(p+"x_v"); xa = x + xx * gg(p+"x_a"); xg = x + xx * gg(p+"x_g")
        }

        let r = pj(xr, p+"r_proj.weight").reshaped([H, S])
        let k0 = pj(xk, p+"k_proj.weight").reshaped([H, S])
        var v = pj(xv, p+"v_proj.weight").reshaped([H, S])

        let gate = linear_(sigmoid(linear_(xg, gg(p+"g_lora_A.weight"))), gg(p+"g_lora_B.weight"))

        if layer == 0 {
            state.vFirst = v.reshaped([1, H, S])
        } else {
            let vv = sigmoid(linear_(linear_(xv, gg(p+"v_lora_A.weight")),
                                     gg(p+"v_lora_B.weight"), gg(p+"v_lora_B.bias"))).reshaped([H, S])
            let vf = state.vFirst!.reshaped([H, S])
            v = v + (vf - v) * vv
        }

        let a = sigmoid(linear_(linear_(xa, gg(p+"a_lora_A.weight")),
                                gg(p+"a_lora_B.weight"), gg(p+"a_lora_B.bias"))).reshaped([H, S])

        var ww = linear_(tanh(linear_(xw, gg(p+"w_lora_A.weight"))),
                         gg(p+"w_lora_B.weight"), gg(p+"w_lora_B.bias"))
        ww = exp(-0.606531 * sigmoid(ww.asType(.float32))).asType(x.dtype).reshaped([H, S])

        let kk = l2normLast(k0 * gg(p+"k_k"))
        let k = k0 * (1.0 + (a - 1.0) * gg(p+"k_a"))

        // WKV-шаг (fp32)
        var (outHD, hNew) = wkvStep(r, ww, k, v, -kk, kk * a, state.wkv[layer])
        state.wkv[layer] = hNew
        // Та же точка течи, что в параллельном пути (см. `castWKVOutput` в
        // RWKVBlock.swift). Состояние `hNew` остаётся fp32 ВСЕГДА — приводится
        // только выход, идущий в остаточный поток.
        if castWKVOutputToComputeDType { outHD = outHD.asType(x.dtype) }

        // ln_x (GroupNorm) для одного токена: [1, D]
        var out = lnXStep(outHD.reshaped([1, D]), gg(p+"ln_x.weight"), gg(p+"ln_x.bias"), H: H)
            .reshaped([H, S])
        // bonus = (r*k*r_k).sum(-1, keepdims) * v
        let bonus = (r * k * gg(p+"r_k")).sum(axis: -1, keepDims: true) * v   // [H,S]
        out = out + bonus

        return pj(out.reshaped([1, D]) * gate, p+"o_proj.weight")
    }

    // cmix для одного токена.
    private func cmixStep(_ x: MLXArray, _ layer: Int, _ state: inout RWKVState) -> MLXArray {
        let p = "blocks.\(layer).cmix."
        let prev = state.cmixPrev[layer]
        let xx = prev - x
        state.cmixPrev[layer] = x
        let xk = x + xx * gg(p+"x_k")
        let h = relu(pj(xk, p+"key.weight"))
        return pj(h * h, p+"value.weight")
    }

    // ─────────────── Публичные методы генерации ───────────────

    /// Обработать промпт ПАРАЛЛЕЛЬНЫМ проходом и оставить состояние, пригодное
    /// для пошагового декода. Возвращает логиты последнего токена `[vocab]`
    /// (не evaluated). Для B=1.
    ///
    /// Промпт — это последовательность, которая целиком известна заранее, то
    /// есть ровно тот случай, ради которого существует параллельный путь.
    /// Рекуррентный проход по промпту стоит столько же за токен, сколько
    /// генерация, и на длинном промпте это доминирующая часть ожидания.
    ///
    /// Своей арифметики здесь нет: `bodyWithState` — тот же проход, на котором
    /// стоит префикс-кэш реранкера. Расхождение с рекуррентным путём того же
    /// порядка, что между `body` и `step` вообще (замерено: 7.4e-7 на
    /// логитах), — это два ядра, считающие одно и то же, а не две разные
    /// модели. Рекуррентный вариант остался как `prefillRecurrent` и служит
    /// эталоном.
    ///
    /// Состояние ПЕРЕЗАПИСЫВАЕТСЯ, а не продолжается: параллельный путь умеет
    /// продолжать с состояния, но `RWKVState` пришлось бы для этого собирать
    /// обратно в батчевый, и молчаливое несовпадение раскладок здесь стоило бы
    /// дороже, чем отсутствие возможности. Продолжение с готового состояния —
    /// это `step`.
    public func prefill(_ ids: [Int], state: inout RWKVState) -> MLXArray {
        precondition(!ids.isEmpty, "prefill: пустой промпт")
        let idsArray = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let (lnOut, batch) = bodyWithState(idsArray)
        state = RWKVState(batch)
        let last = lnOut[0, ids.count - 1].reshaped([1, cfg.nEmbd])
        return pj(last, "head.weight").reshaped([cfg.vocab])
    }

    /// Тот же prefill, но токен за токеном через `step`.
    ///
    /// Оставлен НАМЕРЕННО, и не ради совместимости. Это эталон, относительно
    /// которого проверяется быстрый путь: тест «параллельный prefill согласен
    /// с рекуррентным» без него сравнивал бы параллельный проход сам с собой —
    /// то есть остался бы зелёным при полностью сломанном переносе состояния.
    ///
    /// Для длинных промптов пользоваться им незачем: он в разы медленнее.
    public func prefillRecurrent(_ ids: [Int], state: inout RWKVState) -> MLXArray {
        precondition(!ids.isEmpty, "prefill: пустой промпт")
        var logits = MLXArray.zeros([cfg.vocab])
        for id in ids {
            logits = step(id, state: &state)
            // eval на каждом токене ОБЯЗАТЕЛЕН, и это не «на всякий случай».
            //
            // MLX ленив: без него цикл не считает, а СТРОИТ граф — и к концу
            // длинного промпта это граф на тысячи шагов, который потом
            // разворачивается разом. В `generate` этого не видно, потому что
            // сэмплер читает выбранный id и тем самым вычисляет граф каждый
            // токен; здесь читать нечего.
            //
            // Найдено замером: прогон на 2048 токенов не закончился за
            // четверть часа и съел под два гигабайта, тогда как та же длина
            // по шагам с eval идёт секунды.
            state.eval()
        }
        return logits
    }

    /// Один шаг декодирования: id токена → logits [vocab] (не evaluated).
    ///
    /// `reshaped([1, D])` на обеих остаточных связях — не косметика.
    ///
    /// Множители token-shift (`x_r`, `x_k`, …) в x070 лежат как `[1,1,D]`, и
    /// `x + xx * gg(...)` разворачивает `[1,D]` в `[1,1,D]` по правилам
    /// бродкаста. Лишняя ось едет через gate в результат блока, оттуда в `x`
    /// следующего слоя, а из него — в СОСТОЯНИЕ: сдвиги слоя 0 оставались
    /// `[1,D]`, а всех верхних становились `[1,1,D]`.
    ///
    /// Считалось при этом всё правильно — бродкаст сходится, — и потому
    /// расхождение не проявлялось ничем: логиты верные, продолжение верное.
    /// Нашлось только сверкой ФОРМ с параллельным путём, который отдаёт
    /// честные `[1,D]`. Разъезд форм внутри одного типа оставлять нельзя:
    /// ровно так и появляется бродкаст там, где ждали совпадения размеров.
    ///
    /// Нормировка стоит здесь, а не внутри `tmixStep`/`cmixStep`, и ровно в
    /// двух местах — по одному на каждую запись в остаточный поток. Это
    /// проверяемо: снятие любой из двух даёт `[1,1,D]` в состоянии и роняет
    /// тест форм. Раньше их было четыре, и они друг друга ПОДМЕНЯЛИ — ни одна
    /// мутация не ловилась, потому что оставшиеся три чинили результат.
    public func step(_ id: Int, state: inout RWKVState) -> MLXArray {
        step(MLXArray([Int32(id)]), state: &state)
    }

    /// Тот же шаг, но токен приходит МАССИВОМ, а не числом.
    ///
    /// Разница не косметическая. `step(Int)` вынуждает вызывающего
    /// достать выбранный токен в CPU — то есть синхронизировать
    /// конвейер на каждом шаге. По трейсу это ~2 мс зазора CPU↔GPU, по
    /// прямому замеру 2.75 мс/ток на 2.9B из 32.26. С этой перегрузкой
    /// цикл генерации может держать токен на GPU и читать его назад
    /// реже, чем считает.
    public func step(_ id: MLXArray, state: inout RWKVState) -> MLXArray {
        let D = cfg.nEmbd
        // Через общий путь: таблица может быть квантованной.
        let emb = embedForBlock(id).reshaped([1, D])
        var x = layerNorm_(emb, gg("ln0.weight"), gg("ln0.bias"))
        for layer in 0 ..< cfg.nLayer {
            let h = tmixStep(layerNorm_(x, gg("blocks.\(layer).ln1.weight"),
                                        gg("blocks.\(layer).ln1.bias")), layer, &state)
            x = (x + h).reshaped([1, D])
            x = (x + cmixStep(layerNorm_(x, gg("blocks.\(layer).ln2.weight"),
                                         gg("blocks.\(layer).ln2.bias")), layer, &state))
                .reshaped([1, D])
        }
        let lnOut = layerNorm_(x, gg("ln_out.weight"), gg("ln_out.bias"))
        return pj(lnOut, "head.weight").reshaped([cfg.vocab])
    }
}

// ─────────────── Локальные утилиты (повтор для extension-доступа) ───────────────

private func l2normLast(_ x: MLXArray) -> MLXArray {
    x / sqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)
}
private func layerNorm_(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
                        eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let varc = (x - mean).square().mean(axis: -1, keepDims: true)
    return (x - mean) / sqrt(varc + eps) * weight + bias
}
private func linear_(_ x: MLXArray, _ w: MLXArray) -> MLXArray { matmul(x, w.transposed()) }
private func linear_(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    matmul(x, w.transposed()) + b
}

// Per-token GroupNorm (eps=64e-5, pytorch_compatible): normalizes each head
// independently and ONLY over the current token — causally correct, unlike the
// parallel lnX (MLX GroupNorm on [B,T,D] mixes across all T).
private func lnXStep(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray,
                     H: Int, eps: Float = 64e-5) -> MLXArray {
    let D = x.shape[x.shape.count - 1]
    let S = D / H
    let g = x.reshaped([H, S])
    let mean = g.mean(axis: -1, keepDims: true)
    let varc = (g - mean).square().mean(axis: -1, keepDims: true)
    let normed = ((g - mean) / sqrt(varc + eps)).reshaped([1, D])
    return normed * weight + bias
}
