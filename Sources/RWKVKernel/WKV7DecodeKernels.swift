import Foundation
import MLX
import MLXFast

// ───────────────────────────────────────────────────────────────────────
//  Ядра ДЕКОДА: шаг WKV-7 на один токен и хвост tmix одним запуском.
//
//  Зачем. На декоде шаг упирается не в арифметику, а в число запусков:
//  все проекции одного токена 1.5B (144 GEMV) занимают 9.1 мс из 18.0, а
//  остальные ~9 мс — это десятки поэлементных операций на слой над
//  тензорами в несколько килобайт. У Metal-бэкенда rwkv-quant та же
//  не-GEMV часть стоит 4–5 мс, потому что шаг WKV там — одно ядро
//  (wkv7_infer), и хвост слоя — одно ядро (fused_tail.py). Здесь их порт.
//
//  Оба ядра обслуживают ТОЛЬКО рекуррентный шаг (B·T = 1): на параллельном
//  проходе тензоры крупные, накладные запуска амортизируются, и вторая
//  реализация на горячем пути обучения не нужна.
//
//  Типы входов у обоих ядер — строго fp32 (приведение стоит в обёртках):
//  MLX строит исходник ядра из типов входов, а кеширует по имени, и то же
//  имя с другими типами освобождает конвейер из-под ещё не исполненных
//  запусков (см. WKV7Train.swift).
// ───────────────────────────────────────────────────────────────────────

private var stepKernelCache: [Int: MLXFastKernel] = [:]
private var tailKernelCache: [String: MLXFastKernel] = [:]

/// Один шаг рекуррентности на голову. Арифметика и порядок суммирования —
/// как у первого шага прямого ядра (WKV7.swift), без записи sa наружу:
///   sa   = Σ_dk h[dv,dk]·a[dk]
///   h'   = w[dk]·h[dv,dk] + v[dv]·k[dk] + sa·b[dk]
///   out  = Σ_dk h'[dv,dk]·r[dk]
private func wkv7StepKernel(H: Int) -> MLXFastKernel {
    if let k = stepKernelCache[H] { return k }
    let header = """
    constant uint HEAD_SIZE_C = \(WKV7_HEAD_SIZE);
    constant uint H_C         = \(H);
    """
    let source = """
        uint dv = thread_position_in_grid.y;
        uint hi = thread_position_in_grid.x;

        float h_row[HEAD_SIZE_C];
        uint h_base = hi*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

        uint base = hi*HEAD_SIZE_C;
        float sa = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];

        float v_dv = v[base+dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*b[base+dk];

        float y = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
        out[base+dv] = y;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
    """
    let kern = MLXFast.metalKernel(
        name: "wkv7_step_\(H)",
        inputNames: ["r", "w", "k", "v", "a", "b", "h_in"],
        outputNames: ["out", "h_out"],
        source: source,
        header: header
    )
    stepKernelCache[H] = kern
    return kern
}

/// Шаг WKV-7 на один токен одним запуском. Входы r, w, k, v, a, b — [H, S],
/// состояние h — [H, S, S]. Возвращает (out [H, S], h' [H, S, S]), fp32.
public func wkv7StepForward(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ h: MLXArray
) -> (MLXArray, MLXArray) {
    let H = r.shape[0], S = r.shape[1]
    precondition(S == WKV7_HEAD_SIZE, "S (\(S)) должен равняться HEAD_SIZE \(WKV7_HEAD_SIZE)")
    precondition(h.shape == [H, S, S], "состояние \(h.shape), ожидалось [\(H), \(S), \(S)]")
    let inputs = [r, w, k, v, a, b, h].map { $0.asType(.float32) }
    let out = wkv7StepKernel(H: H)(
        inputs,
        grid: (H, S, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[H, S], [H, S, S]],
        outputDTypes: [.float32, .float32]
    )
    return (out[0], out[1])
}

/// Можно ли считать хвост одним ядром: ширина головы кратна simd-группе.
public func wkvTailApplies(headSize S: Int) -> Bool { S % 32 == 0 }

/// Хвост tmix одним запуском: (group_norm(wkv) + bonus(r, k, r_k)·v)·g.
///
/// Порт rwkv_quant/backends/metal/fused_tail.py. Формулы взяты из
/// покомандного пути буквально, включая два прохода по голове ради
/// mean((x − mean)²): однопроходное E[x²] − mean² — разность близких
/// величин. Порядок суммирования другой (simd-древесный против редукций
/// MLX), поэтому с покомандным путём хвост совпадает в допуске, не побитно.
///
/// wkv, r, k, v — [H, S]; g — [1, D]; r_k, ln_w, ln_b — любой формы на
/// D = H·S элементов. Возвращает [1, D] fp32 — готовый вход o_proj.
public func wkvTail(
    _ wkv: MLXArray, r: MLXArray, k: MLXArray, v: MLXArray,
    rK: MLXArray, lnWeight: MLXArray, lnBias: MLXArray, gate: MLXArray,
    nHead H: Int, headSize S: Int, eps: Float = 64e-5
) -> MLXArray {
    precondition(wkvTailApplies(headSize: S), "хвост: S=\(S) не кратен ширине simd-группы")
    let D = H * S
    let key = "\(H)_\(S)_\(eps)"
    let kern: MLXFastKernel
    if let k = tailKernelCache[key] {
        kern = k
    } else {
        let header = """
        constant uint  H_C = \(H);
        constant uint  S_C = \(S);
        constant float EPS = \(eps)f;
        """
        let source = """
            uint tg   = threadgroup_position_in_grid.x;   // строка*H + голова
            uint lane = thread_position_in_threadgroup.x;
            uint h    = tg % H_C;
            uint base = tg * S_C;                         // [N*H, S] -- плотно
            uint hb   = h * S_C;                          // r_k / ln_x -- [H, S]

            // --- group_norm, два прохода
            float s1 = 0.0f;
            for (uint i = lane; i < S_C; i += 32) s1 += wkv[base + i];
            s1 = simd_sum(s1);
            float mean = s1 / (float)S_C;

            float s2 = 0.0f;
            for (uint i = lane; i < S_C; i += 32) {
                float d = wkv[base + i] - mean;
                s2 += d * d;
            }
            s2 = simd_sum(s2);
            float inv = 1.0f / sqrt(s2 / (float)S_C + EPS);

            // --- bonus: (r * k * r_k) по голове, затем на v
            float dp = 0.0f;
            for (uint i = lane; i < S_C; i += 32)
                dp += r[base + i] * k[base + i] * rk[hb + i];
            dp = simd_sum(dp);

            // --- сборка в том же порядке, что и в покомандном пути:
            //     ((norm * w + b) + dp * v) * g
            for (uint i = lane; i < S_C; i += 32) {
                float nrm = (wkv[base + i] - mean) * inv;
                float y   = nrm * lnw[hb + i] + lnb[hb + i] + dp * v[base + i];
                out[base + i] = y * g[base + i];
            }
        """
        kern = MLXFast.metalKernel(
            name: "wkv_tail_h\(H)_s\(S)",
            inputNames: ["wkv", "r", "k", "v", "rk", "lnw", "lnb", "g"],
            outputNames: ["out"],
            source: source,
            header: header
        )
        tailKernelCache[key] = kern
    }
    let inputs = [wkv.reshaped([1, D]), r.reshaped([1, D]), k.reshaped([1, D]), v.reshaped([1, D]),
                  rK.reshaped([D]), lnWeight.reshaped([D]), lnBias.reshaped([D]),
                  gate.reshaped([1, D])].map { $0.asType(.float32) }
    return kern(inputs,
                grid: (H * 32, 1, 1),
                threadGroup: (32, 1, 1),
                outputShapes: [[1, D]],
                outputDTypes: [.float32])[0]
}
