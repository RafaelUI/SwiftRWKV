import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Сборка x070-словаря весов ЦЕЛИКОМ из одного самодостаточного сайдкара
//  (полный экспорт .rwkvq, format_version 2) — без стороннего dense-файла.
//
//  Точный порт rwkv_metal/model/convert.py::convert(). Переносится один в
//  один, включая три места, где это НЕ простое переименование ключа:
//    1. w1/w2, a1/a2, v1/v2, g1/g2 — TRANSPOSED при переносе в x070-имена
//       (мир хранит их как [in,rank]/[rank,in] под "x @ w1" без транспонирования
//       в оригинальном коде; x070 ждёт Linear-конвенцию [out,in]).
//    2. w0/a0/v0 — форма [1,1,D] в мире, СПЛЮЩИВАЮТСЯ в 1D-bias при переносе
//       (это не отдельные веса, а bias-термы у *_lora_B).
//    3. v_lora (v1/v2/v0) кладётся в x070-словарь ТОЛЬКО для слоёв i>0 — у
//       слоя 0 нет предыдущего слоя, с которым мешать value residual, и
//       модель такого параметра не заводит. В world-манифесте v0/v1/v2 у
//       слоя 0 при этом ЕСТЬ (проверено на реальном экспорте) — их просто
//       не переносим, а не "их там нет".
//
//  sb6-раскладка (4 tmix-проекции + 2 cmix + head + emb на слой) сюда
//  СОЗНАТЕЛЬНО не входит — эти веса остаются квантованными и подключаются
//  отдельно через `attachRwkvq`, иначе смысл сжатой базы теряется.
//  Единственное исключение — emb.weight: см. комментарий в вызывающем коде
//  (X070Backbone.load(fromSidecar:)) о причине эагерной деквантизации.
// ───────────────────────────────────────────────────────────────────────

public enum RwkvqFullConvertError: Error, CustomStringConvertible {
    case incompleteManifest(String)

    public var description: String {
        switch self {
        case .incompleteManifest(let why): return "неполный манифест сайдкара: \(why)"
        }
    }
}

extension RwkvqSidecar {

    /// x070-словарь ВСЕГО, кроме sb6-проекций (тело модели: нормы,
    /// token-shift, внутренние low-rank LoRA-ветки decay/iclr/value/gate).
    ///
    /// `computeDType` — тип хранения результата (веса модели считают в
    /// bf16 по умолчанию — см. X070Backbone.init).
    public func buildDenseWeights(computeDType: DType = .bfloat16) throws -> [String: MLXArray] {
        guard nLayer > 0, nEmbd > 0, headSize > 0 else {
            throw RwkvqFullConvertError.incompleteManifest(
                "n_layer/n_embd/head_size отсутствуют или нулевые — манифест не самоописан")
        }
        let H = nEmbd / headSize
        var out: [String: MLXArray] = [:]

        func dq(_ worldKey: String) throws -> MLXArray {
            try dequantize(worldKey, dtype: computeDType)
        }

        // ── глобальные веса ──
        out["ln0.weight"] = try dq("blocks.0.ln0.weight")
        out["ln0.bias"]   = try dq("blocks.0.ln0.bias")
        out["ln_out.weight"] = try dq("ln_out.weight")
        out["ln_out.bias"]   = try dq("ln_out.bias")

        for i in 0 ..< nLayer {
            let b = "blocks.\(i)."; let att = b + "att."; let ffn = b + "ffn."
            let P = b + "tmix."

            out[b + "ln1.weight"] = try dq(b + "ln1.weight")
            out[b + "ln1.bias"]   = try dq(b + "ln1.bias")
            out[b + "ln2.weight"] = try dq(b + "ln2.weight")
            out[b + "ln2.bias"]   = try dq(b + "ln2.bias")

            for x in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                out[P + x] = try dq(att + x)
            }
            out[P + "k_k"] = try dq(att + "k_k").reshaped([H, headSize])
            out[P + "k_a"] = try dq(att + "k_a").reshaped([H, headSize])
            out[P + "r_k"] = try dq(att + "r_k").reshaped([H, headSize])

            out[P + "w_lora_A.weight"] = try dq(att + "w1").transposed()
            out[P + "w_lora_B.weight"] = try dq(att + "w2").transposed()
            out[P + "w_lora_B.bias"]   = try dq(att + "w0").reshaped([nEmbd])

            out[P + "a_lora_A.weight"] = try dq(att + "a1").transposed()
            out[P + "a_lora_B.weight"] = try dq(att + "a2").transposed()
            out[P + "a_lora_B.bias"]   = try dq(att + "a0").reshaped([nEmbd])

            out[P + "g_lora_A.weight"] = try dq(att + "g1").transposed()
            out[P + "g_lora_B.weight"] = try dq(att + "g2").transposed()

            if i > 0 {
                out[P + "v_lora_A.weight"] = try dq(att + "v1").transposed()
                out[P + "v_lora_B.weight"] = try dq(att + "v2").transposed()
                out[P + "v_lora_B.bias"]   = try dq(att + "v0").reshaped([nEmbd])
            }

            out[P + "ln_x.weight"] = try dq(att + "ln_x.weight")
            out[P + "ln_x.bias"]   = try dq(att + "ln_x.bias")

            out[b + "cmix.x_k"] = try dq(ffn + "x_k")
        }

        return out
    }
}
