import Foundation
import MLX
import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Загрузка X070Backbone из ОДНОГО сайдкара, без стороннего dense-файла.
//
//  С 04.08 экспорт .rwkvq_mlx полный (все раскладки, не только sb6) — этот
//  файл использует это напрямую: собирает тело модели (нормы, token-shift,
//  внутренние LoRA-ветки decay/iclr/value/gate) через
//  RwkvqSidecar.buildDenseWeights, а 194 больших sb6-тензора (4 tmix-
//  проекции + 2 cmix + head + emb на слой) подключает уже существующим
//  attachRwkvq — они остаются квантованными в памяти, ради чего всё и
//  затевалось.
// ───────────────────────────────────────────────────────────────────────

extension X070Backbone {

    /// Собрать бэкбон из самодостаточного сайдкара `.rwkvq_mlx`.
    ///
    /// - useNativeKernel: перекладка sb6 в родной контейнер MLX
    ///   (`quantizedMM`) вместо развёртывания в плотный транзиент на
    ///   каждом проходе — 15-70% быстрее на реальных тензорах при тех же
    ///   числах (см. RwkvqAttachOptions.useNativeKernel), поэтому здесь
    ///   включена по умолчанию в отличие от общего API.
    public static func load(fromSidecar sidecar: RwkvqSidecar,
                            computeDType: DType = .bfloat16,
                            useNativeKernel: Bool = true) throws -> X070Backbone {
        var weights = try sidecar.buildDenseWeights(computeDType: computeDType)

        // emb.weight — единственное исключение: физически это sb6, но
        // разворачивается ОДИН раз здесь, а не на каждом forward.
        // Формат не поддерживает строчную выборку (коды упакованы блоками
        // вдоль входной оси), так что оставить его в сайдкаре означало бы
        // разворачивать таблицу 65536×D целиком на КАЖДЫЙ gather — то же
        // соображение, что у attachRwkvq.quantizeEmbedding=false по
        // умолчанию, только принятое на шаг раньше.
        weights["emb.weight"] = try sidecar.dequantize("emb.weight", dtype: computeDType)

        let cfg = X070Config(nLayer: sidecar.nLayer, nEmbd: sidecar.nEmbd,
                             headSize: sidecar.headSize, vocab: sidecar.vocabSize)
        let bb = X070Backbone(weights: weights, cfg: cfg, computeDType: computeDType)

        // Остальные sb6: 4 tmix-проекции + 2 cmix + head на каждом слое.
        // quantizeEmbedding остаётся false — emb уже плотный (см. выше).
        let info = bb.attachRwkvq(sidecar, options: RwkvqAttachOptions(
            quantizeCmix: true, quantizeHead: true, quantizeEmbedding: false,
            useNativeKernel: useNativeKernel))

        if useNativeKernel && bb.isRawSidecarRetained {
            // Хотя бы один ключ не переложился в native — см.
            // isRawSidecarRetained. Печатаем ИМЕНА непереложенных ключей,
            // не только count: "что-то не так" бесполезно для диагностики,
            // "вот эти конкретные ключи" -- да.
            let notNative = info.attached > 0 ? bb.rwkvqBackedKeys.filter {
                !bb.hasNativeRepack($0)
            } : []
            FileHandle.standardError.write((
                "[X070Backbone.load] ВНИМАНИЕ: сырой сайдкар не освободился "
                + "(isRawSidecarRetained=true) -- держится ОДНОВРЕМЕННО с "
                + "native-буферами. attached=\(info.attached) missing=\(info.missing) "
                + "не переложились в native: \(notNative)\n"
            ).data(using: .utf8)!)
        }

        return bb
    }
}
