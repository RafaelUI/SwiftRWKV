import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Перекладка sb6 в родной контейнер MLX (`quantizedMM`).
//
//  Зачем. Сейчас `.rwkvq`-вес разворачивается целиком на КАЖДУЮ проекцию
//  каждого токена, и это стоит 2.21x против плотного bf16 на декоде 2.9B
//  (SwiftRWKV/decode-bench, 151 против 68 мс/ток). Причина ровно одна и
//  она арифметическая: путь читает 1855 МБ сжатых, ПИШЕТ 5896 МБ
//  плотного транзиента и читает их обратно — 13.6 ГБ против 5.9 у
//  плотного. Родное ядро читает только сжатое и не материализует
//  ничего: замер на формах 2.9B дал 2.1–4.9x против нынешнего пути и
//  1.3–2.1x против плотного.
//
//  ПЕРЕКЛАДКА БЕЗ ПОТЕРЬ. Наша формула `w = q·s + m` при беззнаковых
//  кодах и группе 32 — это и есть affine-модель MLX (`w = q·scale +
//  bias`, та же группа). Меняется только укладка бит; коды, scale и
//  bias остаются те же до последнего разряда.
//
//  ЧЕГО ДЕЛАТЬ НЕЛЬЗЯ: `quantized(денсовый_вес)`. Он пересчитает
//  scale/bias по min/max блока — это своя, никем не откалиброванная
//  схема, и вся разница между REDUCTION и COMPRESSION вместе с
//  измеренной деградацией ppl пропадает. Тем не менее именно так
//  устроен `decode-bench --micro`: там это ПРОБА СКОРОСТИ, и там это
//  помечено.
//
//  Раскладка контейнера MLX проверена для 4, 5, 6 и 8 бит
//  (rwkv-quant/tests/probe_mlx_native_packing.py): группа из 32 кодов —
//  LSB-first битовый поток, поле позиции p начинается на глобальном
//  бите p·bits и переходит границу 32-битного слова без выравнивания.
//
//  Эталон для проверки: `.testdata/mlx_affine_ref.safetensors`,
//  собирается rwkv-quant/tests/test_mlx_affine_repack.py --dump.
// ───────────────────────────────────────────────────────────────────────

/// Вес в родном контейнере MLX: то, что принимает `quantizedMM`.
public struct RwkvqNativeWeight: Sendable {
    public let wq: MLXArray          // uint32 [OUT, NB·bits]
    public let scales: MLXArray      // fp16 (fp32 у тензоров с зажатым scale) [OUT, NB]
    public let biases: MLXArray      // того же типа, что scales             [OUT, NB]
    public let bits: Int
    public let outFeatures: Int
    public let inFeatures: Int
    public var groupSize: Int { 32 }

    /// y = x · Wᵀ без материализации W.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        quantizedMM(x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: 32, bits: bits)
    }

    /// Байты, которые вес занимает в памяти. Для честного сравнения с
    /// `RwkvqSidecar.packedBytes`: контейнер MLX чуть крупнее нашего
    /// (bits + 1 бит на вес против bits + 0.625), и молчать об этом
    /// незачем.
    public var bytes: Int {
        [wq, scales, biases].reduce(0) { $0 + $1.size * $1.dtype.size }
    }
}

extension RwkvqSidecar {

    /// sb6-тензор из сайдкара → родной контейнер MLX.
    ///
    /// Считается один раз при загрузке. Дорогих операций тут нет: всё
    /// векторное, 32 сдвига на тензор.
    ///
    /// `exactScales` — что делать с зажимом scale снизу (1e-8). Формат держит
    /// scale вырожденного блока на 1e-8, а в fp16 это число не представимо
    /// (минимальная субнормаль ~6e-8) и становится нулём. Если коды такого
    /// блока нулевые, веса от этого не меняются, и scale / bias хранятся в
    /// fp16 — вдвое компактнее. Если нет — тензор остаётся в fp32 с зажимом,
    /// как в файле: решает сверка ДЕКВАНТА, а не предположение (так же
    /// поступает rwkv-metal, RwkvqNativeLinear). `false` — прежнее поведение:
    /// fp16 всегда, зажим обнуляется (так устроен эталон
    /// codec.sb6_to_mlx_affine; расхождение не больше 63·1e-8 на вес).
    public func nativeAffine(_ key: String, exactScales: Bool = true) throws -> RwkvqNativeWeight {
        try nativeAffineBanded(key, exactScales: exactScales)
    }

    /// Потолок транзиента перекладки, байт: коды полосы строк, развёрнутые
    /// в uint32. Меняется только тестом (одна полоса на весь тензор против
    /// многих — результат обязан совпасть побитно).
    nonisolated(unsafe) public static var nativeRepackBandBytes = 64 << 20

    private func nativeAffineBanded(_ key: String, exactScales: Bool) throws -> RwkvqNativeWeight {
        guard let info = tensors[key] else {
            throw RwkvqError.missingBuffer(tensor: key, buffer: "manifest")
        }
        guard info.isSb6 else {
            throw RwkvqError.unsupportedKind(tensor: key, kind: info.kind.rawValue)
        }
        guard let qblk = arrays["\(key)::qblk"],
              let qsqm = arrays["\(key)::qsqm"],
              let ddm = arrays["\(key)::ddm"] else {
            throw RwkvqError.missingBuffer(tensor: key, buffer: "qblk/qsqm/ddm")
        }
        let OUT = info.outFeatures, IN = info.inFeatures
        let NB = IN / 32, NSB = NB / info.superBlock
        let xbits = info.xbits
        let bits = 4 + xbits

        // ── масштабы ────────────────────────────────────────────────
        let sm = qsqm.reshaped([OUT, NB, 2])
        let qs = sm[0..., 0..., 0].asType(.float32)
        // qm лежит как int8 в байте uint8. Реинтерпретация делается
        // арифметикой, а не view: (u + 128) % 256 - 128 даёт u при
        // u < 128 и u - 256 при u >= 128, без зависимости от того, как
        // именно конкретная версия MLX трактует смену типа.
        let u = sm[0..., 0..., 1].asType(.int32)
        let qm = (((u + 128) % 256) - 128).asType(.float32)

        let dd = ddm.reshaped([OUT, NSB, 2])
        let sb = info.superBlock
        let d = repeated(dd[0..., 0..., 0].asType(.float32), count: sb, axis: 1)
        let dm = repeated(dd[0..., 0..., 1].asType(.float32), count: sb, axis: 1)

        // half-роундтрип обязателен: writer и ядро считают именно так.
        // Клип снизу 1e-8 в fp16 не представим (минимальная субнормаль
        // ~6e-8) и обращается в ноль — у вырожденных блоков в
        // контейнере окажется scale = 0. Все веса такого блока равны
        // bias, расхождение не больше 63·1e-8; на 2.9B таких блоков
        // 0.0034%, и все в emb (неиспользуемые строки словаря).
        let scales32 = maximum((qs * d).asType(.float16).asType(.float32), 1e-8)
        let biases32 = (qm * dm).asType(.float16).asType(.float32)
        let scales16 = scales32.asType(.float16)
        let biases16 = biases32.asType(.float16)      // точно: значение уже fp16

        eval(scales32, biases32, scales16, biases16)

        // ── коды и упаковка — ПОЛОСАМИ СТРОК ─────────────────────────
        // Перекладка разворачивает каждый код в uint32: [R, NB, 32] слов,
        // то есть 4 байта на вес против 4–6 бит в файле. Целым тензором это
        // давало пик загрузки втрое выше самого файла (1.5B COMPRESSION:
        // 3.4 ГБ при файле 0.95 ГиБ; одна голова 65536×2048 — 537 МБ
        // транзиента, и ещё столько же на сверку зажима). Полосами транзиент
        // ограничен ~64 МБ, результат тот же побитно (сверяется тестом).
        let shifts = MLXArray(Array(0 ..< 8).map { UInt8($0) })
        let bandRows = Swift.max(1, RwkvqSidecar.nativeRepackBandBytes / (NB * 32 * 4))
        let clamped = exactScales
            && !arrayEqual(scales16.asType(.float32), scales32).item(Bool.self)
        var useHalf = true
        var bands: [MLXArray] = []
        var row = 0
        while row < OUT {
            let R = Swift.min(bandRows, OUT - row)
            let rows = row ..< (row + R)
            // qblk[row, blk] = 16Б нибблов [+4Б qh] [+4Б qh2]
            let blk = qblk[rows].reshaped([R, NB, 16 + 4 * xbits])
            let cb = blk[0..., 0..., 0 ..< 16]
            // блок-локальный split: младшие 16 колонок в low-нибблах,
            // старшие 16 — в high (см. schema.py::pack_nib_block)
            var q = concatenated([cb & 0xF, cb >> 4], axis: 2).asType(.uint32)
            for plane in 0 ..< xbits {
                let off = 16 + 4 * plane
                let hb = blk[0..., 0..., off ..< (off + 4)].reshaped([R, IN / 8])
                let b = (hb.expandedDimensions(axis: -1) >> shifts) & 1
                q = q + b.reshaped([R, NB, 32]).asType(.uint32) * UInt32(16 << plane)
            }
            var words = [MLXArray](repeating: MLXArray.zeros([R, NB], dtype: .uint32),
                                   count: bits)
            for p in 0 ..< 32 {
                let start = p * bits
                let w0 = start / 32, offset = start % 32
                let lo = Swift.min(bits, 32 - offset)
                let hi = bits - lo
                let code = q[0..., 0..., p]
                words[w0] = words[w0] | ((code & UInt32((1 << lo) - 1)) << UInt32(offset))
                if hi > 0 {
                    words[w0 + 1] = words[w0 + 1]
                        | ((code >> UInt32(lo)) & UInt32((1 << hi) - 1))
                }
            }
            let band = stacked(words, axis: -1).reshaped([R, NB * bits])
            eval(band)
            if clamped, useHalf {
                // есть блоки на зажиме: fp16 законен, только если деквант тот же
                let a = dequantized(band, scales: scales16[rows].asType(.float32),
                                    biases: biases32[rows], groupSize: 32, bits: bits)
                let b = dequantized(band, scales: scales32[rows], biases: biases32[rows],
                                    groupSize: 32, bits: bits)
                useHalf = arrayEqual(a, b).item(Bool.self)
            }
            bands.append(band)
            row += R
        }
        let wq = bands.count == 1 ? bands[0] : concatenated(bands, axis: 0)
        let scales = useHalf ? scales16 : scales32
        let biases = useHalf ? biases16 : biases32
        eval(wq, scales, biases)
        bands.removeAll()

        return RwkvqNativeWeight(wq: wq, scales: scales, biases: biases,
                                 bits: bits, outFeatures: OUT, inFeatures: IN)
    }
}
