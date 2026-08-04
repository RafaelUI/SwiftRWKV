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
    public let scales: MLXArray      // fp16   [OUT, NB]
    public let biases: MLXArray      // fp16   [OUT, NB]
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
    public func nativeAffine(_ key: String) throws -> RwkvqNativeWeight {
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

        // ── коды ────────────────────────────────────────────────────
        // qblk[row, blk] = 16Б нибблов [+4Б qh] [+4Б qh2]
        let blk = qblk.reshaped([OUT, NB, 16 + 4 * xbits])
        let cb = blk[0..., 0..., 0 ..< 16]
        // блок-локальный split: младшие 16 колонок в low-нибблах,
        // старшие 16 — в high (см. schema.py::pack_nib_block)
        var q = concatenated([cb & 0xF, cb >> 4], axis: 2).asType(.uint32)
        let shifts = MLXArray(Array(0 ..< 8).map { UInt8($0) })
        for plane in 0 ..< xbits {
            let off = 16 + 4 * plane
            let hb = blk[0..., 0..., off ..< (off + 4)].reshaped([OUT, IN / 8])
            let b = (hb.expandedDimensions(axis: -1) >> shifts) & 1
            q = q + b.reshaped([OUT, NB, 32]).asType(.uint32)
                * UInt32(16 << plane)
        }

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
        let scales = maximum((qs * d).asType(.float16).asType(.float32), 1e-8)
            .asType(.float16)
        let biases = (qm * dm).asType(.float16).asType(.float32).asType(.float16)

        // ── упаковка ────────────────────────────────────────────────
        var words = [MLXArray](repeating: MLXArray.zeros([OUT, NB], dtype: .uint32),
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
        let wq = stacked(words, axis: -1).reshaped([OUT, NB * bits])
        eval(wq, scales, biases)

        return RwkvqNativeWeight(wq: wq, scales: scales, biases: biases,
                                 bits: bits, outFeatures: OUT, inFeatures: IN)
    }
}
