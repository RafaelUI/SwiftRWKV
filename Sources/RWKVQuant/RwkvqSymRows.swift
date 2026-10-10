import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Таблица эмбеддинга в раскладке sym при восьми битах: строки
//  деквантуются ПО ТРЕБОВАНИЮ, плотной таблицы не существует.
//
//  Порт rwkv_quant/backends/metal/quant_model.py::SymGatherEmb.
//
//  Зачем. Плотная fp16-таблица — крупнейшая одиночная статья памяти модели
//  из .rwkvq: 65536 × D × 2 байта (268 МБ на 1.5B, 335 МБ на 2.9B) при том,
//  что коды той же таблицы занимают половину. И при загрузке она же давала
//  всплеск: коды и плотная таблица живут рядом, пока идёт разворот.
//
//  Почему это возможно именно здесь. У sym при восьми битах код — один байт
//  на элемент, масштаб — на блок 16 вдоль строки: строка таблицы читается
//  выборкой строк трёх буферов. У sb6 (COMPRESSION) и у sym при шести битах
//  коды упакованы битплоскостями — там таблица остаётся плотной.
//
//  ТОЖДЕСТВО с плотным путём (ядро `rwkvqDequantSym`, тип fp16): масштаб
//  блока s = half(qs · d), элемент = half(код · s). Код — целое до восьми
//  бит, s — half: произведение точно представимо в fp32, округление одно.
//  Равенство ПОБИТНОЕ и проверяется тестом, а не выводится.
// ───────────────────────────────────────────────────────────────────────

public struct RwkvqSymRowTable {
    /// Знаковые байты кодов в uint8-контейнере, [rows, IN].
    let codes: MLXArray
    /// Знаковые байты масштабов блоков, [rows, IN / 16].
    let qs: MLXArray
    /// fp16-масштаб суперблока, [rows, ceil(IN / 256)].
    let d: MLXArray
    public let rowCount: Int
    public let inFeatures: Int

    /// Байт int8 в uint8-контейнере → значение. Арифметикой, как в
    /// `rwkvqDequantSymReference`: не зависит от того, как MLX трактует каст.
    private static func signed(_ u: MLXArray) -> MLXArray {
        ((u.asType(.int32) + 128) % 256) - 128
    }

    /// Строки таблицы: ids любой формы → [..., IN], fp16.
    public func rows(_ ids: MLXArray) -> MLXArray {
        let lead = ids.shape
        let NB = inFeatures / 16
        let c = RwkvqSymRowTable.signed(codes.take(ids, axis: 0)).asType(.float32)
        let q = RwkvqSymRowTable.signed(qs.take(ids, axis: 0)).asType(.float32)
        let dd = repeated(d.take(ids, axis: 0).asType(.float32), count: 16, axis: -1)[.ellipsis, 0 ..< NB]
        let scale = (q * dd).asType(.float16).asType(.float32)
        return (c.reshaped(lead + [NB, 16]) * scale.expandedDimensions(axis: -1))
            .reshaped(lead + [inFeatures]).asType(.float16)
    }

    /// Вся таблица плотной, [rows, IN] fp16 — для тех, кому нужен вес целиком
    /// (копирование наружу, слияние); на горячем пути не зовётся.
    public func dense() -> MLXArray {
        rwkvqDequantSym(qblk: codes, qs: qs, d: d, outFeatures: rowCount,
                        inFeatures: inFeatures, bits: 8, dtype: .float16)
    }

    /// Сколько байт занимают буферы таблицы.
    public var packedBytes: Int { codes.size + qs.size + d.size * 2 }
}

extension RwkvqSidecar {

    /// Таблица строк для тензора `key`; nil — раскладка не sym при восьми
    /// битах, IN не кратен блоку либо буферы не подняты. Отказ не молчаливый
    /// для вызывающего: по nil он обязан пойти плотным путём.
    public func symRowTable(_ key: String) -> RwkvqSymRowTable? {
        guard let info = tensors[key], info.kind == .sym, info.bits == 8,
              info.shape.count == 2, info.inFeatures % 16 == 0,
              let qblk = arrays["\(key)::qblk"], let qs = arrays["\(key)::qs"],
              let d = arrays["\(key)::d"],
              qblk.dtype == .uint8, qs.dtype == .uint8, d.dtype == .float16 else { return nil }
        return RwkvqSymRowTable(codes: qblk.reshaped([info.outFeatures, info.inFeatures]), qs: qs, d: d,
                                rowCount: info.outFeatures, inFeatures: info.inFeatures)
    }
}
