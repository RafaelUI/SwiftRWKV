import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Буферы базы для выгрузки на диск и замены (подкачка весов, RWKVGen).
//
//  Подкачка весов квантованной базы выгружает её буферы в файл и дальше
//  читает их оттуда по частям. Здесь — что выгружать и как собрать ту же
//  базу над другими буферами тех же имён. Срезы сведённых групп (r/k/v слоя
//  в одном буфере, `fusingSym`) не выгружаются: они собираются из буфера
//  группы заново теми же срезами строк — без копии байт, как при сведении.
// ───────────────────────────────────────────────────────────────────────

extension RwkvqSidecar {

    /// Имя буфера сведённой группы в выгрузке: `fused::<группа>::qblk|qs|d`.
    public static func fusedBufferName(_ group: String, _ part: String) -> String { "fused::\(group)::\(part)" }

    /// Буферы для выгрузки: собственные буферы тензоров (без срезов сведённых
    /// групп) и буферы групп.
    public var spillBuffers: [String: MLXArray] {
        var sliced = Set<String>()
        for g in symFused.values { for k in g.keys { for p in ["qblk", "qs", "d"] { sliced.insert("\(k)::\(p)") } } }
        var out = arrays.filter { !sliced.contains($0.key) }
        for (id, g) in symFused {
            out[Self.fusedBufferName(id, "qblk")] = g.qblk
            out[Self.fusedBufferName(id, "qs")] = g.qs
            out[Self.fusedBufferName(id, "d")] = g.d
        }
        return out
    }

    /// Та же база, где буферы с именами из `new` (имена `spillBuffers`) заменены;
    /// остальные — прежние. Срезы сведённой группы пересобираются из её новых буферов.
    public func replacingSpilled(_ new: [String: MLXArray]) -> RwkvqSidecar {
        var a = arrays
        var fused = symFused
        for (name, v) in new {
            let parts = name.components(separatedBy: "::")
            if parts.count == 3, parts[0] == "fused" {
                let id = parts[1], part = parts[2]
                guard let g = fused[id] else { preconditionFailure("подкачка базы: нет сведённой группы \(id)") }
                let ng = RwkvqSymFusedGroup(keys: g.keys, qblk: part == "qblk" ? v : g.qblk,
                                            qs: part == "qs" ? v : g.qs, d: part == "d" ? v : g.d,
                                            outPer: g.outPer, inFeatures: g.inFeatures, bits: g.bits)
                fused[id] = ng
                for (i, key) in g.keys.enumerated() {
                    a["\(key)::\(part)"] = v[(i * g.outPer) ..< ((i + 1) * g.outPer)]
                }
            } else {
                guard a[name] != nil else { preconditionFailure("подкачка базы: нет буфера \(name)") }
                a[name] = v
            }
        }
        return RwkvqSidecar(arrays: a, tensors: tensors, naming: naming, nLayer: nLayer, nEmbd: nEmbd,
                            headSize: headSize, vocabSize: vocabSize, symFused: fused)
    }
}

extension RwkvqSymRowTable {

    /// Буферы таблицы для выгрузки: `codes`, `qs`, `d`.
    public var spillBuffers: [String: MLXArray] { ["codes": codes, "qs": qs, "d": d] }

    /// Та же таблица над другими буферами тех же имён (отсутствующие — прежние).
    public func replacingSpilled(_ new: [String: MLXArray]) -> RwkvqSymRowTable {
        RwkvqSymRowTable(codes: new["codes"] ?? codes, qs: new["qs"] ?? qs, d: new["d"] ?? d,
                         rowCount: rowCount, inFeatures: inFeatures)
    }
}
