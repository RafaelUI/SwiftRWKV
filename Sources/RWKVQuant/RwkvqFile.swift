import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Чтение .rwkvq НАПРЯМУЮ, без сайдкара.
//
//  Порт rwkv_quant/formats/codec.py (контейнер и перекладка в раскладку
//  загрузчика) и rwkv_metal/lora/rwkvq_linear.py::_load_rwkvq_direct.
//
//  .rwkvq — это safetensors: 8 байт длины заголовка (u64 LE), JSON-заголовок,
//  сырые буферы. Манифест формата лежит строкой JSON в
//  "__metadata__"["rwkvq"], буферы адресуются как "<ключ>::<поле>".
//  `loadArrays` из mlx-swift сюда не годится по двум причинам: он выбирает
//  разбор по расширению файла (а оно .rwkvq) и материализует все буферы
//  разом. Здесь файл отображается в память, и каждый тензор забирается
//  отдельно: канонические буферы sb6 живут ровно до конца своей перекладки.
//
//  ДИСК КАНОНИЧЕСКИЙ, ИНТЕРЛИВ — ДЕТАЛЬ ЗАГРУЗЧИКА. Ядра деквантизации
//  читают не дисковую раскладку, а интерлив (у sb6 он называется K3): коды
//  и масштабы одного блока лежат рядом. Его прежде строил только export_mlx
//  и клал в сайдкар; здесь он строится при загрузке. Числа при этом не
//  пересчитываются — это перестановка байт, и что она совпадает с
//  сайдкарной, проверяется тестом на равенство буферов, а не выводится.
//
//  Результат — тот же `RwkvqSidecar`, что даёт сайдкар: всё, что стоит
//  выше (деквантизация, родной контейнер MLX, подключение к бэкбону), не
//  различает источник.
// ───────────────────────────────────────────────────────────────────────

/// Контейнер safetensors, отображённый в память. Буферы отдаются по одному.
public struct RwkvqContainer {

    struct Entry {
        let dtype: DType
        let shape: [Int]
        let range: Range<Int>
    }

    let data: Data
    let entries: [String: Entry]
    /// Секция "__metadata__" заголовка.
    public let metadata: [String: String]

    /// Имена типов safetensors → MLX. Список закрытый: неизвестный тип —
    /// отказ с именем буфера, а не чтение байт «как получится».
    static let dtypes: [String: DType] = [
        "BOOL": .bool, "U8": .uint8, "I8": .int8, "I16": .int16,
        "U16": .uint16, "F16": .float16, "BF16": .bfloat16,
        "I32": .int32, "U32": .uint32, "F32": .float32,
        "F64": .float64, "I64": .int64, "U64": .uint64,
    ]

    public init(path: String) throws {
        let p = (path as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: p) else {
            throw RwkvqError.missingFile(p)
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: p), options: .alwaysMapped)
        guard data.count >= 8 else {
            throw RwkvqError.malformedManifest("\(p): файл короче заголовка safetensors")
        }
        // Первые байты zip'а, который пишет torch.save. Без этой проверки
        // pickle-контейнер прежней эры читается как safetensors с длиной
        // заголовка порядка 10^18 — и вместо внятной ошибки выходит мусор.
        if data.prefix(4).elementsEqual([0x50, 0x4B, 0x03, 0x04]) {
            throw RwkvqError.malformedManifest(
                "\(p): это pickle-контейнер (torch.save), а не safetensors — "
                + "пересохраните файл нынешним rwkv-quant")
        }
        var n: UInt64 = 0
        for i in 0 ..< 8 { n |= UInt64(data[data.startIndex + i]) << UInt64(8 * i) }
        guard n > 0, n <= UInt64(data.count - 8) else {
            throw RwkvqError.malformedManifest("\(p): длина заголовка \(n) вне файла")
        }
        let base = 8 + Int(n)
        guard let header = try JSONSerialization.jsonObject(
            with: data.subdata(in: 8 ..< base)) as? [String: Any] else {
            throw RwkvqError.malformedManifest("\(p): заголовок safetensors — не объект")
        }
        var entries: [String: Entry] = [:]
        entries.reserveCapacity(header.count)
        var meta: [String: String] = [:]
        for (name, value) in header {
            if name == "__metadata__" {
                meta = value as? [String: String] ?? [:]
                continue
            }
            guard let d = value as? [String: Any],
                  let ts = d["dtype"] as? String,
                  let shape = d["shape"] as? [Int],
                  let off = d["data_offsets"] as? [Int], off.count == 2 else {
                throw RwkvqError.malformedManifest("\(p): неполная запись буфера \(name)")
            }
            guard let dt = RwkvqContainer.dtypes[ts] else {
                throw RwkvqError.malformedManifest("\(p): тип \(ts) у буфера \(name) не поддержан")
            }
            let count = shape.reduce(1, *)
            guard off[0] >= 0, off[1] >= off[0], base + off[1] <= data.count,
                  off[1] - off[0] == count * dt.size else {
                throw RwkvqError.malformedManifest(
                    "\(p): буфер \(name) — смещения \(off) не сходятся с формой \(shape) и типом \(ts)")
            }
            entries[name] = Entry(dtype: dt, shape: shape,
                                  range: (base + off[0]) ..< (base + off[1]))
        }
        self.data = data
        self.entries = entries
        self.metadata = meta
    }

    public var names: [String] { entries.keys.sorted() }

    public func contains(_ name: String) -> Bool { entries[name] != nil }

    public func shape(_ name: String) -> [Int]? { entries[name]?.shape }

    /// Буфер копией в память MLX; nil, если такого имени нет.
    ///
    /// `as` — прочитать те же байты другим типом того же размера. Нужно
    /// ровно в одном месте: знаковые байты (коды и масштабы sym) ядро читает
    /// через `char*`, а интерлив хранит их как uint8 — приведение значений
    /// здесь было бы лишней операцией с тем же результатом по байтам.
    public func array(_ name: String, as reinterpret: DType? = nil) -> MLXArray? {
        guard let e = entries[name] else { return nil }
        let dt = reinterpret ?? e.dtype
        precondition(dt.size == e.dtype.size,
                     "\(name): \(e.dtype) нельзя прочитать как \(dt) — разный размер элемента")
        if e.range.isEmpty { return MLXArray.zeros(e.shape, dtype: dt) }
        return MLXArray(data[e.range], e.shape, dtype: dt)
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Перекладка дисковой раскладки в раскладку загрузчика
// ───────────────────────────────────────────────────────────────────────

/// 6-битный битстрим: uint8 [..., 6] → int32 [..., 8], значения 0...63.
/// Порт codec.unpack6 для единственной длины, которая встречается в формате
/// (8 масштабов на суперблок): чанк из 4 значений — 3 байта, LSB-first.
func rwkvqUnpack6x8(_ b: MLXArray) -> MLXArray {
    precondition(b.shape.last == 6, "unpack6: ожидалось 6 байт на суперблок, а не \(b.shape.last ?? -1)")
    let lead = Array(b.shape.dropLast())
    let x = b.asType(.int32).reshaped(lead + [2, 3])
    let x0 = x[.ellipsis, 0], x1 = x[.ellipsis, 1], x2 = x[.ellipsis, 2]
    let v0 = x0 & 0x3F
    let v1 = ((x0 >> 6) | (x1 << 2)) & 0x3F
    let v2 = ((x1 >> 4) | (x2 << 4)) & 0x3F
    let v3 = (x2 >> 2) & 0x3F
    return stacked([v0, v1, v2, v3], axis: -1).reshaped(lead + [8])
}

/// sb6: дисковая раскладка → K3-интерлив (qblk, qsqm, ddm) и число
/// битплоскостей. Порт codec.sb6_to_k3 — перестановка байт, не пересчёт.
///
///   qblk[row, blk] = 16Б нибблов [+4Б qh] [+4Б qh2]
///   qsqm[row, blk] = (qs, qm) двумя байтами; qm как int8 БЕЗ сдвига +31
///   ddm[row, sblk] = (d, dm) двумя fp16
func rwkvqSb6ToK3(codesPacked: MLXArray, qsqm: MLXArray, d: MLXArray, dm: MLXArray,
                  qh: MLXArray?, qh2: MLXArray?,
                  shape: [Int], groupSize gs: Int, superBlock sb: Int)
    -> (qblk: MLXArray, qsqm: MLXArray, ddm: MLXArray, xbits: Int) {
    let OUT = shape[0], IN = shape[1]
    let NB = IN / gs
    let xbits = (qh == nil ? 0 : 1) + (qh2 == nil ? 0 : 1)

    var parts = [codesPacked.reshaped([OUT, NB, 16])]
    if let qh { parts.append(qh.reshaped([OUT, NB, 4])) }
    if let qh2 { parts.append(qh2.reshaped([OUT, NB, 4])) }
    let qblk = concatenated(parts, axis: 2).reshaped([OUT, NB * (16 + 4 * xbits)])

    let qs = rwkvqUnpack6x8(qsqm[.ellipsis, 0 ..< 6]).reshaped([OUT, NB])
    // -31 применяется ЗДЕСЬ и хранится байтом int8: ядро читает его через
    // as_type<char> и второго сдвига не делает. В uint8-контейнере это
    // (qm - 31) по модулю 256.
    let qm = rwkvqUnpack6x8(qsqm[.ellipsis, 6 ..< 12]).reshaped([OUT, NB])
    let qmByte = (qm - 31 + 256) % 256
    let k3 = stacked([qs, qmByte], axis: -1).asType(.uint8).reshaped([OUT, NB * 2])

    let ddm = stacked([d, dm], axis: -1).reshaped([OUT, (NB / sb) * 2])
    return (qblk, k3, ddm, xbits)
}

extension RwkvqSidecar {

    /// Открыть квантованную базу: `.rwkvq` напрямую либо прежний сайдкар
    /// (путь без суффикса, рядом `<path>.safetensors` и `<path>.json`).
    /// Различается по тому, что лежит на диске, чтобы прежние вызовы
    /// работали без правок.
    public static func open(_ path: String) throws -> RwkvqSidecar {
        let p = (path as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        let isFile = FileManager.default.fileExists(atPath: p, isDirectory: &isDir) && !isDir.boolValue
        if !isFile, FileManager.default.fileExists(atPath: p + ".safetensors") {
            return try RwkvqSidecar(path: p)
        }
        return try RwkvqSidecar(rwkvq: p)
    }

    /// Прочитать `.rwkvq` напрямую. Сайдкар не нужен.
    public convenience init(rwkvq path: String) throws {
        let p = (path as NSString).expandingTildeInPath
        let box = try RwkvqContainer(path: p)
        guard let text = box.metadata["rwkvq"], let raw = text.data(using: .utf8),
              let root = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else {
            throw RwkvqError.malformedManifest("\(p): не .rwkvq — в __metadata__ нет ключа rwkvq")
        }
        guard (root["format"] as? String) == "rwkvq" else {
            throw RwkvqError.malformedManifest("\(p): формат \(root["format"] ?? "nil")")
        }
        let version = root["format_version"] as? Int ?? 0
        guard version <= RwkvqSidecar.maxFormatVersion else {
            throw RwkvqError.malformedManifest(
                "\(p): format_version \(version) новее \(RwkvqSidecar.maxFormatVersion)")
        }
        guard let tensorDict = root["tensors"] as? [String: Any] else {
            throw RwkvqError.malformedManifest("\(p): нет секции \"tensors\"")
        }
        let naming = root["naming"] as? String ?? "world"

        var arrays: [String: MLXArray] = [:]
        var infos: [String: RwkvqTensorInfo] = [:]

        for (key, value) in tensorDict {
            guard let m = value as? [String: Any],
                  let shape = m["shape"] as? [Int],
                  let kindName = m["kind"] as? String else {
                throw RwkvqError.malformedManifest("\(p): нет формы или раскладки у \(key)")
            }
            guard let kind = RwkvqKind(rawValue: kindName) else {
                throw RwkvqError.malformedManifest("неизвестная раскладка \(kindName) у \(key)")
            }
            let bits = m["bits"] as? Int ?? 16
            let gs = m["gw_gs"] as? Int ?? 0
            let sb = m["gw_sb"] as? Int ?? 0
            let transposed = (m["transposed"] as? Bool)
                ?? (naming == "world" && RwkvqNaming.isRawLoRAWorld(key))

            func need(_ field: String, as dt: DType? = nil) throws -> MLXArray {
                guard let a = box.array("\(key)::\(field)", as: dt) else {
                    throw RwkvqError.missingBuffer(tensor: key, buffer: field)
                }
                return a
            }
            func nBlocksCheck(_ got: Int, _ what: String) throws {
                if let nb = m["n_blocks"] as? Int, nb > 0, nb != got {
                    throw RwkvqError.malformedManifest(
                        "\(key): \(what) покрывает \(got) блоков, в манифесте \(nb)")
                }
            }

            var xbits = 0, packed = false, outliers = 0
            switch kind {
            case .dense:
                arrays["\(key)::dense"] = try need("dense")

            case .asym:
                guard shape.count == 2, gs > 0 else {
                    throw RwkvqError.malformedManifest("неполная asym-запись для \(key)")
                }
                let scale = try need("gw_scale")
                // блоков CEIL(IN/gs): хвостовой неполный блок имеет свой scale
                let nb = (shape[1] + gs - 1) / gs
                guard scale.shape.last == nb else {
                    throw RwkvqError.shapeMismatch(tensor: key, expected: [shape[0], nb],
                                                   got: scale.shape)
                }
                try nBlocksCheck(nb, "gw_scale")
                arrays["\(key)::codes"] = try need("codes")
                arrays["\(key)::gw_scale"] = scale
                arrays["\(key)::gw_min"] = try need("gw_min")

            case .rtn:
                if box.contains("\(key)::codes_packed") {
                    packed = true
                    arrays["\(key)::codes_packed"] = try need("codes_packed")
                } else {
                    arrays["\(key)::codes"] = try need("codes")
                }
                arrays["\(key)::scale"] = try need("scale")
                outliers = box.shape("\(key)::outlier_indices")?.first ?? 0

            case .sb6:
                guard shape.count == 2, gs == 32, sb == 8,
                      shape[1] % (gs * sb) == 0 else {
                    throw RwkvqError.malformedManifest(
                        "\(key): sb6 читается при блоке 32 и суперблоке 8 "
                        + "(в файле gw_gs=\(gs), gw_sb=\(sb), форма \(shape))")
                }
                try nBlocksCheck(shape[1] / gs, "sb6")
                let qsqm = try need("gw_qsqm")
                guard qsqm.shape == [shape[0], shape[1] / (gs * sb), 12] else {
                    throw RwkvqError.shapeMismatch(
                        tensor: key, expected: [shape[0], shape[1] / (gs * sb), 12], got: qsqm.shape)
                }
                let k3 = rwkvqSb6ToK3(
                    codesPacked: try need("codes_packed"), qsqm: qsqm,
                    d: try need("gw_d"), dm: try need("gw_dm"),
                    qh: box.array("\(key)::gw_qh"), qh2: box.array("\(key)::gw_qh2"),
                    shape: shape, groupSize: gs, superBlock: sb)
                // eval на каждом тензоре обязателен: иначе граф ленив и
                // канонические буферы всех тензоров доживают до конца
                // загрузки рядом с интерливом.
                eval(k3.qblk, k3.qsqm, k3.ddm)
                arrays["\(key)::qblk"] = k3.qblk
                arrays["\(key)::qsqm"] = k3.qsqm
                arrays["\(key)::ddm"] = k3.ddm
                xbits = k3.xbits
                guard bits == 4 + xbits else {
                    throw RwkvqError.malformedManifest(
                        "\(key): bits=\(bits), а битплоскостей в файле \(xbits)")
                }

            case .sym:
                guard shape.count == 2, gs == 16, sb == 16 else {
                    throw RwkvqError.malformedManifest(
                        "\(key): sym читается при блоке 16 и суперблоке 16 "
                        + "(в файле gw_gs=\(gs), gw_sb=\(sb))")
                }
                let OUT = shape[0], IN = shape[1]
                guard IN % gs == 0 else {
                    throw RwkvqError.malformedManifest("\(key): sym, IN=\(IN) не кратен блоку 16")
                }
                let qs = try need("gw_qs", as: .uint8)
                let d = try need("gw_d")
                try nBlocksCheck(qs.shape.last ?? 0, "gw_qs")
                guard qs.shape == [OUT, IN / gs], d.shape.count == 2,
                      d.shape[1] * sb == IN / gs else {
                    throw RwkvqError.shapeMismatch(tensor: key, expected: [OUT, IN / gs],
                                                   got: qs.shape)
                }
                // Битность различается ПО ПРИСУТСТВИЮ БУФЕРОВ, как в codec:
                // `codes` есть только при восьми битах, `codes_packed` —
                // только при шести.
                let qblk: MLXArray
                if box.contains("\(key)::codes") {
                    guard bits == 8 else {
                        throw RwkvqError.malformedManifest("\(key): sym с буфером codes при bits=\(bits)")
                    }
                    qblk = try need("codes", as: .uint8)
                } else {
                    guard bits == 6, IN % 32 == 0 else {
                        throw RwkvqError.malformedManifest(
                            "\(key): sym с codes_packed при bits=\(bits), IN=\(IN)")
                    }
                    let NP = IN / 32
                    qblk = concatenated([
                        try need("codes_packed").reshaped([OUT, NP, 16]),
                        try need("gw_qh").reshaped([OUT, NP, 4]),
                        try need("gw_qh2").reshaped([OUT, NP, 4]),
                    ], axis: 2).reshaped([OUT, NP * 24])
                    eval(qblk)
                }
                arrays["\(key)::qblk"] = qblk
                arrays["\(key)::qs"] = qs
                arrays["\(key)::d"] = d
            }
            infos[key] = RwkvqTensorInfo(kind: kind, shape: shape, bits: bits,
                                         xbits: xbits, groupSize: gs, superBlock: sb,
                                         packed: packed, outlierCount: outliers,
                                         transposed: transposed)
        }
        eval(Array(arrays.values))

        self.init(arrays: arrays, tensors: infos, naming: naming,
                  nLayer: root["n_layer"] as? Int ?? 0,
                  nEmbd: root["n_embd"] as? Int ?? 0,
                  headSize: root["head_size"] as? Int ?? 64,
                  vocabSize: root["vocab_size"] as? Int ?? 0)
    }
}
