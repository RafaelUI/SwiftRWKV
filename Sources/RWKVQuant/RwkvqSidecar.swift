import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Сайдкар .rwkvq_mlx: torch-free представление квантованной базы.
//
//  Порт rwkv_metal/lora/rwkvq_linear.py (часть загрузки).
//
//  Почему сайдкар, а не чтение .rwkvq напрямую: разбор .rwkvq требует torch
//  (rwkv_quant/formats/reader.py), а здесь его нет и быть не должно.
//  Конвертация одноразовая, на стороне rwkv-quant:
//
//      python -m rwkv_quant.formats.export_mlx model.rwkvq out/model.rwkvq_mlx
//
//  и даёт пару файлов: <path>.safetensors с буферами K3-интерлива
//  (qblk/qsqm/ddm на каждый sb6-тензор) и <path>.json с манифестом.
//
//  Экспортируются ТОЛЬКО sb6-тензоры. Низкоранговые w/a/v_lora остаются в
//  fp по конвенции QLoRA-базы: их ранги не кратны размеру группы, а квант
//  портит in-context динамику.
// ───────────────────────────────────────────────────────────────────────

public enum RwkvqError: Error, CustomStringConvertible {
    case missingFile(String)
    case malformedManifest(String)
    case missingBuffer(tensor: String, buffer: String)
    case shapeMismatch(tensor: String, expected: [Int], got: [Int])
    case unsupportedKind(tensor: String, kind: String)

    public var description: String {
        switch self {
        case .missingFile(let p):
            return "нет файла сайдкара: \(p)"
        case .malformedManifest(let why):
            return "манифест сайдкара повреждён: \(why)"
        case .missingBuffer(let t, let b):
            return "в сайдкаре нет буфера \(b) для тензора \(t)"
        case .shapeMismatch(let t, let e, let g):
            return "\(t): ожидалась форма \(e), в сайдкаре \(g)"
        case .unsupportedKind(let t, let k):
            return "\(t): раскладка \(k) здесь не разворачивается "
                 + "(считается только sb6)"
        }
    }
}

/// Раскладка тензора. Манифест полного экспорта содержит все четыре;
/// разворачивать здесь мы умеем только `sb6`, но ОТВЕРГАТЬ файл из-за
/// присутствия остальных нельзя — они там законно.
public enum RwkvqKind: String, Sendable {
    case sb6        // блочная квантизация, единственная, что здесь считается
    case asym       // low-rank LoRA, fp32 scale/min на блок
    case rtn        // per-row, коды int8 либо нибблы
    case dense      // bf16 как есть: нормы, token-shift миксы, bias-термы
}

/// Метаданные одного тензора манифеста.
///
/// Поля `xbits`/`groupSize`/`superBlock` осмысленны только у `sb6`; у
/// остальных раскладок они нулевые, и трогать их незачем. Форма тоже
/// не обязана быть двумерной: в полном экспорте есть одномерные нормы
/// и `(1,1,C)` token-shift миксы.
public struct RwkvqTensorInfo: Sendable {
    public let kind: RwkvqKind
    public let shape: [Int]
    public let bits: Int
    public let xbits: Int
    public let groupSize: Int      // gw_gs
    public let superBlock: Int     // gw_sb

    public var isSb6: Bool { kind == .sb6 }
    /// Только для двумерных. У одномерных записей полного экспорта
    /// обращение к ним — ошибка вызывающего, а не данных.
    public var outFeatures: Int { shape[0] }
    public var inFeatures: Int { shape[1] }
}

/// Загруженный сайдкар: буферы плюс манифест.
public final class RwkvqSidecar {

    public let arrays: [String: MLXArray]
    public let tensors: [String: RwkvqTensorInfo]
    public let naming: String
    public let nLayer: Int
    public let nEmbd: Int
    public let headSize: Int
    public let vocabSize: Int

    /// - path: путь БЕЗ суффикса — рядом должны лежать `<path>.safetensors`
    ///   и `<path>.json` (так их кладёт export_mlx).
    public init(path: String) throws {
        let base = (path as NSString).expandingTildeInPath
        let st = base + ".safetensors"
        let js = base + ".json"
        let fm = FileManager.default
        guard fm.fileExists(atPath: st) else { throw RwkvqError.missingFile(st) }
        guard fm.fileExists(atPath: js) else { throw RwkvqError.missingFile(js) }

        self.arrays = try loadArrays(url: URL(fileURLWithPath: st))

        let raw = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: URL(fileURLWithPath: js)))
        guard let root = raw as? [String: Any],
              let tensorDict = root["tensors"] as? [String: Any] else {
            throw RwkvqError.malformedManifest("нет секции \"tensors\"")
        }
        self.naming = root["naming"] as? String ?? "world"
        self.nLayer = root["n_layer"] as? Int ?? 0
        self.nEmbd = root["n_embd"] as? Int ?? 0
        self.headSize = root["head_size"] as? Int ?? 64
        self.vocabSize = root["vocab_size"] as? Int ?? 0

        // Разбор ПО РАСКЛАДКАМ. Прежняя версия требовала от КАЖДОЙ записи
        // двумерную форму и поля xbits/gw_gs/gw_sb — то есть предполагала,
        // что в манифесте бывает только sb6. Это было верно для первых
        // сайдкаров и перестало быть верным, как только экспорт стал
        // полным: одномерные нормы и (1,1,C) token-shift миксы роняли
        // загрузку целиком, ещё до всякого обращения к весам.
        var infos: [String: RwkvqTensorInfo] = [:]
        for (key, value) in tensorDict {
            guard let d = value as? [String: Any],
                  let shape = d["shape"] as? [Int] else {
                throw RwkvqError.malformedManifest("нет формы у \(key)")
            }
            // манифесты первых сайдкаров поля kind не имели вовсе и
            // содержали только sb6 — отсюда умолчание
            let raw = d["kind"] as? String ?? "sb6"
            guard let kind = RwkvqKind(rawValue: raw) else {
                throw RwkvqError.malformedManifest(
                    "неизвестная раскладка \(raw) у \(key)")
            }
            if kind == .sb6 {
                guard shape.count == 2,
                      let bits = d["bits"] as? Int,
                      let xbits = d["xbits"] as? Int,
                      let gs = d["gw_gs"] as? Int,
                      let sb = d["gw_sb"] as? Int else {
                    throw RwkvqError.malformedManifest(
                        "неполная sb6-запись для \(key)")
                }
                infos[key] = RwkvqTensorInfo(kind: kind, shape: shape, bits: bits,
                                             xbits: xbits, groupSize: gs,
                                             superBlock: sb)
            } else {
                infos[key] = RwkvqTensorInfo(
                    kind: kind, shape: shape,
                    bits: d["bits"] as? Int ?? 16, xbits: 0,
                    groupSize: d["gw_gs"] as? Int ?? 0,
                    superBlock: d["gw_sb"] as? Int ?? 0)
            }
        }
        self.tensors = infos
    }

    public var keys: [String] { tensors.keys.sorted() }

    public func contains(_ key: String) -> Bool { tensors[key] != nil }

    /// Восстановить плотный вес [OUT, IN].
    ///
    /// Результат ТРАНЗИЕНТНЫЙ и намеренно не кэшируется: смысл квантованной
    /// базы в том, что она живёт в памяти сжатой. Кэш плотных весов молча
    /// превращает QLoRA обратно в LoRA, только с лишними шагами.
    ///
    /// `dtype` — тип хранения; арифметика деквантизации всегда float.
    /// Просить `.bfloat16` стоит там, где результат всё равно тут же уйдёт в
    /// bf16-вычисления: транзиент вдвое меньше, а числа те же до бита.
    public func dequantize(_ key: String, dtype: DType = .float32) throws -> MLXArray {
        guard let info = tensors[key] else {
            throw RwkvqError.missingBuffer(tensor: key, buffer: "manifest")
        }
        guard info.isSb6 else {
            // Не «не нашли буфер», а «эта раскладка здесь не считается»:
            // asym/rtn/dense в файле есть законно, просто разворачивать
            // их умеет rwkv-quant, а не этот модуль.
            throw RwkvqError.unsupportedKind(tensor: key, kind: info.kind.rawValue)
        }
        guard let qblk = arrays["\(key)::qblk"] else {
            throw RwkvqError.missingBuffer(tensor: key, buffer: "qblk")
        }
        guard let qsqm = arrays["\(key)::qsqm"] else {
            throw RwkvqError.missingBuffer(tensor: key, buffer: "qsqm")
        }
        guard let ddm = arrays["\(key)::ddm"] else {
            throw RwkvqError.missingBuffer(tensor: key, buffer: "ddm")
        }
        return rwkvqDequantDense(qblk: qblk, qsqm: qsqm, ddm: ddm,
                                 outFeatures: info.outFeatures,
                                 inFeatures: info.inFeatures,
                                 superBlock: info.superBlock,
                                 xbits: info.xbits,
                                 dtype: dtype)
    }

    /// Суммарный размер сжатых буферов в байтах — чтобы можно было честно
    /// сказать, сколько база занимает в памяти.
    public var packedBytes: Int {
        arrays.values.reduce(0) { $0 + $1.size * $1.dtype.size }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Имена тензоров
// ───────────────────────────────────────────────────────────────────────

/// Соответствие имён x070 (как в SwiftRWKV) ↔ world (как в .rwkvq).
///
/// Официальные чекпоинты BlinkDL именуют проекции att.receptance/key/value/
/// output и ffn.key/value; SwiftRWKV — tmix.r_proj/k_proj/v_proj/o_proj и
/// cmix.key/value. Математика одна, различается только раскладка имён,
/// поэтому переименование — чистая таблица, а не преобразование.
public enum RwkvqNaming {

    static let tmix: [String: String] = [
        "r_proj": "receptance", "k_proj": "key",
        "v_proj": "value", "o_proj": "output",
    ]

    /// x070-имя → world-имя, или nil если соответствия нет.
    public static func worldKey(forX070 key: String) -> String? {
        if key == "emb.weight" || key == "head.weight" { return key }
        let parts = key.split(separator: ".").map(String.init)
        // blocks.<i>.<tmix|cmix>.<name>.weight
        guard parts.count == 5, parts[0] == "blocks", parts[4] == "weight",
              let layer = Int(parts[1]) else { return nil }
        switch parts[2] {
        case "tmix":
            guard let mapped = tmix[parts[3]] else { return nil }
            return "blocks.\(layer).att.\(mapped).weight"
        case "cmix":
            guard parts[3] == "key" || parts[3] == "value" else { return nil }
            return "blocks.\(layer).ffn.\(parts[3]).weight"
        default:
            return nil
        }
    }

    /// world-имя → x070-имя.
    public static func x070Key(forWorld key: String) -> String? {
        if key == "emb.weight" || key == "head.weight" { return key }
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count == 5, parts[0] == "blocks", parts[4] == "weight",
              let layer = Int(parts[1]) else { return nil }
        switch parts[2] {
        case "att":
            guard let mapped = tmix.first(where: { $0.value == parts[3] })?.key
            else { return nil }
            return "blocks.\(layer).tmix.\(mapped).weight"
        case "ffn":
            guard parts[3] == "key" || parts[3] == "value" else { return nil }
            return "blocks.\(layer).cmix.\(parts[3]).weight"
        default:
            return nil
        }
    }
}
