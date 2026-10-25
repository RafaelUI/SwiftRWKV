import Foundation

// ───────────────────────────────────────────────────────────────────────
//  BPE-токенизатор HuggingFace (`tokenizer.json`), байтовый уровень GPT-2.
//
//  Нужен моделям со своим словарём — ru60m (16k, `tokenizer_ru16k.json`) и
//  всему, что обучается с нуля не на словаре World. Поддерживается ровно та
//  конфигурация, которую можно проверить побитно против `tokenizers`
//  (эталон — Scripts/dump_bpe_reference.py, тест — BPETokenizerTests):
//
//    model          BPE, без dropout, без continuing_subword_prefix /
//                   end_of_word_suffix, byte_fallback = false
//    normalizer     null | NFC | NFD | NFKC | NFKD | Lowercase
//    pre_tokenizer  ByteLevel (add_prefix_space, use_regex — шаблон GPT-2)
//    decoder        ByteLevel | null
//    post_processor null
//    added_tokens   lstrip / rstrip / single_word = false
//
//  Всё прочее — отказ при загрузке (`BPETokenizerError.unsupported`), а не
//  тихое приближение: токенизатор, который кодирует «почти так же», даёт
//  модели чужие id без единой ошибки формы.
//
//  Порядок, как в `tokenizers`:
//    1. текст режется по добавленным токенам (самый длинный, самый левый);
//       токены с normalized = false ищутся в сыром тексте, с true — после
//       нормализации куска;
//    2. каждый кусок нормализуется;
//    3. ByteLevel: пробел в начало куска, если он не начинается с пробела
//       (add_prefix_space), разбиение шаблоном GPT-2, байты куска →
//       символы байтового уровня;
//    4. BPE по рангам слияний (наименьший ранг, при равенстве — левее).
//
//  Шаблон GPT-2 разобран вручную, без регулярных выражений, и сверен с ним
//  на эталоне: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// ───────────────────────────────────────────────────────────────────────

public enum BPETokenizerError: Error, CustomStringConvertible {
    case unreadable(String)
    case unsupported(String)
    case malformed(String)

    public var description: String {
        switch self {
        case .unreadable(let s): return "tokenizer.json не читается: \(s)"
        case .unsupported(let s):
            return "tokenizer.json: \(s) — не поддерживается (кодировать «почти так же» нельзя: id молча станут чужими)"
        case .malformed(let s): return "tokenizer.json повреждён: \(s)"
        }
    }
}

public final class BPETokenizer: RWKVTokenizer {

    enum Normalizer: String { case none, nfc = "NFC", nfd = "NFD", nfkc = "NFKC", nfkd = "NFKD", lowercase = "Lowercase" }

    struct Added {
        let bytes: [UInt8]
        let id: Int
        let special: Bool
        let normalized: Bool
    }

    /// Байты каждого id (для добавленных — их содержимое в UTF-8).
    private let idBytes: [[UInt8]]
    private let specialIds: Set<Int>
    /// Однобайтовые символы → id (байтовый уровень: 256 штук).
    private let byteId: [Int]
    /// (левый id, правый id) → (ранг, id слияния).
    private let merges: [UInt64: (rank: Int32, id: Int32)]
    private let unkId: Int?
    private let normalizer: Normalizer
    private let addPrefixSpace: Bool
    private let rawAdded: [Added]        // ищутся в сыром тексте
    private let normAdded: [Added]       // ищутся после нормализации
    /// Кэш BPE по куску (байты куска → id), как в `tokenizers`.
    private var cache: [[UInt8]: [Int]] = [:]
    private let cacheLimit = 100_000

    public var count: Int { idBytes.count }
    /// Id добавленных токенов, помеченных special (пропускаются в `decode`).
    public var specialTokenIds: Set<Int> { specialIds }

    // ── Загрузка ──────────────────────────────────────────────────────

    public convenience init(tokenizerURL url: URL) throws {
        let data: Data
        do { data = try Data(contentsOf: url) } catch { throw BPETokenizerError.unreadable("\(url.path): \(error)") }
        try self.init(json: data)
    }

    public init(json data: Data) throws {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw BPETokenizerError.malformed("не JSON-объект")
        }
        guard let model = root["model"] as? [String: Any] else { throw BPETokenizerError.malformed("нет model") }
        guard (model["type"] as? String) == "BPE" else {
            throw BPETokenizerError.unsupported("model.type = \(model["type"] ?? "нет")")
        }
        if let d = model["dropout"], !(d is NSNull) { throw BPETokenizerError.unsupported("dropout") }
        for k in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if let v = model[k], !(v is NSNull) { throw BPETokenizerError.unsupported(k) }
        }
        if (model["byte_fallback"] as? Bool) == true { throw BPETokenizerError.unsupported("byte_fallback") }
        if (model["ignore_merges"] as? Bool) == true { throw BPETokenizerError.unsupported("ignore_merges") }

        // нормализатор
        if let n = root["normalizer"] as? [String: Any] {
            guard let t = n["type"] as? String, let nn = Normalizer(rawValue: t) else {
                throw BPETokenizerError.unsupported("normalizer \(n["type"] ?? "?")")
            }
            normalizer = nn
        } else { normalizer = .none }

        // предтокенизатор: ровно ByteLevel
        guard let pre = root["pre_tokenizer"] as? [String: Any], (pre["type"] as? String) == "ByteLevel" else {
            throw BPETokenizerError.unsupported("pre_tokenizer \((root["pre_tokenizer"] as? [String: Any])?["type"] ?? "null")")
        }
        guard (pre["use_regex"] as? Bool) ?? true else { throw BPETokenizerError.unsupported("ByteLevel без use_regex") }
        addPrefixSpace = (pre["add_prefix_space"] as? Bool) ?? true
        if let dec = root["decoder"] as? [String: Any], (dec["type"] as? String) != "ByteLevel" {
            throw BPETokenizerError.unsupported("decoder \(dec["type"] ?? "?")")
        }
        if let pp = root["post_processor"], !(pp is NSNull) {
            throw BPETokenizerError.unsupported("post_processor")
        }

        // словарь: строки байтового уровня → байты
        guard let vocab = model["vocab"] as? [String: Any] else { throw BPETokenizerError.malformed("нет model.vocab") }
        let dec = Self.byteDecoder
        var maxId = -1
        var tokenId: [String: Int] = [:]
        tokenId.reserveCapacity(vocab.count)
        for (s, v) in vocab {
            guard let id = (v as? NSNumber)?.intValue, id >= 0 else { throw BPETokenizerError.malformed("id токена \(s)") }
            tokenId[s] = id; maxId = max(maxId, id)
        }
        var added: [Added] = []
        for a in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = a["content"] as? String, let id = (a["id"] as? NSNumber)?.intValue else {
                throw BPETokenizerError.malformed("added_tokens")
            }
            for k in ["lstrip", "rstrip", "single_word"] where (a[k] as? Bool) == true {
                throw BPETokenizerError.unsupported("added_token \(content): \(k)")
            }
            added.append(Added(bytes: Array(content.utf8), id: id, special: (a["special"] as? Bool) ?? false,
                               normalized: (a["normalized"] as? Bool) ?? false))
            maxId = max(maxId, id)
        }
        var bytesOf = [[UInt8]](repeating: [], count: maxId + 1)
        for (s, id) in tokenId {
            var b: [UInt8] = []
            b.reserveCapacity(s.unicodeScalars.count)
            for u in s.unicodeScalars {
                // В словаре байтового уровня встречаются и добавленные токены
                // (их содержимое — обычный текст); байты для них — из added.
                guard let x = dec[u] else { b = []; break }
                b.append(x)
            }
            bytesOf[id] = b
        }
        for a in added { bytesOf[a.id] = a.bytes }
        idBytes = bytesOf
        specialIds = Set(added.filter { $0.special }.map { $0.id })
        rawAdded = added.filter { !$0.normalized }.sorted { $0.bytes.count > $1.bytes.count }
        normAdded = added.filter { $0.normalized }.sorted { $0.bytes.count > $1.bytes.count }

        var bid = [Int](repeating: -1, count: 256)
        for b in 0 ..< 256 { if let id = tokenId[String(Self.byteEncoder[b])] { bid[b] = id } }
        byteId = bid
        if let u = model["unk_token"] as? String { unkId = tokenId[u] ?? added.first { $0.bytes == Array(u.utf8) }?.id } else { unkId = nil }

        // слияния: «a b» (старый вид) или ["a", "b"]
        guard let ms = model["merges"] as? [Any] else { throw BPETokenizerError.malformed("нет model.merges") }
        var mm: [UInt64: (rank: Int32, id: Int32)] = [:]
        mm.reserveCapacity(ms.count)
        for (rank, m) in ms.enumerated() {
            let pair: (String, String)
            if let s = m as? String {
                let parts = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
                guard parts.count == 2 else { throw BPETokenizerError.malformed("слияние «\(s)»") }
                pair = (String(parts[0]), String(parts[1]))
            } else if let a = m as? [String], a.count == 2 {
                pair = (a[0], a[1])
            } else { throw BPETokenizerError.malformed("слияние \(rank)") }
            guard let l = tokenId[pair.0], let r = tokenId[pair.1], let to = tokenId[pair.0 + pair.1] else {
                throw BPETokenizerError.malformed("слияние \(rank): \(pair.0) + \(pair.1) не в словаре")
            }
            let key = UInt64(UInt32(l)) << 32 | UInt64(UInt32(r))
            if mm[key] == nil { mm[key] = (Int32(rank), Int32(to)) }
        }
        merges = mm
    }

    // ── Кодирование ───────────────────────────────────────────────────

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        for seg in Self.split(Array(text.utf8), by: rawAdded) {
            switch seg {
            case .added(let id): out.append(id)
            case .text(let bytes):
                let normalized = normalize(bytes)
                for seg2 in Self.split(normalized, by: normAdded) {
                    switch seg2 {
                    case .added(let id): out.append(id)
                    case .text(let b): encodeSegment(b, into: &out)
                    }
                }
            }
        }
        return out
    }

    enum Segment { case text([UInt8]); case added(Int) }

    /// Самый левый, при равных — самый длинный добавленный токен.
    static func split(_ bytes: [UInt8], by added: [Added]) -> [Segment] {
        guard !added.isEmpty, !bytes.isEmpty else { return bytes.isEmpty ? [] : [.text(bytes)] }
        var out: [Segment] = []
        var start = 0, i = 0
        let n = bytes.count
        while i < n {
            var hit: Added? = nil
            for a in added where a.bytes.count <= n - i && a.bytes.first == bytes[i] {
                if bytes[i ..< i + a.bytes.count].elementsEqual(a.bytes) { hit = a; break }   // added — по убыванию длины
            }
            if let a = hit {
                if start < i { out.append(.text(Array(bytes[start ..< i]))) }
                out.append(.added(a.id))
                i += a.bytes.count; start = i
            } else { i += 1 }
        }
        if start < n { out.append(.text(Array(bytes[start ..< n]))) }
        return out
    }

    func normalize(_ bytes: [UInt8]) -> [UInt8] {
        if normalizer == .none { return bytes }
        // Быстрый путь: ASCII не меняется ни одной формой Unicode-нормализации.
        if normalizer != .lowercase, bytes.allSatisfy({ $0 < 0x80 }) { return bytes }
        let s = String(decoding: bytes, as: UTF8.self)
        let r: String
        switch normalizer {
        case .none: r = s
        case .nfc: r = s.precomposedStringWithCanonicalMapping
        case .nfd: r = s.decomposedStringWithCanonicalMapping
        case .nfkc: r = s.precomposedStringWithCompatibilityMapping
        case .nfkd: r = s.decomposedStringWithCompatibilityMapping
        case .lowercase: r = s.lowercased()
        }
        return Array(r.utf8)
    }

    private func encodeSegment(_ bytes: [UInt8], into out: inout [Int]) {
        guard !bytes.isEmpty else { return }
        var b = bytes
        if addPrefixSpace, b[0] != 0x20 { b.insert(0x20, at: 0) }
        for piece in Self.pretokenize(b) {
            let key = Array(b[piece])
            if let hit = cache[key] { out.append(contentsOf: hit); continue }
            let ids = bpe(key)
            if cache.count < cacheLimit { cache[key] = ids }
            out.append(contentsOf: ids)
        }
    }

    /// BPE одного куска: наименьший ранг слияния, при равных — левее.
    func bpe(_ bytes: [UInt8]) -> [Int] {
        var syms: [Int] = []
        syms.reserveCapacity(bytes.count)
        for x in bytes {
            let id = byteId[Int(x)]
            if id >= 0 { syms.append(id) } else if let u = unkId { syms.append(u) }
        }
        while syms.count > 1 {
            var best: (rank: Int32, id: Int32, pos: Int)? = nil
            for p in 0 ..< syms.count - 1 {
                let key = UInt64(UInt32(syms[p])) << 32 | UInt64(UInt32(syms[p + 1]))
                if let m = merges[key], best == nil || m.rank < best!.rank { best = (m.rank, m.id, p) }
            }
            guard let m = best else { break }
            syms[m.pos] = Int(m.id)
            syms.remove(at: m.pos + 1)
        }
        return syms
    }

    // ── Шаблон GPT-2 ──────────────────────────────────────────────────

    enum Cls { case space, letter, number, other }

    /// Класс скаляра в смысле шаблона: \s (White_Space), \p{L}, \p{N}, прочее.
    static func cls(_ u: Unicode.Scalar) -> Cls {
        let v = u.value
        if v < 0x80 {
            if v == 0x20 || (v >= 0x09 && v <= 0x0D) { return .space }
            if (v >= 0x41 && v <= 0x5A) || (v >= 0x61 && v <= 0x7A) { return .letter }
            if v >= 0x30 && v <= 0x39 { return .number }
            return .other
        }
        let p = u.properties
        if p.isWhitespace { return .space }
        switch p.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return .letter
        case .decimalNumber, .letterNumber, .otherNumber: return .number
        default: return .other
        }
    }

    /// Куски по шаблону GPT-2 — диапазоны байтов UTF-8 строки.
    static func pretokenize(_ bytes: [UInt8]) -> [Range<Int>] {
        // скаляры с их байтовыми границами
        var scal: [(u: Unicode.Scalar, c: Cls, at: Int)] = []
        scal.reserveCapacity(bytes.count)
        var it = String(decoding: bytes, as: UTF8.self).unicodeScalars.makeIterator()
        var at = 0
        while let u = it.next() {
            scal.append((u, cls(u), at))
            at += UTF8.width(u)
        }
        let n = scal.count
        func byteAt(_ k: Int) -> Int { k < n ? scal[k].at : at }
        var out: [Range<Int>] = []
        var i = 0
        while i < n {
            let c = scal[i]
            // 's 't 're 've 'm 'll 'd
            if c.u == "'" && i + 1 < n {
                let a = scal[i + 1].u
                let b = i + 2 < n ? scal[i + 2].u : nil
                var len = 0
                if a == "s" || a == "t" || a == "m" || a == "d" { len = 2 }
                else if (a == "r" && b == "e") || (a == "v" && b == "e") || (a == "l" && b == "l") { len = 3 }
                if len > 0 { out.append(byteAt(i) ..< byteAt(i + len)); i += len; continue }
            }
            // ` ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+`: необязательный пробел
            // (только U+0020 и только перед непробельным), затем серия одного
            // класса. Классы не пересекаются, поэтому порядок альтернатив
            // сводится к классу первого символа после пробела.
            let k = (c.u == " " && i + 1 < n && scal[i + 1].c != .space) ? i + 1 : i
            let x = scal[k].c
            if x != .space {
                var j = k + 1
                while j < n && scal[j].c == x { j += 1 }
                out.append(byteAt(i) ..< byteAt(j)); i = j; continue
            }
            // пробельные: \s+(?!\S) | \s+
            var j = i
            while j < n && scal[j].c == .space { j += 1 }
            if j == n || j - i == 1 {
                out.append(byteAt(i) ..< byteAt(j)); i = j
            } else {
                out.append(byteAt(i) ..< byteAt(j - 1)); i = j - 1
            }
        }
        return out
    }

    // ── Декодирование ─────────────────────────────────────────────────

    /// Текст id; специальные токены пропускаются, как в `tokenizers` по
    /// умолчанию. Неполный UTF-8 — заменой (U+FFFD), как from_utf8_lossy.
    public func decode(_ tokens: [Int]) -> String { decode(tokens, skipSpecial: true) }

    public func decode(_ tokens: [Int], skipSpecial: Bool) -> String {
        var b: [UInt8] = []
        for t in tokens where t >= 0 && t < idBytes.count {
            if skipSpecial && specialIds.contains(t) { continue }
            b.append(contentsOf: idBytes[t])
        }
        return String(decoding: b, as: UTF8.self)
    }

    /// Байты токена для потокового вывода. У специальных — пусто: поток
    /// генерации обязан совпадать с `decode`, а тот их пропускает.
    public func rawBytes(_ id: Int) -> [UInt8]? {
        guard id >= 0 && id < idBytes.count else { return nil }
        return specialIds.contains(id) ? [] : idBytes[id]
    }

    // ── Байтовый уровень GPT-2: байт ↔ символ ─────────────────────────

    static let byteEncoder: [Character] = {
        var bs: [Int] = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b); cs.append(256 + n); n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(Unicode.Scalar(UInt32(c))!) }
        return table
    }()

    static let byteDecoder: [Unicode.Scalar: UInt8] = {
        var d: [Unicode.Scalar: UInt8] = [:]
        for b in 0 ..< 256 { d[byteEncoder[b].unicodeScalars.first!] = UInt8(b) }
        return d
    }()
}
