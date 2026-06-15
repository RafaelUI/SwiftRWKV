import Foundation

// ───────────────────────────────────────────────────────────────────────
//  RWKV World Tokenizer — байтовый TRIE-токенизатор (порт world_tokenizer.py).
//  Используется ТОЛЬКО с официальными RWKV-7 World весами (vocab 65536).
//
//  Формат vocab-файла (rwkv_vocab_v20230424.txt): по одной записи в строке
//      "<id> <python-bytes-repr> <length>"
//  где repr — питоновский литерал байтов, напр. b'\\xd0\\x9f' или 'hello'.
//
//  Без внешних зависимостей (только Foundation). Greedy longest-match по байтам.
// ───────────────────────────────────────────────────────────────────────

private final class TrieNode {
    var children: [UInt8: TrieNode] = [:]
    var value: (bytes: [UInt8], id: Int)? = nil
}

/// Байтовый TRIE-токенизатор RWKV World.
public final class WorldTokenizer {
    private let root = TrieNode()
    private let idx2token: [Int: [UInt8]]

    /// Загружает токенизатор из текстового vocab-файла.
    /// Возвращает `nil`, если файл не читается.
    public init?(vocabURL: URL) {
        guard let content = (try? String(contentsOf: vocabURL, encoding: .utf8))
                ?? (try? String(contentsOf: vocabURL, encoding: .isoLatin1))
        else { return nil }

        var map: [Int: [UInt8]] = [:]
        for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
            let s = String(line)
            // "idx repr length" — idx/length числа, repr может содержать пробелы.
            guard let firstSpace = s.firstIndex(of: " "),
                  let lastSpace = s.lastIndex(of: " "),
                  firstSpace < lastSpace else { continue }
            let idxStr = String(s[s.startIndex ..< firstSpace])
            let lenStr = String(s[s.index(after: lastSpace)...])
            guard let idx = Int(idxStr), Int(lenStr) != nil else { continue }
            let repr = String(s[s.index(after: firstSpace) ..< lastSpace])
            guard let bytes = Self.parseRepr(repr) else { continue }

            map[idx] = bytes
            var node = root
            for byte in bytes {
                if node.children[byte] == nil { node.children[byte] = TrieNode() }
                node = node.children[byte]!
            }
            node.value = (bytes: bytes, id: idx)
        }
        self.idx2token = map
    }

    /// Размер словаря (число загруженных токенов).
    public var count: Int { idx2token.count }

    /// Кодирует строку в id токенов (greedy longest-match по байтам).
    public func encode(_ text: String) -> [Int] {
        let src = Array(text.utf8)
        var idx = 0
        var tokens: [Int] = []

        while idx < src.count {
            var node = root
            var lastMatch: (endIdx: Int, id: Int)? = nil
            var i = idx
            while i < src.count {
                guard let next = node.children[src[i]] else { break }
                node = next; i += 1
                if let val = node.value { lastMatch = (endIdx: i, id: val.id) }
            }
            if let match = lastMatch {
                tokens.append(match.id); idx = match.endIdx
            } else {
                idx += 1   // fallback: пропуск байта (не должно происходить с полным vocab)
            }
        }
        return tokens
    }

    /// Декодирует последовательность токенов в строку.
    public func decode(_ tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        for tok in tokens { if let b = idx2token[tok] { bytes.append(contentsOf: b) } }
        return String(bytes: bytes, encoding: .utf8)
            ?? String(bytes: bytes, encoding: .isoLatin1) ?? "?"
    }

    /// Сырые байты токена — для byte-safe стриминга многобайтовых символов.
    public func rawBytes(_ id: Int) -> [UInt8]? { idx2token[id] }

    // Парсинг Python bytes-repr → [UInt8].
    private static func parseRepr(_ repr: String) -> [UInt8]? {
        var s = repr
        if s.hasPrefix("b'") || s.hasPrefix("b\"") { s = String(s.dropFirst(2).dropLast()) }
        else if s.hasPrefix("'") || s.hasPrefix("\"") { s = String(s.dropFirst().dropLast()) }

        var bytes: [UInt8] = []
        var i = s.startIndex
        while i < s.endIndex {
            if s[i] == "\\" {
                let next = s.index(after: i)
                guard next < s.endIndex else { return nil }
                switch s[next] {
                case "x":
                    guard let h1 = s.index(next, offsetBy: 1, limitedBy: s.endIndex),
                          let h2 = s.index(next, offsetBy: 2, limitedBy: s.endIndex),
                          h2 < s.endIndex else { return nil }
                    guard let byte = UInt8(String(s[h1...h2]), radix: 16) else { return nil }
                    bytes.append(byte); i = s.index(after: h2)
                case "n":  bytes.append(0x0A); i = s.index(after: next)
                case "r":  bytes.append(0x0D); i = s.index(after: next)
                case "t":  bytes.append(0x09); i = s.index(after: next)
                case "\\": bytes.append(0x5C); i = s.index(after: next)
                case "'":  bytes.append(0x27); i = s.index(after: next)
                case "\"": bytes.append(0x22); i = s.index(after: next)
                case "0":  bytes.append(0x00); i = s.index(after: next)
                default:
                    bytes.append(contentsOf: String(s[next]).utf8)
                    i = s.index(after: next)
                }
            } else {
                bytes.append(contentsOf: String(s[i]).utf8)
                i = s.index(after: i)
            }
        }
        return bytes
    }
}
