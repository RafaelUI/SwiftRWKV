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
    /// Разбор идёт на уровне БАЙТОВ, а не символов Swift.
    ///
    /// Это не оптимизация, а условие правильности. Swift.String итерируется
    /// графемными кластерами, и в 57 строках словаря repr — это кавычка,
    /// за которой идёт комбинирующий знак (ударение, умляут). Кавычка и знак
    /// сливаются в ОДИН Character, после чего `hasPrefix("'")` ложно, кавычки
    /// не снимаются, и токен уходит в дерево вместе с ними — то есть не
    /// находится никогда. Байты такой склейки не знают.
    ///
    /// Симптом, по которому это нашлось: текст с U+0301 токенизировался
    /// иначе, чем в Python — на один токен длиннее и с другими id.
    public init?(vocabURL: URL) {
        guard let data = try? Data(contentsOf: vocabURL) else { return nil }
        let all = [UInt8](data)

        var map: [Int: [UInt8]] = [:]
        var start = 0
        while start <= all.count {
            var end = start
            while end < all.count, all[end] != 0x0A { end += 1 }
            defer { start = end + 1 }
            if end == start { if end >= all.count { break }; continue }

            let line = all[start ..< end]
            // "idx repr length" — idx и length числа, repr может содержать пробелы.
            guard let firstSpace = line.firstIndex(of: 0x20),
                  let lastSpace = line.lastIndex(of: 0x20),
                  firstSpace < lastSpace else { if end >= all.count { break }; continue }

            guard let idx = Self.parseInt(line[line.startIndex ..< firstSpace]),
                  Self.parseInt(line[(lastSpace + 1)...]) != nil else {
                if end >= all.count { break }; continue
            }
            let repr = Array(line[(firstSpace + 1) ..< lastSpace])
            guard let bytes = Self.parseRepr(repr) else {
                if end >= all.count { break }; continue
            }

            map[idx] = bytes
            var node = root
            for byte in bytes {
                if node.children[byte] == nil { node.children[byte] = TrieNode() }
                node = node.children[byte]!
            }
            node.value = (bytes: bytes, id: idx)
            if end >= all.count { break }
        }
        self.idx2token = map
    }

    private static func parseInt(_ bytes: ArraySlice<UInt8>) -> Int? {
        var v = 0
        var any = false
        for b in bytes {
            guard b >= 0x30, b <= 0x39 else { return nil }
            v = v * 10 + Int(b - 0x30)
            any = true
        }
        return any ? v : nil
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

    /// Разбор Python-repr токена в байты.
    ///
    /// Ключевое различие, которое обязано учитываться: `\xNN` означает РАЗНОЕ
    /// в двух видах repr.
    ///   • `b'\xa0'` — литерал bytes, это ровно байт 0xA0;
    ///   • `'\xa0'`  — литерал str, это код-поинт U+00A0, а в UTF-8 он
    ///                 занимает ДВА байта: C2 A0.
    /// Трактовать второе как сырой байт — значит поставить в дерево путь,
    /// которого во входном тексте не бывает, и одновременно лишить дерево
    /// настоящего пути. В словаре таких токенов 15, и среди них U+00A0
    /// (неразрывный пробел) — символ, который в реальных текстах встречается
    /// постоянно.
    ///
    /// Заявленная в третьей колонке длина здесь не используется как источник
    /// истины, но именно она позволяет это проверить: см. тест сверки всей
    /// таблицы с эталоном из ast.literal_eval.
    private static func parseRepr(_ repr: [UInt8]) -> [UInt8]? {
        var body = repr[...]
        var isBytesLiteral = false

        if body.count >= 3, body.first == 0x62,                        // 'b'
           let second = body.dropFirst().first, second == 0x27 || second == 0x22 {
            isBytesLiteral = true
            body = body.dropFirst(2).dropLast()
        } else if body.count >= 2, let f = body.first, f == 0x27 || f == 0x22 {
            body = body.dropFirst().dropLast()
        }

        var bytes: [UInt8] = []
        var i = body.startIndex
        while i < body.endIndex {
            guard body[i] == 0x5C else {                                // '\'
                bytes.append(body[i]); i += 1; continue
            }
            let next = i + 1
            guard next < body.endIndex else { return nil }
            switch body[next] {
            case 0x78:                                                  // 'x'
                guard next + 2 < body.endIndex,
                      let hi = hexValue(body[next + 1]),
                      let lo = hexValue(body[next + 2]) else { return nil }
                let code = hi << 4 | lo
                if isBytesLiteral {
                    bytes.append(code)
                } else {
                    appendUTF8(codePoint: UInt32(code), to: &bytes)
                }
                i = next + 3
            case 0x75, 0x55:                                            // \uNNNN, \UNNNNNNNN
                // Только в str-repr: у bytes-литерала Python такого escape нет,
                // там \u — это буквально обратный слэш и буква u.
                guard !isBytesLiteral else {
                    bytes.append(0x5C); bytes.append(body[next]); i = next + 1
                    break
                }
                let digits = body[next] == 0x75 ? 4 : 8
                guard next + digits < body.endIndex else { return nil }
                var code: UInt32 = 0
                for d in 1 ... digits {
                    guard let v = hexValue(body[next + d]) else { return nil }
                    code = code << 4 | UInt32(v)
                }
                appendUTF8(codePoint: code, to: &bytes)
                i = next + digits + 1
            case 0x6E: bytes.append(0x0A); i = next + 1                 // \n
            case 0x72: bytes.append(0x0D); i = next + 1                 // \r
            case 0x74: bytes.append(0x09); i = next + 1                 // \t
            case 0x5C: bytes.append(0x5C); i = next + 1                 // \\
            case 0x27: bytes.append(0x27); i = next + 1                 // \'
            case 0x22: bytes.append(0x22); i = next + 1                 // \"
            case 0x30: bytes.append(0x00); i = next + 1                 // \0
            default:
                // Неизвестный escape: Python оставляет обратный слэш на месте.
                bytes.append(0x5C)
                bytes.append(body[next])
                i = next + 1
            }
        }
        return bytes
    }

    private static func hexValue(_ b: UInt8) -> UInt8? {
        switch b {
        case 0x30 ... 0x39: return b - 0x30
        case 0x61 ... 0x66: return b - 0x61 + 10
        case 0x41 ... 0x46: return b - 0x41 + 10
        default: return nil
        }
    }

    private static func appendUTF8(codePoint: UInt32, to bytes: inout [UInt8]) {
        if codePoint < 0x80 {
            bytes.append(UInt8(codePoint))
        } else if codePoint < 0x800 {
            bytes.append(UInt8(0xC0 | (codePoint >> 6)))
            bytes.append(UInt8(0x80 | (codePoint & 0x3F)))
        } else if codePoint < 0x10000 {
            bytes.append(UInt8(0xE0 | (codePoint >> 12)))
            bytes.append(UInt8(0x80 | ((codePoint >> 6) & 0x3F)))
            bytes.append(UInt8(0x80 | (codePoint & 0x3F)))
        } else {
            bytes.append(UInt8(0xF0 | (codePoint >> 18)))
            bytes.append(UInt8(0x80 | ((codePoint >> 12) & 0x3F)))
            bytes.append(UInt8(0x80 | ((codePoint >> 6) & 0x3F)))
            bytes.append(UInt8(0x80 | (codePoint & 0x3F)))
        }
    }
}
