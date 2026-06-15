import Foundation

/// ByteLevel-BPE токенизатор (HF tokenizers формат, как GPT-2).
/// Читает tokenizer.json: vocab (piece→id) + merges (ранги).
/// Без unk, без byte_fallback, без auto BOS/EOS (post_processor=null).
public final class BPETokenizer {
    private let vocab: [String: Int]
    private let mergeRank: [String: Int]          // "A B" -> rank
    private let byteToUnicode: [UInt8: Character]  // GPT-2 byte map
    private let regex: NSRegularExpression
    private let whitespaceMode: Bool      // WhitespaceSplit (NFKC, char-level) vs ByteLevel
    private let nfkc: Bool
    private let unkId: Int?

    // GPT-2 byte→unicode таблица (обратимое отображение 256 байт в видимые символы)
    private static func makeByteToUnicode() -> [UInt8: Character] {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0...255 where !bs.contains(b) {
            bs.append(b); cs.append(256 + n); n += 1
        }
        var map: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) {
            map[UInt8(b)] = Character(UnicodeScalar(c)!)
        }
        return map
    }

    public init?(tokenizerJSONURL url: URL) {
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int],
              let mergesRaw = model["merges"] as? [Any]
        else { return nil }

        self.vocab = vocab

        // merges может быть ["A B", ...] или [["A","B"], ...]
        var ranks: [String: Int] = [:]
        for (i, m) in mergesRaw.enumerated() {
            if let s = m as? String {
                ranks[s] = i
            } else if let pair = m as? [String], pair.count == 2 {
                ranks["\(pair[0]) \(pair[1])"] = i
            }
        }
        self.mergeRank = ranks
        self.byteToUnicode = Self.makeByteToUnicode()

        // GPT-2 ByteLevel regex (ICU-совместимый вариант)
        let pat = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        guard let re = try? NSRegularExpression(pattern: pat) else { return nil }
        self.regex = re

        let preType = (root["pre_tokenizer"] as? [String: Any])?["type"] as? String
        self.whitespaceMode = (preType == "WhitespaceSplit")
        let normType = (root["normalizer"] as? [String: Any])?["type"] as? String
        self.nfkc = (normType == "NFKC")
        if let unkTok = model["unk_token"] as? String { self.unkId = vocab[unkTok] } else { self.unkId = nil }
    }

    // BPE-слияние одного pre-token (строки в byte-level кодировке)
    private func bpe(_ token: String) -> [String] {
        var word = token.map { String($0) }
        if word.count < 2 { return word }

        while true {
            // ищем пару с минимальным рангом
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0 ..< (word.count - 1) {
                if let r = mergeRank["\(word[i]) \(word[i+1])"], r < bestRank {
                    bestRank = r; bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            // сливаем пару на позиции bestIdx
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i == bestIdx {
                    merged.append(word[i] + word[i+1]); i += 2
                } else {
                    merged.append(word[i]); i += 1
                }
            }
            word = merged
        }
        return word
    }

    public func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        let input = nfkc ? text.precomposedStringWithCompatibilityMapping : text

        if whitespaceMode {
            // NFKC → split по пробелам → char-level BPE → [UNK]
            var ids: [Int] = []
            for word in input.split(whereSeparator: { $0.isWhitespace }) {
                for sub in bpe(String(word)) {
                    if let id = vocab[sub] { ids.append(id) }
                    else if let u = unkId { ids.append(u) }
                }
            }
            return ids
        }

        // ByteLevel (GPT-2)
        var ids: [Int] = []
        let ns = input as NSString
        let matches = regex.matches(in: input, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let piece = ns.substring(with: m.range)
            var encoded = ""
            for byte in Array(piece.utf8) { encoded.append(byteToUnicode[byte]!) }
            for sub in bpe(encoded) {
                if let id = vocab[sub] { ids.append(id) }
            }
        }
        return ids
    }
}
