//
//  BPETokenizerTests.swift
//  BPE-токенизатор HuggingFace против `tokenizers` (25.10).
//
//  Эталон — Scripts/dump_bpe_reference.py: крайние случаи (специальные
//  токены, пробелы, NFKC, эмодзи, апострофы) и 1500 текстов корпуса,
//  ~700 тыс. токенов. Утверждается совпадение id ДО ЕДИНОГО и декодирования
//  до символа. Пропускается без фикстур.
//
import XCTest
import MLX
@testable import RWKVGen

final class BPETokenizerTests: XCTestCase {

    struct Ref {
        let texts: [String]
        let ids: [[Int]]
        let decoded: [String]
        let decodedSpecial: [String]
        let tokenText: [String]
        let vocabSize: Int
    }

    static func paths() -> (tok: String, ref: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        return (env["RWKV_BPE_TOKENIZER"] ?? home.appendingPathComponent(
                    "Develop/SwiftRWKV/.testdata/tokenizer_ru16k.json").path,
                env["RWKV_BPE_REFERENCE"] ?? home.appendingPathComponent(
                    "Develop/SwiftRWKV/.testdata/bpe_ru16k_ref.json").path)
    }

    static var cached: (BPETokenizer, Ref)?

    func load() throws -> (BPETokenizer, Ref) {
        if let c = Self.cached { return c }
        let (tp, rp) = Self.paths()
        for p in [tp, rp] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет фикстуры \(p)")
        }
        let tok = try BPETokenizer(tokenizerURL: URL(fileURLWithPath: tp))
        let d = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: rp))) as! [String: Any]
        let ref = Ref(texts: d["texts"] as! [String],
                      ids: (d["ids"] as! [[NSNumber]]).map { $0.map(\.intValue) },
                      decoded: d["decoded"] as! [String],
                      decodedSpecial: d["decoded_special"] as! [String],
                      tokenText: d["token_text"] as! [String],
                      vocabSize: (d["vocab_size"] as! NSNumber).intValue)
        Self.cached = (tok, ref)
        return (tok, ref)
    }

    /// id всех текстов — те же, что у `tokenizers`, до единого.
    func testEncodeMatchesHuggingFace() throws {
        let (tok, ref) = try load()
        XCTAssertGreaterThan(ref.texts.count, 1000)
        var bad: [Int] = []
        var tokens = 0
        for (i, t) in ref.texts.enumerated() {
            let got = tok.encode(t)
            tokens += got.count
            if got != ref.ids[i] { bad.append(i) }
        }
        XCTAssertEqual(tokens, ref.ids.reduce(0) { $0 + $1.count })
        if let i = bad.first {
            let got = tok.encode(ref.texts[i])
            let k = (0 ..< min(got.count, ref.ids[i].count)).first { got[$0] != ref.ids[i][$0] } ?? min(got.count, ref.ids[i].count)
            XCTFail("расходятся \(bad.count) текстов из \(ref.texts.count); первый #\(i) «\(ref.texts[i].prefix(60))»: "
                    + "с позиции \(k) \(Array(got[k...].prefix(6))) против \(Array(ref.ids[i][k...].prefix(6)))")
        }
    }

    /// Повторное кодирование (из кэша кусков) даёт то же.
    func testCacheGivesTheSame() throws {
        let (_, ref) = try load()
        let fresh = try BPETokenizer(tokenizerURL: URL(fileURLWithPath: Self.paths().tok))
        for t in ref.texts.prefix(200) {
            let a = fresh.encode(t)
            XCTAssertEqual(fresh.encode(t), a)
        }
    }

    /// Декодирование: со специальными и без — как в `tokenizers`.
    func testDecodeMatchesHuggingFace() throws {
        let (tok, ref) = try load()
        var bad = 0, badSpecial = 0
        for i in ref.texts.indices {
            if tok.decode(ref.ids[i]) != ref.decoded[i] { bad += 1 }
            if tok.decode(ref.ids[i], skipSpecial: false) != ref.decodedSpecial[i] { badSpecial += 1 }
        }
        XCTAssertEqual(bad, 0, "декодирование без специальных расходится")
        XCTAssertEqual(badSpecial, 0, "декодирование со специальными расходится")
        // различающее: специальные действительно пропускаются
        XCTAssertNotEqual(tok.decode([7295, 3, 4936]), tok.decode([7295, 3, 4936], skipSpecial: false))
    }

    /// Байты каждого токена словаря (где они — целый UTF-8).
    func testRawBytesOfEveryToken() throws {
        let (tok, ref) = try load()
        XCTAssertEqual(tok.count, ref.vocabSize)
        var checked = 0
        for id in 0 ..< ref.vocabSize where !ref.tokenText[id].contains("\u{FFFD}") {
            // специальные — без байтов, как в decode: поток генерации равен decode
            let want = tok.specialTokenIds.contains(id) ? "" : ref.tokenText[id]
            XCTAssertEqual(String(decoding: tok.rawBytes(id)!, as: UTF8.self), want, "id \(id)")
            checked += 1
        }
        XCTAssertEqual(tok.specialTokenIds, [0, 1, 2, 3, 4])
        XCTAssertEqual(tok.rawBytes(3), [])
        XCTAssertGreaterThan(checked, ref.vocabSize / 2)
        XCTAssertNil(tok.rawBytes(ref.vocabSize))
        XCTAssertNil(tok.rawBytes(-1))
    }

    /// Шаблон GPT-2 на крайних случаях — куски, а не только итоговые id.
    func testPretokenizePieces() {
        func pieces(_ s: String) -> [String] {
            let b = Array(s.utf8)
            return BPETokenizer.pretokenize(b).map { String(decoding: b[$0], as: UTF8.self) }
        }
        XCTAssertEqual(pieces(" Привет, мир!"), [" Привет", ",", " мир", "!"])
        XCTAssertEqual(pieces("  два  пробела\n\nи\tтаб"), [" ", " два", " ", " пробела", "\n", "\n", "и", "\t", "таб"])
        XCTAssertEqual(pieces(" don't I'll"), [" don", "'t", " I", "'ll"])
        XCTAssertEqual(pieces(" 123 4.5"), [" 123", " 4", ".", "5"])
        XCTAssertEqual(pieces("end   "), ["end", "   "])
        XCTAssertEqual(pieces("a !'s"), ["a", " !'", "s"])
        XCTAssertEqual(pieces(" x\u{00A0}y"), [" x", "\u{00A0}", "y"])
        // \s — по свойству White_Space и вне ASCII: два неразрывных пробела
        // режутся как пробелы (с откатом перед буквой), а не склеиваются в
        // «прочее»; после NFKC их не бывает, поэтому — напрямую.
        XCTAssertEqual(pieces("a\u{00A0}\u{00A0}b"), ["a", "\u{00A0}", "\u{00A0}", "b"])
        XCTAssertEqual(pieces("!\u{2028}"), ["!", "\u{2028}"])
    }

    /// Неподдерживаемое — отказ при загрузке, а не приближение.
    func testRefusesWhatItCannotReproduce() throws {
        let (tp, _) = Self.paths()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: tp), "нет фикстуры")
        let base = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: tp))) as! [String: Any]
        func refuses(_ edit: (inout [String: Any]) -> Void, _ what: String) {
            var j = base
            edit(&j)
            let data = try! JSONSerialization.data(withJSONObject: j)
            XCTAssertThrowsError(try BPETokenizer(json: data), what)
        }
        refuses({ $0["post_processor"] = ["type": "TemplateProcessing"] }, "post_processor")
        refuses({ $0["normalizer"] = ["type": "Replace"] }, "normalizer Replace")
        refuses({ $0["pre_tokenizer"] = ["type": "Whitespace"] }, "pre_tokenizer Whitespace")
        refuses({ var m = $0["model"] as! [String: Any]; m["type"] = "WordPiece"; $0["model"] = m }, "WordPiece")
        refuses({ var m = $0["model"] as! [String: Any]; m["dropout"] = 0.1; $0["model"] = m }, "dropout")
        refuses({ var m = $0["model"] as! [String: Any]; m["byte_fallback"] = true; $0["model"] = m }, "byte_fallback")
        refuses({ var a = $0["added_tokens"] as! [[String: Any]]; a[0]["lstrip"] = true; $0["added_tokens"] = a }, "lstrip")
        // и исходный файл загружается
        XCTAssertNoThrow(try BPETokenizer(json: JSONSerialization.data(withJSONObject: base)))
    }

    /// Добавленные токены: при пересечении берётся самый длинный, и они
    /// режут текст до нормализации (в словаре ru16k специальные не
    /// пересекаются, поэтому — свой токен поверх него).
    func testAddedTokensLongestFirst() throws {
        let (tp, _) = Self.paths()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: tp), "нет фикстуры")
        var j = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: tp))) as! [String: Any]
        var added = j["added_tokens"] as! [[String: Any]]
        added.append(["id": 16000, "content": "<eos>и", "single_word": false, "lstrip": false, "rstrip": false,
                      "normalized": false, "special": false])
        j["added_tokens"] = added
        let tok = try BPETokenizer(json: JSONSerialization.data(withJSONObject: j))
        XCTAssertEqual(tok.count, 16001)
        XCTAssertEqual(tok.encode("<eos>и"), [16000])
        XCTAssertEqual(tok.encode("<eos>"), [3])
        XCTAssertEqual(tok.encode("а<eos>ив"), tok.encode("а") + [16000] + tok.encode("в"))
        XCTAssertEqual(tok.decode([16000]), "<eos>и", "не специальный — в декодировании остаётся")
    }

    /// Генерация с BPE: протокол проходит через цикл, поток кусков равен
    /// `decode` выданных токенов (в том числе специальных — их нет в обоих).
    func testGenerationStreamEqualsDecode() throws {
        let (tok, _) = try load()
        let (bb, _) = TinyBackbone.make(nLayer: 2, vocab: tok.count, seed: 7)
        var streamed = ""
        var state = RWKVState(cfg: bb.cfg)
        let r = bb.generate(prompt: tok.encode("Привет, мир"), config: GenerationConfig(maxTokens: 24, pipelined: false),
                            tokenizer: tok, state: &state, onToken: { chunk, _ in streamed += chunk; return true })
        XCTAssertEqual(r.tokens.count, 24)
        XCTAssertEqual(r.text, tok.decode(r.tokens))
        XCTAssertEqual(streamed, r.text)
        // принудительно специальный токен в выдаче: в тексте его нет
        XCTAssertEqual(tok.decode(r.tokens + [3]), r.text)
    }

    /// `loadTokenizer` выбирает реализацию по файлу.
    func testLoadTokenizerByFile() throws {
        let (tp, _) = Self.paths()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: tp), "нет фикстуры")
        XCTAssertTrue(try loadTokenizer(URL(fileURLWithPath: tp)) is BPETokenizer)
        let home = FileManager.default.homeDirectoryForCurrentUser
        let vp = home.appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt")
        if FileManager.default.fileExists(atPath: vp.path) {
            XCTAssertTrue(try loadTokenizer(vp) is WorldTokenizer)
        }
    }
}
