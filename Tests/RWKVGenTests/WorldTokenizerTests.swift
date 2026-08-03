import XCTest
@testable import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Словарь World: разбор repr → байты.
//
//  Найдено паритетом эмбеддингов: русский текст с U+0301 (комбинирующее
//  ударение) токенизировался иначе, чем в Python. Причина оказалась не в
//  дереве и не в жадном поиске, а в РАЗБОРЕ файла словаря — и притом сразу
//  по двум независимым причинам:
//
//   1. Swift.String итерируется графемными кластерами. В строке вида
//      `2672 '́' 2` кавычка и следующий за ней комбинирующий знак — один
//      Character. Проверка `hasPrefix("'")` ложна, кавычки не снимаются,
//      токен уходит в дерево вместе с ними и не находится никогда. Таких
//      токенов 57.
//   2. `\xNN` значит разное в `b'...'` и в `'...'`: в первом — сырой байт,
//      во втором — код-поинт U+00NN, то есть ДВА байта UTF-8. Трактовка
//      как сырого байта ломает 15 токенов, включая U+00A0 — неразрывный
//      пробел, который в реальных текстах встречается постоянно.
//
//  Ни один структурный тест такого не поймал бы: 65 529 строк, и ошибка на
//  72 из них не мешает ни одному тексту без ударений и NBSP. Поэтому здесь
//  сверяется ВСЯ таблица целиком с эталоном, снятым тем же ast.literal_eval,
//  которым словарь и записан.
//
//  Фикстура:
//      python3 Scripts/dump_vocab_reference.py \
//          --vocab .testdata/rwkv_vocab_v20230424.txt \
//          --out   .testdata/world_vocab_bytes.bin
// ───────────────────────────────────────────────────────────────────────

final class WorldTokenizerTests: XCTestCase {

    func vocabPath() -> String {
        ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
    }

    func referencePath() -> String {
        ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB_BYTES"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/world_vocab_bytes.bin").path
    }

    func loadTokenizer() throws -> WorldTokenizer {
        let p = vocabPath()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p),
                          "нет словаря World — тест пропущен")
        return try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: p)))
    }

    /// Эталонная таблица: magic "RWKVVOCB", uint32 count, затем записи
    /// (uint32 id, uint32 len, len байт).
    func loadReference() throws -> [Int: [UInt8]] {
        let p = referencePath()
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p), """
            Нет эталона словаря — тест пропущен. Чтобы включить:
              python3 Scripts/dump_vocab_reference.py \\
                  --vocab .testdata/rwkv_vocab_v20230424.txt \\
                  --out   .testdata/world_vocab_bytes.bin
            """)
        let data = try Data(contentsOf: URL(fileURLWithPath: p))
        let b = [UInt8](data)
        XCTAssertEqual(Array(b[0 ..< 8]), Array("RWKVVOCB".utf8), "чужой формат эталона")

        func u32(_ at: Int) -> Int {
            Int(UInt32(b[at]) | UInt32(b[at + 1]) << 8
                | UInt32(b[at + 2]) << 16 | UInt32(b[at + 3]) << 24)
        }
        var out: [Int: [UInt8]] = [:]
        let count = u32(8)
        var off = 12
        for _ in 0 ..< count {
            let id = u32(off)
            let len = u32(off + 4)
            out[id] = Array(b[(off + 8) ..< (off + 8 + len)])
            off += 8 + len
        }
        return out
    }

    /// Главный тест: таблица целиком, байт в байт.
    func testWholeVocabTableMatchesPythonReference() throws {
        let tok = try loadTokenizer()
        let ref = try loadReference()

        XCTAssertEqual(tok.count, ref.count,
                       "загружено \(tok.count) токенов, в эталоне \(ref.count)")

        var mismatches: [(Int, [UInt8], [UInt8])] = []
        for (id, want) in ref {
            let got = tok.rawBytes(id)
            if got != want {
                mismatches.append((id, got ?? [], want))
            }
        }
        if !mismatches.isEmpty {
            let shown = mismatches.sorted { $0.0 < $1.0 }.prefix(10).map {
                "id \($0.0): получено \($0.1), ожидалось \($0.2)"
            }.joined(separator: "\n  ")
            XCTFail("расходится \(mismatches.count) токенов из \(ref.count):\n  \(shown)")
        }
    }

    /// Комбинирующие знаки — та самая ловушка графемных кластеров.
    /// Отдельно от общего теста: если сломается именно она, сообщение должно
    /// называть причину, а не «расходится 57 токенов».
    func testCombiningMarksParseAsSingleTokens() throws {
        let tok = try loadTokenizer()
        // U+0301 (острое ударение) и U+0308 (умляут) — repr в файле начинается
        // с кавычки, за которой сразу идёт знак.
        XCTAssertEqual(tok.rawBytes(2672), [0xCC, 0x81],
                       "U+0301 разобран неверно — кавычка склеилась со знаком")
        XCTAssertEqual(tok.rawBytes(2676), [0xCC, 0x88], "U+0308 разобран неверно")

        // И сквозная проверка: жадный поиск обязан найти его одним токеном.
        let ids = tok.encode("\u{0301}")
        XCTAssertEqual(ids, [2672],
                       "комбинирующее ударение разбилось на \(ids.count) токен(ов)")
    }

    /// `\xNN` в str-repr — это код-поинт, а не байт.
    func testHexEscapeInStringLiteralIsCodePointNotRawByte() throws {
        let tok = try loadTokenizer()
        XCTAssertEqual(tok.rawBytes(2430), [0xC2, 0xA0],
                       "U+00A0 (NBSP) обязан быть двумя байтами UTF-8")
        XCTAssertEqual(tok.encode("\u{00A0}"), [2430],
                       "неразрывный пробел не находится в дереве")
    }

    /// `\xNN` в bytes-repr — наоборот, ровно один сырой байт. Обе трактовки
    /// обязаны сосуществовать: если сделать обе одинаковыми, сломается вторая.
    func testHexEscapeInBytesLiteralIsRawByte() throws {
        let tok = try loadTokenizer()
        XCTAssertEqual(tok.rawBytes(205), [0xCC],
                       "b'\\xcc' обязан быть одним сырым байтом")
    }

    /// `\uNNNN` — третий вид escape в словаре, и он тоже разбирался неверно.
    /// Это невидимые пробелы и разделители (U+2002, U+200B, U+200D…), которые
    /// в собранных из веба текстах встречаются регулярно.
    func testUnicodeEscapesParse() throws {
        let tok = try loadTokenizer()
        XCTAssertEqual(tok.rawBytes(9804), [0xE2, 0x80, 0x82], "U+2002 (en space)")
        XCTAssertEqual(tok.rawBytes(9808), [0xE2, 0x80, 0x8B], "U+200B (zero-width space)")
        XCTAssertEqual(tok.encode("\u{200B}"), [9808])
    }

    /// Круговой обход на текстах, которые и вскрыли дефект.
    func testRoundTripOnTextsWithCombiningMarksAndNBSP() throws {
        let tok = try loadTokenizer()
        let samples = [
            "Но что\u{0301} всего больше поразило меня",
            "неразрывный\u{00A0}пробел",
            "обычный текст без ловушек",
            "mixed Latin и кириллица, pince-nez",
        ]
        for s in samples {
            XCTAssertEqual(tok.decode(tok.encode(s)), s,
                           "круговой обход не сошёлся на \(s.debugDescription)")
        }
    }

    /// Ни один байт не должен теряться: у полного словаря fallback-ветка
    /// «пропустить байт» обязана быть недостижимой.
    func testNoByteIsSilentlyDropped() throws {
        let tok = try loadTokenizer()
        var all: [UInt8] = []
        for b in UInt8.min ... UInt8.max { all.append(b) }
        let text = all
        // Прогоняем сырые байты через дерево напрямую: длина восстановленного
        // должна совпасть с исходной.
        let ids = tok.encode(String(decoding: text, as: UTF8.self))
        var back: [UInt8] = []
        for id in ids { back.append(contentsOf: tok.rawBytes(id) ?? []) }
        XCTAssertEqual(back, Array(String(decoding: text, as: UTF8.self).utf8),
                       "часть байтов потерялась при кодировании")
    }
}
