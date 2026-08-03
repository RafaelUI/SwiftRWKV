//
//  GenerateTests.swift
//  Байтовая механика цикла генерации: границы UTF-8 и поиск стоп-строк.
//
//  Модель не нужна: это чистые функции над байтами, и именно они ломаются
//  тихо. Разрыв посреди двухбайтового символа даёт «?» в потоке, а протёкшая
//  стоп-строка — лишний "\n\n" в ответе; ни то, ни другое не роняет процесс.
//
import XCTest
@testable import RWKVGen

final class GenerateTests: XCTestCase {

    // ─────────────────────── границы UTF-8 ───────────────────────

    /// Любой префикс байтов, обрезанный по границе, декодируется БЕЗ потерь.
    ///
    /// Утверждение сильное намеренно: «нет U+FFFD» проходится реализацией,
    /// которая всегда возвращает 0. Поэтому вторая половина — граница обязана
    /// отставать от запрошенной не более чем на 3 байта, то есть функция
    /// обязана отдавать МАКСИМУМ возможного.
    func testUtf8BoundaryNeverSplitsACodePoint() {
        let text = "Привет, мир! 日本語 🚀 ok"
        let bytes = Array(text.utf8)
        for limit in 0 ... bytes.count {
            let end = utf8Boundary(bytes, upTo: limit)
            XCTAssertLessThanOrEqual(end, limit, "граница уехала за предел")
            XCTAssertGreaterThanOrEqual(end, limit - 3, "граница отстала больше чем на 3 байта")
            let s = String(bytes: bytes[0 ..< end], encoding: .utf8)
            XCTAssertNotNil(s, "префикс длины \(end) не UTF-8")
        }
    }

    /// Склейка всех кусков, нарезанных по границам, равна исходному тексту.
    ///
    /// Это то, что на самом деле обещано вызывающему `onToken`: куски могут
    /// быть любыми, но их конкатенация — исходный текст.
    func testChunksConcatenateBackToTheOriginal() {
        let text = "ёж 🦔 съел ёлку"
        let bytes = Array(text.utf8)
        var emitted = 0, out = ""
        // Нарезаем по два байта — гарантированно попадая в середины символов.
        for limit in stride(from: 0, through: bytes.count, by: 2) {
            let end = max(emitted, utf8Boundary(bytes, upTo: limit))
            if end > emitted {
                out += String(decoding: bytes[emitted ..< end], as: UTF8.self)
                emitted = end
            }
        }
        out += String(decoding: bytes[emitted...], as: UTF8.self)
        XCTAssertEqual(out, text)
    }

    /// Чистый ASCII отдаётся целиком, ничего не придерживается.
    func testAsciiIsNeverHeldBack() {
        let bytes = Array("plain ascii".utf8)
        for limit in 0 ... bytes.count {
            XCTAssertEqual(utf8Boundary(bytes, upTo: limit), limit)
        }
    }

    // ─────────────────────── стоп-строки ───────────────────────

    func testStopHitFindsTheEarliestOccurrence() {
        let bytes = Array("abc STOP def END".utf8)
        let hit = firstStopHit(bytes, ["END", "STOP"])
        XCTAssertEqual(hit?.needle, "STOP", "выбрано не первое вхождение")
        XCTAssertEqual(hit?.start, 4)
    }

    func testStopHitReturnsNilWhenAbsent() {
        XCTAssertNil(firstStopHit(Array("nothing here".utf8), ["\n\n", "STOP"]))
        XCTAssertNil(firstStopHit(Array("x".utf8), ["longer than haystack"]))
        XCTAssertNil(firstStopHit([], ["a"]))
        XCTAssertNil(firstStopHit(Array("abc".utf8), []))
    }

    /// Стоп-строка ложится НА ГРАНИЦУ токенов и обязана находиться.
    ///
    /// Ровно этот случай пропускает реализация, которая ищет стоп-строку в
    /// последнем декодированном токене: "\n\n" в World-словаре обычно два
    /// разных токена, и по отдельности ни один из них стоп-строкой не является.
    func testStopHitSpansTokenBoundary() {
        var bytes = Array("line".utf8)
        let before = bytes.count
        bytes.append(contentsOf: Array("\n".utf8))
        XCTAssertNil(firstStopHit(bytes, ["\n\n"], from: max(0, before - 1)))
        let before2 = bytes.count
        bytes.append(contentsOf: Array("\n".utf8))
        let hit = firstStopHit(bytes, ["\n\n"], from: max(0, before2 - 1))
        XCTAssertEqual(hit?.start, 4, "стоп-строка на границе токенов не найдена")
    }

    /// Сдвинутый старт поиска не теряет вхождение, начавшееся до него.
    func testStopHitFromOffsetStillSeesOverlap() {
        let bytes = Array("aaSTOPaa".utf8)
        XCTAssertEqual(firstStopHit(bytes, ["STOP"], from: 0)?.start, 2)
        XCTAssertEqual(firstStopHit(bytes, ["STOP"], from: 2)?.start, 2)
        // Начав ПОСЛЕ вхождения, найти его нельзя — и это ожидаемо: цикл
        // генерации всегда откатывает старт на длину стоп-строки минус один.
        XCTAssertNil(firstStopHit(bytes, ["STOP"], from: 3))
    }

    /// Многобайтовая стоп-строка.
    func testStopHitHandlesMultibyteNeedles() {
        let bytes = Array("текст КОНЕЦ хвост".utf8)
        let hit = firstStopHit(bytes, ["КОНЕЦ"])
        XCTAssertEqual(hit?.needle, "КОНЕЦ")
        XCTAssertEqual(String(decoding: bytes[0 ..< hit!.start], as: UTF8.self), "текст ")
    }

    // ─────────────────────── конфигурация ───────────────────────

    func testGenerationConfigDefaultsAreGreedy() {
        let c = GenerationConfig()
        XCTAssertTrue(c.sampling.isGreedy)
        XCTAssertFalse(c.penalizePrompt)
        XCTAssertTrue(c.stopStrings.isEmpty)
    }
}
