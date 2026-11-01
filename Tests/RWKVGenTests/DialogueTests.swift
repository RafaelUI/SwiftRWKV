//
//  DialogueTests.swift
//  Свойства сборки многоходового диалога (Dialogue.run). Модель не нужна: генерация —
//  подставная функция, проверяется только то, какой промпт получает каждый ход.
//
import XCTest
@testable import RWKVGen

final class DialogueTests: XCTestCase {
    private let user: (String) -> String = { "User: \($0)\n\nAssistant: <think>\n</think>" }
    private let turns = ["Привет", "Как тебя зовут?", "  Сколько будет 2+2?  ", "Ты Claude?"]

    /// Ответ хода — с пробелами по краям и переводами строк: проверка, что история хранит сырой текст.
    private func fake(_ prompt: String, _ k: Int) -> String { "\n ответ \(k) [\(prompt.count)] \n" }

    /// Первый ход — ровно префикс + вопрос в шаблоне: тот же промпт, что у одиночного вопроса.
    func testFirstTurnIsSingleQuestionPrompt() {
        let r = Dialogue.run(prefix: "System: S\n\n", turns: turns, userPart: user, reply: fake)
        XCTAssertEqual(r[0].prompt, "System: S\n\n" + user(turns[0]))
        let none = Dialogue.run(prefix: "", turns: turns, userPart: user, reply: fake)
        XCTAssertEqual(none[0].prompt, user(turns[0]))
    }

    /// Промпт хода k+1 = промпт хода k + СЫРОЙ ответ k + «\n\n» + вопрос k+1.
    func testHistoryAccumulatesRawRepliesWithSeparator() {
        let r = Dialogue.run(prefix: "System: S\n\n", turns: turns, userPart: user, reply: fake)
        XCTAssertEqual(r.count, turns.count)
        for k in 0 ..< turns.count - 1 {
            XCTAssertEqual(r[k + 1].prompt, r[k].prompt + r[k].reply + "\n\n" + user(turns[k + 1]), "ход \(k + 1)")
        }
    }

    /// Префикс — один раз, в начале; вопросы — каждый ровно один раз и по порядку.
    func testPrefixOnceAndQuestionsInOrder() {
        let prefix = "System: Ты — Neuron.\n\n"
        let last = Dialogue.run(prefix: prefix, turns: turns, userPart: user, reply: fake).last!.prompt
        XCTAssertTrue(last.hasPrefix(prefix))
        XCTAssertEqual(last.components(separatedBy: "System:").count - 1, 1)
        var pos = last.startIndex
        for q in turns {
            let r = last.range(of: user(q), range: pos ..< last.endIndex)
            XCTAssertNotNil(r, "вопрос «\(q)» не найден по порядку"); pos = r?.upperBound ?? pos
        }
    }

    /// Генерация получает ровно промпт хода и его номер; ответы возвращаются как выданы.
    func testReplyReceivesExactPromptAndTurnIndex() {
        var seen: [(String, Int)] = []
        let r = Dialogue.run(prefix: "", turns: turns, userPart: user) { p, k in seen.append((p, k)); return fake(p, k) }
        XCTAssertEqual(seen.map(\.1), Array(0 ..< turns.count))
        XCTAssertEqual(seen.map(\.0), r.map(\.prompt))
        XCTAssertEqual(r.map(\.reply), zip(r.map(\.prompt), 0...).map { fake($0, $1) })
    }

    /// Пустой диалог — пустой результат, генерация не вызывается.
    func testEmptyDialogue() {
        var calls = 0
        XCTAssertTrue(Dialogue.run(prefix: "x", turns: [], userPart: user) { _, _ in calls += 1; return "" }.isEmpty)
        XCTAssertEqual(calls, 0)
    }
}
