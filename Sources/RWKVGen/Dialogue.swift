//
//  Dialogue.swift
//  Многоходовый диалог для проверки устойчивости роли (01.11).
//
//  Ход k — генерация с нуля по промпту из префикса (системная строка или пусто)
//  и ВСЕХ прошлых ходов: вопрос в шаблоне чата, сырой ответ модели, разделитель
//  «\n\n», затем вопрос хода k. Сырой ответ — ровно то, что выдала модель (без
//  обрезки пробелов): ученик состояния учился на `raw + "\n\nUser:"`, и история
//  должна выглядеть так же. Повторный проход по всей истории на каждом ходе
//  дороже продолжения состояния, но точен по построению: промпт хода k — обычный
//  промпт, тот же путь генерации, что у одиночного вопроса.
//

public enum Dialogue {
    /// Разделитель между ответом и следующей репликой — как в обучении состояния.
    public static let separator = "\n\n"

    /// Один ход: промпт, по которому он сгенерирован, и сырой ответ.
    public struct Turn: Equatable, Sendable {
        public let prompt: String
        public let reply: String
    }

    /// Прогон диалога. `userPart` оборачивает вопрос в шаблон чата (`User: q\n\nAssistant: …`);
    /// `reply(prompt, k)` — генерация хода k по полному промпту, возвращает сырой текст ответа.
    public static func run(prefix: String, turns: [String], userPart: (String) -> String,
                           reply: (_ prompt: String, _ turn: Int) -> String) -> [Turn] {
        var history = prefix
        var out: [Turn] = []
        out.reserveCapacity(turns.count)
        for (k, q) in turns.enumerated() {
            let prompt = history + userPart(q)
            let r = reply(prompt, k)
            out.append(Turn(prompt: prompt, reply: r))
            history = prompt + r + separator
        }
        return out
    }
}
