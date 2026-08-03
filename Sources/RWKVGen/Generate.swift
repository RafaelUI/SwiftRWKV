import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Цикл генерации: промпт → текст.
//
//  Поверх `prefill`/`step` и `Sampler`. Своей арифметики здесь нет вообще —
//  и это условие, а не совпадение: вторая копия арифметики уже один раз
//  стоила молчаливого игнорирования LoRA в декоде.
//
//  Четыре вещи, которые цикл обязан делать правильно и которые легко сделать
//  неправильно.
//
//  • UTF-8 на границе токена. Токен World — это БАЙТЫ, и один символ
//    кириллицы это два токена. Декодировать каждый токен в String по
//    отдельности значит выдавать «?» на каждом втором. Поэтому поток идёт по
//    байтам, а в `onToken` уходит только то, что уже составило целые кодовые
//    точки.
//
//  • Стоп-строка не должна протечь наружу. Если стоп — "\n\n", а мы уже
//    отдали "\n" в `onToken`, отменить это нельзя. Поэтому хвост, который
//    ЕЩЁ МОЖЕТ оказаться началом стоп-строки, придерживается до тех пор,
//    пока не станет ясно. И искать её надо по всему потоку, а не по
//    последнему токену: она свободно ложится на границу токенов.
//
//  • Состояние обязано соответствовать выдаче. После возврата `state`
//    поглотил промпт И ВСЕ токены из `tokens` — включая тот, на котором
//    остановились. Иначе продолжение с тем же состоянием теряло бы токен
//    молча, а молчаливая потеря токена в RNN — это расхождение, которое
//    проявится через сотню шагов и будет необъяснимо.
//
//  • Отмена. `onToken` возвращает Bool: false — прекратить. Без этого
//    длинная генерация в интерфейсе неостановима.
// ───────────────────────────────────────────────────────────────────────

/// Почему генерация остановилась.
public enum StopReason: Sendable, Equatable {
    /// Выбран лимит `maxTokens`.
    case maxTokens
    /// Встретился токен из `stopTokens`. Токен есть в `tokens`, но не в `text`.
    case stopToken(Int)
    /// В выдаче появилась одна из `stopStrings`; `text` обрезан ДО неё.
    case stopString(String)
    /// `onToken` вернул false.
    case cancelled
}

/// Результат генерации.
public struct GenerationResult: Sendable {
    /// Выданные токены, все до единого. При остановке по стоп-строке — вместе
    /// с теми, что её составили: обрезан `text`, а не `tokens`.
    public let tokens: [Int]
    /// Текст генерации: без промпта, без стоп-строки, без стоп-токена.
    public let text: String
    public let stopReason: StopReason
}

/// Параметры цикла генерации — что ОСТАНАВЛИВАЕТ, отдельно от параметров
/// сэмплирования, которые говорят, что ВЫБИРАЕТСЯ.
public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    /// Строки, при появлении которых генерация прекращается. Сама строка в
    /// `text` не попадает.
    public var stopStrings: [String]
    /// Токены, при выдаче которых генерация прекращается. Для World-моделей
    /// это обычно `[0]`.
    public var stopTokens: Set<Int>
    /// Засчитывать ли токены промпта в историю штрафов за повтор. По
    /// умолчанию нет — как в ChatRWKV: штрафуется то, что модель
    /// сгенерировала, а не то, что ей дали.
    public var penalizePrompt: Bool
    public var sampling: SamplingConfig

    public init(maxTokens: Int = 128,
                stopStrings: [String] = [],
                stopTokens: Set<Int> = [],
                penalizePrompt: Bool = false,
                sampling: SamplingConfig = .greedy) {
        precondition(maxTokens > 0, "maxTokens обязан быть > 0")
        precondition(!stopStrings.contains(""),
                     "пустая стоп-строка остановила бы генерацию на первом же токене")
        self.maxTokens = maxTokens
        self.stopStrings = stopStrings
        self.stopTokens = stopTokens
        self.penalizePrompt = penalizePrompt
        self.sampling = sampling
    }
}

extension X070Backbone {

    /// Сгенерировать продолжение промпта.
    ///
    /// `state` передаётся `inout` и после возврата пригоден для продолжения:
    /// он поглотил промпт и все токены из `tokens`.
    ///
    /// `onToken` получает (кусок текста, id токена) и возвращает `true`, чтобы
    /// продолжать. Кусок может быть ПУСТЫМ — токен оказался частью
    /// многобайтового символа или попал в придержанный хвост. Может и
    /// содержать несколько символов — когда хвост наконец разрешился.
    @discardableResult
    public func generate(prompt: [Int],
                         config: GenerationConfig,
                         tokenizer: WorldTokenizer,
                         state: inout RWKVState,
                         onToken: ((String, Int) -> Bool)? = nil) -> GenerationResult {
        precondition(!prompt.isEmpty, "generate: пустой промпт")

        var sampler = Sampler(config: config.sampling)
        if config.penalizePrompt { for id in prompt { sampler.observe(id) } }

        var logits = prefill(prompt, state: &state)

        var produced: [Int] = []
        var fed = 0                      // сколько выданных токенов поглотило state
        var bytes: [UInt8] = []          // весь текст генерации, в байтах
        var emitted = 0                  // сколько байт уже ушло в onToken
        var reason: StopReason = .maxTokens

        // Хвост, который придерживаем: стоп-строка длины L может начаться не
        // раньше, чем за L-1 байт до конца.
        let holdBack = config.stopStrings.map { $0.utf8.count }.max().map { $0 - 1 } ?? 0

        loop: for _ in 0 ..< config.maxTokens {
            let id = sampler.next(logits)
            produced.append(id)

            if config.stopTokens.contains(id) {
                reason = .stopToken(id)
                break loop
            }

            let grewFrom = bytes.count
            bytes.append(contentsOf: tokenizer.rawBytes(id) ?? [])

            // Новое вхождение обязано задевать только что добавленные байты,
            // поэтому искать с самого начала не нужно — иначе цикл стал бы
            // квадратичным по длине генерации.
            if let hit = firstStopHit(bytes, config.stopStrings,
                                      from: max(0, grewFrom - holdBack)) {
                bytes.removeSubrange(hit.start ..< bytes.count)
                reason = .stopString(hit.needle)
                break loop
            }

            if let onToken {
                let limit = holdBack == 0 ? bytes.count : bytes.count - holdBack
                let safeEnd = max(emitted, utf8Boundary(bytes, upTo: max(0, limit)))
                let chunk = safeEnd > emitted
                    ? String(decoding: bytes[emitted ..< safeEnd], as: UTF8.self) : ""
                emitted = safeEnd
                if !onToken(chunk, id) {
                    reason = .cancelled
                    break loop
                }
            }

            logits = step(id, state: &state)
            fed = produced.count
        }

        // Догнать состояние: выход по break пропускает step последнего токена.
        while fed < produced.count {
            _ = step(produced[fed], state: &state)
            fed += 1
        }

        // Придержанный хвост отдаём, если остановились не по отмене: при
        // отмене вызывающий уже сказал, что больше не хочет.
        if let onToken, reason != .cancelled, emitted < bytes.count {
            _ = onToken(String(decoding: bytes[emitted...], as: UTF8.self), produced.last ?? -1)
        }

        return GenerationResult(tokens: produced,
                                text: String(decoding: bytes, as: UTF8.self),
                                stopReason: reason)
    }

    /// То же, но промпт — строка, а состояние заводится своё.
    @discardableResult
    public func generate(prompt: String,
                         maxTokens: Int = 128,
                         config: SamplingConfig = .greedy,
                         tokenizer: WorldTokenizer,
                         stopStrings: [String] = [],
                         stopTokens: Set<Int> = [],
                         onToken: ((String, Int) -> Bool)? = nil) -> GenerationResult {
        var state = RWKVState(cfg: cfg)
        let gc = GenerationConfig(maxTokens: maxTokens,
                                  stopStrings: stopStrings,
                                  stopTokens: stopTokens,
                                  sampling: config)
        return generate(prompt: tokenizer.encode(prompt), config: gc,
                        tokenizer: tokenizer, state: &state, onToken: onToken)
    }
}

// ─────────────── Байтовые утилиты потоковой выдачи ───────────────

/// Первое вхождение любой из стоп-строк, начиная с байта `from`.
/// Возвращает начало вхождения и саму строку.
func firstStopHit(_ bytes: [UInt8], _ needles: [String],
                  from: Int = 0) -> (start: Int, needle: String)? {
    guard !needles.isEmpty, from < bytes.count else { return nil }
    var best: (start: Int, needle: String)? = nil
    for needle in needles {
        let pat = Array(needle.utf8)
        guard !pat.isEmpty, pat.count <= bytes.count - from else { continue }
        var start = from
        while start <= bytes.count - pat.count {
            var ok = true
            for j in 0 ..< pat.count where bytes[start + j] != pat[j] { ok = false; break }
            if ok {
                if best == nil || start < best!.start { best = (start, needle) }
                break
            }
            start += 1
        }
    }
    return best
}

/// Наибольшая граница `<= limit`, на которой байты образуют целые кодовые
/// точки UTF-8.
///
/// Без этого поток рвался бы посреди двухбайтового символа, и в `onToken`
/// уходил бы «?» — а собрать его обратно на стороне вызывающего уже нельзя.
func utf8Boundary(_ bytes: [UInt8], upTo limit: Int) -> Int {
    let end = min(limit, bytes.count)
    if end <= 0 { return 0 }
    // Отступаем от конца не более чем на 3 байта: длиннее последовательности
    // в UTF-8 не бывает.
    var back = 0
    while back < 4, end - back > 0 {
        let b = bytes[end - back - 1]
        if b & 0x80 == 0 { return back == 0 ? end : end - back }   // ASCII
        if b & 0xC0 == 0x80 { back += 1; continue }                // продолжающий
        let need: Int                                              // ведущий
        if b & 0xE0 == 0xC0 { need = 2 } else if b & 0xF0 == 0xE0 { need = 3 }
        else if b & 0xF8 == 0xF0 { need = 4 } else { return end }  // мусор — не наше дело
        return back + 1 >= need ? end : end - back - 1
    }
    return end
}
