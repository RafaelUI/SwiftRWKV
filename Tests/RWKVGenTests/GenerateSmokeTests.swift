//
//  GenerateSmokeTests.swift
//  Цикл генерации на РЕАЛЬНОЙ 0.1B: публичный API так, как его увидит
//  пользователь.
//
//  Модульные тесты проверяют сэмплер на известном распределении и байтовые
//  утилиты — на байтах. Здесь проверяется связка: промпт → текст, стоп-условия
//  и главное — что состояние после возврата пригодно для продолжения.
//
//  Пропускается без модели и словаря.
//
import XCTest
import MLX
@testable import RWKVGen

final class GenerateSmokeTests: XCTestCase {

    func backbone() throws -> (X070Backbone, WorldTokenizer, X070Config) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let vp = env["RWKV_WORLD_VOCAB"] ?? home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        for p in [mp, vp] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет фикстуры \(p)")
        }
        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: w["head.weight"]!.shape[0])
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: vp))
        else { throw XCTSkip("словарь не разобрался") }
        return (X070Backbone(weights: w, cfg: cfg), tok, cfg)
    }

    /// Жадная генерация воспроизводима И зависит от промпта.
    func testGreedyGenerationIsReproducibleAndPromptDependent() throws {
        let (bb, tok, _) = try backbone()
        let a = bb.generate(prompt: "The capital of France is", maxTokens: 12, tokenizer: tok)
        let b = bb.generate(prompt: "The capital of France is", maxTokens: 12, tokenizer: tok)
        let c = bb.generate(prompt: "Рецепт борща начинается с", maxTokens: 12, tokenizer: tok)

        XCTAssertEqual(a.tokens, b.tokens, "жадная генерация невоспроизводима")
        XCTAssertEqual(a.text, b.text)
        XCTAssertEqual(a.tokens.count, 12)
        XCTAssertEqual(a.stopReason, .maxTokens)
        XCTAssertFalse(a.text.isEmpty, "жадная генерация не дала текста")
        XCTAssertNotEqual(a.tokens, c.tokens, "продолжение не зависит от промпта")
    }

    /// `generate` жадный совпадает с ручным циклом `prefill` + argmax + `step`.
    ///
    /// То самое утверждение, которое запрещает второй копии арифметики
    /// разойтись с первой. Ручной цикл — это ровно то, что было написано в
    /// docs/Inference.md до появления сэмплера.
    func testGreedyGenerateMatchesTheHandWrittenLoop() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("Bees collect nectar")

        var st = RWKVState(cfg: cfg)
        var logits = bb.prefill(ids, state: &st)
        var manual: [Int] = []
        for _ in 0 ..< 10 {
            let id = logits.argMax().item(Int.self)
            manual.append(id)
            logits = bb.step(id, state: &st)
        }

        let r = bb.generate(prompt: "Bees collect nectar", maxTokens: 10, tokenizer: tok)
        XCTAssertEqual(r.tokens, manual, "generate разошёлся с ручным жадным циклом")
    }

    /// После возврата состояние поглотило промпт И ВСЕ выданные токены.
    ///
    /// Проверяется не «примерно то же», а совпадение с состоянием, собранным
    /// с нуля по той же цепочке токенов. Если бы цикл терял последний токен,
    /// расхождение было бы огромным — и проявилось бы только у того, кто
    /// продолжает генерацию с тем же состоянием.
    func testStateAfterGenerateIsReadyToContinue() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")

        var st = RWKVState(cfg: cfg)
        let r = bb.generate(prompt: ids,
                            config: GenerationConfig(maxTokens: 6),
                            tokenizer: tok, state: &st)

        // Эталон собирается ТЕМ ЖЕ способом, которым его строит generate:
        // промпт параллельным проходом, дальше по шагу на каждый выданный
        // токен. Сравнивать со сплошным `prefill(ids + tokens)` было бы
        // неверно — там весь хвост тоже пошёл бы параллельным путём, а
        // расхождение двух ядер (7.4e-7) превратило бы точное равенство в
        // допуск, то есть ослабило бы ровно то утверждение, ради которого
        // тест написан.
        var ref = RWKVState(cfg: cfg)
        _ = bb.prefill(ids, state: &ref)
        for t in r.tokens { _ = bb.step(t, state: &ref) }

        for layer in 0 ..< cfg.nLayer {
            eval(st.wkv[layer], ref.wkv[layer])
            let d = MLX.abs(st.wkv[layer] - ref.wkv[layer]).max().item(Float.self)
            XCTAssertEqual(d, 0, "состояние слоя \(layer) разошлось на \(d)")
        }
    }

    /// Стоп-строка останавливает генерацию и НЕ попадает в текст.
    func testStopStringStopsAndIsCutOff() throws {
        let (bb, tok, cfg) = try backbone()
        // Стоп-строка выбирается из того, что модель на самом деле выдала:
        // придумывать её заранее нельзя — 0.1B может её просто не породить,
        // и тест станет зелёным ни на чём.
        let base = bb.generate(prompt: "The capital of France is", maxTokens: 24, tokenizer: tok)
        let mid = String(base.text.dropFirst(3).prefix(4))
        try XCTSkipIf(mid.count < 3, "выдача слишком коротка, чтобы резать")

        var st = RWKVState(cfg: cfg)
        let r = bb.generate(prompt: tok.encode("The capital of France is"),
                            config: GenerationConfig(maxTokens: 24, stopStrings: [mid]),
                            tokenizer: tok, state: &st)

        XCTAssertEqual(r.stopReason, .stopString(mid))
        XCTAssertFalse(r.text.contains(mid), "стоп-строка протекла в текст: \(r.text)")
        XCTAssertLessThan(r.tokens.count, 24, "стоп-строка не остановила генерацию")
        XCTAssertTrue(base.text.hasPrefix(r.text), "текст до стоп-строки не совпал с общим")
    }

    /// Стоп-строка, ЛЕЖАЩАЯ НА ГРАНИЦЕ ТОКЕНОВ, всё равно находится.
    ///
    /// Отдельный тест, а не частный случай предыдущего. Реализация, которая
    /// ищет стоп-строку только внутри байтов последнего токена, проходит все
    /// остальные проверки: стоп-строка из середины выдачи обычно целиком
    /// лежит в одном токене. Здесь она собирается НАМЕРЕННО из хвоста одного
    /// токена и головы следующего, так что внутри токена её нет ни разу.
    func testStopStringSpanningATokenBoundaryIsFound() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")
        var probe = RWKVState(cfg: cfg)
        let base = bb.generate(prompt: ids, config: GenerationConfig(maxTokens: 24),
                               tokenizer: tok, state: &probe)

        // Ищем пару соседних токенов, чьи байты склеиваются в строку, которой
        // нет ни в одном токене по отдельности.
        var needle: String? = nil
        for i in 1 ..< base.tokens.count - 1 {
            guard let a = tok.rawBytes(base.tokens[i]), a.count >= 1,
                  let b = tok.rawBytes(base.tokens[i + 1]), b.count >= 1 else { continue }
            let cross = [a[a.count - 1]] + [b[0]]
            guard let s = String(bytes: cross, encoding: .utf8), s.count == 2 else { continue }
            // Внутри отдельных токенов такой пары быть не должно.
            let insideSome = base.tokens.contains { id in
                guard let raw = tok.rawBytes(id), raw.count >= 2 else { return false }
                return (0 ... raw.count - 2).contains { Array(raw[$0 ..< $0 + 2]) == cross }
            }
            if !insideSome { needle = s; break }
        }
        let stop = try XCTUnwrap(needle, "не нашлось пары, ложащейся на границу")

        var st = RWKVState(cfg: cfg)
        let r = bb.generate(prompt: ids,
                            config: GenerationConfig(maxTokens: 24, stopStrings: [stop]),
                            tokenizer: tok, state: &st)
        XCTAssertEqual(r.stopReason, .stopString(stop),
                       "стоп-строка на границе токенов не найдена")
        XCTAssertFalse(r.text.contains(stop))
    }

    /// Поток НЕ отдаёт наружу начало стоп-строки.
    ///
    /// Самая неприятная из возможных протечек: `onToken` уже отдан
    /// вызывающему, отменить его нельзя, и в интерфейсе остаётся кусок
    /// разделителя, которого нет в `text`.
    func testStreamNeverLeaksThePrefixOfAStopString() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")
        var probe = RWKVState(cfg: cfg)
        let base = bb.generate(prompt: ids, config: GenerationConfig(maxTokens: 24),
                               tokenizer: tok, state: &probe)
        let stop = String(base.text.dropFirst(4).prefix(5))
        try XCTSkipIf(stop.count < 4, "выдача слишком коротка")

        var streamed = ""
        var st = RWKVState(cfg: cfg)
        let r = bb.generate(prompt: ids,
                            config: GenerationConfig(maxTokens: 24, stopStrings: [stop]),
                            tokenizer: tok, state: &st) { chunk, _ in streamed += chunk; return true }

        XCTAssertEqual(r.stopReason, StopReason.stopString(stop))
        XCTAssertEqual(streamed, r.text,
                       "поток разошёлся с текстом: отдано «\(streamed)», в тексте «\(r.text)»")
    }

    /// Стоп-токен останавливает генерацию.
    func testStopTokenStops() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")
        let base = bb.generate(prompt: "The capital of France is", maxTokens: 8, tokenizer: tok)
        let victim = base.tokens[3]
        // Первое вхождение, а не позиция 3: тот же id мог встретиться раньше,
        // и тогда остановка произойдёт там. Тест обязан утверждать то, что
        // следует из контракта, а не то, что удобно.
        let cut = base.tokens.firstIndex(of: victim)!

        var st = RWKVState(cfg: cfg)
        let r = bb.generate(prompt: ids,
                            config: GenerationConfig(maxTokens: 8, stopTokens: [victim]),
                            tokenizer: tok, state: &st)
        XCTAssertEqual(r.stopReason, .stopToken(victim))
        XCTAssertEqual(r.tokens, Array(base.tokens.prefix(cut + 1)),
                       "остановка не на первом вхождении стоп-токена")
        XCTAssertEqual(r.text, tok.decode(Array(base.tokens.prefix(cut))),
                       "стоп-токен попал в текст")
    }

    /// Отмена через `onToken` прекращает генерацию немедленно.
    func testCancellationThroughOnToken() throws {
        let (bb, tok, _) = try backbone()
        var seen = 0
        let r = bb.generate(prompt: "The capital of France is", maxTokens: 30,
                            tokenizer: tok) { _, _ in
            seen += 1
            return seen < 5
        }
        XCTAssertEqual(r.stopReason, .cancelled)
        XCTAssertEqual(r.tokens.count, 5)
        XCTAssertEqual(seen, 5, "onToken звали после отмены")
    }

    /// Склейка кусков из `onToken` равна итоговому тексту.
    ///
    /// Контракт потоковой выдачи. Ломается ровно на многобайтовых символах,
    /// поэтому промпт — русский.
    func testStreamedChunksConcatenateToTheResultText() throws {
        let (bb, tok, _) = try backbone()
        var streamed = ""
        let r = bb.generate(prompt: "Пчёлы собирают нектар и", maxTokens: 20,
                            tokenizer: tok) { chunk, _ in streamed += chunk; return true }
        XCTAssertEqual(streamed, r.text, "поток не сложился в итоговый текст")
        XCTAssertFalse(r.text.contains("\u{FFFD}"), "в тексте есть символ замены")
    }

    /// Состояние догоняет выдачу и при РАННЕЙ остановке.
    ///
    /// Предыдущий тест этого не проверяет: без стоп-условий цикл доходит до
    /// конца и последний токен скармливается состоянию сам собой. Догоняющий
    /// шаг нужен только на выходе по `break` — и мутационная проверка это
    /// показала: удаление догоняющего цикла не поймал НИ ОДИН тест.
    func testStateCatchesUpAfterAnEarlyStop() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")
        var probe = RWKVState(cfg: cfg)
        let base = bb.generate(prompt: ids, config: GenerationConfig(maxTokens: 10),
                               tokenizer: tok, state: &probe)
        let victim = base.tokens[4]

        var st = RWKVState(cfg: cfg)
        let r = bb.generate(prompt: ids,
                            config: GenerationConfig(maxTokens: 10, stopTokens: [victim]),
                            tokenizer: tok, state: &st)
        XCTAssertLessThan(r.tokens.count, 10, "остановка не была ранней — тест ничего не проверил")

        var ref = RWKVState(cfg: cfg)
        _ = bb.prefill(ids, state: &ref)
        for t in r.tokens { _ = bb.step(t, state: &ref) }
        for layer in 0 ..< cfg.nLayer {
            eval(st.wkv[layer], ref.wkv[layer])
            let d = MLX.abs(st.wkv[layer] - ref.wkv[layer]).max().item(Float.self)
            XCTAssertEqual(d, 0, "состояние слоя \(layer) отстало на \(d)")
        }
    }

    /// Придержанный хвост не рвёт символы и досылается в конце.
    ///
    /// Стоп-строка выбрана заведомо невозможной: тогда генерация доходит до
    /// `maxTokens`, но `holdBack` всё равно ненулевой — а значит граница
    /// выдачи стоит на ПРОИЗВОЛЬНОМ байте, а не на границе токена. Именно
    /// там рвутся многобайтовые символы, и именно там теряется хвост, если
    /// его не досылать.
    ///
    /// Оба дефекта до этого теста мутационная проверка находила
    /// непойманными: границы токенов World на русском тексте сами по себе
    /// символы не режут, и обычный поток проходил без разрывов.
    func testHeldBackTailIsCharacterSafeAndFlushed() throws {
        let (bb, tok, _) = try backbone()
        var streamed = ""
        let r = bb.generate(prompt: "Столица России —", maxTokens: 24,
                            tokenizer: tok,
                            stopStrings: ["###ЭТОГО-НЕ-БУДЕТ###"]) { chunk, _ in
            streamed += chunk
            return true
        }
        XCTAssertEqual(r.stopReason, .maxTokens, "невозможная стоп-строка сработала")
        // Различающее: на чисто ASCII-выдаче тест проверял бы пустоту.
        XCTAssertTrue(r.text.contains { $0.utf8.count > 1 },
                      "выдача оказалась однобайтовой — тест ничего не проверил: \(r.text)")
        XCTAssertFalse(r.text.contains("\u{FFFD}"))
        XCTAssertEqual(streamed, r.text,
                       "поток разошёлся с текстом: отдано «\(streamed)»")
    }

    /// Сид определяет выдачу при температуре, разные сиды расходятся.
    func testSeedDeterminesSampledOutput() throws {
        let (bb, tok, _) = try backbone()
        func run(_ seed: UInt64) -> [Int] {
            bb.generate(prompt: "Once upon a time", maxTokens: 16,
                        config: SamplingConfig(temperature: 1.0, topP: 0.9, seed: seed),
                        tokenizer: tok).tokens
        }
        XCTAssertEqual(run(1), run(1), "сид не воспроизводит выдачу")
        XCTAssertNotEqual(run(1), run(2), "разные сиды дали одно и то же")
    }

    /// topK = 1 на реальной модели совпадает с жадным декодом.
    ///
    /// Через общий путь отбора, с настоящим словарём в 65536: отсечение по
    /// topK на синтетике из пяти элементов и на реальном словаре — это разные
    /// объёмы работы для сортировки.
    func testTopKOneMatchesGreedyOnTheRealVocabulary() throws {
        let (bb, tok, _) = try backbone()
        let greedy = bb.generate(prompt: "The capital of France is", maxTokens: 10,
                                 tokenizer: tok).tokens
        for seed in UInt64(0) ..< 3 {
            let k1 = bb.generate(prompt: "The capital of France is", maxTokens: 10,
                                 config: SamplingConfig(temperature: 1, topK: 1, seed: seed),
                                 tokenizer: tok).tokens
            XCTAssertEqual(k1, greedy, "topK=1 разошёлся с жадным (сид \(seed))")
        }
    }
}
