import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Поток токенов и предобучение.
//
//  Датасет — место, где ошибка тише всего: сдвиг y на один токен, порядок
//  байт, заворот на конце файла. Ничего из этого не роняет обучение, всё
//  просто делает его бессмысленным. Поэтому здесь проверяется точное
//  содержимое батчей, а не только их формы.
// ───────────────────────────────────────────────────────────────────────

final class PretrainTests: XCTestCase {

    var tmp: URL!

    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("rwkv_pretrain_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    /// Поток вида 0,1,2,…,n−1 — по значению токена сразу видно его позицию,
    /// поэтому любой сдвиг или перепутанный порядок байт виден глазами.
    @discardableResult
    func writeRamp(_ n: Int, name: String = "train.bin") throws -> URL {
        let url = tmp.appendingPathComponent(name)
        try TokenStreamWriter.write(tokens: Array(0 ..< n), to: url)
        return url
    }

    // ── Формат и чтение ──────────────────────────────────────────────

    func testWriteReadRoundTrip() throws {
        let url = try writeRamp(1000)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)
        XCTAssertEqual(s.count, 1000)

        let b = s.batch(batchSize: 1, step: 0)
        eval(b.x, b.y)
        let xs = b.x.asArray(Int32.self), ys = b.y.asArray(Int32.self)
        XCTAssertEqual(xs, (0 ..< 16).map { Int32($0) },
                       "x не совпал с исходной последовательностью — "
                       + "подозрение на порядок байт")
        XCTAssertEqual(ys, (1 ... 16).map { Int32($0) },
                       "y должен быть сдвинут ровно на один токен вперёд")
    }

    /// Значения выше 255 проверяют СТАРШИЙ байт: при перепутанном порядке
    /// байт тест на маленьких токенах прошёл бы.
    func testHighByteSurvivesRoundTrip() throws {
        let url = tmp.appendingPathComponent("hi.bin")
        let tokens = [0, 1, 255, 256, 257, 4096, 65535, 1000, 12345, 60000,
                      7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        try TokenStreamWriter.write(tokens: tokens, to: url)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)
        let b = s.batch(batchSize: 1, step: 0)
        eval(b.x)
        XCTAssertEqual(b.x.asArray(Int32.self), tokens.prefix(16).map { Int32($0) },
                       "старший байт потерян или переставлен")
    }

    func testMissingFileAndOddSizeAreReported() throws {
        XCTAssertThrowsError(try BinTokenStream(path: tmp.appendingPathComponent("nope.bin").path,
                                                ctxLen: 16)) { e in
            guard case TokenStreamError.fileNotFound = e else {
                return XCTFail("ожидалась fileNotFound, получено \(e)")
            }
        }
        let odd = tmp.appendingPathComponent("odd.bin")
        try Data([1, 2, 3]).write(to: odd)
        XCTAssertThrowsError(try BinTokenStream(path: odd.path, ctxLen: 16)) { e in
            guard case TokenStreamError.oddByteCount = e else {
                return XCTFail("ожидалась oddByteCount, получено \(e)")
            }
        }
    }

    func testTooShortStreamIsRejected() throws {
        let url = try writeRamp(10, name: "tiny.bin")
        XCTAssertThrowsError(try BinTokenStream(path: url.path, ctxLen: 16)) { e in
            guard case TokenStreamError.tooShort = e else {
                return XCTFail("ожидалась tooShort, получено \(e)")
            }
        }
    }

    // ── Раскладка батчей ─────────────────────────────────────────────

    /// Строки батча идут подряд с шагом ctxLen+1 и не перекрываются:
    /// перекрытие означало бы, что модель видит одни и те же токены дважды
    /// за шаг, а окна внутри батча коррелируют.
    func testBatchRowsAreConsecutiveNonOverlappingWindows() throws {
        let url = try writeRamp(10_000)
        let ctx = 16
        let s = try BinTokenStream(path: url.path, ctxLen: ctx)
        let b = s.batch(batchSize: 4, step: 0)
        eval(b.x)
        let xs = b.x.asArray(Int32.self)
        for row in 0 ..< 4 {
            let start = Int32(row * (ctx + 1))
            XCTAssertEqual(xs[row * ctx], start,
                           "строка \(row) начинается не там, где ожидается")
            XCTAssertEqual(xs[row * ctx + ctx - 1], start + Int32(ctx) - 1,
                           "строка \(row) не непрерывна")
        }
    }

    /// Соседние шаги читают разные окна и продолжают друг друга.
    func testConsecutiveStepsAdvance() throws {
        let url = try writeRamp(10_000)
        let ctx = 16, bs = 2
        let s = try BinTokenStream(path: url.path, ctxLen: ctx)
        let b0 = s.batch(batchSize: bs, step: 0)
        let b1 = s.batch(batchSize: bs, step: 1)
        eval(b0.x, b1.x)
        XCTAssertEqual(b1.x.asArray(Int32.self)[0], Int32(bs * (ctx + 1)),
                       "шаг 1 должен продолжать шаг 0")
    }

    /// Заворот на конце корпуса не выходит за границы и не падает.
    func testWrapAroundStaysInBounds() throws {
        let url = try writeRamp(200)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)
        for step in [0, 5, 50, 5000, 100_000] {
            let b = s.batch(batchSize: 4, step: step)
            eval(b.x, b.y)
            let xs = b.x.asArray(Int32.self) + b.y.asArray(Int32.self)
            XCTAssertTrue(xs.allSatisfy { $0 >= 0 && $0 < 200 },
                          "шаг \(step): токены вне корпуса")
        }
    }

    /// Детерминизм: тот же шаг даёт тот же батч. Без этого возобновление с
    /// чекпоинта не воспроизводит непрерывный прогон.
    func testBatchesAreDeterministic() throws {
        let url = try writeRamp(10_000)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)
        let a = s.batch(batchSize: 3, step: 42)
        let b = s.batch(batchSize: 3, step: 42)
        eval(a.x, b.x)
        XCTAssertEqual(a.x.asArray(Int32.self), b.x.asArray(Int32.self))
    }

    /// source() ведёт собственный счётчик и выдаёт ту же последовательность,
    /// что и явные вызовы batch(step:).
    func testSourceMatchesExplicitSteps() throws {
        let url = try writeRamp(10_000)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)
        let src = s.source(batchSize: 2)
        for step in 0 ..< 4 {
            let got = src(), want = s.batch(batchSize: 2, step: step)
            eval(got.x, want.x)
            XCTAssertEqual(got.x.asArray(Int32.self), want.x.asArray(Int32.self),
                           "source разошёлся с batch(step: \(step))")
        }
    }

    // ── OOV ──────────────────────────────────────────────────────────

    func testValidateDetectsOutOfVocabulary() throws {
        let url = try writeRamp(5000)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)

        let ok = s.validate(vocabSize: 5000)
        XCTAssertTrue(ok.ok, "ложная тревога: \(ok.issues)")
        XCTAssertEqual(ok.maxToken, 4999)

        let bad = s.validate(vocabSize: 1000)
        XCTAssertFalse(bad.ok, "OOV не обнаружены, хотя токены доходят до 4999")
        XCTAssertEqual(bad.maxToken, 4999)
        XCTAssertFalse(bad.issues.isEmpty)

        XCTAssertThrowsError(try s.validateOrThrow(vocabSize: 1000)) { e in
            guard case PretrainError.outOfVocabulary = e else {
                return XCTFail("ожидалась outOfVocabulary, получено \(e)")
            }
        }
        XCTAssertNoThrow(try s.validateOrThrow(vocabSize: 5000))
    }

    /// OOV в КОНЦЕ файла тоже обязаны находиться: выборочная проверка легко
    /// вырождается в «посмотрели только начало».
    func testValidateSamplesEndOfStream() throws {
        let url = tmp.appendingPathComponent("tail.bin")
        var tokens = Array(repeating: 5, count: 20_000)
        tokens[19_990] = 60_000                      // выброс у самого конца
        try TokenStreamWriter.write(tokens: tokens, to: url)
        let s = try BinTokenStream(path: url.path, ctxLen: 16)
        let v = s.validate(vocabSize: 1000)
        XCTAssertFalse(v.ok, "выброс в хвосте файла не найден")
        XCTAssertEqual(v.maxToken, 60_000)
    }

    // ── Конфигурация ─────────────────────────────────────────────────

    func testMaxStepsFromTokenBudget() {
        var c = PretrainConfig(ctxLen: 512, batchSize: 8, maxSteps: nil,
                               maxTokens: 1_000_000)
        XCTAssertEqual(c.resolvedMaxSteps(), 245)      // ceil(1e6 / 4096)
        c.gradAccum = 2
        XCTAssertEqual(c.resolvedMaxSteps(), 123)      // ceil(1e6 / 8192)
        c.maxSteps = 77
        XCTAssertEqual(c.resolvedMaxSteps(), 77, "явные шаги имеют приоритет")
    }

    func testContextLengthMustDivideChunk() throws {
        let url = try writeRamp(5000)
        let cfg = PretrainConfig(nLayer: 1, nEmbd: 128, vocab: 256,
                                 trainData: url.path, ctxLen: 20, batchSize: 1,
                                 maxSteps: 1)
        XCTAssertThrowsError(try Pretrain.run(config: cfg)) { e in
            guard case PretrainError.badContextLength = e else {
                return XCTFail("ожидалась badContextLength, получено \(e)")
            }
        }
    }

    func testPretrainRejectsOutOfVocabularyData() throws {
        let url = try writeRamp(5000)
        let cfg = PretrainConfig(nLayer: 1, nEmbd: 128, vocab: 256,
                                 trainData: url.path, ctxLen: 16, batchSize: 1,
                                 maxSteps: 1)
        XCTAssertThrowsError(try Pretrain.run(config: cfg)) { e in
            guard case PretrainError.outOfVocabulary = e else {
                return XCTFail("ожидалась outOfVocabulary, получено \(e)")
            }
        }
    }

    // ── Сквозной прогон ──────────────────────────────────────────────

    /// Маленькое, но настоящее предобучение: случайные веса → поток из .bin →
    /// падающий лосс. Данные периодические, так что модель обязана уловить
    /// период и уйти заметно ниже ln(vocab).
    func testEndToEndPretrainLearnsPeriodicData() throws {
        let url = tmp.appendingPathComponent("periodic.bin")
        // период 8: предсказуемо, но не тривиально
        try TokenStreamWriter.write(tokens: (0 ..< 20_000).map { $0 % 8 }, to: url)

        let cfg = PretrainConfig(nLayer: 2, nEmbd: 128, headSize: 64, vocab: 16,
                                 trainData: url.path, ctxLen: 16, batchSize: 4,
                                 maxSteps: 60, maxTokens: nil,
                                 lr: 1e-3, schedule: .cosine, warmupSteps: 5,
                                 adamEps: 1e-8,
                                 computeDType: .float32, cacheLimitGB: 0,
                                 evalEvery: 0, checkpointDir: nil,
                                 resume: false, logEvery: 1)

        var losses: [Float] = []
        let res = try Pretrain.run(config: cfg, onStep: { losses.append($0.loss) })

        XCTAssertEqual(res.steps, 60)
        XCTAssertTrue(losses.allSatisfy { $0.isFinite }, "NaN/inf при предобучении")
        XCTAssertEqual(losses.first!, log(Float(16)), accuracy: 0.5,
                       "старт далёк от ln(vocab)")
        XCTAssertLessThan(losses.last!, 0.5,
                          "модель не выучила период 8: \(losses.first!) → \(losses.last!)")
    }

    /// Валидация считается и возвращается.
    func testValidationLossIsReported() throws {
        let train = tmp.appendingPathComponent("t.bin")
        let val = tmp.appendingPathComponent("v.bin")
        try TokenStreamWriter.write(tokens: (0 ..< 5000).map { $0 % 8 }, to: train)
        try TokenStreamWriter.write(tokens: (0 ..< 2000).map { $0 % 8 }, to: val)

        let cfg = PretrainConfig(nLayer: 1, nEmbd: 128, vocab: 16,
                                 trainData: train.path, valData: val.path,
                                 ctxLen: 16, batchSize: 2,
                                 maxSteps: 10, maxTokens: nil,
                                 computeDType: .float32, cacheLimitGB: 0,
                                 evalEvery: 5, evalBatches: 2,
                                 checkpointDir: nil, resume: false, logEvery: 1)

        var evals: [(Int, Float)] = []
        let res = try Pretrain.run(config: cfg, onEval: { evals.append(($0, $1)) })
        XCTAssertEqual(evals.count, 2, "валидация должна пройти на шагах 5 и 10")
        XCTAssertTrue(evals.allSatisfy { $0.1.isFinite })
        XCTAssertNotNil(res.bestValLoss)
    }

    /// Чекпоинт пишется и подхватывается: второй запуск с resume стартует не
    /// с нуля, а продолжает.
    func testCheckpointSavedAndResumed() throws {
        let data = tmp.appendingPathComponent("c.bin")
        try TokenStreamWriter.write(tokens: (0 ..< 5000).map { $0 % 8 }, to: data)
        let ckptDir = tmp.appendingPathComponent("ckpt")

        func makeConfig(steps: Int, resume: Bool) -> PretrainConfig {
            PretrainConfig(nLayer: 1, nEmbd: 128, vocab: 16,
                           trainData: data.path, ctxLen: 16, batchSize: 2,
                           maxSteps: steps, maxTokens: nil,
                           computeDType: .float32, cacheLimitGB: 0,
                           evalEvery: 0, checkpointDir: ckptDir.path,
                           saveEvery: 0, resume: resume, logEvery: 1)
        }

        let first = try Pretrain.run(config: makeConfig(steps: 5, resume: false))
        XCTAssertEqual(first.steps, 5)
        let files = try FileManager.default.contentsOfDirectory(atPath: ckptDir.path)
        XCTAssertEqual(files.count, 1, "чекпоинт не сохранён: \(files)")

        // продолжение до 8 шагов: должно доделать 3, а не начать заново
        var stepsSeen: [Int] = []
        let second = try Pretrain.run(config: makeConfig(steps: 8, resume: true),
                                      onStep: { stepsSeen.append($0.step) })
        XCTAssertEqual(second.steps, 8)
        XCTAssertEqual(stepsSeen, [6, 7, 8],
                       "возобновление не продолжило с шага 5, а начало заново")
    }
}
