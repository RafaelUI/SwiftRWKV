import XCTest
import MLX
@testable import RWKVEmbedding
@testable import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Данные для эмбеддингов: загрузка LitRetrieval, разбор кандидатов,
//  батчеры.
//
//  Часть тестов не требует файлов вообще (разбор строки, форма батча на
//  синтетике), часть — среза корпуса. Фикстуры:
//      .testdata/litretrieval_slice.jsonl        — 120 строк, по 40 на задачу
//      .testdata/rwkv_vocab_v20230424.txt        — словарь World
//  Оба генерируются локально (см. Scripts), в репозиторий не кладутся.
// ───────────────────────────────────────────────────────────────────────

final class EmbeddingDatasetTests: XCTestCase {

    static var testdata: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata")
    }
    static var slicePath: String {
        ProcessInfo.processInfo.environment["RWKV_LITRETRIEVAL_SLICE"]
            ?? testdata.appendingPathComponent("litretrieval_slice.jsonl").path
    }
    static var vocabPath: String {
        ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? testdata.appendingPathComponent("rwkv_vocab_v20230424.txt").path
    }

    func loadTokenizer() throws -> WorldTokenizer {
        try XCTSkipIf(!FileManager.default.fileExists(atPath: Self.vocabPath),
                      "нет словаря World — тест пропущен")
        let t = WorldTokenizer(vocabURL: URL(fileURLWithPath: Self.vocabPath))
        try XCTSkipIf(t == nil, "словарь не разобрался")
        return t!
    }

    func loadSlice(limit: Int? = nil,
                   tasks: Set<EmbeddingTask>? = nil) throws -> [EmbeddingSample] {
        try XCTSkipIf(!FileManager.default.fileExists(atPath: Self.slicePath),
                      "нет среза LitRetrieval — тест пропущен")
        return try EmbeddingDataset.loadJSONL(path: Self.slicePath,
                                              limit: limit, tasks: tasks)
    }

    // ── Разбор кандидатов (файлы не нужны) ───────────────────────────

    func testParseCandidatesFromInstruction() {
        let anchor = "Instruct: Classify the emotion into one of the following "
                   + "categories: longing, guilt, joy, awe, shame, fear, bitterness\n"
                   + "Query: Северные сумерки..."
        let cands = ClassificationLabels.parseCandidates(from: anchor)
        XCTAssertEqual(cands, ["longing", "guilt", "joy", "awe",
                               "shame", "fear", "bitterness"])
    }

    func testParseCandidatesHandlesTrailingPeriodAndSpaces() {
        let anchor = "categories:  joy ,  fear,shame.\nQuery: x"
        XCTAssertEqual(ClassificationLabels.parseCandidates(from: anchor),
                       ["joy", "fear", "shame"])
    }

    func testParseCandidatesReturnsNilWithoutInstruction() {
        XCTAssertNil(ClassificationLabels.parseCandidates(from: "Query: просто текст"))
    }

    /// Пул закрыт и состоит ровно из 25 меток — проверено по данным.
    func testLabelPoolShape() {
        XCTAssertEqual(ClassificationLabels.pool.count, 25)
        XCTAssertEqual(Set(ClassificationLabels.pool).count, 25, "дубликаты в пуле")
        for l in ["joy", "contempt", "anticipation", "bitterness"] {
            XCTAssertTrue(ClassificationLabels.pool.contains(l), "нет метки \(l)")
        }
    }

    // ── Загрузка ─────────────────────────────────────────────────────

    func testLoadsAllThreeTasks() throws {
        let s = try loadSlice()
        let byTask = Dictionary(grouping: s, by: \.task).mapValues(\.count)
        XCTAssertEqual(Set(byTask.keys), Set(EmbeddingTask.allCases),
                       "не все задачи представлены: \(byTask)")
        for (task, n) in byTask {
            XCTAssertGreaterThan(n, 0, "\(task) пуст")
        }
        XCTAssertTrue(s.allSatisfy { !$0.anchor.isEmpty && !$0.positive.isEmpty })
    }

    func testTaskFilterAndLimit() throws {
        let only = try loadSlice(tasks: [.retrieval])
        XCTAssertTrue(only.allSatisfy { $0.task == .retrieval })
        XCTAssertGreaterThan(only.count, 0)

        let limited = try loadSlice(limit: 7)
        XCTAssertEqual(limited.count, 7, "limit не соблюдён")
    }

    func testMissingFileReported() {
        XCTAssertThrowsError(try EmbeddingDataset.loadJSONL(path: "/nope.jsonl")) { e in
            guard case EmbeddingDataError.fileNotFound = e else {
                return XCTFail("ожидалась fileNotFound, получено \(e)")
            }
        }
    }

    /// В реальных данных ответ classification всегда внутри предъявленного
    /// набора. Если это перестанет быть так, обучение начнёт выбирать из
    /// множества, где верного варианта нет, — и лосс будет «работать»,
    /// обучая ерунде.
    func testClassificationTargetIsAlwaysAmongCandidates() throws {
        let s = try loadSlice(tasks: [.classification])
        XCTAssertGreaterThan(s.count, 0)
        for row in s {
            guard let cands = ClassificationLabels.parseCandidates(from: row.anchor) else {
                return XCTFail("не разобрались кандидаты: \(row.anchor.prefix(80))")
            }
            XCTAssertEqual(cands.count, 7, "ожидались 7 кандидатов, получено \(cands.count)")
            XCTAssertTrue(cands.contains(row.positive.trimmingCharacters(in: .whitespaces)),
                          "верная метка \(row.positive) вне набора \(cands)")
            XCTAssertTrue(cands.contains(row.negative.trimmingCharacters(in: .whitespaces)),
                          "hard-negative \(row.negative) вне набора")
            XCTAssertTrue(ClassificationLabels.pool.contains(
                            row.positive.trimmingCharacters(in: .whitespaces)),
                          "метка вне закрытого пула из 25")
        }
    }

    // ── Токенизация батча ────────────────────────────────────────────

    /// poolIndex обязан указывать на терминатор СВОЕЙ строки, а не на конец
    /// тензора: иначе вектор короткой строки читается с добивки.
    func testEncodeBatchPoolIndexPointsAtOwnTerminator() throws {
        let tok = try loadTokenizer()
        let (idx, pool) = encodeBatch(tokenizer: tok,
                                      texts: ["короткий", "существенно более длинный текст"],
                                      terminator: 0)
        eval(idx, pool)
        let poolVals = pool.asArray(Int32.self)
        XCTAssertLessThan(poolVals[0], poolVals[1], "короткая строка должна иметь меньший poolIndex")

        // на указанной позиции обязан стоять терминатор
        let ids = idx.asArray(Int32.self)
        let T = idx.shape[1]
        for b in 0 ..< 2 {
            XCTAssertEqual(ids[b * T + Int(poolVals[b])], 0,
                           "poolIndex строки \(b) указывает не на терминатор")
        }
    }

    func testEncodeBatchTruncatesToMaxTokens() throws {
        let tok = try loadTokenizer()
        let long = String(repeating: "слово ", count: 500)
        let (idx, pool) = encodeBatch(tokenizer: tok, texts: [long],
                                      terminator: 0, maxTokens: 32)
        eval(idx, pool)
        XCTAssertLessThanOrEqual(idx.shape[1], 32)
        XCTAssertEqual(pool.item(Int32.self), Int32(idx.shape[1] - 1))
    }

    // ── Батчеры ──────────────────────────────────────────────────────

    func testTripletBatcherShapesAndDeterminism() throws {
        let tok = try loadTokenizer()
        let samples = try loadSlice(tasks: [.retrieval])
        var a = TripletBatcher(samples: samples, tokenizer: tok, batchSize: 4,
                               maxTokens: 64, seed: 42)
        var b = TripletBatcher(samples: samples, tokenizer: tok, batchSize: 4,
                               maxTokens: 64, seed: 42)
        let x = a.next(), y = b.next()
        eval(x.anchorIdx, y.anchorIdx)
        XCTAssertEqual(x.batchSize, 4)
        XCTAssertEqual(x.anchorPool.shape, [4])
        XCTAssertEqual(x.anchorIdx.asArray(Int32.self), y.anchorIdx.asArray(Int32.self),
                       "одинаковый seed дал разные батчи")

        var c = TripletBatcher(samples: samples, tokenizer: tok, batchSize: 4,
                               maxTokens: 64, seed: 7)
        let z = c.next()
        eval(z.anchorIdx)
        XCTAssertNotEqual(x.anchorIdx.asArray(Int32.self), z.anchorIdx.asArray(Int32.self),
                          "разный seed дал одинаковые батчи — перемешивания нет")
    }

    /// Батчер обязан циклиться, а не падать, когда примеры кончились.
    func testTripletBatcherCyclesPastEnd() throws {
        let tok = try loadTokenizer()
        let samples = Array(try loadSlice(tasks: [.sts]).prefix(3))
        var b = TripletBatcher(samples: samples, tokenizer: tok, batchSize: 4,
                               maxTokens: 32, seed: 1)
        for _ in 0 ..< 5 {
            let batch = b.next()
            eval(batch.anchorIdx)
            XCTAssertEqual(batch.batchSize, 4)
        }
    }

    func testClassificationBatcherShapesAndMask() throws {
        let tok = try loadTokenizer()
        let samples = try loadSlice(tasks: [.classification])
        var b = ClassificationBatcher(samples: samples, tokenizer: tok,
                                      batchSize: 3, maxTokens: 64, seed: 5)
        XCTAssertGreaterThan(b.count, 0, "ни одной пригодной строки")
        let batch = b.next()
        eval(batch.candidateIdx, batch.mask, batch.targetIndex)

        XCTAssertEqual(batch.batchSize, 3)
        XCTAssertEqual(batch.candidateCount, 7, "в этих данных ровно 7 кандидатов")
        XCTAssertEqual(batch.candidateIdx.shape.prefix(2).map { $0 }, [3, 7])
        XCTAssertEqual(batch.candidatePool.shape, [3, 7])
        XCTAssertEqual(batch.mask.shape, [3, 7])

        // все кандидаты реальные ⇒ маска целиком из единиц
        XCTAssertEqual(batch.mask.sum().item(Float.self), 21)

        // цель обязана быть внутри диапазона кандидатов
        let t = batch.targetIndex.asArray(Int32.self)
        XCTAssertTrue(t.allSatisfy { $0 >= 0 && $0 < 7 }, "цель вне диапазона: \(t)")
    }

    /// Режим полного пула предъявляет все 25 меток — это другой, более
    /// трудный вопрос к модели, и он должен отражаться в форме батча.
    func testClassificationFullPoolMode() throws {
        let tok = try loadTokenizer()
        let samples = try loadSlice(tasks: [.classification])
        var b = ClassificationBatcher(samples: samples, tokenizer: tok,
                                      batchSize: 2, maxTokens: 64, seed: 5,
                                      useFullPool: true)
        let batch = b.next()
        eval(batch.candidateIdx, batch.mask)
        XCTAssertEqual(batch.candidateCount, 25)
        XCTAssertEqual(batch.mask.sum().item(Float.self), 50)
    }

    /// Маскировка добивки — на СИНТЕТИКЕ, и это не прихоть.
    ///
    /// В LitRetrieval у всех строк ровно 7 кандидатов, поэтому добивки не
    /// возникает никогда и маска честно состоит из единиц. Проверено
    /// мутацией: замена маски на единичную не роняет ни один тест на этих
    /// данных. То есть путь маскировки реальным корпусом не задействуется
    /// вовсе — а он сработает на любом другом наборе с разным числом меток.
    /// Поэтому здесь строки с РАЗНЫМ K собраны вручную.
    func testClassificationMaskMarksPaddingOnMixedCandidateCounts() throws {
        let tok = try loadTokenizer()
        let three = EmbeddingSample(
            anchor: "categories: joy, fear, shame\nQuery: текст",
            positive: "fear", negative: "joy", task: .classification)
        let five = EmbeddingSample(
            anchor: "categories: joy, fear, shame, guilt, pride\nQuery: текст",
            positive: "guilt", negative: "joy", task: .classification)

        var b = ClassificationBatcher(samples: [three, five], tokenizer: tok,
                                      batchSize: 2, maxTokens: 32,
                                      shuffle: false, seed: 0)
        let batch = b.next()
        eval(batch.mask, batch.targetIndex)

        XCTAssertEqual(batch.candidateCount, 5, "K должен добиться до максимума")
        let m = batch.mask.asArray(Float.self)
        XCTAssertEqual(Array(m[0 ..< 5]), [1, 1, 1, 0, 0],
                       "строка с тремя метками должна иметь две пад-позиции")
        XCTAssertEqual(Array(m[5 ..< 10]), [1, 1, 1, 1, 1],
                       "строка с пятью метками не должна маскироваться")
        XCTAssertEqual(batch.mask.sum().item(Float.self), 8)

        // цели указывают на РЕАЛЬНЫЕ позиции своих наборов
        let t = batch.targetIndex.asArray(Int32.self)
        XCTAssertEqual(t[0], 1, "fear — второй кандидат в наборе из трёх")
        XCTAssertEqual(t[1], 3, "guilt — четвёртый кандидат в наборе из пяти")
    }

    /// Строки, чей ответ не попадает в набор кандидатов, отбрасываются:
    /// цель обязана быть достижима.
    func testClassificationBatcherDropsUnreachableRows() throws {
        let tok = try loadTokenizer()
        let bad = EmbeddingSample(
            anchor: "categories: joy, fear\nQuery: текст",
            positive: "melancholy",           // нет среди кандидатов
            negative: "joy", task: .classification)
        let good = EmbeddingSample(
            anchor: "categories: joy, fear\nQuery: текст",
            positive: "fear", negative: "joy", task: .classification)
        let b = ClassificationBatcher(samples: [bad, good, bad],
                                      tokenizer: tok, batchSize: 1, seed: 0)
        XCTAssertEqual(b.count, 1, "непригодные строки не отброшены")
    }
}
