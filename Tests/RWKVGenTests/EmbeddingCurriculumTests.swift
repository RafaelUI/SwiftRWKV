//
//  EmbeddingCurriculumTests.swift
//  Прогонщик curriculum эмбеддера (`EmbeddingCurriculum`, 17.10): отчёт на
//  диске после каждой стадии, возобновление после обрыва.
//
//  Эталон — непрерывный прогон: оборванный посреди стадии и продолженный
//  НОВОЙ моделью (как после падения процесса) обязан дать побитно те же
//  параметры и те же метрики. И сам прогонщик чисел не меняет: без обрыва он
//  равен `EmbeddingFinetune.run`.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding

final class EmbeddingCurriculumTests: XCTestCase {

    func makeModel() -> EmbeddingModel {
        let (bb, _) = TinyBackbone.make(nLayer: 2, vocab: 65536)
        let head = EmbeddingHead(dim: bb.cfg.nEmbd, seed: 3)
        MLXRandom.seed(103)
        head.fc2 = MLXRandom.normal([bb.cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        return EmbeddingModel(backbone: bb, head: head)
    }

    func rows(_ task: EmbeddingTask, _ n: Int, offset: Int = 0) -> [EmbeddingSample] {
        (0 ..< n).map { j in
            let i = j + offset
            return EmbeddingSample(anchor: "вопрос \(i) про пчёл и мёд", positive: "ответ \(i): пчёлы делают мёд из нектара",
                                   negative: "паровая машина номер \(i)", task: task)
        }
    }

    func stages(steps: Int = 3) -> [EmbeddingStage] {
        let plan: [(EmbeddingTask, Int)] = [(.retrieval, 0), (.sts, 1), (.retrieval, 2)]
        return plan.map { task, k in
            var cfg = TrainingConfig(lr: 1e-2, maxSteps: steps)
            cfg.logEvery = 1
            return EmbeddingStage(task: task, train: rows(task, 6, offset: 10 * k),
                                  heldOut: rows(task, 4, offset: 100 + 10 * k),
                                  config: cfg, batchSize: 2, maxTokens: 48, maxChars: 200,
                                  seed: UInt64(k))
        }
    }

    func tmpDir() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("curr_\(UUID().uuidString)")
    }

    func tensors(_ url: URL) throws -> [String: MLXArray] { try loadArrays(url: url) }

    func sameBits(_ a: [String: MLXArray], _ b: [String: MLXArray]) -> Bool {
        a.keys.sorted() == b.keys.sorted() && a.allSatisfy { k, x in
            let y = b[k]!
            return x.shape == y.shape && MLX.abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self) == 0
        }
    }

    /// Отчёт без времени (время от прогона к прогону своё).
    func numbers(_ r: EmbeddingCurriculumReport) -> [[String]] {
        r.stages.map { ["\($0.index)", $0.task, "\($0.steps)", "\($0.finalLoss)", "\($0.before ?? .nan)",
                        "\($0.after ?? .nan)", $0.beforeText ?? "", $0.afterText ?? ""] }
    }

    /// Обрыв посреди второй стадии, продолжение новой моделью — побитно как
    /// без обрыва; во всех судьбах обучаемого множества.
    func testResumeAfterCrashIsBitIdentical() throws {
        let tok = try worldTokenizer()
        for mode in [BaseTrainingMode.frozen, .topLayers(1), .state(learningRateScale: 10)] {
            let ref = tmpDir(), cut = tmpDir()
            defer { try? FileManager.default.removeItem(at: ref); try? FileManager.default.removeItem(at: cut) }
            let r0 = try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: mode,
                                                 stages: stages(), directory: ref)
            XCTAssertTrue(r0.finished)

            var stop = false
            let r1 = try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: mode,
                                                 stages: stages(), directory: cut,
                                                 isCancelled: { stop },
                                                 onStep: { i, s in if i == 1 && s.step >= 2 { stop = true } })
            XCTAssertEqual(r1.stages.count, 1, "\(mode): оборванная стадия записана")
            XCTAssertFalse(r1.finished)
            XCTAssertFalse(FileManager.default.fileExists(
                atPath: EmbeddingCurriculum.paramsURL(cut, 1).path))

            // Второй обрыв — посреди третьей стадии: продолжение обязано взять
            // параметры ВТОРОЙ, а не первой.
            var resumedAt = -1
            var ran: [Int] = []
            stop = false
            let r15 = try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: mode,
                                                  stages: stages(), directory: cut,
                                                  isCancelled: { stop },
                                                  onResume: { resumedAt = $0 },
                                                  onStage: { i, _ in ran.append(i) },
                                                  onStep: { i, s in if i == 2 && s.step >= 2 { stop = true } })
            XCTAssertEqual(resumedAt, 1)
            XCTAssertEqual(ran, [1])
            XCTAssertEqual(r15.stages.count, 2)
            ran = []
            let r2 = try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: mode,
                                                 stages: stages(), directory: cut,
                                                 onResume: { resumedAt = $0 },
                                                 onStage: { i, _ in ran.append(i) })
            XCTAssertEqual(resumedAt, 2)
            XCTAssertEqual(ran, [2], "\(mode): продолжение прогнало не те стадии")
            XCTAssertTrue(r2.finished)
            XCTAssertEqual(numbers(r2), numbers(r0), "\(mode): метрики продолжения не те")
            for i in 0 ..< 3 {
                XCTAssertTrue(sameBits(try tensors(EmbeddingCurriculum.paramsURL(ref, i)),
                                       try tensors(EmbeddingCurriculum.paramsURL(cut, i))),
                              "\(mode): параметры после стадии \(i) не те")
            }
            // Отчёт на диске — тот же, что вернулся.
            XCTAssertEqual(try EmbeddingCurriculum.report(in: cut), r2)
        }
    }

    /// Без обрыва прогонщик — это `EmbeddingFinetune.run`: те же числа.
    func testRunnerEqualsLibraryCurriculum() throws {
        let tok = try worldTokenizer()
        let dir = tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let a = makeModel(), b = makeModel()
        let lib = try EmbeddingFinetune.run(model: a, tokenizer: tok, mode: .topLayers(1), stages: stages())
        let rep = try EmbeddingCurriculum.run(model: b, tokenizer: tok, mode: .topLayers(1), stages: stages(),
                                              directory: dir)
        XCTAssertEqual(lib.map { $0.after?.headline }, rep.stages.map { $0.after })
        XCTAssertEqual(lib.map { Double($0.finalLoss) }, rep.stages.map { $0.finalLoss })
        XCTAssertTrue(sameBits(Dictionary(uniqueKeysWithValues: zip(EmbeddingHead.parameterNames, a.head.parameters)),
                               Dictionary(uniqueKeysWithValues: zip(EmbeddingHead.parameterNames, b.head.parameters))))
        let top = BackboneWeightsTrainableSet.topLayerKeys(a.backbone, from: 1)
        XCTAssertFalse(top.isEmpty)
        XCTAssertTrue(sameBits(Dictionary(uniqueKeysWithValues: top.map { ($0, a.backbone.w[$0]!) }),
                               Dictionary(uniqueKeysWithValues: top.map { ($0, b.backbone.w[$0]!) })))
    }

    /// Отчёт пишется после КАЖДОЙ стадии — к моменту onStage он уже на диске.
    func testReportIsOnDiskAfterEachStage() throws {
        let tok = try worldTokenizer()
        let dir = tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        var seen: [Int] = []
        try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .frozen, stages: stages(),
                                    directory: dir, onStage: { i, _ in
            seen.append((try? EmbeddingCurriculum.report(in: dir))??.stages.count ?? -1)
            XCTAssertTrue(FileManager.default.fileExists(atPath: EmbeddingCurriculum.paramsURL(dir, i).path))
        })
        XCTAssertEqual(seen, [1, 2, 3])
    }

    /// Другой план в ту же папку — отказ; тот же законченный — ни шага, тот же отчёт.
    func testOtherPlanIsRefused() throws {
        let tok = try worldTokenizer()
        let dir = tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let r = try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .frozen,
                                            stages: stages(), directory: dir)
        XCTAssertThrowsError(try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .frozen,
                                                         stages: stages(steps: 4), directory: dir)) {
            XCTAssertTrue("\($0)".contains("другого плана"), "\($0)")
        }
        func planMismatch(_ e: Error) -> Bool {
            if case EmbeddingCurriculumError.planMismatch = e { return true }
            return false
        }
        XCTAssertThrowsError(try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .topLayers(1),
                                                         stages: stages(), directory: dir)) {
            XCTAssertTrue(planMismatch($0), "\($0)")
        }
        // Режим с тем же составом параметров и другим множителем шага.
        let sdir = tmpDir()
        defer { try? FileManager.default.removeItem(at: sdir) }
        try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .state(learningRateScale: 10),
                                    stages: stages(), directory: sdir)
        XCTAssertThrowsError(try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok,
                                                         mode: .state(learningRateScale: 20),
                                                         stages: stages(), directory: sdir)) {
            XCTAssertTrue(planMismatch($0), "\($0)")
        }
        var other = stages()
        other[1].train[0] = EmbeddingSample(anchor: "другое", positive: "другое", negative: "иное", task: .sts)
        XCTAssertThrowsError(try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .frozen,
                                                         stages: other, directory: dir), "другие данные приняты")
        var steps = 0
        let again = try EmbeddingCurriculum.run(model: makeModel(), tokenizer: tok, mode: .frozen,
                                                stages: stages(), directory: dir, onStep: { _, _ in steps += 1 })
        XCTAssertEqual(steps, 0)
        XCTAssertEqual(again, r)
    }

    private func worldTokenizer() throws -> WorldTokenizer {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = ProcessInfo.processInfo.environment["RWKV_WORLD_VOCAB"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет словаря World — тест пропущен")
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: path)) else {
            throw XCTSkip("словарь World не разобрался")
        }
        return tok
    }
}
