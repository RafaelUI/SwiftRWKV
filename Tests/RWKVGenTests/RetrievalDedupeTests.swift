//
//  RetrievalDedupeTests.swift
//  Оценка retrieval со слиянием одинаковых кандидатов (dedupeCandidates).
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVEmbedding

final class RetrievalDedupeTests: XCTestCase {

    func model() -> EmbeddingModel {
        let (bb, _) = TinyBackbone.make(nLayer: 2, vocab: 65536)
        let head = EmbeddingHead(dim: bb.cfg.nEmbd, seed: 5)
        MLXRandom.seed(105)
        head.fc2 = MLXRandom.normal([bb.cfg.nEmbd, head.hidden]) * 0.05
        eval(head.fc2)
        return EmbeddingModel(backbone: bb, head: head)
    }

    private func tok() throws -> WorldTokenizer {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет словаря World")
        return try XCTUnwrap(WorldTokenizer(vocabURL: URL(fileURLWithPath: path)))
    }

    /// Пять запросов на три различных документа: позитив повторяется, негатив
    /// одной строки — позитив другой. Со слиянием в пуле ровно различные
    /// тексты (3), ранги — по матрице над ними, верный индекс — индекс текста
    /// своего позитива; без слияния — прежний пул 2N.
    func testDedupeRanksAmongDistinctTexts() throws {
        let t = try tok(), m = model()
        let docs = ["book a flight reservation", "cancel an existing reservation", "look up user details"]
        let rows: [EmbeddingSample] = [
            .init(anchor: "please book me a flight", positive: docs[0], negative: docs[1], task: .retrieval),
            .init(anchor: "cancel my trip", positive: docs[1], negative: docs[0], task: .retrieval),
            .init(anchor: "my user id is 42", positive: docs[2], negative: docs[0], task: .retrieval),
            .init(anchor: "I want another flight", positive: docs[0], negative: docs[2], task: .retrieval),
            .init(anchor: "drop that booking", positive: docs[1], negative: docs[2], task: .retrieval),
        ]
        var sims: MLXArray? = nil
        EmbeddingMetrics.similarityObserver = { sims = $0 }
        defer { EmbeddingMetrics.similarityObserver = nil }

        let plain = EmbeddingMetrics.evaluateRetrieval(model: m, tokenizer: t, rows: rows)
        XCTAssertEqual(sims!.shape, [5, 10], "без слияния пул обязан остаться 2N")

        let ded = EmbeddingMetrics.evaluateRetrieval(model: m, tokenizer: t, rows: rows, dedupeCandidates: true)
        XCTAssertEqual(sims!.shape, [5, 3], "в пуле не различные тексты")
        let a = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: rows.map { $0.anchor }, role: .query)
        let c = EmbeddingMetrics.embedAll(model: m, tokenizer: t, texts: docs, role: .document)
        let ref = EmbeddingMetrics.rankingMetrics(
            similarities: matmul(a.asType(.float32), c.asType(.float32).transposed()),
            correctIndex: [0, 1, 2, 0, 1])
        XCTAssertEqual(ded, ref)
        XCTAssertEqual(ded.count, 5)
        _ = plain
    }

    /// Без повторов слияние ничего не меняет — метрики побитно те же.
    func testDedupeIsNoOpWithoutDuplicates() throws {
        let t = try tok(), m = model()
        let rows = (0 ..< 4).map { i in
            EmbeddingSample(anchor: "запрос \(i)", positive: "документ \(i)", negative: "чужой \(i)", task: .retrieval)
        }
        XCTAssertEqual(EmbeddingMetrics.evaluateRetrieval(model: m, tokenizer: t, rows: rows),
                       EmbeddingMetrics.evaluateRetrieval(model: m, tokenizer: t, rows: rows, dedupeCandidates: true))
    }

    /// Стадия передаёт выключатель в оценку.
    func testStagePassesDedupe() throws {
        let t = try tok(), m = model()
        let rows = (0 ..< 3).map { i in
            EmbeddingSample(anchor: "q\(i)", positive: "один и тот же документ", negative: "другой \(i)", task: .retrieval)
        }
        var stage = EmbeddingStage(task: .retrieval, train: rows, heldOut: rows,
                                   config: TrainingConfig(lr: 1e-3, maxSteps: 1), batchSize: 2)
        var width = 0
        EmbeddingMetrics.similarityObserver = { width = $0.shape[1] }
        defer { EmbeddingMetrics.similarityObserver = nil }
        _ = EmbeddingFinetune.evaluate(model: m, tokenizer: t, stage: stage)
        XCTAssertEqual(width, 6)
        stage.dedupeEvaluationPool = true
        _ = EmbeddingFinetune.evaluate(model: m, tokenizer: t, stage: stage)
        XCTAssertEqual(width, 4, "1 общий позитив + 3 негатива")
    }
}
