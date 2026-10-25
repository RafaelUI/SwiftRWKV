//
//  DocExamples.generated.swift
//  СГЕНЕРИРОВАН Scripts/check_docs.sh из docs/*.md — не править руками.
//
//  Файл существует ради одного: примеры в документации обязаны
//  собираться. Функции никто не вызывает, и это не упущение — проверка
//  здесь именно на тайпчек, потому что протухают у примеров имена и
//  сигнатуры, а не поведение.
//
import Foundation
import MLX
@testable import RWKVGen
@testable import RWKVQuant
@testable import RWKVEmbedding
@testable import RWKVRerank

func doc_Embedding_0() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    let embedder = Embedder(model: EmbeddingModel(backbone: base), tokenizer: tok)

    let v = embedder.embed("bees collect nectar from flowers")
    print(v.shape)                       // [768], L2-normalised

    let m = embedder.embed(["bees and honey", "Watt's steam engine"])
    let sim = cosineSimilarity(m)        // [2, 2]
    print(sim.shape)
}

func doc_Embedding_1() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    // Recommended: model and contract are both rebuilt FROM the file.
    let embedder = try Embedder.fromCheckpoint(
        backbone: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/embedding_head.safetensors"))
    print(embedder.contract.pooling, embedder.maxTokens as Any)
}

func doc_Embedding_2() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let rows = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl",
                                              limit: 2000,
                                              tasks: [.retrieval, .sts])
    print(rows.count, rows.first?.task as Any)
}

func doc_Embedding_3() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let sample = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl",
                                                limit: 2000, tasks: [.retrieval],
                                                sampleSeed: 1)
    print(sample.count)
}

func doc_Embedding_4() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let model = EmbeddingModel(backbone: base)

    // .frozen — head only; cheap, but the ceiling is low: the head only sees
    //           the pooled last layer and cannot change the representations.
    // .topLayers(N) — the top N backbone layers plus ln_out, and the head.
    // .full   — the whole backbone and the head.
    // .lora   — attached LoRA/QLoRA adapters and the head.
    // .state  — the initial state of the recurrence and the head; the base stays
    //           frozen (see state-tuning.md).
    let trainable = EmbeddingTrainable.make(model: model, mode: .topLayers(4))
    print(trainable.parameterNames.count)
}

func doc_Embedding_5() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let model = EmbeddingModel(backbone: base)
    let rows = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl",
                                              limit: 2000)
    let retrieval = rows.filter { $0.task == .retrieval }

    let stage = EmbeddingStage(
        task: .retrieval,
        train: Array(retrieval.dropLast(200)),
        heldOut: Array(retrieval.suffix(200)),
        config: TrainingConfig(lr: 2e-5, lrMin: 1e-6, schedule: .cosine,
                               warmupSteps: 100, maxSteps: 1500),
        batchSize: 8,
        gradCacheChunk: 0,       // >0 turns on GradCache for triplet tasks
        temperature: 0.05,
        maxTokens: 512)

    let results = try EmbeddingFinetune.run(
        model: model, tokenizer: tok, mode: .frozen, stages: [stage],
        onStage: { r in print(r.task, r.delta as Any) })
    print(results.count)

    try model.saveHead(to: URL(fileURLWithPath: "runs/embedding_head.safetensors"))
}

func doc_Embedding_6() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let model = EmbeddingModel(backbone: base)
    let rows = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl", limit: 2000)
    func stage(_ task: EmbeddingTask, steps: Int) -> EmbeddingStage {
        let r = rows.filter { $0.task == task }
        return EmbeddingStage(task: task, train: Array(r.dropLast(100)), heldOut: Array(r.suffix(100)),
                              config: TrainingConfig(lr: 2e-5, maxSteps: steps), batchSize: 8)
    }
    let report = try EmbeddingCurriculum.run(
        model: model, tokenizer: tok, mode: .topLayers(4),
        stages: [stage(.retrieval, steps: 600), stage(.sts, steps: 200)],
        directory: URL(fileURLWithPath: "runs/curriculum"),
        onStage: { i, r in print(i, r.after?.description ?? "—") })
    print(report.finished, report.stages.map { $0.after ?? .nan })
}

func doc_Embedding_7() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let (state, _) = try StateFile.load(
        URL(fileURLWithPath: "runs/query.state.safetensors"), base: base)
    base.initialState = state
    // The head's checkpoint records the scope it was trained with.
    let embedder = try Embedder.fromCheckpoint(
        backbone: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/embedding_head.safetensors"))
    let docs = embedder.embed(["Bees make honey from nectar."], role: .document)
    let query = embedder.embed("how is honey made", role: .query)
    print(cosineSimilarity(query.reshaped([1, -1]), docs).shape)
}

func doc_Embedding_8() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let model = EmbeddingModel(backbone: base)
    let cache = InstructionCache(model: model, tokenizer: tok)
    let instruct = "Instruct: Pick the tool that serves the user's request.\n"
    let queries = try cache.embedQueries([
        instruct + "Query: cancel my order",
        instruct + "Query: where is my parcel",
        "a query without an instruction",
    ])
    print(queries.shape, cache.foldedCount, cache.foldedTokens)   // [3, D] 1 …
}

func doc_Embedding_9() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    // The bank is for queries only: documents never see it.
    let model = EmbeddingModel(backbone: base, stateScope: .queriesOnly)
    let instructions = ["Instruct: Pick the airline tool for the request.\n",
                        "Instruct: Pick the banking tool for the request.\n"]
    // Start from the folded instructions: exactly the quality with them inline.
    let folded = InstructionCache(model: model, tokenizer: tok).bank(instructions)
    let rows = [
        EmbeddingSample(anchor: "Query: change my flight", positive: "change_flight: …",
                        negative: "get_balance: …", task: .retrieval, state: 0),
        EmbeddingSample(anchor: "Query: what is on my card", positive: "get_balance: …",
                        negative: "change_flight: …", task: .retrieval, state: 1),
    ]
    let bank = StateBankTrainableSet(base, initial: folded)
    let stage = EmbeddingStage(task: .retrieval, train: rows, heldOut: rows,
                               config: TrainingConfig(lr: 1e-4, maxSteps: 200),
                               batchSize: 4)
    let result = try EmbeddingFinetune.runStage(model: model, tokenizer: tok,
                                                trainable: bank, stage: stage)
    print(result.before as Any, result.after as Any)
    // Serving: a query with the row of its instruction.
    let query = EmbeddingMetrics.embedAll(model: model, tokenizer: tok,
                                          texts: ["Query: cancel my trip"],
                                          role: .query, states: [0])
    print(query.shape)
}

func doc_Embedding_10() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let model = EmbeddingModel(backbone: base)
    let rows = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl",
                                              limit: 500)

    let r = EmbeddingMetrics.evaluateRetrieval(
        model: model, tokenizer: tok,
        rows: rows.filter { $0.task == .retrieval })
    print(r.mrr, r.recall[1] as Any, r.ndcg10)

    let s = EmbeddingMetrics.evaluateSTS(
        model: model, tokenizer: tok, rows: rows.filter { $0.task == .sts })
    print(s.accuracy, s.meanSimilarityPositive, s.meanSimilarityNegative)

    let c = EmbeddingMetrics.evaluateClassification(
        model: model, tokenizer: tok,
        rows: rows.filter { $0.task == .classification },
        useFullPool: true)
    print(c.accuracy, c.candidatesPerRow, c.predictions.count)
}

func doc_Embedding_11() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let embedder = try Embedder.fromCheckpoint(
        backbone: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/embedding_head.safetensors"))

    let corpus = ["bees collect nectar", "steam engines and railways"]
    let vectors = embedder.embed(corpus)          // [N, D]
    let query = embedder.embed("how is honey made").reshaped([1, -1])
    let scores = cosineSimilarity(query, vectors) // [1, N]
    print(scores.shape)
}

func doc_Inference_0() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    let result = base.generate(prompt: "The capital of France is",
                               maxTokens: 32, tokenizer: tok)
    print(result.text, result.tokens.count, result.stopReason)
}

func doc_Inference_1() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")

    var state = RWKVState(cfg: base.cfg)
    var logits = base.prefill([1, 2, 3], state: &state)
    for _ in 0 ..< 32 {
        let id = logits.argMax().item(Int.self)
        logits = base.step(id, state: &state)
    }
}

func doc_Inference_2() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    let sampling = SamplingConfig(temperature: 1.0,
                                  topK: 0,              // 0 = off
                                  topP: 0.9,
                                  frequencyPenalty: 0.4,
                                  penaltyDecay: 0.996,
                                  seed: 1234)

    var state = RWKVState(cfg: base.cfg)
    let out = base.generate(prompt: tok.encode("Once upon a time"),
                            config: GenerationConfig(maxTokens: 200,
                                                     stopStrings: ["\n\n"],
                                                     stopTokens: [0],
                                                     sampling: sampling),
                            tokenizer: tok,
                            state: &state) { chunk, _ in
        print(chunk, terminator: "")
        return true          // return false to cancel
    }
    print(out.stopReason)
}

func doc_Inference_3() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let url = URL(fileURLWithPath: "chat.state.safetensors")

    var state = base.startState()
    _ = base.generate(prompt: tok.encode("User: hi\n\nAssistant:"),
                      config: GenerationConfig(maxTokens: 64, stopStrings: ["\n\n"]),
                      tokenizer: tok, state: &state)
    try StateFile.saveSnapshot(state, base: base, to: url, note: "chat 42")

    // later, possibly in another process
    var resumed = try StateFile.loadSnapshot(url, base: base).state
    let next = base.generate(prompt: tok.encode("\n\nUser: and then?\n\nAssistant:"),
                             config: GenerationConfig(maxTokens: 64, stopStrings: ["\n\n"]),
                             tokenizer: tok, state: &resumed)
    print(next.text)
}

func doc_Inference_4() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let results = base.generateBatch(prompts: ["The capital of France is", "Two plus two is"],
                                     config: GenerationConfig(maxTokens: 32, stopTokens: [0]),
                                     tokenizer: tok)
    for r in results { print(r.text, r.stopReason) }
}

func doc_Inference_5() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    // The text-feeding contract is read FROM the checkpoint, never from defaults.
    let inf = try RerankerInference.fromCheckpoint(
        base: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/reranker_head.safetensors"))

    let docs = ["a passage about bees", "a passage about steam engines"]
    for (i, score) in try inf.rank(query: "how do bees make honey", docs: docs) {
        print(docs[i], score)
    }
}

func doc_Inference_6() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    let embedder = try Embedder.fromCheckpoint(
        backbone: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/embedding_head.safetensors"))

    let corpus = ["bees collect nectar", "steam engines and railways"]
    let vectors = embedder.embed(corpus)
    let query = embedder.embed("how is honey made").reshaped([1, -1])
    print(cosineSimilarity(query, vectors).shape)
}

func doc_Inference_7() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let model = try X070Backbone.load(fromRwkvq: "~/models/rwkv7-g1d-0.1b-reduction.rwkvq")
    var st = RWKVState(cfg: model.cfg)
    print(model.prefill([1, 2, 3], state: &st).shape)
}

func doc_Inference_8() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    // `open` takes a .rwkvq or a sidecar path (without the suffix)
    let sidecar = try RwkvqSidecar.open("~/models/0.1B_reduction.rwkvq_mlx")

    let info = base.attachRwkvq(sidecar)
    print(info.attached,            // tensors now backed by the sidecar
          info.missing,             // wanted but absent — should be empty
          info.freedDenseBytes,     // dense copies dropped
          info.packedBytes)         // what the sidecar holds instead

    // Everything downstream is unchanged: decode, reranker, embeddings.
    var state = RWKVState(cfg: base.cfg)
    let logits = base.prefill([1, 2, 3], state: &state)
    print(logits.shape)

    base.detachRwkvq()   // only meaningful with dropDenseWeights: false
}

func doc_Inference_9() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    var opts = X070Backbone.RwkvqAttachOptions()
    opts.quantizeCmix = true        // also cmix key/value
    opts.quantizeEmbedding = true   // see the caveat below
    opts.layers = 4 ..< 12          // leave the early layers dense
    opts.dropDenseWeights = false   // keep dense copies so detach can restore
    opts.useNativeKernel = true     // run through MLX's quantised matmul, see below
    print(base.attachRwkvq(try RwkvqSidecar(path: "~/models/q.rwkvq_mlx"),
                           options: opts).attached)
}

func doc_Train_0() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let trainable = BackboneWeightsTrainableSet(base, keys: ["ln_out.weight"])

    let config = TrainingConfig(
        lr: 1e-4,
        lrMin: 1e-6,
        schedule: .cosine,        // .constant | .cosine | .linear
        warmupSteps: 100,
        gradClip: 1.0,            // global norm; <= 0 disables
        weightDecay: 0.01,
        maxSteps: 1000,
        gradAccum: 1,             // effective batch = batch × gradAccum
        cacheLimitGB: 2.0,        // Metal buffer-cache ceiling; <= 0 leaves it alone
        logEvery: 10)

    let trainer = Trainer<(x: MLXArray, y: MLXArray)>(
        trainable: trainable,
        objective: { batch in
            // Return a scalar. Anything differentiable w.r.t. the injected
            // parameters will be trained.
            (base.body(batch.x).sum() * 0).sum()
        },
        nextBatch: { (x: MLXArray.zeros([1, 16], dtype: .int32),
                      y: MLXArray.zeros([1, 16], dtype: .int32)) },
        config: config)

    let result = trainer.run(onStep: { step in
        print(step.step, step.loss, step.gradNorm, step.learningRate,
              step.peakMemoryMB)
    })
    print(result.finalLoss, result.steps)
}

func doc_Train_1() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let trainer = Trainer<Int>(
        trainable: BackboneWeightsTrainableSet(base, keys: ["ln_out.weight"]),
        objective: { _ in base.weight("ln_out.weight").sum() },
        nextBatch: { 0 },
        config: TrainingConfig(maxSteps: 10))
    _ = trainer.run()

    let url = URL(fileURLWithPath: "runs/step_1000.safetensors")
    try trainer.saveCheckpoint(to: url)
    try trainer.loadCheckpoint(from: url)
    print(trainer.completedSteps, trainer.currentParameters.count)
}

func doc_Train_2() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")

    var spec = LoRASpec()
    spec.rank = 16
    spec.alpha = 32                          // scale = alpha / rank
    spec.tmixTargets = LoRATargets.tmix      // r_proj, k_proj, v_proj, o_proj
    spec.cmixTargets = LoRATargets.cmix      // key, value
    spec.layers = nil                        // nil ⇒ every block
    spec.quantizeBits = 0                    // 4 or 8 ⇒ QLoRA on those targets
    spec.quantGroupSize = 64

    let info = LoRA.add(to: base, spec: spec)
    print(info.numAdapters, info.trainablePct, base.loraTargets.count)
}

func doc_Train_3() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    _ = LoRA.add(to: base, spec: LoRASpec())
    base.trainLayers = []      // inference: no chunk-alignment requirement
    var state = RWKVState(cfg: base.cfg)
    print(base.prefill([1, 2, 3], state: &state).shape)
}

func doc_Train_4() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    _ = LoRA.add(to: base, spec: LoRASpec())

    var cfg = LoRAConfig()
    cfg.lr = 1e-4
    cfg.maxSteps = 500
    cfg.warmupSteps = 50
    cfg.gradAccum = 4
    cfg.useBlockCheckpoint = true    // trade compute for activation memory
    cfg.cacheLimitGB = 2.0

    let result = LoRAFinetune.run(
        base,
        nextBatch: { (x: MLXArray.zeros([2, 64], dtype: .int32),
                      y: MLXArray.zeros([2, 64], dtype: .int32)) },
        config: cfg,
        onStep: { step, loss, gradNorm, peakMB in
            print(step, loss, gradNorm, peakMB)
        })
    print(result.finalLoss)
}

func doc_Train_5() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    _ = LoRA.add(to: base, spec: LoRASpec())
    let url = URL(fileURLWithPath: "runs/adapters.safetensors")
    try LoRA.save(base, to: url)
    try LoRA.load(base, from: url)
    print(LoRA.adapterState(base).count)
}

func doc_Train_6() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let model = try X070Backbone.load(fromRwkvq: "~/models/rwkv7-g1d-0.1b-reduction.rwkvq")
    let added = LoRA.add(to: model, spec: LoRASpec(rank: 16, alpha: 32))
    print(added.numAdapters)        // every tmix target, quantised or dense
    // ... train as above ...
    LoRA.merge(model)               // merged targets become dense weights
}

func doc_Train_7() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let ids = MLXArray((0 ..< 65).map { Int32($0 % 60000) }, [1, 65])
    let batch: LoRABatch = (x: ids[0..., 0 ..< 64], y: ids[0..., 1 ... 64])   // T % 16 == 0
    let (_, state) = StateTuning.run(base, nextBatch: { batch }, config: StateTuneConfig())
    base.initialState = state                // body / prefill / embedder start from it
}

func doc_Train_8() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let examples = [X070Example(ids: [1, 2, 3], label: 0),
                    X070Example(ids: [4, 5, 6], label: 1)]

    // The frozen prefix is computed ONCE and cached on disk: with `freeze` layers
    // never changing, their output for a given example never changes either.
    let train = X070PartialFinetune.buildBoundaryCache(
        backbone: base, examples: examples, freeze: 8, name: "train", ctxLen: 128)
    let val = X070PartialFinetune.buildBoundaryCache(
        backbone: base, examples: examples, freeze: 8, name: "val", ctxLen: 128)

    let result = X070PartialFinetune.train(
        backbone: base, trainCache: train, valCache: val, cfg: base.cfg,
        numClasses: 2, freeze: 8, epochs: 5, batchSize: 8, lr: 1e-4, seed: 0,
        onEpoch: { epoch, acc in print(epoch, acc) })
    print(result.valAcc, result.params.count)

    X070PartialFinetune.clearCache()
}

func doc_Train_9() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    var cfg = PretrainConfig()
    cfg.nLayer = 12
    cfg.nEmbd = 768
    cfg.headSize = 64
    cfg.vocab = 65536
    cfg.trainData = "~/data/train.bin"
    cfg.valData = "~/data/val.bin"
    cfg.ctxLen = 512                 // must be a multiple of 16
    cfg.batchSize = 8
    cfg.maxSteps = 10_000
    cfg.lr = 6e-4
    cfg.warmupSteps = 500
    cfg.schedule = .cosine
    cfg.gradAccum = 4
    cfg.useBlockCheckpoint = true
    cfg.checkpointDir = "runs/pretrain"
    cfg.saveEvery = 1000
    cfg.resume = true
    cfg.evalEvery = 500

    print(cfg.modelConfig.nLayer, cfg.resolvedMaxSteps())
}

func doc_Train_10() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let cfg = X070Config(nLayer: 2, nEmbd: 128, headSize: 64, vocab: 1024)
    var ic = X070InitConfig()
    ic.seed = 0
    let weights = X070Init.weights(cfg: cfg, init: ic)
    let backbone = X070Init.makeBackbone(cfg: cfg, init: ic)
    print(weights.count, backbone.cfg.nLayer)
}

func doc_Train_11() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let stream = try BinTokenStream(path: "~/data/train.bin", ctxLen: 512)
    print(stream.count, stream.ctxLen)

    // Out-of-range ids are caught by sampling rather than by a full scan.
    try stream.validateOrThrow(vocabSize: 65536)

    let batch = stream.batch(batchSize: 8, step: 0)
    print(batch.x.shape, batch.y.shape)

    let source = stream.source(batchSize: 8, startStep: 0)
    print(source().x.shape)
}

func doc_Train_12() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let url = URL(fileURLWithPath: "/tmp/tokens.bin")
    try TokenStreamWriter.write(tokens: [1, 2, 3, 4], to: url)
    print(url.lastPathComponent)
}

func doc_reranker_0() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    // The text-feeding contract comes FROM the checkpoint, not from defaults.
    let inf = try RerankerInference.fromCheckpoint(
        base: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/rerank/reranker_head.safetensors"))

    let docs = ["a passage about bees", "a passage about steam engines"]
    for (i, score) in try inf.rank(query: "how do bees make honey", docs: docs) {
        print(docs[i], score)
    }
}

func doc_reranker_1() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let model = try Reranker(base: base, cfg: RerankerConfig(
        layerIdx: [5],       // which backbone layers to read; negatives count back
        sharedState: false,  // all blocks read the state of the LAST layerIdx entry
        nProbe: 1,           // number of probe tokens
        headHidden: nil))    // MLP width; nil ⇒ D
    print(model.head.uniqueSources, model.head.parameterCount)
}

func doc_reranker_2() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let head = URL(fileURLWithPath: "runs/rerank/reranker_head.safetensors")

    // Correct: values come from the file.
    let good = try RerankerInference.fromCheckpoint(base: base, tokenizer: tok,
                                                    head: head)

    // Overriding is possible, but it has to be written out explicitly.
    let tuned = try RerankerInference.fromCheckpoint(
        base: base, tokenizer: tok, head: head,
        overrides: { $0.encode.docBatch = 16 })

    print(good.config.contract, tuned.config.encode.docBatch)
}

func doc_reranker_3() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }

    // 1. Data. Reservoir sampling in a single pass — the file may be gigabytes.
    let rows = try RerankDataset.loadJSONL(path: "~/data/train.jsonl",
                                           limit: 300, seed: 0)

    // 2. Candidates: positive + mined negatives + top-up from the shared pool.
    //    The position of the positive is SHUFFLED: with a listwise loss a fixed
    //    position is a shortcut the head will learn instead of the task.
    let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 8, seed: 0)
    let (trainSamples, evalSamples) =
        RerankCandidates.splitTrainEval(samples, nEval: 80, seed: 0)

    // 3. State cache. `path:` writes straight to disk — the cache need not fit
    //    in memory, it is read by pages.
    let model = try Reranker(base: base, cfg: RerankerConfig(layerIdx: [5]))
    let encCfg = RerankEncodeConfig(maxDocTokens: 384, maxQueryTokens: 96)
    let trainCache = try RerankEncoder.encodePairs(
        model, tokenizer: tok, pool: pool, samples: trainSamples,
        config: encCfg, path: URL(fileURLWithPath: "runs/cache_train"))
    let evalCache = try RerankEncoder.encodePairs(
        model, tokenizer: tok, pool: pool, samples: evalSamples,
        config: encCfg, path: URL(fileURLWithPath: "runs/cache_eval"))

    // 4. Training. The contract is stated EXPLICITLY: when reusing a ready-made
    //    cache the caller must be told about a mismatch rather than silently
    //    inherit someone else's truncations.
    let result = try RerankTraining.train(
        model, trainCache: trainCache, evalCache: evalCache,
        config: RerankTrainConfig(lr: 2e-4, batchSize: 32, epochs: 8,
                                  loss: .listwise, keepBest: true, seed: 0),
        contract: encCfg.contract)

    // The cheapest check in the whole pipeline.
    precondition(abs(result.firstLoss - result.expectedFirstLoss) < 1e-4,
                 "starting loss is not ln(C) — the wiring is broken, not the LR")

    // 5. Checkpoint. The contract is taken from the RESULT, not from our own
    //    configuration: they coincide only when the cache was built by this run.
    try model.saveHead(to: URL(fileURLWithPath: "runs/head.safetensors"),
                       extra: result.contract)
    print(result.after?.summary ?? "no evaluation was run")
}

func doc_reranker_4() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let model = try Reranker(base: base, cfg: RerankerConfig(layerIdx: [5]))
    let trainCache = try StateCache.load(URL(fileURLWithPath: "runs/cache_train"))
    let evalCache = try StateCache.load(URL(fileURLWithPath: "runs/cache_eval"))
    let result = try RerankTraining.train(
        model, trainCache: trainCache, evalCache: evalCache,
        config: RerankTrainConfig(epochs: 8),
        checkpoint: URL(fileURLWithPath: "runs/rerank.ckpt.safetensors"))
    print(result.completed, result.epochs.count)
}

func doc_reranker_5() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let cache = try StateCache.load(URL(fileURLWithPath: "runs/cache_eval"))

    // From a checkpoint: the contract is read FROM THE FILE and checked
    // against the cache.
    let (metrics, contract) = try RerankTraining.evaluate(
        base: base, head: URL(fileURLWithPath: "runs/head.safetensors"),
        cache: cache)
    print(contract["max_doc_tokens"] ?? "?", metrics.summary)
}

func doc_reranker_6() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let rows = try RerankDataset.loadJSONL(path: "~/data/train.jsonl", limit: 300)
    let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 8)
    let (tr, ev) = RerankCandidates.splitTrainEval(samples, nEval: 80)
    let probe = try Reranker(base: base, cfg: RerankerConfig(layerIdx: [5]))

    // Encode BOTH layers once.
    let trainCache = try RerankEncoder.encodePairs(
        probe, tokenizer: tok, pool: pool, samples: tr, sources: [5, 11])
    let evalCache = try RerankEncoder.encodePairs(
        probe, tokenizer: tok, pool: pool, samples: ev, sources: [5, 11])

    let sweep = try RerankSweep.run(
        base: base,
        configs: [RerankerConfig(layerIdx: [5]), RerankerConfig(layerIdx: [11])],
        trainCache: trainCache, evalCache: evalCache,
        config: RerankTrainConfig(batchSize: 32, epochs: 8),
        seeds: [0, 1, 2])
    print(sweep.summary)
}

func doc_reranker_7() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let withLayer5 = try StateCache.load(URL(fileURLWithPath: "runs/cache_l5"))
    let withLayer11 = try StateCache.load(URL(fileURLWithPath: "runs/cache_l11"))
    let both = try withLayer5.merged(with: withLayer11,
                                     to: URL(fileURLWithPath: "runs/cache_l5_l11"))
    print(both.sources ?? [])   // [5, 11]
}

func doc_reranker_8() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let inf = try RerankerInference.fromCheckpoint(
        base: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/head.safetensors"))

    let docs = ["a passage about bees", "a passage about steam engines"]
    let index = try inf.buildIndex(docs: docs)
    print(index.count, index.nbytes)

    // The index survives a restart: states and TEXTS live in one file.
    let url = URL(fileURLWithPath: "runs/docs.index")
    try index.save(to: url)
    let reloaded = try DocIndex.load(url)

    for (i, score) in try inf.rankIndexed(query: "how do bees make honey",
                                          index: reloaded, topK: 10) {
        print(reloaded.docs[i], score)
    }
}

func doc_reranker_9() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        // Geometry is derived from weight shapes. There is simply no
        // configuration here that could be specified incorrectly.
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    guard let tok = WorldTokenizer(
            vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
        fatalError("cannot read vocabulary")
    }
    let inf = try RerankerInference.fromCheckpoint(
        base: base, tokenizer: tok,
        head: URL(fileURLWithPath: "runs/head.safetensors"))

    let index = try inf.buildIndexToDisk(
        docs: ["a passage about bees", "a passage about steam engines"],
        at: URL(fileURLWithPath: "runs/docs.index"),
        progress: { done, total in print(done, "/", total) })

    // Many queries against one index in a single pass. Tails are batched across
    // QUERIES, not just across candidates: the one partial batch is then one for
    // the whole job rather than one per query.
    let scores = try inf.scoreIndexedBatch(
        queries: ["how do bees make honey", "who invented the steam engine"],
        index: index)
    print(scores.count)
}

func doc_state_tuning_0() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let ids = MLXArray((0 ..< 65).map { Int32($0 % 60000) }, [1, 65])
    let batch: LoRABatch = (x: ids[0..., 0 ..< 64], y: ids[0..., 1 ... 64])   // T % 16 == 0

    var config = StateTuneConfig()           // lr 3e-4 -> 3e-5 (cosine), WKV only
    config.maxSteps = 500
    let (result, state) = StateTuning.run(base, nextBatch: { batch }, config: config)
    print(result.finalLoss, state.wkv.shape)

    // its own file, with the fingerprint of the base it was trained on
    let url = URL(fileURLWithPath: "/tmp/task.state.safetensors")
    try StateFile.save(state, base: base, to: url, note: "world 0.1b")
}

func doc_state_tuning_1() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try X070Backbone.loadStreaming(safetensors: URL(fileURLWithPath: "/tmp/g1j_1.5b.safetensors"))
    let ids = MLXArray((0 ..< 33).map { Int32($0) }, [1, 33])
    let batch: LoRABatch = (x: ids[0..., 0 ..< 32], y: ids[0..., 1 ... 32])   // T % 16 == 0
    let (_, state) = StateTuning.run(base, nextBatch: { batch }, config: StateTuneConfig())
    print(state.wkv.shape, base.isWeightStreaming)
}

func doc_state_tuning_2() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try X070Backbone.load(fromRwkvq: "/tmp/1p5b_reduction.rwkvq")
    try base.enableWeightStreaming(cache: URL(fileURLWithPath: "/tmp/1p5b_reduction.spill.safetensors"))
    print(base.isWeightStreaming)
}

func doc_state_tuning_3() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let ids = MLXArray((0 ..< 130).map { Int32($0 % 60000) }, [2, 65])
    let train: LoRABatch = (x: ids[0 ..< 1, 0 ..< 64], y: ids[0 ..< 1, 1 ... 64])
    let held: LoRABatch = (x: ids[1 ..< 2, 0 ..< 64], y: ids[1 ..< 2, 1 ... 64])

    var config = StateTuneConfig()
    config.maxSteps = 500
    config.evalEvery = 25            // held-out loss before training and every 25 steps
    config.patience = 3              // stop after 3 checks in a row without improvement
    let (result, state) = StateTuning.run(base, nextBatch: { train }, config: config, heldOut: [held],
                                          onEval: { step, loss, best in print(step, loss, best) })
    print(result.heldOut!.bestStep, result.heldOut!.stoppedEarly, state.wkv.shape)
}

func doc_state_tuning_4() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let url = URL(fileURLWithPath: "/tmp/task.state.safetensors")
    let (loaded, info) = try StateFile.load(url, base: base)   // checks the base
    base.initialState = loaded               // body / prefill / embedder start from it
    print(info.kind, info.trained)

    let cached = base.foldPrompt([510, 261, 0])                 // attached -> prompt -> cache
    let x = MLXArray((0 ..< 16).map { Int32($0 + 100) }, [1, 16])
    _ = base.bodyWithState(x, state: cached)                    // an explicit state wins
    var decode = base.startState()                              // step-by-step decoding from it
    _ = decode
    base.initialState = nil                                     // detach
}

func doc_state_tuning_5() throws {
    func loadBackbone(_ path: String) throws -> X070Backbone {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let weights = try loadArrays(url: url)
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return X070Backbone(weights: weights, cfg: cfg)
    }
    let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
    let (state, _) = try StateFile.load(URL(fileURLWithPath: "/tmp/task.state.safetensors"), base: base)
    try StateFile.exportPEFT(state, to: URL(fileURLWithPath: "/tmp/task.pth"))
    let (back, ignored) = try StateFile.importPEFT(URL(fileURLWithPath: "/tmp/task.pth"), base: base)
    print(back.wkv.shape, ignored)          // keys other than blocks.{i}.att.time_state
}
