# Embedding

Text vectors: a string in, one L2-normalised vector out. Vectors are compared
by cosine, so the module covers first-stage retrieval, semantic similarity and
zero-shot classification.

A frozen or fine-tuned RWKV backbone produces hidden states, a pooling step
collapses them to one vector per text, and a small residual head reshapes the
geometry. Only the head is required to be trained; the backbone may be frozen,
partially trained, fully trained or adapted with LoRA.

For second-stage reranking see [reranker.md](reranker.md). The two are
independent: the reranker does not compute vectors at all.

**About the numbers.** Everything measured below was taken on RWKV-7 World
0.1B (L=12, D=768) on Apple silicon unless stated otherwise. Code examples are
compiled by `Scripts/check_docs.sh`; it does **not** re-verify the numbers.

---

## Quick start

```swift
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
```

The backbone loader used by the examples below:

```swift-helper
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
```

### A raw backbone is not an embedder

This matters more than anything else on this page. Measured on a
LitRetrieval slice, 40 rows per task, **untrained** head:

| | raw base |
|---|---|
| retrieval MRR / R@1 | 0.0611 / 0.0000 |
| STS accuracy | 0.4750 |
| STS mean cosine, positive / negative | 0.8619 / **0.8723** |
| classification accuracy (pool of 25) | 0.0000 |
| cosine between *different* documents | 0.79…0.998, mean **0.9127** |

The positive mean cosine is **lower** than the negative one: separation is
worse than chance. And every document sits in a narrow cone — that is
anisotropy, and it is what contrastive fine-tuning exists to fix.

So: fine-tune before using this for anything. The numbers above are the
"before" column, measured, not assumed.

---

## The text-feeding contract

A vector is a function of exactly the text that was fed and of how it was
folded. Three things change it, and all three diverge **silently** — the
vector is normalised in every case, so no shape error occurs anywhere:

| | what it does |
|---|---|
| `pooling` | `.last` takes the vector at the final real token, `.mean` averages over real tokens. Different geometry entirely, not a detail at the edges |
| `terminator` | a token appended before pooling; `0` is reserved in the World vocabulary and carries no meaning of its own |
| `maxTokens` | input truncation. A long text truncated and not truncated are different vectors |

These travel with the head in its checkpoint:

```swift
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
```

Loading a head trained with `.last` into a model configured for `.mean` is
refused. Without that check it would load fine and quietly produce plausible
numbers from different geometry.

`maxTokens` defaults to **512 everywhere** — serving, metrics and training
stages alike. That was not always true: serving used to not truncate at all
while evaluation truncated at 512, so on long texts the measured quality
described something other than what serving did. The divergence was measured,
then removed, and a test now holds both sides together.

Truncation happens **before** the terminator is appended, so the terminator
stays last. Otherwise the pooling index would point at an ordinary token and
the vector would be taken from the wrong place — again with no error anywhere.

---

## Training

### Tasks and data

Three tasks share one sample shape (`anchor`, `positive`, `negative`):

| task | what the loss does |
|---|---|
| `.retrieval` | InfoNCE over in-batch negatives plus the explicit hard negative |
| `.sts` | triplet over a pool: pull the positive in, push the negative out |
| `.classification` | zero-shot: the anchor carries its own candidate labels |

```swift
let rows = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl",
                                          limit: 2000,
                                          tasks: [.retrieval, .sts])
print(rows.count, rows.first?.task as Any)
```

Rows with broken UTF-8 or an unknown task are skipped silently: in a corpus of
this size isolated garbage is normal, and aborting a load because of it is
worse than skipping it.

For classification the candidate labels are parsed out of the anchor. Note
that the README of LitRetrieval claims 7 emotions while the data holds a
closed pool of **25**, with 7 per row — the data is the source of truth, and
`ClassificationLabels.pool` follows the data.

### What is trained

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
let model = EmbeddingModel(backbone: base)

// .frozen — head only; cheap, but the ceiling is low: the head only sees
//           the pooled last layer and cannot change the representations.
// .topLayers(N) — the top N backbone layers plus ln_out, and the head.
// .full   — the whole backbone and the head.
// .lora   — attached LoRA/QLoRA adapters and the head.
let trainable = EmbeddingTrainable.make(model: model, mode: .topLayers(4))
print(trainable.parameterNames.count)
```

`.full` deliberately excludes `emb.weight` and `head.weight`: logits are never
computed in an embedding task, and the LM head at a vocabulary of 65536 is
about a third of the parameters of a 0.1B model.

The head is a **sibling** of the backbone, not a submodule. That is a
correctness requirement, not taste: freezing or quantising the backbone walks
its parameter tree, and a head living inside would be frozen along with it —
silently.

`fc2` is zero-initialised, so before training the head is the identity. A
randomly initialised head would shift the vectors on the very first step, and
contrastive training would start from a damaged space rather than the
pretrained one.

### A stage, and a curriculum

```swift
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
```

Each stage evaluates before and after, so `StageResult.delta` says what the
stage actually did rather than what it was supposed to do.

Across a curriculum the trainable set is built **once**. It holds an fp32
master copy of the parameters; rebuilding it per stage would take the master
from the model again — i.e. discard the precision accumulated by the previous
stage by rounding back to the weights' bf16 representation.

The tokeniser is checked against the model vocabulary once per stage. Without
it, an out-of-range embedding lookup does not fail in MLX: the loss becomes
NaN and training spins uselessly until the run ends. This has happened twice
in this codebase, which is why the check exists.

### GradCache

Contrastive losses want large batches, because the batch *is* the pool of
negatives. GradCache splits the batch into chunks, so activation memory stops
scaling with batch size.

Measured in-process: peak **19 MB at chunk 8** versus **77 MB at chunk 256**
on the same batch. With a single chunk it is bit-for-bit the ordinary path
(< 1e-5 across parameters after two AdamW steps), which is what makes it safe
to switch on.

Classification has no GradCache path, and that is not a gap: each row carries
its own candidate pool, so a larger batch adds it no negatives at all.

---

## Evaluation

```swift
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
```

Classification is scored against the **full pool of 25 labels** by default.
The seven candidates named in a row's instruction are that row's own, so
accuracy over them also measures how easy that particular seven happened to
be.

`AccuracyMetrics` exposes `predictions` and `poolSizes` rather than accuracy
alone. That is deliberate: a mutation that argmaxed over all labels instead of
the row's pool left accuracy unchanged on this data, and the only visible
trace was the chosen index itself.

Note that `RankingMetrics` exists in both `RWKVEmbedding` and `RWKVRerank` and
they are different types. If you import both, qualify the name.

---

## Serving

```swift
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
```

`embed(_ texts:)` runs each text as its **own** pass, with no padding. Short
and long strings therefore cannot influence each other through a shared batch.
A batched path is possible — `buildMask` and `lastRealIndex` in `RWKVGen`
exist for it — but for serving the gain rarely justifies the risk of quietly
corrupting a short string's vector.

For training and evaluation there is a batched path, `encodeBatch`, with
`padMultiple` for the differentiable kernel's `T % 16 == 0` requirement.
Padding there is provably harmless: RWKV is causal, position `t` depends only
on positions ≤ `t`, and the vector is taken at `poolIndex` — so no padded
token enters it. That is an equality, not an approximation, and a test holds
it.

---

## Measured numbers

| | value | measured on |
|---|---|---|
| retrieval MRR, raw base | 0.0611 | 40 rows, single run |
| STS cos+ / cos− , raw base | 0.8619 / 0.8723 | 40 rows, single run |
| cosine between different documents | 0.79…0.998, mean 0.9127 | 12 documents, single run |
| GradCache peak memory | 19 MB (chunk 8) vs 77 MB (chunk 256) | in-process |
| tokenisation vs Python | **exact** — ids and pooling position | parity fixture |
| anchor / positive / negative vectors vs Python | < 1% | parity fixture |
| retrieval / STS / classification losses vs Python | < 1% | parity fixture |
| MRR, recall@k, nDCG@10 vs Python | < 0.01 absolute | parity fixture |
| content right of `poolIndex` does not affect the vector | **zero** | any overwrite |

The last row is causality, and it is exact rather than approximate.

Two things worth knowing before benchmarking anything here:

**Batch length alignment is not a bitwise identity.** Training batches are
padded to a multiple of 16, and changing `T` changes the result by about
1e-7 — matmuls decompose differently for a different input shape. Causality
still holds exactly. The practical consequence: reproducible runs require the
**same** alignment, not merely a sufficient one.

**Read memory as physical footprint, not RSS.** Metal buffers live in
IOAccelerator and only partly show up in `ps -o rss`. Use
`vmmap --summary <pid>`.

---

## Not done

- **No runner target.** The reranker has `rerank-run` end to end; embeddings
  are library-only. A curriculum has no incremental report either, so a run
  killed halfway leaves no trace.
- **Reservoir sampling is not wired in.** `EmbeddingDataset.loadJSONL` reads
  from the start up to `limit`, so a slice of a 2.6 GB corpus is its
  beginning, not a sample of it. `RerankDataset.loadJSONL` already does this
  properly and the code transfers by copying.
- **The backbone is not recorded in the checkpoint.** A head trained on one
  backbone and loaded onto another differs in neither shape nor contract. A
  weight fingerprint would close this.
- **GradCache throughput is not measured.** Peak memory is; the Python
  figures (batch 48: eager 9.68 GB vs 3.20 GB) did not reproduce in Swift and
  are not claimed here.
- **`X070PartialFinetune` is not on the shared trainer.** Its loop is
  epoch-based with shuffling and per-epoch accuracy — a different control
  structure. `BackboneWeightsTrainableSet` is ready for it.
