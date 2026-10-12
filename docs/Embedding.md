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

### Memory: backward block by block

A stage computes its gradient block by block by default
(`EmbeddingStage.sequentialBackward`, see [Train.md](Train.md#long-sequences)).
Autograd over the whole graph keeps the activations of every layer of every
batch part alive until the backward pass ends; here only block inputs are
kept, the loss is taken from the outputs of the last block, and blocks are
differentiated top-down with an evaluation after each. The gradient is the
same in every mode — head only, top layers, the whole base, adapters.

0.1B, 8 triplets of 256 tokens, step time and peak physical footprint of the
process (`embed-run --seq 0|1`, M4):

| mode | whole-graph autograd | block by block |
|---|---|---|
| LoRA, dense base | 2.18 s, 8.76 GB | 2.98 s, 2.40 GB |
| whole base | 2.97 s, 11.77 GB | 3.51 s, 3.31 GB |
| top 4 layers | 1.40 s, 5.02 GB | 1.71 s, 2.57 GB |
| QLoRA, REDUCTION | 2.17 s, 7.91 GB | 2.99 s, 1.74 GB |

The price is a second forward pass (+18…37% time). Set
`stage.sequentialBackward = false` when memory is not the constraint. Under
GradCache the switch has no effect.

The footprints in that table are peaks of the whole process, and with a
held-out set the peak was not in training at all: the evaluation before
training ran before the MLX cache limit was applied (LoRA row: 2.40 GB for the
process, 1.83 GB in steady training). The stage now applies
`config.cacheLimitGB` before the first evaluation.

### Memory: fitting a budget

Where a step's peak goes (0.1B dense, LoRA, 4 triplets of 512 tokens, all
three parts at full length — 1.52 GB of MLX memory):

| | GB | scales with |
|---|---|---|
| model, adapter masters, Adam moments | 0.31 | the trainable set (whole base: 1.56 + 0.38 of gradients) |
| inputs of every block, all three parts | 0.36 | layers × tokens of the **whole batch** × D × 4 bytes |
| backward of one block: recomputed activations and cotangents | 0.87 | ~140 fp32 tensors of [tokens, D] — tokens of **one part** |

The activations are the peak, and they are fp32 from layer 0 on (the WKV
kernel returns fp32 into the residual stream). Two knobs shrink them without
touching the arithmetic, a third trades memory back for time:

```swift-skip
stage.bodyChunkTokens = 1024      // rows of a part go through the body in pieces of ≤ 1024 tokens
stage.traceStride = 2             // keep every 2nd block input, recompute the rest
stage.config.pipelined = true     // asyncEval after each block instead of eval (off by default)
```

- `bodyChunkTokens` — rows inside the body do not interact, so a part can be
  cut into row pieces; the loss still sees the whole batch. The backward peak
  is then set by the piece, not by the batch. Per-token cost on 0.1B is flat
  down to ~1024 tokens per piece and +16% at 512; on 0.4B it is flat (+3% at
  512). Same gradient up to summation order (1.5e-6 against autograd).
- `traceStride` — block inputs are kept for every N-th block and the others
  are recomputed segment by segment during the backward pass. Bit-identical
  gradient; costs up to one more forward pass.
- `config.pipelined` (`TrainingConfig`; also on `LoRAConfig` and
  `StateTuneConfig`) — the CPU builds the next block's graph while the GPU
  computes the current one. Bit-identical; −8…−14% step time, but two blocks
  overlap in memory (about what doubling `bodyChunkTokens` costs). Off by
  default, since memory is the scarce resource on a phone. It applies to its
  own run only: the trainer switches it on for the loop and restores the
  previous setting after. `SequentialBody.pipelined` is the underlying
  process-wide switch; leave it alone unless you drive `SequentialBody`
  yourself.

Step time and peak physical footprint of the process, training only, all three
parts padded to the full length (worst case; `embed-run --eval 0`, M4):

| 8 triplets × 1024 tokens | as is | chunk 2048 | chunk 1024 | chunk 1024, stride 2 | chunk 1024, stride 4 | chunk 2048, pipelined |
|---|---|---|---|---|---|---|
| 0.1B dense, LoRA | 15.7 s, 3.62 GB | 15.5 s, 2.45 GB | 17.5 s, 2.09 GB | | 22.3 s, 1.38 GB | 14.1 s, 2.83 GB |
| 0.1B dense, whole base | 18.5 s, 5.12 GB | | 20.3 s, 3.50 GB | 21.8 s, 3.07 GB | | |
| 0.1B dense, top 4 layers | 8.9 s, 3.71 GB | | 10.4 s, 1.95 GB | | | |
| 0.1B REDUCTION, QLoRA | 16.4 s, 3.62 GB | | 17.5 s, 2.06 GB | | | |
| 0.4B REDUCTION, QLoRA | 49.3 s, 5.84 GB | | 50.3 s, 4.01 GB | 58.4 s, 2.81 GB | 62.5 s, 2.21 GB | |

| 8 triplets × 512 tokens | as is | chunk 1024 | chunk 512 | chunk 1024, pipelined |
|---|---|---|---|---|
| 0.1B dense, LoRA | 7.9 s, 2.46 GB | 7.8 s, 1.50 GB | 9.2 s, 1.33 GB | 7.1 s, 1.84 GB |
| 0.1B dense, whole base | 9.1 s, 3.94 GB | 9.0 s, 2.93 GB | | 8.4 s, 3.43 GB |
| 0.4B REDUCTION, QLoRA | 24.0 s, 3.69 GB | 24.0 s, 2.68 GB | 24.8 s, 2.46 GB | |

Without the knobs, memory and time follow the tokens of the batch: 8 × 512 and
4 × 1024 cost the same. Head-only training needs no backward pass through the
body and stays under 1.9 GB in every cell above.

"As is" in these tables means `bodyChunkTokens = 0`; the default is 2048.

Evaluation has its own budget. `EmbeddingMetrics.embedAll` fills a batch by
rows and by tokens (`EmbeddingMetrics.evaluationBatchTokens`, 4096) and runs
the body block by block. 32 held-out rows of 1024 tokens, peak MLX memory:
0.1B 1.85 → 0.84 GB, 0.4B REDUCTION 2.17 → 1.15 GB; evaluation takes 10…29%
longer than one graph over 16 rows did.

### Training top layers over a quantized base

Adapters on a 0.1B or 0.4B model carry too few parameters to be worth it; the
useful mode there is training the top layers themselves. A `.rwkvq` base
supports it once those layers are unpacked — the bottom stays compressed and
frozen:

```swift-skip
let base = try X070Backbone.load(fromRwkvq: path)
base.unpackTopLayers(4)                       // projections of the top 4 layers become dense weights
let model = EmbeddingModel(backbone: base)
try EmbeddingFinetune.run(model: model, tokenizer: tok, mode: .topLayers(4), stages: [stage])
```

The unpacked projections are the same dequantized numbers the compressed base
computed with (the embedding moves by 1.7e-4). Training speed in tokens per
second and peak physical footprint of the process — retrieval stage, top N
layers and the head trained, three texts of 512 tokens per triplet, tokens
counted over the whole step including the frozen bottom (M4 16 GB, defaults,
`embed-run --mode top:N --eval 0`; bold is above 3 GB):

| model | layers | batch 1 | batch 2 | batch 4 |
|---|---|---|---|---|
| 0.1B bf16 | 2 | 3072 tok/s, 1.05 GB | 3657 tok/s, 1.24 GB | 3840 tok/s, 1.74 GB |
| 0.1B bf16 | 4 | 2259 tok/s, 1.32 GB | 2719 tok/s, 1.48 GB | 2831 tok/s, 2.01 GB |
| 0.1B bf16 | 6 | 1766 tok/s, 1.64 GB | 2148 tok/s, 1.74 GB | 2259 tok/s, 2.28 GB |
| 0.1B REDUCTION | 2 | 3268 tok/s, 0.98 GB | 3889 tok/s, 1.21 GB | 4267 tok/s, 1.68 GB |
| 0.1B REDUCTION | 4 | 2363 tok/s, 1.26 GB | 2844 tok/s, 1.46 GB | 3027 tok/s, 1.96 GB |
| 0.1B REDUCTION | 6 | 1807 tok/s, 1.59 GB | 2210 tok/s, 1.73 GB | 2336 tok/s, 2.25 GB |
| 0.4B bf16* | 2 | 1313 tok/s, 1.91 GB | 1477 tok/s, 2.13 GB | 1466 tok/s, 2.70 GB |
| 0.4B bf16* | 4 | 1067 tok/s, 2.32 GB | 1209 tok/s, 2.56 GB | 1212 tok/s, **3.16** GB |
| 0.4B bf16* | 6 | 909 tok/s, 2.86 GB | 1031 tok/s, 3.00 GB | 1027 tok/s, **3.62** GB |
| 0.4B REDUCTION | 2 | 1463 tok/s, 1.60 GB | 1707 tok/s, 1.88 GB | 1736 tok/s, 2.41 GB |
| 0.4B REDUCTION | 4 | 1164 tok/s, 2.07 GB | 1347 tok/s, 2.30 GB | 1368 tok/s, 2.90 GB |
| 0.4B REDUCTION | 6 | 966 tok/s, 2.61 GB | 1113 tok/s, 2.77 GB | 1129 tok/s, **3.39** GB |

\* measured on the dequantized REDUCTION saved as bf16; a rerun on the original
dense checkpoint gave the same numbers within 2%.

With `config.pipelined = true` the same grid runs 2…15% faster for
0.2…0.6 GB more.

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

### On a quantised base (`.rwkvq`)

`EmbeddingModel(backbone:)` takes a backbone opened with
`X070Backbone.load(fromRwkvq:)`. Two modes apply: `.frozen` and `.lora`
(QLoRA — the base stays compressed, gradients reach the adapters through the
quantised projections). `.full` and `.topLayers` are refused: the projections
are codes, not weights (`EmbeddingTrainable.unsupportedReason`).

`embed-run` measures one stage end to end. 0.1B, LitRetrieval retrieval rows:
2000 train, 300 held out (a pool of 600 candidates), 200 steps of 8 triplets,
256 tokens, rank-8 adapters on every layer, lr 5e-4, **single run**:

| base | MRR before | MRR after | recall@10 | s / step | MLX peak |
|---|---|---|---|---|---|
| dense + LoRA | 0.014 | 0.614 | 0.800 | 3.59 | 7.80 GB |
| REDUCTION + QLoRA | 0.022 | 0.645 | 0.847 | 3.52 | 6.93 GB |
| COMPRESSION + QLoRA | 0.029 | 0.693 | 0.873 | 3.47 | 6.93 GB |

What this establishes is that a quantised base fine-tunes no worse than the
dense one; one seed does not order the three rows. Training the head alone on
a frozen base does not move the metric in 300 steps on either base.

QLoRA on larger REDUCTION bases, 128 tokens, 200 held out, M4 16 GB, single
run — a feasibility check, not a recipe:

| base | batch | steps | MRR before → after | s / step | MLX peak |
|---|---|---|---|---|---|
| 0.4B | 4 | 100 | 0.020 → 0.518 | 2.93 | 5.79 GB |
| 1.5B | 2 | 50 | 0.031 → 0.158 | 4.14 | 7.41 GB |

1.5B fits, barely: batch 2 already peaks at 7.4 GB and touches swap. A
contrastive batch that small has almost no in-batch negatives; use GradCache
for anything real at this size.

Vectors from the quantised path stay at cosine 0.9999 to the same weights run
densely.

Evaluation and serving always take the forward path, whatever is being
trained (`X070Backbone.withoutGradientPath`): the differentiable kernel needs
a length that is a multiple of 16, and a text has whatever length it has.

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
