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
| `stateScope` | which texts an attached initial state applies to: `.allTexts`, or `.queriesOnly` — documents are then embedded exactly as with no state. Only matters when a state is attached (`backbone.initialState`) |

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
refused, and so is loading a head trained with one `stateScope` into a model
configured with the other (`initial_state_scope` in the checkpoint; a checkpoint
without it means `.allTexts`). Without that check it would load fine and quietly produce plausible
numbers from different geometry.

The checkpoint also records the base it was trained on (`base_fingerprint`,
`StateFile.baseFingerprint` — the dense model and its `.rwkvq` files are
different bases; top layers committed by training are part of the base, an
active trainable override is not). Loading onto another base is refused with
`EmbeddingCheckpointError.otherBase`; `allowOtherBase: true` on
`Embedder.fromCheckpoint`, `EmbeddingModel.fromHead` and `loadHead` (and
`embed-run --allow-other-base` for `--head-in`) loads it anyway, for a
deliberate transfer. A checkpoint without the key is not checked.

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

`limit` alone takes the first `limit` matching rows and stops reading there. With
`sampleSeed:` it takes a uniform sample of the whole file instead (reservoir
sampling, reproducible per seed; the task filter applies first) — a slice of a
large corpus is then a sample rather than its beginning, at the price of
reading the whole file. `embed-run --limit N --sample-seed S` does the same.

```swift
let sample = try EmbeddingDataset.loadJSONL(path: "~/data/litretrieval.jsonl",
                                            limit: 2000, tasks: [.retrieval],
                                            sampleSeed: 1)
print(sample.count)
```

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
// .state  — the initial state of the recurrence and the head; the base stays
//           frozen (see state-tuning.md).
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

Across a curriculum the trainable set is built **once**, and each stage
starts from what the set reads back from the model. That is exact for the head
(fp32), for a state (fp32) and for top layers unpacked from a `.rwkvq` (fp32);
top layers of a bf16 base and LoRA adapters are committed in bf16 at the end of
a stage, so the next stage starts from the bf16-rounded values, not from the
fp32 master of the previous one. (An earlier version of this paragraph claimed
an fp32 master survives between stages; reading the code, it does not for
those two. Changing it would change the numbers of every multi-stage run, so it
is recorded, not changed.)

#### Running a curriculum with a report and resumption

`EmbeddingCurriculum.run` runs the same stages in a run directory: after
**every** stage it writes `curriculum.json` (before / after metrics, final
loss, steps, time) and `stage<i>.params.safetensors` (the trainable set's
parameters by name). Run it again with the same plan and it continues with the
first unfinished stage; a stage interrupted half-way starts over from the
previous stage's parameters. Each stage has its own data seed and its own
optimiser (as in `EmbeddingFinetune.run`), so a run killed in the middle and
resumed with a freshly built model gives **bit-identical** parameters and
metrics to one that never stopped — a test holds this for the head, top layers
and a state, and that the runner equals `EmbeddingFinetune.run`. A different
plan (stages, data, steps, mode, state scope) in the same directory is refused.
Build the model the same way both times (same base, same unpacked layers, same
attached state): the runner stores the trainable set, not the base.

```swift
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
```

From the command line: `embed-run --curriculum plan.json --run-dir dir`, where
`plan.json` is an array of stages whose keys (`data`, `task`, `train`, `eval`,
`steps`, `lr`, `seed`, `batch`, `max_tokens`, `max_chars`, `sample_seed`,
`limit`) override the command-line ones.

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

Most of that time is not the block-by-block pass but MLX's buffer-cache
ceiling. Freed GPU buffers normally stay in a cache for reuse; a ceiling makes
MLX hand them back to the system and allocate fresh ones for the next block.
Evaluation is forward passes only, so that churn is a visible share of it; a
training step is dominated by the backward pass and does not notice (≤ 1% for
top-layer training). The cache counts towards the process footprint, so lifting
the ceiling costs memory. `stage.evaluationCacheLimitGB` sets a ceiling for
evaluation alone (nil — the training one, `.infinity` — none). Retrieval
evaluation of 300 queries (900 texts up to 512 tokens), two rounds with arms
interleaved, time and peak footprint of the whole process:

| ceiling | 0.1B REDUCTION | 0.4B REDUCTION |
|---|---|---|
| 0.25 GB (default) | 45.2 / 45.4 s, 1.04 GB | 137.3 / 143.7 s, 1.47 GB |
| 1 GB | 42.8 / 44.2 s, 1.78 GB | 129.7 / 139.5 s, 2.22 GB |
| none | 42.3 / 43.6 s, 3.30 GB | 130.1 / 137.0 s, 4.58 GB |

The ceiling costs evaluation 4…7% of time and saves 2.3…3.1 GB. 1 GB is as
fast as no ceiling within noise and stays under 2.3 GB; without a ceiling a
0.4B evaluation goes past a 3 GB budget. The default is left at the training
ceiling because memory is the constraint this library is tuned for.

### A trained state in place of the instruction

The query side of a task carries an instruction line (`Instruct: …\nQuery: …`).
An initial state (`BaseTrainingMode.state`, see [state-tuning.md](state-tuning.md)) can stand in for
it: train the embedder as usual, then remove the line from the queries, freeze
the base and the head, and train only the state on the same rows; attach the
state (`backbone.initialState`) when embedding queries. `embed-run --then-state N`
runs exactly this after the first stage (`--then-head` is the control: the same
steps train the head instead).

0.1B, top 4 layers, retrieval, three seeds, MRR over 300 queries, paired
bootstrap:

| base | with instruction | without | without + state | without + head (control) | state − with instruction |
|---|---|---|---|---|---|
| dense | 0.759 | 0.752 | 0.758 | 0.754 | −0.001 [−0.012; +0.010] |
| REDUCTION | 0.757 | 0.748 | 0.756 | 0.750 | −0.000 [−0.012; +0.011] |

The state restores the quality with the instruction and saves 17.8 tokens per
query. On this data the instruction is the same for every query, so dropping it
costs little in the first place (−0.007…−0.009), and the state's edge over the
control (+0.004…+0.007) is at the edge of significance. State lr 1e-4; 1e-3
made things worse (0.704 against 0.745 without the instruction). In this
measurement the state is attached to the model, so it applies to every text —
documents included, which never had the instruction.

Applied to queries only (below), the same measurement on `reduction` gives
0.752 against 0.756 for the all-texts state and 0.750 for the control: queries
only − all texts −0.004 [−0.010; +0.002], queries only − control +0.003
[−0.004; +0.010] (three seeds, paired bootstrap over 300 queries; the all-texts
arm reproduces the table above exactly). On this data restricting the state to
queries does not help; details in
[state-tuning.md](state-tuning.md#in-place-of-the-instruction).

On data where instructions matter more — tool selection with ~1 thousand-token
agent policies, and the 126 instructed tasks of MAIR — dropping the instruction
costs −0.253 and −0.025 MRR. A trained state wins back part of it (Tool_Use:
0.394 → 0.552…0.570), but a head trained on the same rows wins back as much
(0.557), and no arm reaches the quality with the instruction: one state cannot
stand for many different instructions. Same section of state-tuning.md. A state
per instruction — the instruction folded once — keeps the quality with the
instruction and drops its tokens from every query (below).

#### Attaching the state to queries only

`EmbeddingModel.stateScope = .queriesOnly` applies the attached state to
queries and embeds documents as if no state were attached — bit for bit the
same vectors (`X070Backbone.withoutInitialState` builds their passes). Every
place that embeds then needs the role of the text: `embed(_:role:)`,
`Embedder.embed(_:role:)`, `EmbeddingMetrics.embedAll(role:)`. Without a role
it stops rather than guess. Training and evaluation pass the roles
themselves: in retrieval the anchor is a query and the positives and negatives
are documents; in classification the label texts are documents; in sts both
sides are queries. Block-by-block backward recomputes each piece under its
batch's role, so the gradient matches autograd on the same batches (checked
for the sequential path, row chunks and GradCache), and a state learns only
from the queries. `embed-run --state-scope queries` sets it.

```swift
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
```

### Folding the instruction into a state

A query with an instruction is `Instruct: X\nQuery: q`. RWKV reads it left to
right, and all the instruction gives the query is the state after the
instruction's tokens. `InstructionCache` folds each distinct instruction into a
state once (`foldPrompt`) and embeds the queries of that instruction starting
from it, so a query pays only for its own tokens. With agent policies of about a
thousand tokens that is most of what a query costs.

```swift
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
```

- **Where it is cut.** The prefix is `Instruct: X\n`, the tail `Query: q`.
  Cut there, the two pieces tokenize exactly as the whole line (checked on
  1,500 rows of the Tool_Use set below and 1,500 of MAIR); cut after `Query: `,
  World's greedy vocabulary glues the space to the next word and nearly every
  row tokenizes differently. Rows without an instruction of this form are
  embedded as they are; the order of the rows is kept.
- **The same numbers**, up to summation order at the seam — with one
  exception: `maxTokens` now cuts the tail, not the whole line, so a long
  instruction can no longer push the query out of the window.
- **Pooling `.last` only** (`.mean` throws): a mean over the tail is not the
  mean over the line.
- **The attached state is the start.** With a trained state attached the order
  is trained state → instruction → query, as it is inline. The cache notices a
  different attached state by its fingerprint and folds again; after the
  weights change (training), make a new cache. The attached state is restored
  after each call — the same arrays, not a copy
  (`X070Backbone.withInitialState`).
- **Folding goes in pieces** of `foldChunkTokens` (512) chained through the
  state: one pass over a 3,000-token instruction builds the graph of every
  layer at once and raised the evaluation peak from 0.69 to 1.27 GB (60
  queries, untrained head); in pieces the peak stays at the inline level
  (table below).

Evaluation: `evaluateRetrieval(foldInstructions: true)`,
`EmbeddingStage.foldInstructions`, `embed-run --fold-instructions`;
`embed-run --compare-folded` evaluates both ways on the same weights,
interleaved, and writes both rankings to the report.

Measured on Tool_Use (the tool-selection set in
[state-tuning.md](state-tuning.md#many-and-long-instructions-1810)): 0.1B
`reduction`, top 4 layers trained 200 steps with the instruction inline, then
both evaluations on the same weights, three seeds, 300 held-out queries with
23 distinct instructions, `--max-tokens 2816`, duplicate candidates merged;
release build, M4, arms interleaved, the first round a warm-up:

| seed | MRR inline | MRR folded | rows ranked differently | evaluation inline | folded | peak MLX inline / folded |
|---|---|---|---|---|---|---|
| 0 | 0.6439 | 0.6439 | 0 | 78.9 s | 13.1 s | 0.72 / 0.74 GB |
| 1 | 0.6470 | 0.6470 | 2 (rank 30→31, 86→87) | 80.6 s | 13.2 s | 0.72 / 0.74 GB |
| 2 | 0.6522 | 0.6522 | 1 (rank 34→33) | 84.2 s | 13.7 s | 0.72 / 0.74 GB |

Folded − inline: −3.5·10⁻⁷ [−3.7·10⁻⁶; +2.8·10⁻⁶] MRR (paired bootstrap over
queries, three seeds); the three moved rows are last-bit ties deep in the
ranking. Query tokens: 1,112 per query inline against 48 per query plus 20,563
folded once — 9.5 times fewer — and the evaluation, which also embeds the
documents, runs 6.0–6.1 times faster. No row was cut by `--max-tokens` here.
The inline arm reproduces, row for row, the "with instruction" evaluation of
the earlier measurement on the same seeds. Process footprint 2.10–2.12 GB
(training dominates), swap unchanged over each run.

Where the window is shorter than the instruction, folding is a gain in quality
as well. The same weights evaluated with `--max-tokens 512` (an iPhone budget;
`embed-run --compare-max-tokens 512`): inline, 197 of the 300 queries lose the
query itself — the instruction fills the window — while folded queries are
untouched (the folded rankings are, row for row, those at 2816):

| seed | MRR inline, 512 | MRR folded, 512 | inline on the 197 cut queries | folded on them |
|---|---|---|---|---|
| 0 | 0.3451 | 0.6439 | 0.2921 | 0.7472 |
| 1 | 0.2993 | 0.6470 | 0.2054 | 0.7350 |
| 2 | 0.3250 | 0.6522 | 0.2317 | 0.7300 |

Folded − inline +0.325 [+0.282; +0.366]: +0.494 on the cut queries, 0.0000 on
the other 103.

Not for training with trainable layers: the folded state depends on the weights,
which change every step. With a frozen base (only the head trains) it would
hold for the whole run; that is not implemented.

### A trained state per instruction (a state bank)

Folding reproduces the instruction; training a state of its own for each
instruction is the step after it. An attached state is one per model, so the
states live in a **bank**: `X070Backbone.initialStateBank` holds K states, and
each row of a batch names its own (`EmbeddingSample.state` →
`TripletBatch.anchorRows` → `withInitialRows`). The block picks the rows of the
bank itself, so a batch mixes instructions exactly as it did without states —
the in-batch negatives do not change — and the gradient of a bank row is the sum
over the batch rows that use it. `StateBankTrainableSet` trains the bank
(`blocks.N.state0bank.*`); a row no training batch uses gets no gradient and,
under Adam without weight decay, does not move.

```swift
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
```

Rules: the bank is for queries only, under `stateScope = .queriesOnly`, and for
retrieval only; a bank without rows, rows without a bank, a batch where some rows
have a state and some do not — all stop rather than guess. Folding with a bank
attached stops too (which row would it start from?): fold first, attach after.
Size: a 0.1B state is 2.4 MB in fp32, so a bank of 150 instructions is 350 MB
before the optimizer's moments. `embed-run --then-bank N` runs this as a second
stage after the first (instructions inline), `--then-head` being the control.

Measured on Tool_Use: the same first stage (0.1B `reduction`, top 4 layers,
200 steps with the instruction inline), then 200 steps at lr 1e-4 of either the
bank (WKV, starting from the folded instructions) or — the control — the head
with the folded bank fixed; three seeds, the same 300 held-out queries, both
arms of a seed sharing the first stage:

| | with instruction | folded bank | after the second stage |
|---|---|---|---|
| bank trained | 0.6477 | 0.6477 | 0.6737 |
| head trained (control) | 0.6477 | 0.6477 | 0.6571 |

| difference (Δ MRR [95 %]) | |
|---|---|
| folded bank − with instruction | −0.0000 [−0.0000; +0.0000] |
| trained bank − with instruction | +0.026 [+0.015; +0.037] |
| control − with instruction | +0.009 [+0.002; +0.017] |
| trained bank − control | +0.017 [+0.006; +0.027] |

96 instructions in the bank (234 MB); 10 held-out queries have an instruction
no training row has, and on them the bank stays at the folded state — exactly
the quality with the instruction (Δ 0). On the other 290 the bank adds +0.027
over the instruction and +0.018 over the control. The second stage takes
~175 s against ~78 s for the head (the gradient goes through every layer), and
the process footprint is 2.72 GB against 2.11 GB (the bank and its Adam
moments); swap unchanged over every run. The bank carries far more numbers than
the head (56.6 M), and roughly 20 training queries per instruction.

**The strong control beats it (21.10).** `embed-run --then-top 200` spends the
same 200 extra steps on what the first stage trains — the top 4 layers and the
head, instruction inline, same rows, same batch order as the bank (seed + 1),
the first stage's schedule; the second stage continues the first on the same
trainable set. Three seeds:

| difference (Δ MRR [95 %]) | |
|---|---|
| top layers, 200 more steps − with instruction | +0.049 [+0.030; +0.068] |
| trained bank − top layers, 200 more steps | **−0.023 [−0.042; −0.004]** |

(0.696 against 0.674 for the bank.) The bank's gain over the head-only control
is extra training with more parameters; the top layers trained further give
twice as much. Use a bank where its locality matters — it touches no shared
weights and reproduces the instruction exactly for instructions it was not
trained on — not for quality.

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
computed with (the embedding moves by 1.7e-4).

**Quality.** Against the same dense checkpoint in fp16 — the type a `.rwkvq`
base computes in — top-layer training over a compressed bottom loses nothing
once training has converged (retrieval, MRR over 300 queries, three to five
seeds, paired bootstrap):

- top 4 and 6 layers: differences −0.007…+0.007, intervals around zero, on
  LitRetrieval and on RuLitSearch (hard negatives mined by Harrier);
- top 2 layers: differences appear only while training is short (0.1B,
  200 steps: −0.015; 0.4B, 300 steps: +0.03…+0.04) and vanish or flip sign at
  three times the steps (0.1B, 600: +0.013 / +0.010; 0.4B, 900: +0.003 /
  +0.004);
- `compression` and `reduction` do not differ in any cell.

Mind the type when comparing: a dense base loaded in bf16 (the default for a
dense checkpoint) is a different arm — on 0.4B with two layers it scored 0.015
below the same base in fp16. The full tables are in rwkv-quant
`docs/ARTICLE-1.md`, section 15.

**Speed and memory.** Tokens per second and peak physical footprint of the
process — retrieval, top N layers and the head trained, three texts of 512
tokens per triplet, tokens counted over the whole step including the frozen
bottom; the four arms interleaved as separate processes with pauses, a control
run after each model (M4 16 GB, defaults, `embed-run --mode top:N --eval 0`;
bold is above 3 GB). Dense bf16 and dense fp16 run at the same speed within 2%
and are shown once:

| model | layers | batch | dense | `reduction` | `compression` |
|---|---|---|---|---|---|
| 0.1B | 2 | 1 / 2 / 4 | 3339 / 3572 / 3792 tok/s, 1.08 / 1.23 / 1.74 GB | 3657 / 3938 / 4237, 1.03 / 1.20 / 1.67 | 3657 / 3938 / 4237, 1.02 / 1.21 / 1.68 |
| 0.1B | 4 | 1 / 2 / 4 | 2477 / 2671 / 2818, 1.32 / 1.47 / 2.01 | 2603 / 2792 / 2982, 1.27 / 1.43 / 1.96 | 2603 / 2792 / 2997, 1.28 / 1.45 / 1.97 |
| 0.1B | 6 | 1 / 2 / 4 | 1725 / 2104 / 2234, 1.63 / 1.74 / 2.28 | 1765 / 2163 / 2318, 1.59 / 1.70 / 2.25 | 1765 / 2178 / 2318, 1.60 / 1.72 / 2.26 |
| 0.4B | 2 | 1 / 2 / 4 | 1409 / 1484 / 1452, 1.91 / 2.13 / 2.70 | 1462 / 1725 / 1721, 1.60 / 1.86 / 2.41 | 1476 / 1765 / 1725, 1.55 / 1.81 / 2.36 |
| 0.4B | 4 | 1 / 2 / 4 | 1089 / 1209 / 1197, 2.33 / 2.56 / **3.16** | 1154 / 1341 / 1356, 2.06 / 2.30 / 2.90 | 1154 / 1341 / 1356, 2.02 / 2.26 / 2.86 |
| 0.4B | 6 | 1 / 2 / 4 | 925 / 1027 / 1017, 2.84 / 3.00 / **3.62** | 960 / 1125 / 1119, 2.62 / 2.77 / **3.38** | 960 / 1109 / 1121, 2.60 / 2.73 / **3.35** |

A compressed bottom trains 3–12% (0.1B) and 4–19% (0.4B) faster than a dense
one and takes 0.03–0.07 GB (0.1B) and 0.22–0.36 GB (0.4B) less.

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

Retrieval ranks each query's positive among **all** positives and negatives
of the held-out rows (2N candidates). Where documents repeat — one tool is the
answer to many queries, a document is one query's positive and another's
negative — pass `dedupeCandidates: true` (`EmbeddingStage.dedupeEvaluationPool`,
`embed-run --dedupe-pool`): identical texts are merged and the rank is taken
among distinct documents. Without it, copies of the same text embedded in
different evaluation batches differ in the last bits, and a copy of the correct
document can rank above the document itself. The default keeps the 2N pool.

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
`X070Backbone.load(fromRwkvq:)`. `.frozen`, `.lora` (QLoRA — the base stays
compressed, gradients reach the adapters through the quantised projections) and
`.state` apply as they are. `.topLayers(N)` applies once the top N layers are
unpacked (`unpackTopLayers(N)`, [above](#training-top-layers-over-a-quantized-base));
without that, and `.full` always, are refused: the projections are codes, not
weights (`EmbeddingTrainable.unsupportedReason`).

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

- **A state bank has no file.** `StateFile` holds one state; a trained bank
  (`initialStateBank`) lives in memory, and its rows are known only by the order
  of the instructions it was built from.
- **GradCache throughput is not measured.** Peak memory is; the Python
  figures (batch 48: eager 9.68 GB vs 3.20 GB) did not reproduce in Swift and
  are not claimed here.
