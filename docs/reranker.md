# Reranker

Second-stage retrieval: a query and a list of documents in, a sorted list out.

The first stage (an embedder) compares vectors, so it must compress a whole
document into one vector *before* it ever sees the query. The reranker has no
such constraint: it reads query and document together and produces a single
number. It pays for that with time linear in the number of candidates, which
is why it runs over a shortlist rather than a corpus.

It is not built like a usual cross-encoder. No vectors are computed at all: a
small trainable head reads the **state** of a frozen RWKV backbone, folded
from the text of the pair. Everything else in this document follows from
that — the cache, the text-feeding contract, and why training takes seconds.

**About the numbers.** Everything measured below was taken on RWKV-7 World
0.1B (L=12, D=768) on Apple silicon unless stated otherwise. Code examples
are compiled by `Scripts/check_docs.sh`; it does **not** re-verify the
numbers — that needs the model and minutes of compute.

---

## Quick start

Train a head with the executable target, from data to checkpoint:

```swift-skip
swift run -c release rerank-run \
  --model ~/models/world_0.1b_x070.safetensors \
  --vocab .testdata/rwkv_vocab_v20230424.txt \
  --data  ~/data/reranker-triples-multi/train.jsonl \
  --queries 300 --candidates 8 --eval-queries 80 \
  --layers 5 --epochs 8 --lr 2e-4 \
  --out runs/rerank
```

Apply the trained head:

```swift
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
```

The score is a raw logit. It is comparable **within** one query and not
comparable across queries: the head was trained with a listwise loss, which
fixes only the ordering. It has no absolute scale, and a sigmoid of this
number would mean a probability only if training included a non-zero BCE
term.

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

---

## How it works

### A pair is a prefix and a tail

The text of a pair is assembled like this:

```
Instruct: <instruction>
Document: <document>
Query: <query><terminator>
```

The document comes **before** the query, and that is not cosmetic. RWKV state
accumulates left to right, so the "Instruct + Document" part does not depend
on the query and can be folded **once**. Swap the order and there is nothing
left to cache.

The reverse order (`PairTemplate(docFirst: false)`) is kept for honest
comparison: there the document is read already knowing the query, which is
theoretically better, but the prefix is uncacheable and a document index
cannot be built at all.

Measured: a prefix costs **72.0 ± 0.2 ms**, a tail **6.1 ± 0.0 ms** (0.1B,
batches of 8 and 16, three runs). So giving a query eight more candidates is
nearly free once the documents are folded.

### The head

The head reads backbone state at selected layers and emits a scalar:
trainable probe tokens → `ln0` → a stack of RWKV blocks on top of the state →
`ln_out` → an MLP down to one number.

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
let model = try Reranker(base: base, cfg: RerankerConfig(
    layerIdx: [5],       // which backbone layers to read; negatives count back
    sharedState: false,  // all blocks read the state of the LAST layerIdx entry
    nProbe: 1,           // number of probe tokens
    headHidden: nil))    // MLP width; nil ⇒ D
print(model.head.uniqueSources, model.head.parameterCount)
```

`score_fc2` is **zero-initialised**. That gives the property it exists for:
before training every score is exactly zero, so the listwise loss starts at
exactly `ln(C)`. If the first logged loss is not `ln(C)`, the data or the
head's wiring is broken, and that is where to look — not at the LR schedule.
This is the cheapest detector in the whole pipeline.

**The default `layerIdx` is `[-1]`, the last layer, and it is the most
degenerate configuration available.** It stays that way so numbers remain
comparable with the reference implementation. Start from the middle of the
stack instead: it measures noticeably better (see "Measured numbers").

---

## The text-feeding contract

The most important section of this document.

State is a function of *exactly* the text that was fed. Template, document
truncation, query truncation, terminator, instruction — all of them change
the state. And they diverge **silently**: no shape error occurs anywhere, the
scores stay plausible, they are just worse.

So the contract is not a comment but a checked value, and it travels with the
artefacts:

| where | what is checked |
|---|---|
| `StateCache` | the cache remembers the contract it was built with and which layers it holds |
| `RerankTraining.train` | training and held-out caches must agree; the caller may state its own expectation |
| `RerankTrainResult.contract` | the contract the head **actually** trained on |
| head checkpoint | saved with the contract taken from the training result |
| `RerankerInference.fromCheckpoint` | serving configuration is read **from** the checkpoint |
| `DocIndex` | the index remembers instruction, template and document truncation |

One rule: **never assemble a serving configuration by hand when a checkpoint
exists.** Defaults are the most likely source of silent quality loss here,
precisely because they look plausible.

```swift
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
```

What is deliberately **not** part of the contract, and why:

- **batch length alignment** — pad positions are neutral for the recurrence
  (`w←1, k←0, b←0`), so state does not depend on them. Requiring a match
  would forbid reusing a cache after changing a performance knob;
- **query truncation, for an index** — it lives in the tail, the prefix does
  not depend on it, and re-indexing a corpus to change query length would be
  pointless;
- **the instruction, during training** — there it is a field of the sample,
  and a corpus may contain several. The cache records the instruction only if
  there is exactly **one**; otherwise it stays silent, which is more honest
  than any default.

---

## Training

### Why it takes seconds

The backbone is frozen, so the mapping "pair text → state" is **fixed**.
Training the head therefore need not recompute the long pass on every step:
pairs are folded once into a cache, and from then on the head — one or two
blocks over a single token — learns on ready-made states.

Measured: encoding 2400 pairs takes **3:46**; eight epochs of training on
them take **2.7 s**. The cache exists for that asymmetry.

Only the layers the head reads are stored, not all of them: for a single
layer that is **96 KB per pair** in fp16 versus 2.4 MB for full state in
fp32.

### The full path

```swift
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
```

### Evaluating without training

Comparing two saved heads, re-scoring an old one on a new held-out set,
checking a checkpoint after a transfer — all of this used to require running
training again, i.e. changing the very thing you meant to measure.

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
let cache = try StateCache.load(URL(fileURLWithPath: "runs/cache_eval"))

// From a checkpoint: the contract is read FROM THE FILE and checked
// against the cache.
let (metrics, contract) = try RerankTraining.evaluate(
    base: base, head: URL(fileURLWithPath: "runs/head.safetensors"),
    cache: cache)
print(contract["max_doc_tokens"] ?? "?", metrics.summary)
```

From the runner: `rerank-run --eval-head <path>`. The cache must **already
exist**; a missing one is an error, not a reason to quietly build a new one.
A cache built right now would answer a different question.

### Metrics

`MRR`, `recall@k`, `nDCG@10`, plus a breakdown into "against the mined
negative" and "against the pool top-up". The first is the column the reranker
exists for: an overall MRR across eight candidates, six of which were drawn at
random, flatters everyone.

Ties are resolved by **average** rank. That is not a detail: with an
untrained head all scores are equal, and optimistic tie-breaking would yield a
perfect MRR of 1.0, turning the "before" column of every table into a lie.
Averaging honestly reports `2/(C+1)` — exactly random guessing, 0.2222 with
eight candidates.

---

## Sweeps: which configuration is better

A single run with a hundred held-out queries is **noisy**, and differences
between configurations cannot be judged from it — yet they look exactly as
convincing as real ones.

A cache can hold a **superset** of layers, so ten configurations cost one
encoding pass: each head takes its own slice.

```swift
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
```

Every seed gets a **fresh** head — otherwise the second run starts from the
first one's weights and measures further training, not spread. The deviation
is the sample one (divisor n−1), and with a single run it is **NaN, not
zero**: zero would read as "there is no spread", turning one noisy run into an
established fact.

A head reading a layer the cache does not hold aborts training. The check is
not a formality: states of different layers are indistinguishable by shape,
and without it the head would silently train on the wrong layer.

### Topping up a cache with more layers

If a layer is missing from a cache, there is no need to re-encode everything.
Only the missing layer is encoded and merged into the existing cache:
encoding is GPU-bound and costs minutes, merging shuffles bytes and costs
seconds.

```swift
let withLayer5 = try StateCache.load(URL(fileURLWithPath: "runs/cache_l5"))
let withLayer11 = try StateCache.load(URL(fileURLWithPath: "runs/cache_l11"))
let both = try withLayer5.merged(with: withLayer11,
                                 to: URL(fileURLWithPath: "runs/cache_l5_l11"))
print(both.sources ?? [])   // [5, 11]
```

Merging rejects caches covering different pairs or with contradicting
contracts, and it compares **overlapping layers numerically** on a sample of
rows. That last check is the only thing that catches merging caches produced
by **different backbones**: the model is not recorded in the contract, and the
states of two models are indistinguishable by shape and by contract alike. If
there is no overlap, there is nothing to check against — and that is stated
out loud rather than passed over.

---

## Serving

Two modes, and the choice between them is not about speed in general but
about whether documents repeat.

**Direct path.** Every pair is computed in full. Stores nothing; right when
the documents are new every time.

**Indexed.** The prefix state is computed once and kept; after that a query
costs only its own length.

```swift
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
```

For a large index there is a streaming build: batches go to disk as they are
produced and are released, so the memory peak is one batch rather than the
whole index.

```swift
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
```

**The index holds FULL state — every layer**, because continuing the prefix
with a query requires the whole depth. For 0.1B that is roughly 2.4 MB per
document in fp32. A thousand-document index is 2.4 GB, so it is meant for a
"hot" subset (say the top-100 from an embedder), not for a whole corpus.
`dtype: .float16` halves it.

That is also what distinguishes it from `StateCache`: there the unit is a
pair, and only the final state of the layers the head reads is stored.

An index built with a different instruction or a different document
truncation is rejected at scoring time. Its shapes are perfectly valid and
its scores are plausible — they just come from different text.

---

## Measured numbers

Everything below is on the real 0.1B. Single-run numbers are marked as such:
they must not be used to compare configurations.

### Quality

300 queries × 8 candidates (220 train, 80 held out), 25 languages, 8 epochs,
lr 2e-4, **three seeds**:

| | MRR | R@1 | vs. mined negative |
|---|---|---|---|
| layer 5 (mid-stack) | **0.9812 ± 0.0063** | 0.9667 ± 0.0072 | 0.9908 ± 0.0052 |
| layer 11 (last) | 0.8536 ± 0.0234 | 0.7750 ± 0.0331 | 0.9000 ± 0.0043 |

The MRR gap of +0.1277 against a pooled spread of 0.0171 is larger than two
spreads. What is established is the **direction**; the magnitude depends on
the amount of training data (on a four times larger corpus the same gap was
0.024) and is not usable as "the value of the gap".

The spread for layer 11 is four times that of layer 5. The worse
configuration is also the noisier one — exactly the case where a single run
misleads the most.

### Serving versus training

The head trained on **states** from a cache and is applied to **text**. 80
held-out queries, single run:

| | MRR | R@1 | nDCG@10 | vs. mined negative |
|---|---|---|---|---|
| evaluation on the cache | 0.9750 | 0.9625 | 0.9812 | 0.9850 |
| serving, direct path | 0.9750 | 0.9625 | 0.9812 | 0.9850 |
| serving, via prefix index | 0.9750 | 0.9625 | 0.9812 | 0.9850 |

The direct and indexed paths differ by **6.5e-4** absolute (1.1e-3 relative)
while agreeing on the ordering across all 640 pairs. The correct phrasing is
"the paths produce the same ordering", not "the paths are identical".

### Speed and memory

| | value | measured on |
|---|---|---|
| prefix | 72.0 ± 0.2 ms | 200 prefixes, 3 runs |
| tail | 6.1 ± 0.0 ms | same |
| encoding 2400 pairs, 2 layers | 3:46 | single run |
| 8 epochs of training on them | 2.7 s | single run |
| cache | 96 KB per pair per layer (fp16) | computed, checked against the file |
| encoding memory | flat 4.3 GB for the whole run | single run |

Against the Python implementation (`rwkv-metal`) on identical input, with
warm-up and a cache limit: **1.37×** on total encoding time.

**Read memory as physical footprint, not RSS.** Metal buffers live in
IOAccelerator and only partly show up in `ps -o rss`: on a run whose
footprint reached 11.6 GB, RSS reported 0.4 GB. The tool is
`vmmap --summary <pid>`.

**A buffer-cache ceiling is mandatory.** Batch shapes vary, so there is
nothing to reuse, and without `cacheLimitGB` the cache grows linearly with the
number of batches until the machine starts swapping. The default is 2 GB.

Batch length rounding (`lengthBucket`) applies only to batches of at least
`4 × lengthBucket` tokens. Rounding everything cost 38% of encoding time and
saved no memory at all: with the ceiling in place the peak is the same either
way.

---

## Not done

- **Resuming training.** `Trainer` supports checkpoints with Adam moments;
  `RerankTraining` does not expose them. Deliberately deferred: training takes
  seconds, the mechanism is needed for a corpus that does not exist yet, and
  building it now would mean building for an imagined load. It becomes
  necessary once a cache stops fitting into a single session.
- **The backbone is not recorded in the cache contract.** Caches from
  **different** models are indistinguishable by shape and by contract;
  merging catches this numerically via overlapping layers, but training on a
  substituted cache does not. A weight fingerprint in the contract would close
  the hole completely.
- **Memory peak during merging.** `merged` assembles the result in row
  batches, but a batch itself lives in MLX in full; for very wide states that
  is noticeable.
- **The index cannot be updated incrementally.** Adding documents to an
  existing index is not possible, only rebuilding it.
