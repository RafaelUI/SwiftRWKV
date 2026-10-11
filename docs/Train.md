# Training

Every training path in this package — LoRA, partial fine-tuning, pretraining
from scratch, embedding curricula, reranker heads — runs through one loop and
one abstraction. This page covers that shared machinery first, then each
specific path.

Task-specific detail for embeddings and reranking lives in
[Embedding.md](Embedding.md) and [reranker.md](reranker.md); the inference
side is in [Inference.md](Inference.md).

**About the numbers.** Everything measured below was taken on RWKV-7 World
0.1B (L=12, D=768) on Apple silicon unless stated otherwise. Code examples are
compiled by `Scripts/check_docs.sh`; it does **not** re-verify the numbers.

```swift-helper
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
```

---

## The two pieces everything is built from

### `TrainableSet` — what differentiates

A training run needs to know which tensors are parameters. That is the whole
job of `TrainableSet`:

```swift-skip
public protocol TrainableSet: AnyObject {
    func initialParameters() -> [MLXArray]   // fp32 master, deterministic order
    func inject(_ ps: [MLXArray])            // substitute before forward
    func commit(_ ps: [MLXArray])            // write back when done
    var parameterNames: [String] { get }      // for checkpoints and diagnostics
}
```

Two properties of this protocol are load-bearing and easy to get wrong if you
write your own:

**The master copy is fp32.** Official RWKV weights are bf16 — 8 mantissa bits.
At `lr ≈ 1e-4` and weights around 0.05, part of every AdamW step is smaller
than the representable quantum and rounds away. The loss keeps falling
because the last layer of the head still moves, so the problem is invisible
unless you look for it.

**`inject` is called INSIDE the gradient closure.** The dtype cast has to
happen there. Cast outside and the chain back to the fp32 master is severed —
gradients then flow to a temporary, and training silently does nothing.

Implementations shipped:

| type | trains |
|---|---|
| `LoRATrainableSet` | attached LoRA/QLoRA adapters |
| `BackboneWeightsTrainableSet` | backbone weights by key (top-N layers, or all) |
| `EmbeddingHeadTrainableSet` | the embedding head |
| `RerankerHeadTrainableSet` | the reranker head |
| `CompositeTrainableSet` | several sets as one, parameters end to end |

`CompositeTrainableSet` is how "backbone plus head" is expressed. The head is
a **separate** set added to whichever backbone set you chose, rather than a
special case inside each — which is why every combination of base treatment
and head works without new branches.

### `Trainer` — the loop

```swift
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
```

`Trainer` owns AdamW, the LR schedule, gradient accumulation, global-norm
clipping and checkpointing. It is generic over the batch type, so the data
side is entirely yours.

Checkpoints include Adam moments, so resuming continues rather than restarts:

```swift
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
```

### A gradient hook, when the default is not enough

`Trainer` accepts a `GradientProvider` that replaces the built-in
`valueAndGrad`. That is how GradCache plugs into the ordinary loop instead of
growing a second one. Default is `nil` and the standard path is untouched.

---

## Advice that cost something to learn

**Set a Metal buffer-cache ceiling.** Batch shapes vary, so freed buffers are
rarely reusable, and MLX's cache grows linearly with the number of batches
until the machine swaps. `cacheLimitGB: 2.0` is the default in every config
here. Measured: without it, an encoding run reached 13 GB physical footprint
and went to swap.

**Do not round batch lengths up "for shape stability".** It costs real work
and saves no memory once the ceiling is set. Measured on the reranker encoder:
rounding every batch to a multiple of 64 cost **38%** of total encoding time
while peak memory stayed identical. Round only batches that are already long
(see `lengthBucketMinTokens`).

**Reproducibility requires identical alignment, not merely sufficient
alignment.** Training batches are padded to a multiple of `WKV7_CHUNK` (16)
because the differentiable kernel requires it. Changing `T` changes results by
about 1e-7 — matmuls decompose differently for different input shapes.
Causality is exact regardless; the numbers are not.

**Watch the first logged loss.** For any head with a zero-initialised final
layer the starting loss is analytically known — `ln(C)` for a listwise loss
over C candidates. If the first loss is not that number, the data or the
wiring is broken, and no amount of LR tuning will help. This one check has
caught more than every metric combined.

**Verify the tokeniser against the model vocabulary once per run.** An
out-of-range embedding lookup does not fail in MLX. It returns garbage that
depends on the batch shape — sometimes NaN, sometimes plausible numbers. The
plausible case is worse. Every training entry point here does this check; if
you write your own, do it too.

**Measure memory as physical footprint.** Metal buffers live in IOAccelerator
and show up only partly in `ps -o rss`: on one run where footprint reached
11.6 GB, RSS reported 0.4 GB. Use `vmmap --summary <pid>`.

**A run taken while the machine swaps is invalid, not merely slow.** Count
swap pages before and after; discard the run if they moved.

**Single runs cannot compare configurations.** With a hundred held-out rows
the noise is comparable to the effects you are looking for. Use several seeds
and report the spread — and note that a spread computed from one run is
undefined, not zero.

---

## LoRA and QLoRA

Adapters on the projection weights, with the base frozen. The cheapest way to
adapt a model that will not fit full fine-tuning.

```swift
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
```

`LoRA.add` also marks the touched layers trainable, which routes the forward
pass through the differentiable kernel. That kernel requires `T % 16 == 0`,
so batches must be aligned — and if you want to *infer* with adapters
attached, clear `trainLayers` first:

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
_ = LoRA.add(to: base, spec: LoRASpec())
base.trainLayers = []      // inference: no chunk-alignment requirement
var state = RWKVState(cfg: base.cfg)
print(base.prefill([1, 2, 3], state: &state).shape)
```

Adapters are applied by the shared projection path, so they affect both the
parallel pass and recurrent decode. That was not always true — decode used to
ignore them silently, and a fine-tuned model generated as if it had never
been fine-tuned. Measured divergence after the fix: **1.1e-5**.

Training:

```swift
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
```

`LoRABatch` is `(x: ids [B,T], y: targets [B,T])` and the objective is the
language-modelling loss. `T` must be a multiple of 16; the first batch is
checked and the run aborts with a clear message if it is not.

#### Long sequences

With `useBlockCheckpoint` the whole backward pass is still one lazy MLX graph,
and MLX, not the caller, picks the order in which it is evaluated: recomputed
activations of several blocks end up alive at once. Peak memory grows by about
1.9 GB per 1024 tokens on a 1.5B model, and 4096 tokens do not fit in 16 GB.
Two switches fix the order instead of the granularity:

```swift-skip
cfg.sequentialBackward = true   // backward block by block, evaluated after each
cfg.lossChunkTokens = 256       // head and cross-entropy in pieces
cfg.cacheLimitGB = 0.25         // the MLX cache counts toward the footprint
```

`sequentialBackward` runs a gradient-free forward pass that keeps each block's
input, then the head, then the blocks top-down, evaluating after every block.
The gradient is **bit-for-bit** the one autograd gives over the whole graph.
It replaces `useBlockCheckpoint` and costs the same: every block is computed
twice.

1.5B REDUCTION, rank-16 adapters on every layer, batch 1, M4 16 GB — step time
and peak physical footprint (`/usr/bin/time -l`):

| tokens | block checkpoint | sequential, chunks of 256, cache 0.25 GB |
|---|---|---|
| 1024 | 5.2 s, 5.66 GB | 5.5 s, 3.16 GB |
| 2048 | 10.3 s, 7.18 GB | 10.7 s, 3.92 GB |
| 4096 | does not fit | 21.3 s, 5.17 GB |
| 8192 | does not fit | 42.7 s, 8.16 GB |

`lossChunkTokens` applies only while the head is a constant (no adapter, no
trained override); with a trained head the loss takes the ordinary path.

Both are off by default. `decode-bench --train-bench m.rwkvq --tb-t 2048
--tb-seq --tb-loss-chunk 256` measures a step on your own model.

Adapters persist separately from the base:

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
_ = LoRA.add(to: base, spec: LoRASpec())
let url = URL(fileURLWithPath: "runs/adapters.safetensors")
try LoRA.save(base, to: url)
try LoRA.load(base, from: url)
print(LoRA.adapterState(base).count)
```

**QLoRA note.** `spec.quantizeBits > 0` quantises the target weights through
MLX's own affine quantisation, which is a *different* mechanism from the
`.rwkvq` base described in [Inference.md](Inference.md). The two coexist.

### Adapters on a `.rwkvq` base

A backbone loaded from a `.rwkvq` takes adapters the same way; the base stays
compressed and frozen, and the gradient flows through it:

```swift
let model = try X070Backbone.load(fromRwkvq: "~/models/rwkv7-g1d-0.1b-reduction.rwkvq")
let added = LoRA.add(to: model, spec: LoRASpec(rank: 16, alpha: 32))
print(added.numAdapters)        // every tmix target, quantised or dense
// ... train as above ...
LoRA.merge(model)               // merged targets become dense weights
```

Three things differ from a dense base, all deliberate:

- **The backward pass through the base is wider than the forward.** The base
  computes in fp16 (the rwkv-quant norm); differentiating straight through it
  loses small cotangents to fp16's narrow exponent. On 0.1B the relative error
  of adapter gradients against an fp32 matmul goes from 4.8e-3 to 0.65 when the
  loss is scaled down by 4096 — which is what a long averaged batch does to it.
  So the base carries its own VJP: forward in fp16, backward `dX = dY · W` with
  `W` dequantised in bf16 (`rwkvqBackwardDType`, default `.bfloat16`; `nil` is
  plain autograd, `.float32` is more exact and costs memory). Its error stays
  at 7.0e-3 at both scales, and it costs about 2% of a training step.
- **Adapters are never created in fp16**, even though the model's dense
  parameters are: they are bf16, as in rwkv-metal. In fp16 the adapters of
  layer 0, whose input is still fp16, overflow in the backward pass.
- **`merge` produces dense weights.** A merged target is dequantised, the
  adapter is added, and the result lives as an ordinary dense weight; nothing
  is baked back into the quantised file.

The `sym` decode kernel (see Inference.md) is not on this path at all: it is
opaque to autograd and runs only in the recurrent step.

---

## Partial fine-tuning: top-N layers plus a classifier

For classification, where only the upper layers need to move.

```swift
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
    numClasses: 2, freeze: 8, epochs: 5, batchSize: 8, lr: 1e-4,
    onEpoch: { epoch, acc in print(epoch, acc) })
print(result.valAcc, result.params.count)

X070PartialFinetune.clearCache()
```

The boundary cache is the same idea as the reranker's state cache: a frozen
prefix is a fixed function, so compute it once. `ctxLen` must be a multiple of
16.

This path has its own epoch loop rather than using `Trainer` — shuffling and
per-epoch accuracy are a different control structure.
`BackboneWeightsTrainableSet` is ready for a port when one is wanted.

---

## Pretraining from scratch

```swift
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
```

Initialisation is a port of the **official** `RWKV-LM` (`RWKV-v7/train_temp`),
not of rwkv-metal: the latter's `init_weights` belongs to a different,
simplified model (fixed rank 64, no bias on `w_lora_B`), and the x070 model
there only ever loads weights.

```swift
let cfg = X070Config(nLayer: 2, nEmbd: 128, headSize: 64, vocab: 1024)
var ic = X070InitConfig()
ic.seed = 0
let weights = X070Init.weights(cfg: cfg, init: ic)
let backbone = X070Init.makeBackbone(cfg: cfg, init: ic)
print(weights.count, backbone.cfg.nLayer)
```

Note that orthogonal initialisation uses QR, and **QR does not run on the GPU
in MLX** — an explicit CPU stream is required. For `head` at a vocabulary of
65536 that is tens of seconds, once.

### Token streams

Training data is a flat `uint16` file, memory-mapped, format-compatible with
rwkv-metal in both directions:

```swift
let stream = try BinTokenStream(path: "~/data/train.bin", ctxLen: 512)
print(stream.count, stream.ctxLen)

// Out-of-range ids are caught by sampling rather than by a full scan.
try stream.validateOrThrow(vocabSize: 65536)

let batch = stream.batch(batchSize: 8, step: 0)
print(batch.x.shape, batch.y.shape)

let source = stream.source(batchSize: 8, startStep: 0)
print(source().x.shape)
```

Writing one:

```swift
let url = URL(fileURLWithPath: "/tmp/tokens.bin")
try TokenStreamWriter.write(tokens: [1, 2, 3, 4], to: url)
print(url.lastPathComponent)
```

---

## Embedding and reranker training

Both sit on the same `Trainer`; their specifics are documented separately.

**Embeddings** — [Embedding.md](Embedding.md). Contrastive fine-tuning with
retrieval / STS / classification stages, `BaseTrainingMode` choosing what
moves (`.frozen`, `.topLayers(N)`, `.full`, `.lora`), and GradCache for large
batches. Measured: peak **19 MB at chunk 8** versus **77 MB at chunk 256** on
the same batch, and with a single chunk it is bit-for-bit the ordinary path
(< 1e-5 across parameters after two AdamW steps).

`.full` deliberately excludes `emb.weight` and `head.weight`: logits are never
computed in an embedding task, and the LM head at a 65536 vocabulary is about
a third of a 0.1B model's parameters.

**Reranker** — [reranker.md](reranker.md). A head over frozen-backbone state,
trained on a cache of precomputed states. Because the base is frozen, the map
"pair text → state" is fixed, so states are computed once and the head then
trains on them. Measured: encoding 2400 pairs takes **3:46**, eight epochs of
training on them take **2.7 s**. That asymmetry is the entire point of the
cache.

---

## Measured numbers

| | value | measured on |
|---|---|---|
| encoding 2400 pairs (reranker) | 3:46 | single run |
| 8 epochs of head training on them | 2.7 s | single run |
| GradCache peak memory | 19 MB (chunk 8) vs 77 MB (chunk 256) | in-process |
| GradCache, one chunk, vs ordinary path | < 1e-5 after two AdamW steps | — |
| rounding every batch length to 64 | +38% encoding time, no memory saved | 200 prefixes, 3 runs |
| encoding memory with a 2 GB ceiling | flat 4.3 GB for a whole run | single run |
| encoding memory without a ceiling | 13 GB, machine swapped | single run |
| grad accumulation, accum=2 vs accum=1 on equal microbatches | identical | — |
| trainer refactor, training trajectory | matched to 9 significant figures | via `git stash` |

Two exactness claims worth knowing when you benchmark:

**Splitting a sequence through state is bit-exact.** Any split, including one
not aligned to `CHUNK`, gives exactly zero difference. The recurrence is
sequential within a stream, carrying state is a copy with no arithmetic, and
the padding token is neutral under IEEE-754. Practical consequence: a prefix
cache need not align documents to `CHUNK`.

**The frozen path and the training path produce identical state**, despite
using different kernels. That is what makes a cache built by a frozen pass
legitimate training input.

---

## Not done

- **No sampler**, so nothing here can be evaluated by generating text; see
  [Inference.md](Inference.md).
- **No runner for embeddings.** The reranker has `rerank-run` end to end;
  embedding curricula are library-only and leave no incremental report, so a
  run killed halfway leaves no trace.
- **Reranker training does not expose resumption.** `Trainer` supports it;
  `RerankTraining` does not surface it. Deliberately deferred while training
  takes seconds.
- **`X070PartialFinetune` is not on the shared trainer.**
- **The backbone is not fingerprinted in any checkpoint.** A head trained on
  one backbone and loaded onto another differs in neither shape nor contract.
- **Reservoir sampling is missing from `EmbeddingDataset.loadJSONL`**, so a
  slice of a large corpus is its beginning rather than a sample.
  `RerankDataset` does this correctly and the code transfers by copying.
- **No tokeniser for non-World vocabularies**, which limits pretraining on a
  custom vocabulary: Swift has no equivalent of HuggingFace `tokenizers`.
- **Fixed 10.10: NaN gradients and GPU faults on real models.** Training a
  real model at T >= 32 used to fail intermittently — NaN in the gradients of
  the lowest layers, or a dead process with a Metal command-buffer error
  (Hang / PageFault / InnocentVictim). The cause was in the WKV backward
  kernel's call, not in the data: MLX generates a custom kernel's source from
  the *types of its inputs* but caches the library by *name*, and a call with
  the same name and other types makes it drop the old library and pipeline at
  once — while dispatches already encoded in the command buffer still use
  them. The backward kernel received its primals as they came: layer 0 in the
  parameter type (bf16 / fp16), the layers above in fp32, so in one backward
  pass one layer kept pulling the pipeline from under another. The inputs are
  now cast to fp32, as the forward kernel's always were; results are
  bit-identical where they were correct. A second, independent defect went
  with it: the kernel declared more threadgroup memory than Metal allows
  (37376 bytes against 32768; Metal counts it double). Before the fix: NaN or
  a crash in 3 runs of 3 on a dense 0.1B at T = 48, and a crash in 3 of 4 on a
  1.5B at T = 64; after: 18 rounds and 5 runs clean. Any new custom kernel
  must keep its input types fixed (the quantised-base kernels now assert
  theirs).
