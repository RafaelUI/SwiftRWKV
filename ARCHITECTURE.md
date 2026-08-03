# SwiftRWKV — Architecture

Design record for porting `rwkv-metal` (Python/MLX) to SwiftRWKV (Swift/mlx-swift):
inference, pretraining, fine-tuning, embeddings, and reranking on one backbone.

This document records **decisions and their reasons**, not API. It deliberately
contains no code snippets: signatures churn while the code is being written, and
a design note that churns with them stops being a design note. Per-module guides
with worked examples live in `docs/*.md` and are written *after* each module
works, so their examples can be compiled and their numbers measured.

Code comments are Russian; documentation is English. That matches both
repositories today.

- [Scope](#scope)
- [Where things stand](#where-things-stand)
- [The central observation](#the-central-observation)
- [Module graph](#module-graph)
- [RWKVKernel](#rwkvkernel)
- [RWKVCore](#rwkvcore)
- [RWKVQuant](#rwkvquant)
- [RWKVTraining](#rwkvtraining)
- [RWKVGen](#rwkvgen)
- [RWKVEmbedding](#rwkvembedding)
- [RWKVReranker](#rwkvreranker)
- [SwiftRWKV umbrella](#swiftrwkv-umbrella)
- [Cross-cutting decisions](#cross-cutting-decisions)
- [Python to Swift: the real gaps](#python-to-swift-the-real-gaps)
- [Open questions](#open-questions)
- [Order of work](#order-of-work)

---

## Scope

`rwkv-metal` is ~7,000 lines of Python across seven concerns: kernel, model,
tokenizer, pretraining, LoRA/QLoRA, embeddings, reranking. SwiftRWKV currently
covers roughly the first third of that — about 1,900 lines.

The target is feature parity for everything that makes sense on-device, exposed
through an API simple enough to drop into an app without reading the source.
Two properties are non-negotiable, because both repos already depend on them:

- **Numerical parity with the Python path.** The WKV-7 kernel is verified
  bit-exact against a reference; the quantized dequant path is verified
  bit-exact against a PyTorch reference. Ports that "look right" but drift
  silently are worse than missing features, because a drifting model still
  produces plausible output.
- **Bounded memory.** Every design choice here that looks over-engineered
  (disk-backed caches, gradient checkpointing, cache limits, GradCache) exists
  because a 16 GB Mac swaps otherwise, and swapping ruins both throughput and
  every measurement taken while it happens.

---

## Where things stand

Present in SwiftRWKV:

| Component | Status |
|---|---|
| WKV-7 Metal kernel, forward | complete, chunked, state-threading internally |
| WKV-7 Metal kernel, backward | complete, gradient-checkpointed, parity-tested |
| `X070Backbone` | complete — canonical x070, LoRA hooks, quant hooks, block checkpoint |
| Recurrent generation | complete, but `B = 1` only |
| `WorldTokenizer` | complete (World vocabulary only — no BPE path) |
| LoRA / QLoRA engine + LM fine-tune loop | complete |
| N-layer partial fine-tune | complete, classification objective only |

Absent: pretraining, embeddings, reranking, the `.rwkvq` quantization backend,
batched state, and any shared training infrastructure.

---

## The central observation

The two existing Swift training paths each fuse three independent concerns into
one monolith. Separating them makes the remaining work small and mostly
combinatorial rather than novel.

The three axes:

- **What is differentiated** — LoRA adapters, `.rwkvq`-backed adapters, the top
  N layers as full fp32 weights, every weight, or a head alone.
- **What the loss is** — LM cross-entropy, classification, contrastive, listwise
  or BCE ranking.
- **Where inputs come from** — a token stream, a disk-backed boundary cache, or
  a state cache.

Filled in, the matrix shows how little is genuinely new:

| trainable ↓ / objective → | LM CE | classification | contrastive | listwise + BCE |
|---|---|---|---|---|
| LoRA / QLoRA / `.rwkvq` | **done** | — | needed | — |
| top N, full fp32 | — | **done** | needed | — |
| all weights | needed | — | — | — |
| head only, frozen base | — | — | — | needed |

Two of eight cells exist, each written as if it were the only one.

**Decision: one `Trainer`, parameterized by three protocols — not one config
struct with flags.** A flag-based configuration for this matrix would carry
roughly forty fields of which most combinations are invalid, and the invalid
ones would fail at runtime deep inside a training loop. Protocol conformance
moves that failure to compile time.

**Consequence for pretraining.** Pretraining is not "partial fine-tuning with
N set to every layer", which was the initial hypothesis. `X070PartialFinetune`
is the *classification* path: it optimizes a pooled classifier head against
labels, scores itself with `argMax` accuracy, and its entire structure exists to
serve `buildBoundaryCache` — a disk cache of activations at the frozen/trainable
boundary. At `freeze = 0` that cache degenerates to storing the embedding layer's
output, and the objective is wrong regardless.

The actual donor is `LoRAFinetune`, which is already an LM loop: token pairs,
cross-entropy over `[B, T, vocab]`, gradient accumulation with an `eval` between
micro-steps, global-norm clipping, warmup, an fp32 master copy, and a Metal
cache limit. Pretraining differs from it in exactly three ways — the trainable
set is every weight rather than the adapters; there is no post-warmup decay
schedule and no resume; and there is no weight initialization path at all, since
`X070Backbone` is constructed *from* loaded weights.

That last one is the only part requiring genuinely new code, and it is small:
RWKV-7 initialization zeroes every LoRA-B matrix (making all dynamic parameters
neutral at step 0), scales `k_proj` by 0.1 to damp the WKV recurrence, scales
`r_proj` and `v_proj` by `1/sqrt(n_layer)`, and scales the output head by
`1/sqrt(n_embd)`. Without it, NaN on the first step is guaranteed.

---

## Module graph

```
RWKVKernel ──┬──────────────────────────────────────────┐
             │                                          │
          RWKVCore ──┬── RWKVQuant                      │
             │       │                                  │
             │   RWKVTraining ─┬── RWKVGen ─────────────┘
             │                 ├── RWKVEmbedding
             │                 └── RWKVReranker
             │
        (all re-exported by the SwiftRWKV umbrella target)
```

**Decision: split `RWKVGen` rather than keep one library.** Today `RWKVGen`
holds backbone, generation, LoRA, and partial fine-tuning together, and "import
one module" is a real virtue of the current API. But an app that only reranks
should not link a pretraining loop, and an embedding module that must import a
generation module to reach the backbone has its dependency arrow backwards.

The cost — losing the single-import story — is paid back by an umbrella target
that re-exports everything, so the simple case stays one line while a
size-sensitive app can depend on exactly what it uses.

---

## RWKVKernel

**Responsibility.** The WKV-7 recurrence as Metal kernels, forward and backward,
plus the constants that define chunking. Nothing above the recurrence.

The forward kernel already threads state chunk-to-chunk: `wkv7ChunkForward` is
public, accepts `h_in`, and returns `h_out` alongside the output and the `sa`
buffer. The backward kernel already computes `dh_in_out`. Both are parity-tested.

**This is the most important finding of the survey.** The perceived headline
risk of this port — "batched recurrent state through a hand-written Metal
kernel" — is largely already built. What is missing is not kernel work but
plumbing above it:

- `wkv7Forward` initializes `h` to zeros and discards the final `h` after its
  chunk loop. Both ends need to become parameters.
- `wkv7Train` hardcodes a zero `h_in` and explicitly drops `dh_in` from the
  gradient tuple. Exposing both makes the recurrence differentiable with respect
  to its initial state, which is what lets a head train starting from a cached
  state.

**Decision: extend signatures, do not fork the kernels.** The Metal sources stay
untouched. Two correctness surfaces for the same recurrence is exactly the
failure mode this project has avoided so far, and the parity test is written
against the existing kernels.

**Constraint that propagates everywhere.** `CHUNK = 16`, and any `T` entering a
trainable layer must be a multiple of it. 32 diverged on 1.5B; 16 is numerically
equivalent and stable. This constraint currently surfaces as scattered
`precondition` calls at call sites — it belongs in one place, enforced where
batches are constructed rather than where they are consumed.

---

## RWKVCore

**Responsibility.** The backbone, the recurrent state, the reusable block, the
tokenizer, logging. Everything that both inference and training need and neither
owns.

Three things must change or arrive here, and together they are the largest piece
of work in the port.

**1. Batched `RWKVState`.** The existing `RWKVState` lives in the generation file
and is single-sequence: per-layer arrays sized for `B = 1`, built for
token-at-a-time decoding. Everything downstream of it — reranker state caches,
prefix reuse, batched scoring — needs `[L, B, H, S, S]` with slicing, gathering,
repetition, concatenation along the batch axis, and `stopGradient`.

Two details from the Python type are load-bearing and easy to lose:

- **Token-shift state belongs in the state.** Per layer, the state carries not
  just the WKV matrix but the last `ln1(x)` and `ln2(x)` values. Without them a
  continuation differs from a contiguous pass at the first token of every layer,
  precisely where token-shift reaches for a predecessor and finds zero. This is a
  silent, small, plausible-looking error — the worst kind.
- **`v_first` deliberately does *not* belong in it.** In x070 `v_first` is not a
  running quantity; layer 0 recomputes it per position and layers above consume
  it at the same position. A continuation computes its own from its own tokens.
  Note that the current Swift generation state *does* carry `vFirst`, which is
  correct for its use (one uninterrupted sequence, `step` by `step`) and wrong as
  a general continuation boundary. The two types should not be merged
  carelessly; the batched type is a new type with different semantics, and the
  generation type may end up expressed in terms of it.

**2. A chunked forward that accepts and returns state.** `body` today takes ids
and returns `ln_out`, with no way to seed or retrieve the recurrence. The
reranker's entire economics — a document folded once, each query costing only its
own tokens — depends on this existing and running through the Metal kernel.
Doing it by stepping one token at a time would be correct and useless: a
512-token document would be encoded at decode speed.

**3. Right-padding with an exact mask.** Absent from Swift entirely; present and
carefully specified in Python, where pad positions are made no-ops for the
recurrence so a row's final state is independent of both its padding and its
batch neighbours, and the state is read at the last *real* token rather than at
the end of padding. Without this, batching sequences of unequal length silently
corrupts the shorter ones.

**4. A non-World tokenizer.** `WorldTokenizer` covers the official 65536-token
World vocabulary and nothing else. Pretraining produces from-scratch checkpoints
with their own vocabularies, and the Python side handles this with a thin
wrapper over HuggingFace `tokenizers` presenting the same `encode` / `decode`
interface, so every downstream consumer works with either. Swift has no
equivalent library, which makes this a genuine open question rather than a
transcription: either a minimal BPE implementation, or restricting pretraining
to a byte-level or World vocabulary. It does not block anything except
pretraining on a custom vocabulary, so it should not be resolved early — but it
should not be discovered late either.

**Decision: promote the block to a first-class type.** Blocks currently exist as
internal methods on the backbone. The reranker head is a short stack of RWKV
blocks initialized from selected base layers and run over a cached state — it
needs blocks as constructible, runnable objects. This is a refactor of existing
correct code, not new math, but it is a prerequisite for the reranker.

**Logging.** Progress callbacks (`onStep`, `onEpoch`) are the right shape for UI
binding and stay as they are. What is missing is diagnostics: a logger protocol
defaulting to `os.Logger`, so an embedding app's training run is filterable by
subsystem in Console.app instead of being lost to stdout. Structured events —
gradient norm, tokens per second, peak resident MB — should be emitted as values
rather than formatted strings, or they cannot be plotted.

---

## RWKVQuant

**Responsibility.** The `.rwkvq` (`gw_mode="sb6"`) quantized-base backend: load
the exported sidecar and reconstruct dense weights on the fly.

Kept separate from `RWKVCore` because it owns a file format, a manifest, and its
own Metal kernel. A build that never touches quantized weights should not carry
any of it.

**Risk assessment: low, contrary to first impressions.** `MLXFastKernel` in
mlx-swift takes the same name / input names / output names / source / header
shape as `mx.fast.metal_kernel`, and `WKV7.swift` already builds a kernel from a
source string with generated header constants. The fused dequant kernel ports as
the same Metal source with different host binding.

**Two invariants that must survive the port**, both already paid for in Python
with bit-exact verification against a PyTorch reference:

- The final combine is fp32, not half. The half path is ~18% divergent on one
  bit of bf16 mantissa; the REDUCTION preset is calibrated against the fp32
  math, so a half combine would add noise on top of a calibration that assumes
  its absence.
- Dequantized weights are transient and must not be cached. The entire point of
  a quantized base is that it lives compressed in memory; a dense cache silently
  converts QLoRA back into LoRA with extra steps.

**Decision: two-stage conversion stays.** Reading `.rwkvq` natively requires
torch; both repos are deliberately torch-free at runtime. Export remains a
one-time step performed in `rwkv-quant`, and this module consumes only the
resulting safetensors plus JSON manifest.

---

## RWKVTraining

**Note on the name.** Not `RWKVTrain`: that name belonged to a module that was
removed (a from-scratch backbone plus a classification trainer, whose
capability now lives on `X070Backbone`), and [docs/Train.md](docs/Train.md) still documents its
removal. Reusing the name for something structurally different would make the
repository's own history misleading.

**Responsibility.** Everything common to all four training tasks: the optimizer,
the learning-rate schedule, gradient accumulation, global-norm clipping,
checkpointing and resume, and the training loop itself. It owns no objective and
no model.

The three protocols from [the central observation](#the-central-observation)
live here. Concretely they must account for how mlx-swift differs from Python
MLX: gradients are taken over a flat array of tensors with explicit argument
numbers, not over a module tree. So the trainable-set abstraction has to flatten
its parameters into an ordered array and reassemble them into the backbone's
weight-override map — which is exactly what both existing Swift training paths
already do by hand, with their own bookkeeping each.

**What must be added beyond what exists.** The LoRA loop implements warmup and
then holds the learning rate flat; there is no cosine or linear decay and no
`lr_min`, both of which pretraining needs. Neither existing path has checkpoint
and resume. Both hand-roll AdamW inline. All three are single implementations
serving four callers.

**Decision: hand-rolled AdamW stays.** Both paths already do this, decoupled
weight decay included, and it composes with the fp32-master pattern that
adapter training requires — bf16 updates are lost to rounding, so the master
copy is fp32 and a bf16 copy is injected inside the loss closure so the cast
keeps the gradient chain intact. This is a validated recipe, not an accident.

---

## RWKVGen

**Responsibility.** Generation and the LM objective — including pretraining,
which is an LM objective over a token stream with the trainable set widened to
everything.

Also home to weight initialization, since pretraining is the only caller.

Generation should gain batching by expressing its state in terms of the batched
type from `RWKVCore`, but the `B = 1` decode path is correct today and is not on
the critical path for any other module.

**Dataset streaming** — `.bin` token streams, on-the-fly tokenization of raw
text, and the out-of-vocabulary check that currently warns before training
rather than producing NaN mid-run — belongs here as a `BatchSource`.

---

## RWKVEmbedding

**Responsibility.** Text-to-vector: pooling, the trainable head, contrastive
objectives, and GradCache.

**The cheap half already works.** `body` returns `ln_out`, so extracting vectors
from a base model needs only pooling and L2 normalization. RWKV folds a sequence
into a fixed-size state by construction, so last-position pooling gives a
usable, if unpolished, embedding with no training at all.

**The head is a sibling of the base, never a child.** This is a correctness
requirement, not a style preference: freezing or quantizing the base walks the
base's parameter tree, and a head living inside that tree would be silently
frozen along with it. The head is zero-initialized so it is the identity at step
0 and does not disturb the pretrained geometry before training has said anything.

**GradCache is the substantive piece.** The negative pool in the contrastive
loss *is* the batch, so retrieval quality scales with batch size — but so does
activation memory, and RWKV-7 activations over long passages exhaust unified
memory quickly. GradCache decouples the two: three phases (embed all chunks
without gradient and cache the vectors; compute the loss and its gradient with
respect to those cached vectors, where the whole batch interacts at once;
re-forward each chunk with gradient seeded by its slice of that gradient).
Activation memory then scales with chunk size while the loss still sees the
entire pool.

It is exact, not an approximation, and the Python measurements are unusually
convincing: at one chunk the gradient difference sits at the run-to-run noise
floor, meaning cutting the graph and re-seeding introduces no error whatsoever;
the residual at four chunks is floating-point summation order alone. Plain
gradient accumulation, by contrast, deviates by roughly 400% — because it
genuinely changes the contrastive math, each micro-batch seeing only its own
negatives. That contrast is the reason this module exists.

Memory measured in Python, bf16, 800-character passages: eager grows from 3.4 GB
at batch 8 to 9.7 GB at batch 48; GradCache stays flat at ~3.2 GB throughout,
costing about 30% wall-clock.

**Decision: GradCache is a scheduling algorithm and gets no Metal kernel.** Its
per-chunk loop *is* the algorithm — that loop is what bounds activation memory.
The arithmetic already runs on the GPU: the recurrence through the existing
kernel, the similarity and softmax through MLX ops. A custom kernel would add a
second correctness surface and no speed.

---

## RWKVReranker

**Responsibility.** Cross-encoder scoring of query-document pairs read from the
recurrent state, the state cache that makes training it cheap, and inference
against a prepared document index.

**The idea.** A transformer cross-encoder concatenates the pair, runs the stack,
and reads a score off `[CLS]`. RWKV has already folded the pair into a
fixed-size state per layer, so the score can be read from the state directly:
push a few learnable probe tokens through a short stack of RWKV blocks whose
recurrence *starts* from that state, and project to a scalar. Reading the state
this way is one attention-like query against everything it accumulated.

Three consequences follow, and each is an architectural constraint rather than a
nice property. The head sees a full `[S, S]` matrix per head instead of a pooled
`[D]` vector. Head cost is independent of pair length — one or two tokens,
always. And the prefix state is cacheable, which is what makes both training and
serving affordable.

**This module is the reason `RWKVCore` needs batched state.** It cannot be built
first and it cannot be built without that work.

**The state cache.** With a frozen base, the map from pair text to state is
fixed, so pairs are encoded once and the head trains on stored states. The head
is tiny, so an epoch over tens of thousands of pairs takes seconds instead of
tens of minutes — which is what makes large batches, many epochs, and honest
hyperparameter search possible on a laptop.

Only the layers the head actually reads are stored, not the full state. For the
default single-layer head that is 1 layer of 12 — roughly 98 KB per pair in bf16
against 2.4 MB for a full fp32 state.

**Decision: the cache is memory-mapped and lives outside MLX.** In Python it is
numpy, for reasons that apply identically in Swift: the cache is gigabytes while
a step touches hundreds of rows, so holding it as MLX arrays forces a full copy
at creation and offers no way to avoid holding it resident. A mapped file lets
batches be gathered cheaply and converted to MLX arrays at batch size, and lets
the cache exceed RAM. Swift has no numpy, but `X070BoundaryCache` already
demonstrates the pattern with a memory-mapped `Data`.

**Document first, then query.** The ordering is not arbitrary: it is what makes
the prefix cacheable across queries. Documents are many and queries per document
are few, so the expensive prefix is computed once per document and each query
continues from it at the cost of its own tokens. Python measures ~73 ms for a
512-token document against ~3.5 ms per additional pair — a full pass at 584 ms
versus 28 ms for a 16-token continuation from cache.

**Checkpoints must carry their configuration.** A head trained on layer 5 and a
head trained on layer 11 have identical tensor shapes. Loading one into the
other therefore succeeds silently and produces confident nonsense. The Python
implementation stores `layer_idx`, template order, truncation limits, terminator
and instruction text in safetensors metadata and refuses a mismatch on load.
This must port — it is the difference between an error message and a week of
debugging a model that merely scores badly.

---

## SwiftRWKV umbrella

**Responsibility.** Re-export every module so the common case remains a single
import, and hold the compiled documentation snippets.

**Decision: documentation examples are compiled.** The failure mode of written
documentation is not that it needs rewriting — it is that it silently stops
being true. If every `docs/*.md` quick-start exists as a compiled snippet, a
changed signature breaks the build and names the file to fix. This also makes
the public API's ergonomics a build-time concern rather than an opinion, and
gives the package a free smoke test of every documented entry point.

---

## Cross-cutting decisions

**Precision.** bf16 weights with fp32 accumulation where it matters — loss, the
WKV recurrence, optimizer master copies. This is mixed precision, not "bf16
training", and matches how RWKV-7 World was trained. The WKV state stays fp32
unconditionally; its precision is what determines whether a continuation matches
a contiguous pass.

**Memory.** Every task must run within a phone or a 16 GB Mac. The levers, in
the order they should be reached for: gradient checkpointing at block level
(roughly −45% peak for +22% time), a Metal cache limit, gradient accumulation,
disk-backed caches for frozen prefixes, and GradCache where the objective's
batch coupling forbids plain accumulation.

**Cancellation and progress.** Every long-running operation takes a cancellation
predicate and a progress callback. Both existing training paths already do this;
it is a requirement for anything invoked from a UI, and retrofitting it is
harder than starting with it.

**Errors.** Invalid configurations should be unrepresentable where possible and
caught at construction otherwise. The specific trap this project has already hit
is silent shape-compatible mismatch — see reranker checkpoints above. Shape
compatibility is not correctness.

---

## Python to Swift: the real gaps

MLX's Python and Swift APIs are close enough that most of this port is
transcription. Four places are not.

**Gradients over flat arrays.** Python takes gradients over a module tree and is
`freeze()`-aware; mlx-swift takes them over an ordered array with explicit
argument numbers. This shapes the trainable-set protocol, and it removes the
convenience that keeps GradCache composing automatically with frozen layers and
LoRA — that composition must be arranged explicitly.

**Compilation with captured state.** The Python pretraining loop compiles its
step function with the model and optimizer state declared as inputs and outputs.
Whether mlx-swift offers an equivalent needs verifying before the pretraining
loop is designed; neither existing Swift path uses one. If it does not, the loop
is straightforward but slower, and that cost should be measured rather than
assumed.

**No numpy.** Both large caches lean on numpy memory-mapping. `Data` with
`.alwaysMapped` plus explicit offset arithmetic is the replacement, already
demonstrated in the existing boundary cache.

**Closure capture in the loss.** The existing LoRA loop routes the current batch
through file-private mutable globals to avoid capturing `inout` state in the
gradient closure. It works, but it is not thread-safe and will not survive two
trainers in one process. A shared trainer should solve this once, properly.

---

## Open questions

To be resolved by experiment, before the affected module is designed:

- Does mlx-swift expose compilation with captured model/optimizer state? If not,
  what does the pretraining loop actually cost?
- What is the real throughput of a chunked state-returning forward versus the
  Python measurements, on the same hardware? The reranker's economics are quoted
  from M4 Air numbers that have not been reproduced in Swift.
- Does exposing `dh_in` through the training kernel change any existing parity
  test result? It should not — the value is already computed and discarded — but
  "should not" is not "does not".
- Can the generation state be expressed in terms of the batched state without
  regressing the recurrent-versus-parallel parity test?

---

## Order of work

Sequenced so that each step de-risks the next, and so that the two modules with
real unknowns come before the two that merely have volume.

1. **Kernel plumbing.** Expose `h_in` / `h_out` on the forward path and `h_in` /
   `dh_in` on the training path. Existing parity tests must still pass;
   add one for state continuity — a split pass through the state must equal a
   contiguous pass.
2. **`RWKVCore` state work.** Batched `RWKVState`, chunked forward with state,
   right-padding with mask, block as a type. Largest single piece; everything
   downstream waits on it.
3. **`RWKVTraining` extraction.** Refactor the two existing paths onto the shared
   trainer without changing their behaviour. Their current results are the
   regression test — if LoRA fine-tuning does not reproduce, the abstraction is
   wrong.
4. **Pretraining.** Weight initialization, decay schedules, resume, `.bin`
   streaming. Small once step 3 exists.
5. **`RWKVQuant`.** Independent of steps 2–4; can be done in parallel by
   whoever is not blocked. Bit-exactness against the Python path is the
   acceptance test.
6. **`RWKVEmbedding`.** Head, contrastive objectives, GradCache. Verify
   GradCache against eager at one chunk first — that equality is the test that
   proves the implementation.
7. **`RWKVReranker`.** State cache, head, inference, checkpoint metadata.
8. **`docs/*.md`** per module, written as each lands, with compiled examples and
   measured numbers.
