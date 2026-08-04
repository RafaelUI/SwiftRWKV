# Inference

Running a trained RWKV-7 model: text generation, reranking, embeddings, and
the quantised `.rwkvq` backend that all three can sit on.

The four sections are independent to read but share one backbone and one
weight-access path. That sharing is the point of this page: the interesting
failures happen where a task-specific path diverges from the common one.

**About the numbers.** Everything measured below was taken on RWKV-7 World
0.1B (L=12, D=768) on Apple silicon unless stated otherwise. Code examples are
compiled by `Scripts/check_docs.sh`; it does **not** re-verify the numbers.

**Timings marked *(debug build)* are not what you will see.** They were taken
under `swift test`, which builds unoptimised, and on a dispatch-heavy workload
that costs a lot. Worse, it does not cost a *constant* amount: on the two
branches of the decode loop measured below, debug inflated one by 2.2x and the
other by 5.0x. Debug numbers therefore cannot be rescaled — they can only be
replaced. The one figure below taken in release is labelled as such.

Task-specific detail lives in [reranker.md](reranker.md) and
[Embedding.md](Embedding.md); this page covers the inference side of each.

---

## Two ways through the model

Everything here is one of two passes.

**Parallel** — `body(ids)` over a whole sequence at once. Used for prefill,
for encoding pairs, for embedding a text. Throughput-oriented; memory scales
with sequence length.

**Recurrent** — `step(id, state:)`, one token at a time carrying `RWKVState`.
Used for generation. Constant memory per step regardless of how much came
before, which is the property RWKV exists for.

They compute the same function and are held together by a test: measured
**7.4e-7** relative on logits, and the same top-1 token. Not zero, and it
cannot be — the two use different kernels — but far below anything that
changes an argmax.

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

## Decode: generating text

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
guard let tok = WorldTokenizer(
        vocabURL: URL(fileURLWithPath: "rwkv_vocab_v20230424.txt")) else {
    fatalError("cannot read vocabulary")
}

let result = base.generate(prompt: "The capital of France is",
                           maxTokens: 32, tokenizer: tok)
print(result.text, result.tokens.count, result.stopReason)
```

That is the whole surface for the common case. `generate` defaults to **greedy**
decoding — `SamplingConfig()` has `temperature = 0` — because a package that
produces different text on every run by default cannot be debugged.

Underneath it are the two primitives, and they remain public:

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")

var state = RWKVState(cfg: base.cfg)
var logits = base.prefill([1, 2, 3], state: &state)
for _ in 0 ..< 32 {
    let id = logits.argMax().item(Int.self)
    logits = base.step(id, state: &state)
}
```

`prefill` folds the prompt and returns the logits of its last token; `step`
advances by a single token. `RWKVState` holds the WKV state and the
token-shift buffers for every layer — nothing else is needed to continue,
which is why generation is O(1) in memory per step. A test holds that greedy
`generate` produces token-for-token the same sequence as this hand-written
loop; that is what keeps the convenience API from drifting into a second
implementation.

### Prefill is a parallel pass, not a loop of steps

A prompt is known in full before anything is generated, which is exactly the
case the parallel pass exists for. So `prefill` runs `bodyWithState` — the
same pass the reranker's prefix cache is built on — and converts the batch
state it returns into the single-sequence `RWKVState` that `step` continues
from.

The difference is not marginal. Measured on 0.1B, swap unmoved either side:

| prompt | recurrent | parallel | speed-up |
|---|---|---|---|
| 16 | 493 ms | 50 ms | 9.8× |
| 64 | 1728 ms | 70 ms | 24.9× |
| 256 | 6975 ms | 155 ms | 44.9× |
| 512 | 14249 ms | 171 ms | 83.5× |
| 1024 | 30262 ms | 321 ms | 94.4× |

The recurrent path sits at a flat ~29.6 ms per token — the same cost as
generating one. The parallel path falls from 3.1 ms/token at length 16 to
0.31 ms/token at 1024, because its fixed startup (twelve layers of kernel
launches) is amortised. A 1024-token prompt went from half a minute to a
third of a second.

The token-by-token version is still there as `prefillRecurrent`, and it is
kept deliberately: it is the reference the fast path is checked against.
Tests compare the two on the last-token logits, on the state layer by layer,
and on sixteen tokens of greedy continuation. Comparing only the logits would
not be enough — logits are read from the network's output and do not depend
on the token-shift buffers at all, so a prefill that dropped them would return
a correct next token and a broken continuation.

`RWKVState.vFirst` is left `nil` after a parallel prefill, and nothing is lost
by that. In x070 `v_first` is not a running value: layer 0 recomputes it at
every position and the layers above consume it at that same position, so it
never crosses a call boundary. A test asserts this directly — clearing
`vFirst` before a `step` changes the logits by exactly zero.

Greedy decoding is deterministic, and a test holds that too: the same prompt
yields the same continuation, and a *different* prompt yields a different one.
The second half matters — without it the test would pass for an implementation
that ignored the prompt entirely.

### Sampling

```swift
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
```

**The same seed gives the same text.** Randomness comes from a local
SplitMix64, not from `MLXRandom` — the latter is process-global state shared
with weight initialisation and training, so a sampler that drew from it would
make generation depend on whatever ran before it, and would perturb a training
run happening alongside. Reproducibility could not be claimed at all.

Two degenerate configurations are worth knowing because they are the ones
people reach for to check that the knobs are wired up at all:

| configuration | result |
|---|---|
| `temperature = 0` | argmax, and the RNG is **not** advanced |
| `temperature → 0⁺` | argmax, reached through the general selection path |
| `topK = 1` | argmax, also through the general path |
| `topP` below the mode's probability | the mode alone — never an empty set |

Neither `topK = 1` nor a tiny temperature is special-cased in the code. That
is deliberate: a short circuit would make the test "top-k = 1 equals greedy"
compare a branch against itself, which is how a test quietly stops testing
anything.

`topK` and `topP` **intersect**; they do not override one another. The
surviving candidates are always renormalised, so with `p = [0.5, 0.3, …]` and
`topP = 0.75` the two survivors are drawn at 0.625 / 0.375, not 0.5 / 0.3.

### Repetition penalties, and which one actually works

Three knobs, and they are not interchangeable.

- `presencePenalty` — subtract a constant from a token's logit once it has
  appeared.
- `frequencyPenalty` — subtract in proportion to how often it has appeared.
- `penaltyDecay` — multiply the accumulated counts by this each step, so old
  tokens fade. This is the ChatRWKV convention.
- `repetitionPenalty` — the multiplicative CTRL-style penalty: divide the
  logit if positive, multiply it if negative. Both directions lower it.

**Prefer the additive ones.** The CTRL penalty has a property that surprises
people, and it is a property of the formula rather than of this
implementation: logits are essentially `log p`, so they are almost all
negative, and once the history covers most of the candidates, multiplying
*every* logit by 3 is arithmetically the same as dividing the temperature by
3. The distribution gets **sharper**, and repetition goes **up**. Measured
here on a five-token synthetic distribution over 300 steps: 128 immediate
repeats without any penalty, **164 with `repetitionPenalty = 3`**. A test
pins this behaviour down so it is discovered once rather than repeatedly.
`frequencyPenalty` is a shift, is independent of logit scale, and reduces
repeats as expected.

### Streaming, stop conditions and state

`onToken` receives `(chunk, tokenId)` and returns `false` to stop. The chunk
may be **empty** and it may be **several characters long**, for two reasons
that both have to be handled by the loop rather than by the caller:

- World tokens are *bytes*. A single Cyrillic character is two tokens; decoding
  each token to a `String` on its own yields `?` for half of them. The loop
  emits only bytes that already form complete code points.
- If a stop string is `"\n\n"` and the first `"\n"` has already been handed to
  `onToken`, it cannot be taken back. So a tail that could still turn out to be
  the start of a stop string is held until it is settled.

Stop strings are searched across the whole byte stream, not inside the last
token — `"\n\n"` is normally two separate tokens and appears in neither of them
alone. The stop string is cut from `text`; the tokens that formed it stay in
`tokens`.

**After `generate` returns, `state` has absorbed the prompt and every token in
`tokens`**, including the one that triggered the stop. A test compares that
state against one built from scratch over the same token sequence and requires
exact equality. Without this, continuing generation from the returned state
would silently drop one token — an error that surfaces a hundred steps later
as text that is merely a bit wrong.

### What sampling costs

Measured on 0.1B, **debug build**, 64 tokens each, after warm-up (see the note
on debug timings at the top — the absolute values are inflated, the ratio is
the point):

| | ms/token |
|---|---|
| greedy | 34.30 |
| `temperature = 1`, `topP = 0.9` | 34.93 |
| the same plus `frequencyPenalty` | 35.37 |

So sampling adds roughly **0.6 ms per token, under 2%**. In isolation one
`pick` over the full 65536 vocabulary takes **1.54 ms**, of which the sort is
a small part — most of it is MLX dispatch and the synchronisation needed to
read the chosen id back. The selection itself is a GPU `argSort` plus a CPU
walk over the sorted prefix: the walk has an early exit, which masks express
only as full-width operations.

### Decode goes through the shared projection path

`step` uses the same weight access as the parallel pass, which is why it works
with a quantised backbone and applies LoRA adapters. That was not always true,
and the way it failed is worth keeping in mind:

- on a quantised backbone it **crashed** with a bare force-unwrap, because it
  read the dense weight dictionary directly while `attachRwkvq` had dropped
  the dense copies;
- with LoRA adapters attached it **silently ignored them** — a fine-tuned
  model generated as if it had never been fine-tuned. No error, plausible
  text. Measured divergence from the parallel path was over 10%; after the fix
  it is 1.1e-5.

The second is the worse failure, and it is the reason a second copy of the
arithmetic is a liability rather than an optimisation.

---

## Reranker inference

Scoring a query against candidate documents. Full detail in
[reranker.md](reranker.md); the inference-relevant parts:

```swift
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
```

Two modes: the direct path recomputes every pair in full, the indexed path
folds the "Instruct + Document" prefix once and continues it with each query.
Measured on 0.1B: **72.0 ± 0.2 ms** per prefix versus **6.1 ± 0.0 ms** per
tail, so extra candidates are nearly free once documents are folded.

The paths agree on ordering across all 640 measured pairs while differing by
6.5e-4 on the raw scores. The honest phrasing is "same ordering", not
"identical".

Scores are raw logits: comparable within one query, not across queries.

---

## Embedding inference

Turning texts into vectors to compare by cosine. Full detail in
[Embedding.md](Embedding.md).

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
let vectors = embedder.embed(corpus)
let query = embedder.embed("how is honey made").reshaped([1, -1])
print(cosineSimilarity(query, vectors).shape)
```

Each text is a separate parallel pass with no padding, so a short string
cannot be affected by a long one sharing its batch.

**A raw backbone is not an embedder.** Measured on a LitRetrieval slice, the
mean cosine between *positive* pairs (0.8619) is *lower* than between negative
ones (0.8723) — separation worse than chance — and all documents sit in a cone
with mean pairwise cosine 0.9127. Fine-tune before using this for retrieval.

---

## The `.rwkvq` quantised backend

`.rwkvq` is the checkpoint format of [rwkv-quant], a quantisation toolkit for
RWKV-7 on Apple silicon. It is a **sidecar**: quantised weights live in their
own file, and a dense backbone is told to read from it instead of from its own
copies. Nothing about it is specific to generation, reranking or embeddings —
it replaces weight access underneath all three.

[rwkv-quant]: https://github.com/RafaelUI/rwkv-quant

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
let sidecar = try RwkvqSidecar(path: "~/models/0.1B_reduction.rwkvq_mlx")

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
```

Measured on 0.1B with the `reduction` preset: **73 tensors attached, 270 MB of
dense weights freed, 153 MB packed in their place.**

By default `attachRwkvq` covers the four tmix projections on every layer plus
the LM head. What it covers is configurable:

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
var opts = X070Backbone.RwkvqAttachOptions()
opts.quantizeCmix = true        // also cmix key/value
opts.quantizeEmbedding = true   // see the caveat below
opts.layers = 4 ..< 12          // leave the early layers dense
opts.dropDenseWeights = false   // keep dense copies so detach can restore
opts.useNativeKernel = true     // run through MLX's quantised matmul, see below
print(base.attachRwkvq(try RwkvqSidecar(path: "~/models/q.rwkvq_mlx"),
                       options: opts).attached)
```

`useNativeKernel` is the one option that changes speed rather than what is
quantised. Off (the default), every projection unpacks the whole matrix into a
dense transient and then does an ordinary matmul. On, the sb6 data is
relayouted once at load into the container MLX's own `quantizedMM` expects, and
no dense matrix is ever materialised. The numbers are identical — it is a
relayout, not a requantisation, and the parity is asserted rather than assumed
— while decode on 2.9B goes from 151 to 30 ms/token. It defaults to off because
the unpack-per-call behaviour is deliberate for QLoRA, where the frozen base
has to stay compressed in memory; for inference it is simply a cost.

**The embedding table is not quantised by default**, and that is a
considered choice rather than an omission: the sb6 format packs codes in
blocks along the input axis, so a single row cannot be fetched more cheaply
than a block. Reading one token's embedding means unpacking the whole
65536×768 table — about 200 MB of transient per pass. It works, and a test
covers it, but it costs memory to save memory.

### What quantisation does and does not change

It **does** change the numbers, materially. Measured on this 0.1B with the
`reduction` sidecar: relative difference on logits versus the dense model is
**0.62**, and the top-1 token differs on a short prompt. Do not expect
identical output from a quantised and a dense model.

On the default path it also costs about **2.4× on every projection**, and that
cost does not go away with scale. Each projection unpacks the whole matrix and
then does an ordinary matmul; on a `[1, D]` vector both halves are
bandwidth-bound over the same matrix, so unpacking is roughly a second pass.
(`useNativeKernel` removes this entirely — the table below describes the
default.) Measured on the sidecars directly, with swap unmoved either side:

| | dequantise | matmul | together | unpack share |
|---|---|---|---|---|
| 0.1B, `768×768` (×48) | 0.355 ms | 0.334 ms | 0.413 ms | 86% |
| 0.1B, `65536×768` (×2) | 3.919 ms | 2.487 ms | 5.177 ms | 76% |
| 2.9B, `2560×2560` (×128) | 0.624 ms | 0.641 ms | 1.003 ms | 62% |
| 2.9B, `65536×2560` (×2) | 9.423 ms | 7.109 ms | 16.564 ms | 57% |

All of one token's projections under a single `eval` — the four tmix
projections per layer plus the head, which is what `attachRwkvq` covers by
default — come to **8.4 ms on 0.1B and 99.6 ms on 2.9B**. The second number
is a ceiling of about 10 tokens/second before normalisation, WKV, cmix and
sampling are counted at all.

The natural expectation is that unpacking becomes dominant only once matrices
are large. It does not: it is already 76–86% of the work at 0.1B, and its
*share* is slightly lower at 2.9B. The penalty is a roughly constant factor,
not a scale-dependent one.

What a fused decode-time GEMV would remove is visible in the traffic. Today
each projection reads the packed data, writes a dense float32 copy, and reads
it back — for the whole 2.9B model that is 1795 MB + 11 328 MB + 11 328 MB,
about **13.6× the packed size**. A kernel decoding the packed format in place
reads the packed data and nothing else.

`dequantize` takes a `dtype`: the combine is always computed in float, but the
result can be stored as bfloat16 instead of float32, halving the transient. It
is bit-for-bit identical to the float32 result rounded to bfloat16 — which is
what the caller did anyway — and a test asserts that on every tensor of the
sidecar. On a synthetic bfloat16 input the whole token's projections go from
14.2 ms to 7.9 ms on 0.1B and from 139.8 ms to 54.6 ms on 2.9B.

**That saving is not realised today, and the reason matters more than the
saving.** The WKV step computes its recurrence in float32 — it has to — and
returns float32 into the residual stream, from layer 0 onward. So the
activation reaching every projection is float32, the weight is promoted to
match, and asking for bfloat16 changes nothing. `body()` returns float32 on a
dense backbone too.

That promotion is expensive on its own, quite apart from quantisation. With a
float32 activation MLX raises the whole bfloat16 weight matrix to float32
before multiplying. Measured on 0.1B *(debug build)*:

| projection | float32 activation | bfloat16 activation | |
|---|---|---|---|
| tmix `768×768` | 0.302 ms | 0.256 ms | 1.18× |
| head `65536×768` | **5.503 ms** | **1.283 ms** | **4.29×** |

Casting the WKV output back to the compute dtype right after the recurrence is
how the reference implementations do it (state in float32, activations in the
model dtype). It sits behind `X070Backbone.castWKVOutputToComputeDType`,
**off by default**, and the flag flips on a live object so both branches can be
timed in one process.

End to end, **release build**, A/B interleaved, two independent runs:

| | off | on | |
|---|---|---|---|
| decode, dense | 15.74 ms/token | **5.58** | **2.82×** |
| decode, quantised | 15.27 | **9.45** | **1.62×** |

The dense figure is 179 tokens/second on a 0.1B. The quantised one is where the
bfloat16 dequantisation path above finally does something: it was blocked by
this leak, not useless.

**The price shows up in the intermediate numbers.** Activations become
bfloat16 — seven mantissa bits — and the agreement between the recurrent and
parallel paths falls from **3.75e-6 to 9.90e-3**, a factor of 2640. Logits move
by 6.9e-3 relative to the current behaviour.

**In perplexity, though, it costs almost nothing.** Measured on 0.1B over
`test.txt` — mixed domains, Russian, Serbian and English, 29 225 tokens in 57
independent 512-token chunks, loss accumulated in float32 for both branches,
both branches in one process:

| | perplexity |
|---|---|
| off (float32 activations) | 16.8459 |
| off again, as a control | 16.8459 |
| on (bfloat16 activations) | **16.8610** |
| | **+0.090%** |

For scale: rwkv-quant's `reduction` preset costs +0.12% and is described there
as "degradation near zero, suitable as a QLoRA base". So the trade is **2.82×
decode for 0.090% perplexity**, and the cost is smaller than what this family
of projects already treats as negligible.

The flag is still off by default, for two reasons that are about process
rather than merit: the parity tolerances against the Python reference have to
be re-derived deliberately rather than loosened until green, and rwkv-metal
has the same leak — so today this package matches its reference and with the
flag on it would not. Perplexity has also only been measured on 0.1B; a deeper
model accumulates more bfloat16 rounding and should be checked before the same
number is promised for it.

It does **not** change which pass you are in. Recurrent decode on a quantised
backbone matches the parallel pass on the *same* quantised backbone to
**8.0e-7** — the same order as dense (7.4e-7). That comparison is the right
one: comparing quantised decode against the *dense* model would measure the
quality of the quantisation, not the fidelity of the decoder.

Quality is a property of the preset and the calibration, measured in
rwkv-quant on its 1.5B reference: `reduction` at 2.35× size costs +0.12%
perplexity, `compression` at 3.04× costs +2.47%. Those numbers come from that
project's benchmarks, not from measurements here.

### The sidecar carries its own names

`.rwkvq` uses the "world" naming convention; the x070 backbone uses its own.
`RwkvqNaming.worldKey(forX070:)` is the mapping, and without it `attachRwkvq`
would find nothing and quietly report zero tensors attached. `info.missing`
exists to make that visible — an empty `missing` is the thing to check after
attaching.

---

## Measured numbers

| | value | measured on |
|---|---|---|
| recurrent vs parallel, dense | 7.4e-7 relative, same top-1 | single prompt |
| recurrent vs parallel, quantised | 8.0e-7 relative, same top-1 | single prompt |
| decode with LoRA vs parallel | 1.1e-5 relative | single prompt |
| quantised vs dense, logits | 0.62 relative, different top-1 | single prompt |
| `.rwkvq` attach, 0.1B `reduction` | 73 tensors, 270 MB freed, 153 MB packed | — |
| reranker prefix / tail | 72.0 ± 0.2 ms / 6.1 ± 0.0 ms | 200 prefixes, 3 runs |
| reranker direct vs indexed | 6.5e-4 absolute, same ordering on 640 pairs | single run |
| embedding, raw base cos+ / cos− | 0.8619 / 0.8723 | 40 rows, single run |
| **decode, greedy, release build** | **5.58 ms/token (179 tok/s)** | 48 tokens, WKV output cast on |
| perplexity, WKV cast off / on | 16.8459 / 16.8610 (+0.090%) | test.txt, 29 225 tokens, 57 chunks |
| decode, greedy, release, cast off | 15.74 ms/token | same run, A/B interleaved |
| prefill, 1024 tokens, parallel vs recurrent | 321 ms vs 30262 ms (94×) | single run, debug build |
| decode, greedy | 34.30 ms/token | 64 tokens, debug build |
| decode, `T = 1` + `topP = 0.9` | 34.93 ms/token | 64 tokens, debug build |
| one `pick`, vocab 65536 | 1.54 ms | 500 draws, debug build |
| CTRL penalty on an all-negative, fully-covered distribution | 164 repeats vs 128 unpenalised | 5 tokens, 300 steps |

**Read memory as physical footprint, not RSS.** Metal buffers live in
IOAccelerator and only partly show up in `ps -o rss`: on one run whose
footprint reached 11.6 GB, RSS reported 0.4 GB. Use `vmmap --summary <pid>`.

**Any measurement taken while the machine swaps is invalid, not merely
degraded.** Count swap pages before and after and discard the run if they
moved.

---

## Not done

- **No batched decode.** `RWKVState` is single-sequence. Serving several
  independent generations means several states and several passes. Sampling
  also synchronises once per token — it has to read the chosen id back to the
  CPU — which is cheap now but becomes the shared bottleneck under batching.
- **No beam search or the more exotic selection schemes** (typical sampling,
  mirostat). Temperature, top-k, top-p and the penalties are there; each
  further scheme adds a branch to one loop, and is worth adding against a
  request rather than in advance.
- **`weight(_:)` force-unwraps.** The public accessor crashes on a quantised
  backbone where the dense copy was dropped. Callers that may see a quantised
  model should go through the projection path instead.
- **Decode dequantises per projection per token — unless you ask it not to.**
  This is the default and it is what makes a quantised backbone usable at all,
  but it costs 2.21× against dense bf16 at 2.9B, because writing and re-reading
  a dense transient turns a 1.9 GB read into 13.6 GB of traffic per token.
  `RwkvqAttachOptions(useNativeKernel: true)` relayouts sb6 into MLX's own
  quantised container once at load and runs `quantizedMM` instead: same
  numbers to the last bit, no dense transient, 5.1× faster than the default
  path and 2.3× faster than dense, at the same memory. It is off by default
  because the dequantise-per-call path is the right trade for *QLoRA* (the
  base must stay compressed) and the wrong one for inference; the two uses
  have not been separated in the API yet.
- **A quantised model still needs the dense weights.** `attachRwkvq` attaches
  to an already-loaded dense backbone, and the sidecar this package reads
  carries only the sb6 tensors. Note this is now a limitation *of this
  package*, not of the format: `.rwkvq` has held every tensor since the full
  export landed, and `rwkv-quant`'s `codec` can read it and build either
  loader layout without torch. Wiring that up here is pending — see
  `RwkvqSidecar`, whose manifest parser also still assumes every entry is a
  2-D quantised tensor and will reject a full-export manifest.
- **The backbone is not recorded in any checkpoint.** A reranker or embedding
  head loaded onto the wrong backbone differs in neither shape nor contract.
- **Prefill is single-sequence.** It runs the parallel pass, but for one
  prompt at a time; several prompts mean several passes rather than one padded
  batch. `bodyWithState` already accepts a mask and per-row end indices, so
  the pieces are there — what is missing is the batched decode to continue
  with.
