# State-tuning and pluggable initial states

RWKV is an RNN: a sequence does not have to start from zeros. In SwiftRWKV the
starting state is a pluggable object — trained (state-tuning), folded from a
prompt, or one on top of the other — kept in its own small file, separate from
the model.

Code: `Sources/RWKVGen/InitialState.swift`. The reference implementation is
rwkv-metal (`docs/state-tuning.md` there); the file format and the base
fingerprint are the same in both, bit for bit, so a state trained in Python
loads here and the other way round.

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

## What is trained

A trained state and a state after a prompt are one type, `RWKVBatchState` with
batch 1:

| tensor | shape | |
|---|---|---|
| `wkv` | `[L, 1, H, S, S]` | the WKV recurrence matrix, always fp32 |
| `tmixShift` | `[L, 1, 1, D]` | token-shift input of the time mix |
| `cmixShift` | `[L, 1, 1, D]` | token-shift input of the channel mix |

State-tuning trains only `wkv` by default, as RWKV-PEFT does. The shift vectors
add 3% and act on the first token of each layer only; `train:` turns them on.

| model | `wkv` numbers | both shifts | for scale: top 2 layers |
|---|---|---|---|
| 0.1B (12 layers, D 768) | 589,824 | 18,432 | ~15 M |
| 0.4B (24 layers, D 1024) | 1,572,864 | 49,152 | ~25 M |
| 1.5B (24 layers, D 2048) | 3,145,728 | 98,304 | ~101 M |

The whole base stays frozen — dense or `.rwkvq`. Nothing is merged into it.

## Training a state on a language model

```swift
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
```

- **The learning rate is the sensitive part.** The schedule often quoted for
  state-tuning, 1 -> 0.01, destroys the model here: Adam moves every one of the
  state's hundreds of thousands of numbers by about `lr` per step, and the
  useful shift is small. Measured on g1d 0.1B, Russian prose, 100 steps of
  4 × 256, held-out loss 3.228 without a state: lr 1e-2 — 3.374 (worse),
  3e-3 — 3.307, 1e-3 — 3.197, 3e-4 — 3.176. The default is 3e-4 -> 3e-5.
- **Memory.** The state of a layer is a parameter of its own block
  (`blocks.N.state0.wkv`), so the step runs backward block by block
  (`SequentialBody`) with one block in memory; the gradient equals autograd
  over the whole graph.
- **Cost.** The state enters every layer from layer 0 up, so the gradient goes
  through the full depth: there is no frozen bottom running on fast kernels,
  and a step costs about as much as adapters on every layer (tables below).
- `config.train = StateTrainableSet.Part.allCases` also trains the shifts;
  `initial:` starts from a given state, e.g. a folded prompt.

## Attaching, prompts and caches

```swift
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
```

- `backbone.initialState = state` splits the state by layer. A block takes
  `h_in` and the shifts from it only where the caller passed none — each part
  separately — so an explicit state always wins.
- `body`, the embedder, `prefill` and the reranker's prefix encoding therefore
  start from the attached state, and "trained state -> system prompt -> cache"
  is plain continuation with no code of its own (`foldPrompt`).
- `startState()` gives the same start to step-by-step decoding.
- An attached zero state changes nothing, bit for bit.
- Caches that store computed states — the reranker's serving index and pair
  cache — record the attached state's fingerprint (`initial_state`:
  `none`, or `sha256:…` from `StateFile.stateFingerprint`; a zero state is
  `none`) and refuse to be used under another state. See
  [reranker.md](reranker.md#the-text-feeding-contract).

## The file and the base fingerprint

`StateFile` writes three fp32 tensors and metadata: `format` (`rwkv-state`),
`version`, the contract (layers, D, heads, head size), `kind` (`tuned`,
`prompt`, `tuned+prompt`), `trained`, a free-form `note`, and the base
fingerprint.

A state means something only on the base it was made on, so
`StateFile.load(_:base:)`

- always refuses a different architecture (`Failure.otherArchitecture`);
- refuses a different base (`Failure.otherBase`) unless `allowOtherBase: true`.

`StateFile.baseFingerprint` is a sha256 over the contract, the small tensors of
the base (norms, token-shift mixes, `k_k` / `k_a` / `r_k`) and row 0 of two
projections, through bf16 — rows of compressed `.rwkvq` projections through
fp16, the lattice they live on. The same base loaded as bf16 and as fp32 gives
one value; a dense model, its `reduction` file and its `compression` file are
three different bases. Adapters, a trainable override and the attached state
are not part of it.

As the measurements below show, the refusal is stricter than quality requires
between a model and its own quantized files; it is kept so that a state is never
silently used on an unrelated model. Pass `allowOtherBase: true` to move a
state between a dense model and its `.rwkvq` files.

## Embeddings

`BaseTrainingMode.state(learningRateScale:parts:)` trains the state and the
pooling head of an embedder; the base stays frozen. The two want very different
step sizes, so the mode carries a multiplier on the stage learning rate
(`TrainableSet.learningRateScales`). `embed-run --mode state --state-lr …`
runs it from the command line.

A trained state can also stand in for the instruction line of the queries —
see [Embedding.md](Embedding.md#a-trained-state-in-place-of-the-instruction)
and the measurement below.

## Measured

Mac M4, 16 GB. Quality: three seeds, 95% intervals from a paired bootstrap
over held-out items.

### Language modelling

Russian prose, 300 steps of 4 × 256, lr 3e-4, loss on 96 held-out windows of
256 tokens (measured with the rwkv-metal reference, which computes the same
gradient):

| model | base | loss without state | with state | change |
|---|---|---|---|---|
| 0.1B | dense (g1d) | 3.221 | 3.155 | −0.066 [−0.073; −0.060] |
| 0.1B | `reduction` | 3.224 | 3.157 | −0.066 [−0.073; −0.060] |
| 0.1B | `compression` | 3.251 | 3.179 | −0.072 [−0.078; −0.065] |
| 0.4B | dense (g1d) | 2.946 | 2.895 | −0.051 [−0.059; −0.043] |
| 0.4B | `reduction` | 2.947 | 2.897 | −0.050 [−0.058; −0.042] |
| 0.4B | `compression` | 2.965 | 2.925 | −0.040 [−0.056; −0.016] |

- A compressed base gains as much as the dense one.
- **A state moves between a model and its quantized files almost for free.**
  Trained on one of the three 0.1B bases and attached to another, it is
  0.000…0.004 worse than a state trained on the target itself — 94…100% of
  the gain is kept. On 0.4B the difference is −0.008…+0.012.

### As an embedder, from a language-model checkpoint

State and pooling head trained, 0.1B, retrieval, 200 steps, state lr 1e-3,
MRR over 300 queries with a pool of 600:

| base | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| dense | 0.627 | **0.020** | 0.736 |
| `reduction` | 0.468 | 0.414 | 0.616 |
| `compression` | 0.538 | 0.513 | 0.423 |

For comparison, training the top 2 / 4 / 6 layers on the same data gives
0.46 / 0.76 / 0.78. A state reaches the top-2 level with 24 times fewer trainable
numbers, but it is unstable — one run of nine diverged (MRR as before training)
— and the useful step is narrow (seed 0, dense: lr 1e-3 — 0.627, 3e-4 — 0.253,
1e-2 and above — nothing learnt). For a new embedder prefer top-layer training;
a state suits adapting one that already works.

### In place of the instruction

0.1B, top 4 layers trained with `Instruct: … Query: …` queries; then the
instruction (17.8 tokens per query) is removed, base and head are frozen and
only the state is trained, lr 1e-4. Control: the same steps on the same rows,
training the head instead. MRR, three seeds:

| base | with instruction | without | without + state | without + head (control) | state − with instruction |
|---|---|---|---|---|---|
| dense | 0.759 | 0.752 | 0.758 | 0.754 | −0.001 [−0.012; +0.010] |
| `reduction` | 0.757 | 0.748 | 0.756 | 0.750 | −0.000 [−0.012; +0.011] |

The state restores the quality with the instruction while every query gets
17.8 tokens shorter. On this data the instruction is the same for every query,
so dropping it costs little to begin with (−0.007…−0.009), and the state's edge
over the control (+0.004…+0.007) is at the edge of significance. The state was
attached to the model, so it applied to the documents too, not only to the
queries.

**Query-only state (17.10).** The same measurement on `reduction`, re-run with
the state applied to queries only (`stateScope = .queriesOnly`: documents are
embedded exactly as with no state, and the state learns only from queries),
next to a re-run of the all-texts variant and of the control on the same seeds
(first stage 200 steps, second 200 steps at lr 1e-4, 300 held-out queries, 10 000
paired bootstrap resamples over queries; the first stage is identical across the
three arms of a seed):

| state applied to | with instruction | without | after the second stage |
|---|---|---|---|
| all texts | 0.757 | 0.748 | 0.756 |
| queries only | 0.757 | 0.748 | 0.752 |
| — (control: head) | 0.757 | 0.748 | 0.750 |

| difference | Δ MRR [95 %] |
|---|---|
| queries only − all texts | −0.004 [−0.010; +0.002] |
| queries only − with instruction | −0.004 [−0.014; +0.005] |
| all texts − with instruction | −0.000 [−0.012; +0.011] |
| queries only − control | +0.003 [−0.004; +0.010] |
| all texts − control | +0.007 [−0.000; +0.013] |

The all-texts arm reproduces the earlier numbers exactly. Restricting the state
to queries — the way the instruction itself was used — does **not** help: it is
slightly below the all-texts state and indistinguishable from the control. All
intervals contain zero. A plausible reading is that a state on both sides can
move documents and queries together and so has more room to fit, while the
query-only state carries the whole adjustment alone; on single-instruction data
there is little for either to recover (−0.009 from dropping the instruction).

#### Many and long instructions (18.10)

LitRetrieval has one short instruction for every query, so there was little to
replace. The same two-stage measurement on two sets where instructions matter
more (0.1B `reduction`, top 4 layers, 200 + 200 steps, state lr 1e-4, three
seeds, three arms sharing the first stage of each seed, 300 held-out queries,
evaluation with duplicate candidates merged — `--dedupe-pool`, since documents
repeat in both):

- **Tool_Use** — tool selection built from
  [UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609):
  query = the agent's system policy + the last user turn, positive = the
  description of the tool the agent called next, negative = another tool from
  the same list. 6000 rows from seven sources, 203 distinct instructions,
  ~1065 instruction tokens per query (up to ~2.2 thousand; `--max-tokens 2816`).
- **MAIR** — [MAIR](https://github.com/sunnweiwei/Mair) (126 instructed
  retrieval tasks; 10 031 queries, 805 distinct instructions, median 141
  characters); positive = the top-graded relevant document, negative = a random
  other document of the same task.

| set | with instruction | without | + state, all texts | + state, queries only | + head (control) |
|---|---|---|---|---|---|
| Tool_Use | 0.648 | 0.394 | 0.552 | 0.570 | 0.557 |
| MAIR | 0.772 | 0.746 | 0.758 | 0.751 | 0.749 |

| difference (Δ MRR [95 %]) | Tool_Use | MAIR |
|---|---|---|
| without − with instruction | −0.253 [−0.288; −0.219] | −0.025 [−0.040; −0.011] |
| queries only − all texts | +0.018 [+0.003; +0.034] | −0.006 [−0.012; −0.000] |
| state, all texts − control | −0.006 [−0.026; +0.015] | +0.009 [+0.002; +0.016] |
| state, queries only − control | +0.013 [−0.006; +0.032] | +0.003 [−0.003; +0.008] |
| best arm − with instruction | −0.077 [−0.105; −0.050] | −0.014 [−0.028; −0.001] |

Where the instruction carries real information — long agent policies that tell
which domain and which tools a request belongs to — dropping it costs a quarter
of MRR. A second stage on instruction-free queries wins back about 60 % of that,
but a head trained the same way wins back as much: what recovers is the
embedder adapting to the shorter queries, not something only a state can do. A
single state is one object for every query, so it cannot know which of 203
instructions a query had; no arm reaches the quality with the instruction. On
MAIR the instruction is worth less (−0.025) and the picture is the same, with
the all-texts state slightly above the control. Which side the state applies
to flips between the sets (queries only better on Tool_Use, worse on MAIR), so
neither is a default to recommend.

The direction this points to is a state **per instruction** — trained, or simply
the instruction folded into a state (`foldPrompt`) and attached to queries only,
which gives the instruction's effect without its tokens in every query. Not
measured yet.

### Cost of a step

Embedder, state and head trained; all three texts of a triplet at full length
(worst case); tokens/s = 3 · batch · T / step; peak footprint of the whole
process; arms interleaved as separate processes with pauses, a control run
between them stayed within 0.5%.

| model | 512 tok, batch 1 | batch 2 | batch 4 | 1024 tok, batch 1 | batch 2 | batch 4 |
|---|---|---|---|---|---|---|
| 0.1B dense | 1583 tok/s, 0.74 GB | 1687, 1.02 | 1721, 1.55 | 1484, 1.01 | 1701, 1.55 | 1699, 1.82 |
| 0.1B `reduction` | 1409, 0.67 | 1706, 0.96 | 1740, 1.57 | 1498, 0.96 | 1711, 1.57 | 1706, 1.84 |
| 0.1B `compression` | 1422, 0.67 | 1706, 0.96 | 1740, 1.57 | 1505, 0.96 | 1716, 1.57 | 1704, 1.85 |
| 0.4B dense | 508, 1.43 | 567, 1.87 | 557, 2.68 | 535, 1.87 | 568, 2.68 | 567, **3.33** |
| 0.4B `reduction` | 537, 1.14 | 585, 1.61 | 566, 2.36 | 551, 1.61 | 577, 2.36 | 577, **3.01** |
| 0.4B `compression` | 537, 1.09 | 586, 1.56 | 572, 2.38 | 552, 1.56 | 583, 2.37 | 581, **3.03** |
| 1.5B `reduction` | 203, 2.91 | | | | | |
| 1.5B `compression` | 204, 2.65 | | | | | |

Against top-layer training over `reduction` at batch 4 × 512, a state step is
2.5–3 times slower than two layers (0.1B: 1740 against 4267 tok/s; 0.4B: 566
against 1736) while memory is about the same as two layers (1.57 against 1.68 GB;
2.36 against 2.41). Only 0.4B at 4 × 1024 exceeds a 3 GB budget; 1.5B over a
compressed base fits at 1 × 512, where its top layers do not.

## Not done

- A state **per instruction** (trained, or the instruction folded into a state
  and attached to queries only) — the next step after the many-instruction
  measurement above — is not implemented: an attached state is one per model,
  and an embedding batch cannot mix states.
- Neuro (the iOS app) does not use states yet; the file format and API are ready.
