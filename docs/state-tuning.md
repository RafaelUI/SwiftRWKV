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

- **Gradients are clipped at a global norm of 1.0 by default.** Without
  clipping, a state trained from zero on a 0.4B model diverged at step 13 of
  300 (loss 1.03 → 10.98, gradient norm 233) and never recovered; clipping at
  10 did not prevent it (the norm before the jump was 12.2). Typical norms are
  3–17, so 1.0 clips almost every step; Adam rescales by its own running
  magnitude, and the step stays bounded by `lr`.
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

### Weight streaming (opt-in)

Block-by-block backward keeps one block's activations in memory, but the weights of
the whole model still sit there: 3 GB for a 1.5B model in bf16. A backbone built
with `X070Backbone.loadStreaming(safetensors:)` reads its weights lazily from the
file (MLX reads a tensor when it is first evaluated), and the block-by-block paths
let go of what they read: a block's weights after that block in the forward and in
the backward pass, the embedding table after the stem, the head after the loss.
What was let go is read from the file again the next time it is needed.

```swift
let base = try X070Backbone.loadStreaming(safetensors: URL(fileURLWithPath: "/tmp/g1j_1.5b.safetensors"))
let ids = MLXArray((0 ..< 33).map { Int32($0) }, [1, 33])
let batch: LoRABatch = (x: ids[0..., 0 ..< 32], y: ids[0..., 1 ... 32])   // T % 16 == 0
let (_, state) = StateTuning.run(base, nextBatch: { batch }, config: StateTuneConfig())
print(state.wkv.shape, base.isWeightStreaming)
```

The numbers are those of the same file loaded into memory, bit for bit. g1j 1.5B
bf16, one sequence of 320 tokens, block by block, M4, separate processes, two
rounds:

| | step | peak process footprint |
|---|---|---|
| weights in memory | 1.683 / 1.636 s | 3.84 / 3.79 GB |
| streaming | 1.816 / 1.765 s (+8 %) | **0.95 / 0.93 GB** |

The file was in the operating system's cache here; on a phone it is read from
storage on every pass. Only dense `.safetensors` checkpoints stream; a `.rwkvq` base
is kept in memory. A whole-graph pass (`bb(ids)`, training with
`sequentialBackward = false`) reads every weight and keeps it until
`releaseAllStreamedWeights()`.

### Early stopping on held-out loss

Pass held-out batches and `evalEvery`, and training stops when the held-out loss
stops improving; the state that comes back (and stays attached) is the best one,
not the last.

```swift
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
```

- The held-out loss is computed block by block, like the training step: on the
  same batch with the same state it is the training loss bit for bit. Batches are
  weighted by the number of tokens in the loss (targets ≥ 0).
- Training runs in segments of `evalEvery` steps on one trainer (same step
  counter, same Adam moments, same schedule): the checks do not change training —
  every step's loss is the same bit for bit as without them.
- An improvement is a loss below the best by more than `minDelta`; `patience`
  checks without one stop training. Cancelling (`isCancelled`) mid-segment still
  checks the steps done.
- `result.heldOut` holds every check (step 0 is before training), the best step
  and loss, and whether patience stopped the run. Works with weight streaming;
  the check lets go of the weights it read.

Measured on g1j 1.5B, "Neuron" (a system prompt distilled into a state, folded
start, batch 4, 300 steps), with 40 of the 366 teacher answers held out and a
check every 25 steps: held-out loss 1.044 before training, 0.890 at step 100,
0.873 at step 200 (best), 0.875–0.876 from 225 to 300. No overfitting by step
300, a plateau from about 200; `patience = 3` would have stopped at 275 and
returned step 200. A check of 10 batches took ~12 s (a step ~4.5 s).

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

### RWKV-PEFT `.pth`

`StateFile.exportPEFT` writes a state the way RWKV-PEFT state tuning saves it, a
torch file read by RWKV Runner, Ai00 and other runners; `StateFile.importPEFT`
reads one. No torch is needed on either side.

```swift
let base = try loadBackbone("~/models/world_0.1b_x070.safetensors")
let (state, _) = try StateFile.load(URL(fileURLWithPath: "/tmp/task.state.safetensors"), base: base)
try StateFile.exportPEFT(state, to: URL(fileURLWithPath: "/tmp/task.pth"))
let (back, ignored) = try StateFile.importPEFT(URL(fileURLWithPath: "/tmp/task.pth"), base: base)
print(back.wkv.shape, ignored)          // keys other than blocks.{i}.att.time_state
```

- Keys `blocks.{i}.att.time_state`, shape (heads, S, S), fp32, in the same layout
  as here (x070 runners take it untransposed). Checked on g1j 1.5B: a 64-token
  prompt folded here and in mlx-lm (the path Preen exports through, checked in
  RWKV Runner) — relative difference 8.4e-7 as is, 1.40 transposed, in both
  directions; `torch.load(weights_only=True)` reads the file.
- The reader understands what `torch.save` writes for a dict of tensors (fp32,
  bf16, fp16 storages, shared storage with offsets, strided views,
  `nn.Parameter`, `_metadata`) and executes nothing: an unknown global or opcode
  is refused, as with `weights_only=True`. Compressed or zip64 files are refused.
- The file has no base fingerprint: `importPEFT(base:)` checks the architecture
  (layers, heads, head size) only.
- **The format has no token shifts.** Exporting a state with non-zero shifts is
  refused unless `dropShifts: true`; importing gives zero shifts. A state trained
  only on WKV from zero has none, but one trained from a folded prompt keeps the
  fold's shifts, and the trained matrix leans on them. g1j 1.5B, "Neuron" (system
  prompt distilled into a state, 300 steps, folded start): the persona answers
  the name questions 10/10 with its shifts and 0/10 with them dropped. Started
  from the fold with the shifts zeroed (`gen-run --init fold-wkv`), the same
  training reaches a lower loss (0.391 against 0.419) and, loaded from the
  `.pth`, answers the four name questions 10/10 each again ("what can you do?"
  3/10 against 10/10). General questions (199 checkable answers, one training
  seed, 4 samples each): 0.828 against 0.818 with shifts (+0.010 [−0.014;
  +0.035]) and 0.799 for the model without a prompt. For a state that will be
  exported, start from zero or from a fold without shifts.

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
which gives the instruction's effect without its tokens in every query. The
folded form is done (19.10, `InstructionCache`, see
[Embedding.md](Embedding.md#folding-the-instruction-into-a-state)): on Tool_Use
it gives the same MRR as the instruction inline (−3.5·10⁻⁷, three seeds) with
9.5 times fewer query tokens and a 6 times faster evaluation. It is a cache, not
a gain in quality — except where the window is shorter than the instruction
(at 512 tokens: +0.325). A **trained** state per instruction (20.10, a state
bank, see [Embedding.md](Embedding.md#a-trained-state-per-instruction-a-state-bank)),
started from the folded instructions, adds +0.026 [+0.015; +0.037] over the
instruction inline and +0.017 [+0.006; +0.027] over a head trained the same
way (three seeds) — the first arm in this series that beats the instruction.

**The strong control (21.10) explains that gain away.** The same 200 extra steps
spent on what the first stage trains — the top 4 layers and the head, with the
instruction inline, on the same rows in the same batch order (`embed-run
--then-top 200`; the second stage continues the first with the same trainable
set and its fp32 master) — reach 0.696 against 0.648 with the instruction after
the first stage: +0.049 [+0.030; +0.068]. The trained bank is **below** it:
−0.023 [−0.042; −0.004] (three seeds, 300 held-out queries). The bank's edge over
the head-only control came from extra training with more parameters, not from a
state per instruction as such; ordinary steps on the top layers give twice as
much. What the bank still offers is locality: on queries whose instruction had
no training rows it reproduces the instruction exactly, and it leaves the shared
weights untouched, so it can be added or dropped per task.

### In place of a system prompt (generation, 20–23.10)

g1d 0.4B with the system line «Ты — Neuron, языковая модель. Ты умеешь писать
стихи.» ("You are Neuron, a language model. You can write poems."). The state
is distilled from the same model *with* the line: the teacher answers training
questions (4 samples, temperature 1.0, top-p 0.6), answers are filtered by kind
— questions about who it is keep only answers naming Neuron, factual questions
keep only **correct** answers, answers that recite the system line are dropped —
and the student, the model without the line, trains only its WKV state on them
(loss on the answer only, ending with `\n\nUser:`), starting from the folded
system line, 300 steps, lr 3e-4, batch 4, clipping 1.0; three seeds. `gen-run
--distill` in this repository runs exactly this.

General ability is measured on 199 questions with a checkable answer (capitals,
authors, science, geography, arithmetic, units, translation both ways; 130
Russian, 69 English; disjoint from training by question and by fact), a regex
on each answer, 12 answers per question; paired bootstrap over questions:

| arm | all 199 | translation (32) | arithmetic (36) | other facts (131) |
|---|---|---|---|---|
| no system line | 0.443 | 0.281 | 0.688 | 0.415 |
| system line in the prompt | 0.314 | 0.253 | 0.690 | 0.225 |
| state, 31 correct facts in training | 0.399 | 0.495 | 0.812 | 0.263 |
| state, 109 correct facts in training | **0.506** | 0.661 | 0.889 | 0.363 |

| difference (Δ [95 %]) | all | other facts |
|---|---|---|
| prompt − no line | −0.129 [−0.164; −0.095] | −0.190 [−0.234; −0.146] |
| state (109 facts) − no line | **+0.063 [+0.020; +0.105]** | **−0.052 [−0.090; −0.016]** |
| state (109 facts) − prompt | +0.192 [+0.153; +0.234] | +0.137 [+0.094; +0.181] |
| state: 109 facts − 31 facts | +0.106 [+0.075; +0.139] | +0.100 [+0.062; +0.140] |

- **The prompt itself damages facts**, and a state learned from a prompted
  teacher inherits that unless the data keeps only correct answers. The more
  correct facts in the data, the smaller the damage.
- **The state's gain over the model without the line is form, not knowledge.**
  It comes entirely from question templates that also occur in the training
  facts — "translate the word …" (+0.380) and "how much is …" (+0.201), where
  the bare model often answers with a definition or a derivation and the state
  answers with the word or the number. On other factual questions the state is
  still below the bare model (−0.052), though a third of the prompt's damage.
- **The persona holds better than with the prompt.** Naming itself Neuron
  (10 samples, 5 questions): 3–10 of 10 with the state, mostly 7–10, against 2–4 with the
  prompt; on "Are you ChatGPT?" it says yes in 0–2 of 10 samples, the prompt
  in 2, the bare model in 5 (counted as an answer starting with "yes" or saying
  "I am ChatGPT" without "not"). English questions about the name need English examples in
  the data (without them the state answered "Qwen"). The more facts in the
  data, the smaller the persona's share and the weaker it holds.
- Measure general ability next to the target behaviour, and per kind of
  question: the overall difference hid two opposite effects.
- **On 0.4B the form is tied to the question template.** 40 new items (words to
  translate, sums), each asked twice — in the template of the training facts
  and in a new wording ("How do you say «нож» in English?", "Multiply 5 by
  8."): the state gains +0.242 [+0.160; +0.325] on the template and nothing on
  the new wording (−0.008 [−0.083; +0.067]); new wordings of translation even
  lose (−0.158), the "I write poems" persona leaking into the answer once the
  template no longer holds it. For a small model the data has to cover
  wordings, not only kinds of questions.

**1.5B (g1j, `reduction` base, 23.10)** — the same data, protocol and seeds. The
teacher answers half the facts correctly (243 of 484 samples), so the data has
366 answers. A step takes ~4 s (300 steps in 20 minutes), peak process
footprint 3.0–3.1 GB; the state is trained through the compressed projections.

| arm | all 199 | translation | arithmetic | other facts | Russian | English |
|---|---|---|---|---|---|---|
| no system line | 0.799 | 0.836 | 0.951 | 0.748 | 0.710 | 0.967 |
| system line in the prompt | 0.706 | 0.706 | 0.942 | 0.641 | 0.663 | 0.786 |
| state | **0.816** | 0.833 | 0.986 | 0.765 | 0.759 | 0.923 |

- prompt − no line: −0.093 [−0.122; −0.064]; state − no line: +0.017
  [−0.009; +0.041] (other facts +0.017 [−0.011; +0.046]); state − prompt:
  +0.110 [+0.084; +0.136]. The larger model's state costs nothing measurable
  in general ability, while the prompt still does.
- The form transfers: on the new wordings the state gains +0.102 [+0.052;
  +0.152] over the bare model (+0.046 on the template).
- The persona: Neuron 10 of 10 on four of five questions about itself on all
  three seeds (prompt: 1–9 of 10); on "Are you ChatGPT?" the prompt says yes
  in 6 of 10 samples, the state in 1–2.

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

**Types on a dense bf16 base (23.10).** The WKV kernel returns fp32, and without a
cast that fp32 flows into the residual stream, so every projection then lifts its
whole bf16 weight to an fp32 copy on every pass. `StateTuneConfig` now controls this
for the duration of training (the backbone's own flags are restored afterwards):

- `castProjectionInput` (**on** by default) — a dense projection runs its matmul in
  the weight's type and returns the input's type, `(x.asType(W.dtype) @ Wᵀ).asType(x.dtype)`;
  the residual stream stays fp32.
- `castWKVOutput` (off) — the WKV output is cast to the compute type, the way the
  official implementation and mlx-lm do it.

g1j 1.5B bf16, one sequence of 320 tokens, M4, arms interleaved in one process:

| | step, block by block | step, whole graph | state gradient vs all-fp32, zero / random S₀ |
|---|---|---|---|
| no cast (before 23.10) | 1.767 s | swaps (12 GB) | 5.1 % / 6.6 % |
| `castProjectionInput` (default) | **1.475 s** | 1.158 s, 7.5 GB | 5.7 % / 7.0 % |
| `castWKVOutput` | 1.291 s | **1.013 s**, 5.1 GB | 12.0 % / 9.3 % |

The WKV-output cast is the fastest and matches Preen (RWKV-7 state tuning in
mlx-lm, 0.98 s per step on the same machine), but it doubles the gradient's error;
the projection cast keeps it close to the uncast path, which is why it is the
default. Block by block the step stays slower than one graph — each block is
recomputed once — in exchange for memory (4.5–4.9 GB against 7.5); compiling the
whole step (`mx.compile`, measured in rwkv-metal) gives another 5.5 %.

## Not done

- Weight streaming covers dense `.safetensors` only; a `.rwkvq` base (re-laid out at
  load) is not streamed, and rwkv-metal has no streaming.
- The `.pth` export and early stopping are in SwiftRWKV only; rwkv-metal has
  neither.

- A state per instruction is done (folded: `InstructionCache`; trained: a
  state bank, `StateBankTrainableSet`) for retrieval queries only. Not done: a
  bank for sts and classification, a bank file format (`StateFile` holds one
  state). The strong control for the trained bank is done (21.10): the same
  steps on the top layers beat the bank (−0.023 [−0.042; −0.004]).
- Neuro (the iOS app) does not use states yet; the file format and API are ready.
