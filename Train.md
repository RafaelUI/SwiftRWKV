# SwiftRWKV

On-device **RWKV-7** training and inference for Apple Silicon (iOS / iPadOS /
macOS / visionOS), powered by [MLX](https://github.com/ml-explore/mlx) and a
custom Metal **WKV-7** kernel.

Everything is built on one canonical backbone, `X070Backbone` — a faithful port
of the official RWKV-7 "Goose" x070 architecture. It serves inference,
generation, LoRA/QLoRA fine-tuning, and N-layer partial fine-tuning from a single
weight format.

---

## Modules & imports

| Module | What it gives you | Import when |
|---|---|---|
| `RWKVGen` | Backbone, generation, LoRA/QLoRA, N-layer partial fine-tune, tokenizer | Almost always |
| `RWKVKernel` | Raw WKV-7 kernels + constants | Only for low-level / custom kernel work |

`RWKVGen` depends on `RWKVKernel`, so importing `RWKVGen` is enough for every
high-level workflow. The old `RWKVTrain` module (from-scratch backbone +
classification Trainer) has been **removed**; its partial-fine-tune capability
now lives on the canonical `X070Backbone` inside `RWKVGen`.

```swift
import RWKVGen          // backbone, generation, LoRA, partial fine-tune
import RWKVKernel       // only if you call wkv7* directly
```

### Public API by use case

**Core model**
- `X070Config(nLayer:nEmbd:headSize:vocab:)`
- `X070Backbone(weights:cfg:computeDType:)`
  - `callAsFunction(ids) -> logits [B,T,vocab]`
  - `body(ids) -> ln_out [B,T,D]`
  - split: `boundaryState(ids, upTo:) -> (x, vFirst?)`, `forwardFrom(x, vFirst?, from:) -> ln_out`, `vFirstFrom(ids) -> vFirst`
  - hooks: `trainLayers: Set<Int>`, `wOverride: [String:MLXArray]?`, `useBlockCheckpoint: Bool`
- `WorldTokenizer(vocabURL:)` — `encode` / `decode` / `rawBytes` (RWKV **World** trie tokenizer only)

**Generation (streaming, recurrent)**
- `RWKVState(cfg:dtype:)` — per-layer recurrent state
- `X070Backbone.prefill(ids, state:&) -> logits`
- `X070Backbone.step(id, state:&) -> logits`

**LoRA / QLoRA fine-tune** (LM objective)
- `LoRATargets.tmix` `.cmix`, `LoRASpec(rank:alpha:tmixTargets:cmixTargets:layers:quantizeBits:quantGroupSize:)`
- `LoRA.add(to:spec:) -> LoRAInfo`, `LoRA.merge`, `LoRA.save` / `.load`, `LoRA.adapterState`, `LoRA.quantizeBaseModel`
- `LoRAFinetune.run(bb, nextBatch:config:isCancelled:onStep:) -> LoRATrainResult`
- `LoRAConfig(...)`, `LoRABatch = (x: MLXArray, y: MLXArray)`, `bigQuantTargets`

**N-layer partial fine-tune** (classification / feature head)
- `X070Example(ids:label:)`, `X070PoolKind` (`.mean` / `.last`)
- `X070PartialFinetune.run(backbone:cfg:numClasses:trainSet:valSet:freeze:ctxLen:epochs:batchSize:lr:onStep:onEpoch:) -> TrainResult`

**Kernel (RWKVKernel)**
- `WKV7_CHUNK = 32`, `WKV7_HEAD_SIZE = 64`
- `wkv7Forward`, `wkv7Train`, `wkv7ChunkForward`, `wkv7Reference`

---

## Requirements

| | Minimum |
|---|---|
| OS | iOS 17 / iPadOS 17 / macOS 14 / visionOS 1 |
| Hardware | Apple Silicon with a GPU |
| Toolchain | Swift 5.10+ |
| Dependency | `mlx-swift` ≥ 0.31.4 |

The WKV-7 kernel runs on the **Metal GPU**. Test on a real device or macOS, not
the iOS Simulator's software renderer.

---

## Loading a model

A model is a flat `safetensors` of x070-named tensors plus a tokenizer. Load the
weights and build the backbone:

```swift
import RWKVGen
import MLX

let weights = try loadArrays(url: URL(fileURLWithPath: ".../model_x070.safetensors"))
let cfg = X070Config(nLayer: 18, nEmbd: 448, headSize: 64, vocab: 16000)
let bb  = X070Backbone(weights: weights, cfg: cfg)
```

Weight naming the backbone expects (x070 / official RWKV layout):

```
emb.weight, ln0.{weight,bias}, ln_out.{weight,bias}, head.weight
blocks.N.ln1.{weight,bias}, blocks.N.ln2.{weight,bias}
blocks.N.tmix.{x_r,x_w,x_k,x_v,x_a,x_g}
blocks.N.tmix.{r,k,v,o}_proj.weight
blocks.N.tmix.{k_k,k_a,r_k}                 # per-head, shape [H,S]
blocks.N.tmix.{a,g,w}_lora_{A,B}.weight     # (+ a/w lora_B.bias)
blocks.N.tmix.v_lora_{A,B}.weight (+bias)   # layers > 0 only
blocks.N.tmix.ln_x.{weight,bias}
blocks.N.cmix.{x_k, key.weight, value.weight}
```

### Tokenizers

- **World models** use the RWKV trie tokenizer: `WorldTokenizer(vocabURL:)`.
- **Custom models** (e.g. trained via FLA) ship their own HF `tokenizer.json`.
  The framework does not bundle an HF BPE tokenizer — encode/decode with your own
  pipeline (e.g. `swift-transformers`, or tokenize offline and feed token ids).

---

## Converting FLA checkpoints → x070

If you pretrain with **flash-linear-attention**, the checkpoint uses FLA's module
names (`model.layers.N.attn.*`, `lm_head`, `model.norm`, …) and stores the
per-head vectors `k_k` / `k_a` flat as `[D]`. `X070Backbone` expects official
RWKV names and `k_k` / `k_a` shaped `[H,S]`. Convert with the bundled utility in
the `rwkv-metal` repo:

```bash
python tools/convert_fla_to_x070.py model.pt -o model_x070.safetensors
```

It auto-detects config from tensor shapes, remaps every key, reshapes `k_k`/`k_a`
to `[H,S]`, and verifies the produced key set exactly matches what the backbone
reads. Works for `.pt` (incl. `{'model':..,'step':..}` wrappers) and
`.safetensors` inputs, at any model size.

> Architectural note: FLA RWKV-7 and x070 are the **same** architecture — only
> naming and the `k_k`/`k_a` layout differ. After conversion, the formulas
> (decay, iclr, gate, head GroupNorm) match; verified by next-token loss on real
> text matching the trained model (not random).

---

## Inference & generation

### One-shot logits

```swift
let ids = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, tokenIds.count])
let logits = bb(ids)               // [1, T, vocab]
```

### Streaming generation (recurrent, O(1) memory per step)

```swift
var state = RWKVState(cfg: cfg)
_ = bb.prefill(promptIds, state: &state)     // consume the prompt
state.eval()

var next = argmax(lastLogits)                // your sampling
for _ in 0 ..< maxNewTokens {
    let logits = bb.step(next, state: &state)
    next = sample(logits)                    // greedy / top-p / temperature
    state.eval()                             // fix state between steps
    // append `next`, decode incrementally
}
```

`prefill` / `step` use the recurrent WKV form (no full-sequence kernel), so
memory is constant in sequence length. `RWKVState` holds per-layer `wkv` (fp32),
token-shift previous tokens, and `vFirst`.

---

## LoRA / QLoRA fine-tune

Best for **larger models (≈600M+)** where adapting 0.1–3% of parameters is
enough. LM objective (next-token).

```swift
// 1. Attach adapters (optionally quantize the frozen base → QLoRA).
let spec = LoRASpec(
    rank: 16, alpha: 16,
    tmixTargets: LoRATargets.tmix,     // ["r_proj","k_proj","v_proj","o_proj"]
    cmixTargets: [],                    // add ["key","value"] for more capacity
    layers: nil,                        // nil = all blocks
    quantizeBits: 0,                    // 0 = bf16 base; 4 or 8 = QLoRA
    quantGroupSize: 64
)
let info = LoRA.add(to: bb, spec: spec)
print(info.trainablePct, info.numAdapters)

// 2. Train. You supply batches; x = ids [B,T] i32, y = next-token targets [B,T].
let cfg2 = LoRAConfig(lr: 1e-4, maxSteps: 1000, gradAccum: 1,
                      useBlockCheckpoint: true)   // checkpoint: −~45% peak / +~22% time
let res = LoRAFinetune.run(bb, nextBatch: { myLoader.next() }, config: cfg2,
    onStep: { step, loss, gnorm, peakMB in
        if step % 10 == 0 { print(step, loss, gnorm, peakMB) }
    })
print("final loss:", res.finalLoss)

// 3. Persist / apply adapters.
try LoRA.save(bb, to: adaptersURL)     // adapters only (small)
// LoRA.merge(bb)                       // fold adapters into base for plain inference
```

For QLoRA on a big model, also quantize the heavy frozen matrices:

```swift
LoRA.quantizeBaseModel(bb, bits: 4)    // affects bigQuantTargets: cmix.key/value, head, emb
```

---

## N-layer partial fine-tune

Best for **smaller models (≤400M)**, where LoRA's 0.1–3% is too little. Trains
the **full weights of the top N layers + a fresh head** on a frozen lower stack.
Classification head with pooling.

```swift
let examples = texts.map { X070Example(ids: tokenize($0.text), label: $0.label) }

let res = X070PartialFinetune.run(
    backbone: bb, cfg: cfg, numClasses: 3,
    trainSet: examples, valSet: valExamples,
    freeze: 12,            // freeze layers [0,12); train [12, nLayer) + head
    ctxLen: 64,            // MUST be a multiple of WKV7_CHUNK (32)
    epochs: 5, batchSize: 8, lr: 1e-3,
    onStep:  { step, loss, peakMB in if step % 20 == 0 { print(step, loss) } },
    onEpoch: { epoch, valAcc in print("epoch", epoch, "valAcc", valAcc) }
)
print("val accuracy:", res.valAcc)
let trainedParams = res.params      // top-layer weights + head, for saving
```

### Choosing `freeze` (18-layer model)

| `freeze` | Trains | Trade-off |
|---|---|---|
| 17 | head + 1 layer | fastest, least memory, least capacity |
| ~12–14 | head + 4–6 layers | good default |
| small | many layers | most capacity, slowest, most memory |

Constraint: `0 <= freeze < nLayer`; `ctxLen % 32 == 0`.

### How the split works

Partial fine-tune relies on cutting the network at layer `freeze`:

1. **Boundary pass (frozen).** `boundaryState(ids, upTo: freeze)` runs
   `emb → ln0 → blocks[0..<freeze]` and returns `(x, vFirst)`. Because x070's
   token-shift is **within-block** (zero-pad at t=0, no cross-block carry), the
   only state crossing the boundary is `x` and `vFirst` — there is no `xPrev`.
   `x` is cached to a memory-mapped bf16 file on disk; `vFirst` is recomputed
   cheaply from ids (`vFirstFrom`, layer 0) at train time, so it is not stored.
2. **Trainable tail.** `forwardFrom(x, vFirst, from: freeze)` runs
   `blocks[freeze..<nLayer] → ln_out`. Trainable fp32 weights are injected via
   `wOverride`; trainable layers run the differentiable `wkv7Train` kernel
   (selected by `trainLayers`), optionally gradient-checkpointed
   (`useBlockCheckpoint`).

`boundaryState + forwardFrom` is **bit-exact** equal to `body` (Δ = 0), so the
cache scheme never changes the math.

---

## Memory & performance

- **Block gradient checkpoint** (`useBlockCheckpoint = true`): ≈ −45% peak memory
  for ≈ +22% step time. Recommended on phones.
- **MLX buffer cache**: `LoRAConfig.cacheLimitGB` caps `GPU.cacheLimit` during
  LoRA training (default 1.5 GB); set ≤ 0 to disable the cap.
- **fp32 training**: trainable params + optimizer state are fp32 (bf16 Adam
  updates lose precision); frozen base stays bf16. This matches how RWKV-7 World
  was trained with the fp32 kernel.
- **Fused cross-entropy** was evaluated and **dropped**: in MLX-Swift (no
  Triton-level allocation control) it raised peak memory rather than lowering it,
  and at vocab ≤ 32k the `[N,V]` logits are not the dominant term.

---

## Validation status

Everything below is covered by tests (local `Tests/`, gitignored):

| Check | Result |
|---|---|
| WKV-7 kernel forward vs Python reference (1 chunk) | Δ = 0 (bit-exact) |
| WKV-7 custom backward vs autograd reference (`wkv7Reference`) | rel-L2 ≈ 5e-7 (dk,dv,db); ≈ 5–9e-4 (dr,dw,da, recompute-by-`/w`) |
| x070 logits/ln_out vs Python World reference | parity (existing fixtures) |
| Split `boundaryState+forwardFrom` vs `body` (real ru60m weights) | Δ = 0 at f = 0,6,12,17 |
| ru60m next-token loss on real Russian text | ≈ 2.46 (ppl ≈ 11.7; random ≈ 9.68) |
| Partial fine-tune end-to-end (top layers + head, ru60m) | loss 0.84 → 0.0001, valAcc 1.0 |

---

## Constants & gotchas

- **`headSize` must be 64**, **`ctxLen` / context must be a multiple of 32**
  (the kernel's `WKV7_HEAD_SIZE` and `WKV7_CHUNK`). The trainable path uses
  `wkv7Train`, which requires `T % 32 == 0`.
- **Simulator**: needs a real Metal GPU; run on device or macOS.
- **FLA checkpoints** must be converted (naming + `k_k`/`k_a` reshape) before
  loading — see [Converting FLA → x070](#converting-fla-checkpoints--x070).
- **World vs HF tokenizer**: `WorldTokenizer` is only for RWKV World vocab;
  custom models use their own `tokenizer.json`.
- **MLX is not thread-safe**: never run two trainings concurrently; serialize on
  one queue and report progress via the `onStep` / `onEpoch` callbacks.
- **Which fine-tune?** LoRA/QLoRA for ≈600M+; N-layer partial for ≤400M. Both run
  on the same `X070Backbone`, just different entry points
  (`LoRAFinetune` vs `X070PartialFinetune`).

---

*SwiftRWKV — ImpulseLeap / Alexei Goncharov · <https://www.impulseleap.com> ·
<https://github.com/RafaelUI/SwiftRWKV>*
