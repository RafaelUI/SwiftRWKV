# SwiftRWKV

On-device **RWKV-7** training and inference for the Apple ecosystem
(iOS · iPadOS · macOS · visionOS), built on
[MLX](https://github.com/ml-explore/mlx) and a custom Metal **WKV-7** kernel.

RWKV-7 is a linear-attention (RNN-style) language model: constant memory per
generated token and no quadratic attention, which makes it a good fit for running
*and adapting* models directly on a phone or Mac. SwiftRWKV exposes one canonical
backbone (`X070Backbone`, a faithful port of official RWKV-7 "Goose" x070) and
builds inference, generation, embeddings, reranking and several ways of
adapting a model on top of it — over a dense base or a quantised `.rwkvq` one.

```swift
import RWKVGen
import MLX

let weights = try loadArrays(url: modelURL)                 // x070-named safetensors
let bb = X070Backbone(weights: weights,
                      cfg: X070Config(nLayer: 12, nEmbd: 768, headSize: 64, vocab: 65536))
let tok = WorldTokenizer(vocabURL: vocabURL)!

// Greedy by default — a package that answers differently on every run
// cannot be debugged.
let out = bb.generate(prompt: "The capital of France is",
                      maxTokens: 64, tokenizer: tok)
print(out.text)
```

## Features

- **Text generation** — `generate(prompt:maxTokens:config:onToken:)` with
  temperature, top-k, top-p and repetition penalties, seeded and reproducible,
  streaming with stop strings and cancellation; the sampler can run entirely on
  the GPU. Greedy by default.
- **Inference** — a parallel pass for whole sequences and a recurrent one with
  O(1) memory per token; prefill uses the parallel pass and is measured 94×
  faster than folding the prompt token by token.
- **Reranking** — a cross-encoder head over backbone state, with a state cache,
  listwise training and an indexed serving path.
- **Embeddings** — pooling, a trainable head, contrastive losses, GradCache,
  curriculum stages and retrieval metrics.
- **Quantised `.rwkvq` base** — the format from
  [rwkv-quant](https://github.com/RafaelUI/rwkv-quant), opened directly with
  `X070Backbone.load(fromRwkvq:)` — no dense checkpoint needed — with fused
  Metal dequantisation (an `export_mlx` sidecar is an optional cache, and
  self-contained only when it is a full export); inference, reranking and
  embeddings run on it unchanged, and it can be trained over — adapters, the
  top layers (unpacked to dense, the bottom stays compressed) or a state.
- **LoRA / QLoRA fine-tuning** — for larger models (≈600M+); 4/8-bit quantized
  base or a `.rwkvq` one.
- **Top-layer training** — full-weight training of the top N layers for smaller
  models (≤400M), where adapters carry too few parameters; also over a `.rwkvq`
  base. The classic path (`X070PartialFinetune`) caches the frozen lower stack
  on disk.
- **State-tuning** — the base frozen, only the initial state of the recurrence
  trained (0.1B: 0.59 M numbers), kept in its own file with a fingerprint of the
  base; a trained state and a folded prompt are one pluggable object.
- **Pre-training from scratch** — x070 initialisation, `.bin` token streams, a
  shared trainer with LR schedules and resumable checkpoints.
- **Custom Metal WKV-7 kernel** — forward + a hand-written differentiable
  (gradient-checkpointed) backward, verified bit-exact against a reference;
  dedicated decode kernels for one token.
- **Memory-bounded training** — backward block by block (one block in memory,
  the same gradient), rows of a batch through the body in pieces, block inputs
  recorded with a stride; fp32 trainable weights over a bf16 (dense) or fp16
  (`.rwkvq`) base.

## Requirements

iOS 17 / iPadOS 17 / macOS 14 / visionOS 1 · Apple Silicon GPU · Swift 5.10+ ·
`mlx-swift` ≥ 0.31.4 (resolved automatically). The kernel runs on the Metal GPU —
use a real device or macOS, not the iOS Simulator.

## Installation

Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/RafaelUI/SwiftRWKV", from: "0.2.4"),
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "RWKVGen", package: "SwiftRWKV"),
        // and, if you need them:
        // .product(name: "RWKVRerank",    package: "SwiftRWKV"),
        // .product(name: "RWKVEmbedding", package: "SwiftRWKV"),
        // .product(name: "RWKVQuant",     package: "SwiftRWKV"),
    ]),
]
```

Or in Xcode: **File → Add Package Dependencies…** → paste the repo URL.

The latest tag is `0.2.4`. State-tuning, top-layer training over `.rwkvq` and
the memory levers above are on `main` and not tagged yet.

## Project layout

```
Sources/
├── RWKVKernel/                  # custom Metal WKV-7 kernels (low-level)
│   ├── WKV7.swift               #   forward + chunk forward + constants (CHUNK=16, HEAD_SIZE=64)
│   ├── WKV7Step.swift           #   single-token step, boundary state
│   ├── WKV7DecodeKernels.swift  #   decode: one-token WKV step and tmix tail in one launch
│   ├── WKV7Train.swift          #   differentiable checkpointed backward (CustomFunction)
│   ├── WKV7Reference.swift      #   naive recurrent DPLR — autograd ground-truth for tests
│   └── Ablation.swift           #   subtractive ablation of a training step (instrument)
├── RWKVQuant/                   # .rwkvq quantised base (independent of the backbone)
│   ├── RwkvqFile.swift          #   reading .rwkvq directly
│   ├── RwkvqSidecar.swift       #   the opened base (from .rwkvq or an export_mlx sidecar), manifest, naming
│   ├── RwkvqFullConvert.swift   #   the whole x070 weight dictionary from a full export
│   ├── RwkvqDequant.swift       #   fused Metal dequantisation, one launch
│   ├── RwkvqNative.swift        #   repacking into MLX's native quantised container
│   └── RwkvqSb6GEMV / SymGEMV / SymRows.swift   # projection kernels per preset
├── RWKVGen/                     # backbone, inference, generation, training
│   ├── X070Backbone.swift       #   canonical RWKV-7 backbone + split + training hooks
│   ├── RWKVBlock.swift          #   one block, shared by the backbone and the rerank head
│   ├── RWKVBatchState.swift     #   batched boundary state, masks, right-padding
│   ├── InitialState.swift       #   pluggable initial state, state-tuning, state files
│   ├── X070Generation.swift     #   RWKVState + prefill / step (streaming)
│   ├── Sampling.swift           #   SamplingConfig, seeded sampler, penalties
│   ├── GPUSampler.swift         #   the same sampler without a CPU read per token
│   ├── Generate.swift           #   generation loop: stop conditions, byte-safe streaming
│   ├── WorldTokenizer.swift     #   RWKV World trie tokenizer
│   ├── RwkvqBase.swift          #   attaching a .rwkvq base to an already-loaded dense backbone
│   ├── RwkvqStandaloneLoad.swift#   a backbone from one .rwkvq file, no dense file needed
│   ├── LoRA.swift               #   LoRA/QLoRA spec, add/merge/save/load, base quantization
│   ├── LoRAFinetune.swift       #   LoRA training loop (LM objective)
│   ├── X070PartialFinetune.swift#   N-layer partial fine-tune (boundary cache + train)
│   ├── Checkpoint.swift         #   gradient-checkpoint primitive
│   └── Training/                #   shared trainer, LR schedules, trainable sets,
│                                #   backward block by block, x070 init, token
│                                #   streams, pre-training
├── RWKVEmbedding/               # text vectors: pooling, head, contrastive losses,
│                                # GradCache, curriculum, metrics, checkpoints
├── RWKVRerank/                  # cross-encoder reranker: head over state, state
│                                # cache, listwise training, indexed serving
├── RerankRun/                   # executable rerank-run: data → cache → training → report
├── EmbedRun/                    # executable embed-run: one embedding stage, evaluation, A/B instruments
└── DecodeBench/                 # executable decode-bench: decode and training-step benchmarks
```

Four library products. `import RWKVGen` covers inference, generation and every
training path — it links `RWKVKernel` and `RWKVQuant` itself. Import
`RWKVRerank` or `RWKVEmbedding` for those tasks, and `RWKVQuant` alone only to
read a `.rwkvq` or a sidecar without a model. `RWKVKernel` is an internal target, not a
product: the kernels are reached through `RWKVGen`. Three executables —
`rerank-run`, `embed-run`, `decode-bench` — are the runners and instruments the
numbers in the docs come from.

## Documentation

Guides live in [docs/](docs), each with examples that are compiled on every test
run (`Scripts/check_docs.sh`) so they cannot silently rot:

- **[docs/Inference.md](docs/Inference.md)** — the two passes, generation and
  sampling, streaming and stop conditions, the quantised `.rwkvq` backend, and
  the measured numbers with the conditions they were measured under.
- **[docs/Train.md](docs/Train.md)** — the shared trainer and trainable sets,
  LoRA/QLoRA (also over `.rwkvq`), N-layer partial fine-tuning, pre-training.
- **[docs/state-tuning.md](docs/state-tuning.md)** — training the initial state,
  attaching it and folding prompts, the state file and base fingerprint, and
  what it buys: quality, transfer between a model and its `.rwkvq` files, cost.
- **[docs/reranker.md](docs/reranker.md)** — data, state cache, training the
  head, layer sweeps, serving.
- **[docs/Embedding.md](docs/Embedding.md)** — pooling, heads, losses,
  GradCache, curriculum, metrics, memory of a training step, training over a
  `.rwkvq` base, a state in place of the instruction.

Design decisions and their reasoning are in
[ARCHITECTURE.md](ARCHITECTURE.md); the working log of what is measured, what
is missing and what was tried and rejected is in
[NEXT_SESSION.md](NEXT_SESSION.md).

## Status

Experimental, but validated end to end on real models rather than on toys —
RWKV-7 World 0.1B, g1d 0.1B / 0.4B dense and as `.rwkvq`, and 1.5B / 2.9B for
inference. 566 tests, plus mutation suites that deliberately break the code and
require the tests to notice — the interesting failures in this problem are
silent, so "the tests pass" is not by itself evidence.

What is checked: WKV-7 kernel parity (bit-exact against a reference), the
hand-written backward against autograd, logits parity with the Python
implementation on real weights, the recurrent decode against the parallel pass
(7.4e-7 relative), the partial-fine-tune split (Δ = 0), `.rwkvq` dequantisation
bit-for-bit against a Python reference, gradients of the block-by-block
backward and of the state against autograd, state files and base fingerprints
against the rwkv-metal reference, and end-to-end fine-tuning runs.

API may still change before a stable release.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*SwiftRWKV — ImpulseLeap / Alexei Goncharov · <https://www.impulseleap.com>*
