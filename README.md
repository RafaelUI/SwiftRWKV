# SwiftRWKV

On-device **RWKV-7** training and inference for the Apple ecosystem
(iOS · iPadOS · macOS · visionOS), built on
[MLX](https://github.com/ml-explore/mlx) and a custom Metal **WKV-7** kernel.

RWKV-7 is a linear-attention (RNN-style) language model: constant memory per
generated token and no quadratic attention, which makes it a good fit for running
*and adapting* models directly on a phone or Mac. SwiftRWKV exposes one canonical
backbone (`X070Backbone`, a faithful port of official RWKV-7 "Goose" x070) and
builds inference, generation, and two fine-tuning paths on top of it.

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
  streaming with stop strings and cancellation. Greedy by default.
- **Inference** — a parallel pass for whole sequences and a recurrent one with
  O(1) memory per token; prefill uses the parallel pass and is measured 94×
  faster than folding the prompt token by token.
- **Reranking** — a cross-encoder head over backbone state, with a state cache,
  listwise training and an indexed serving path.
- **Embeddings** — pooling, a trainable head, contrastive losses, GradCache,
  curriculum stages and retrieval metrics.
- **Quantised `.rwkvq` base** — sidecar format from
  [rwkv-quant](https://github.com/RafaelUI/rwkv-quant) with a fused Metal
  dequantisation kernel; inference, reranking and embeddings all run on it
  unchanged.
- **LoRA / QLoRA fine-tuning** — for larger models (≈600M+); 4/8-bit quantized base.
- **N-layer partial fine-tuning** — full-weight training of the top N layers on a
  frozen, disk-cached lower stack; for smaller models (≤400M).
- **Pre-training from scratch** — x070 initialisation, `.bin` token streams, a
  shared trainer with LR schedules and resumable checkpoints.
- **Custom Metal WKV-7 kernel** — forward + a hand-written differentiable
  (gradient-checkpointed) backward, verified bit-exact against a reference.
- **fp32 training, bf16 weights** — matches how RWKV-7 World was trained.

## Requirements

iOS 17 / iPadOS 17 / macOS 14 / visionOS 1 · Apple Silicon GPU · Swift 5.10+ ·
`mlx-swift` ≥ 0.31.4 (resolved automatically). The kernel runs on the Metal GPU —
use a real device or macOS, not the iOS Simulator.

## Installation

Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/RafaelUI/SwiftRWKV", from: "0.2.0"),
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

## Project layout

```
Sources/
├── RWKVKernel/                  # custom Metal WKV-7 kernel (low-level)
│   ├── WKV7.swift               #   forward + chunk forward + constants (CHUNK=16, HEAD_SIZE=64)
│   ├── WKV7Step.swift           #   single-token step, boundary state
│   ├── WKV7Train.swift          #   differentiable checkpointed backward (CustomFunction)
│   └── WKV7Reference.swift      #   naive recurrent DPLR — autograd ground-truth for tests
├── RWKVQuant/                   # .rwkvq quantised base (independent of the backbone)
│   ├── RwkvqSidecar.swift       #   format, manifest, x070↔world naming
│   └── RwkvqDequant.swift       #   fused Metal dequantisation, one launch
├── RWKVGen/                     # backbone, inference, generation, training
│   ├── X070Backbone.swift       #   canonical RWKV-7 backbone + split + training hooks
│   ├── RWKVBlock.swift          #   one block, shared by the backbone and the rerank head
│   ├── RWKVBatchState.swift     #   batched boundary state, masks, right-padding
│   ├── X070Generation.swift     #   RWKVState + prefill / step (streaming)
│   ├── Sampling.swift           #   SamplingConfig, seeded sampler, penalties
│   ├── Generate.swift           #   generation loop: stop conditions, byte-safe streaming
│   ├── WorldTokenizer.swift     #   RWKV World trie tokenizer
│   ├── RwkvqBase.swift          #   attaching a .rwkvq sidecar to the backbone
│   ├── LoRA.swift               #   LoRA/QLoRA spec, add/merge/save/load, base quantization
│   ├── LoRAFinetune.swift       #   LoRA training loop (LM objective)
│   ├── X070PartialFinetune.swift#   N-layer partial fine-tune (boundary cache + train)
│   ├── Checkpoint.swift         #   gradient-checkpoint primitive
│   └── Training/                #   shared trainer, LR schedules, trainable sets,
│                                #   x070 init, token streams, pre-training
├── RWKVEmbedding/               # text vectors: pooling, head, contrastive losses,
│                                # GradCache, curriculum, metrics, checkpoints
├── RWKVRerank/                  # cross-encoder reranker: head over state, state
│                                # cache, listwise training, indexed serving
└── RerankRun/                   # executable: data → cache → training → report
```

Five importable modules. `import RWKVGen` covers inference, generation and every
training path — it pulls in `RWKVKernel` and `RWKVQuant` itself. Import
`RWKVRerank` or `RWKVEmbedding` for those tasks, `RWKVQuant` alone only to read
a sidecar without a model, and `RWKVKernel` only to call the `wkv7*` kernels
directly.

## Documentation

Guides live in [docs/](docs), each with examples that are compiled on every test
run (`Scripts/check_docs.sh`) so they cannot silently rot:

- **[docs/Inference.md](docs/Inference.md)** — the two passes, generation and
  sampling, streaming and stop conditions, the quantised `.rwkvq` backend, and
  the measured numbers with the conditions they were measured under.
- **[docs/Train.md](docs/Train.md)** — loading and converting checkpoints,
  LoRA/QLoRA, N-layer partial fine-tuning, pre-training, the shared trainer.
- **[docs/reranker.md](docs/reranker.md)** — data, state cache, training the
  head, layer sweeps, serving.
- **[docs/Embedding.md](docs/Embedding.md)** — pooling, heads, losses,
  GradCache, curriculum, metrics.

Design decisions and their reasoning are in
[ARCHITECTURE.md](ARCHITECTURE.md); the working log of what is measured, what
is missing and what was tried and rejected is in
[NEXT_SESSION.md](NEXT_SESSION.md).

## Status

Experimental, but validated end to end on a real RWKV-7 World 0.1B rather than
on toys. 418 tests, plus mutation suites that deliberately break the code and
require the tests to notice — the interesting failures in this problem are
silent, so "the tests pass" is not by itself evidence.

What is checked: WKV-7 kernel parity (bit-exact against a reference), the
hand-written backward against autograd, logits parity with the Python
implementation on real weights, the recurrent decode against the parallel pass
(7.4e-7 relative), the partial-fine-tune split (Δ = 0), `.rwkvq` dequantisation
bit-for-bit against a Python reference, and end-to-end fine-tuning runs.

API may still change before a stable release.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*SwiftRWKV — ImpulseLeap / Alexei Goncharov · <https://www.impulseleap.com>*
