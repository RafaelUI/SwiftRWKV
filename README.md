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
                      cfg: X070Config(nLayer: 18, nEmbd: 448, headSize: 64, vocab: 16000))

let logits = bb(idsTensor)                                   // [1, T, vocab]
```

## Features

- **Inference & streaming generation** — recurrent state with O(1) memory per token.
- **LoRA / QLoRA fine-tuning** — for larger models (≈600M+); 4/8-bit quantized base.
- **N-layer partial fine-tuning** — full-weight training of the top N layers on a
  frozen, disk-cached lower stack; for smaller models (≤400M).
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
    .package(url: "https://github.com/RafaelUI/SwiftRWKV", from: "0.1.0"),
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "RWKVGen", package: "SwiftRWKV"),
    ]),
]
```

Or in Xcode: **File → Add Package Dependencies…** → paste the repo URL.

## Project layout

```
Sources/
├── RWKVKernel/                  # custom Metal WKV-7 kernel (low-level)
│   ├── WKV7.swift               #   forward + chunk forward + constants (CHUNK=32, HEAD_SIZE=64)
│   ├── WKV7Train.swift          #   differentiable checkpointed backward (CustomFunction)
│   └── WKV7Reference.swift      #   naive recurrent DPLR — autograd ground-truth for tests
└── RWKVGen/                     # everything high-level (import this)
    ├── X070Backbone.swift       #   canonical RWKV-7 backbone + split + training hooks
    ├── X070Generation.swift     #   RWKVState + prefill / step (streaming)
    ├── WorldTokenizer.swift     #   RWKV World trie tokenizer
    ├── LoRA.swift               #   LoRA/QLoRA spec, add/merge/save/load, base quantization
    ├── LoRAFinetune.swift       #   LoRA training loop (LM objective)
    ├── X070PartialFinetune.swift#   N-layer partial fine-tune (boundary cache + train)
    └── Checkpoint.swift         #   gradient-checkpoint primitive
```

Two modules: `import RWKVGen` for every high-level workflow (it depends on
`RWKVKernel`); `import RWKVKernel` only if you call the `wkv7*` kernels directly.

## Documentation

See **[Train.md](Train.md)** for the full guide: loading models, the public API
by use case, FLA→x070 checkpoint conversion, inference/generation, LoRA/QLoRA,
N-layer partial fine-tuning, the network-split mechanism, memory/perf, and the
validation status.

## Status

Experimental but validated end-to-end on a real model (61M, Russian). Covered by
tests: WKV-7 kernel parity (bit-exact), custom backward vs autograd, x070 logits
parity vs the Python reference, the partial-fine-tune split (Δ=0), and an
end-to-end fine-tune run. API may still change before a stable release.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*SwiftRWKV — ImpulseLeap / Alexei Goncharov · <https://www.impulseleap.com>*
