# Training Guide — `RWKVTrain`

On-device fine-tuning of **RWKV-7** classifiers and feature extractors on Apple
Silicon (iOS / iPadOS / macOS / visionOS), powered by
[MLX](https://github.com/ml-explore/mlx) and a custom Metal **WKV-7** kernel.

This module trains the **top layers + a classification head** on top of a frozen
pretrained backbone. The frozen part of the network is run once and its boundary
features are cached to disk; only the small upper part is trained. This keeps
memory low enough to fine-tune on a phone.

- Module: `import RWKVTrain`
- Tasks: **text classification**, **feature extraction**
- Hot path (WKV-7 recurrence): a hand-written Metal kernel, bit-exact against the
  Python reference (`Δ = 0`, see [Kernel parity](#kernel-parity)).

> For large-model inference and LoRA/QLoRA fine-tuning, see the `RWKVGen` module
> (coming later).

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [The model package format](#the-model-package-format)
4. [Quick start](#quick-start)
5. [Providing data](#providing-data)
6. [`ModelConfig`](#modelconfig)
7. [`TrainingConfig`](#trainingconfig)
8. [Logging & metrics](#logging--metrics)
9. [Cancellation](#cancellation)
10. [Saving & loading trained models](#saving--loading-trained-models)
11. [Inference](#inference)
12. [How it works](#how-it-works)
13. [Memory & performance](#memory--performance)
14. [Error handling](#error-handling)
15. [Kernel parity](#kernel-parity)
16. [FAQ / gotchas](#faq--gotchas)

---

## Requirements

| | Minimum |
|---|---|
| OS | iOS 17 / iPadOS 17 / macOS 14 / visionOS 1 |
| Hardware | Apple Silicon (M-series / A-series with a GPU) |
| Toolchain | Swift 5.10+ |
| Dependency | `mlx-swift` ≥ 0.31.4 (resolved automatically) |

> The WKV-7 kernel runs on the **GPU via Metal**. It will not run on the iOS
> Simulator's software renderer reliably — test on a real device or on macOS.

---

## Installation

Swift Package Manager. In `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/RafaelUI/SwiftRWKV", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "RWKVTrain", package: "SwiftRWKV"),
        ]
    ),
]
```

Or in Xcode: **File → Add Package Dependencies…** and paste the repo URL.

```swift
import RWKVTrain
```

---

## The model package format

A model is a **self-contained folder** with exactly three files:

```
my-model/
├── config.json          # ModelConfig (architecture + optional task/training fields)
├── model.safetensors    # backbone weights (+ head, if trained)
└── tokenizer.json        # HF-format byte-level BPE tokenizer
```

This format lets you keep any number of models and load each one uniformly by
its folder URL. `RWKVModel(directory:)` reads all three and validates geometry.

A minimal `config.json` for a base (untrained) backbone:

```json
{
  "arch": "rwkv7",
  "nLayer": 18,
  "nEmbd": 256,
  "headSize": 64,
  "vocab": 16000,
  "contextSize": 128,
  "language": "ru",
  "tokenizer": "ru16k"
}
```

After training, `Trainer.save(_:to:)` writes a richer `config.json` that also
carries `task`, `pooling`, `numClasses`, `classes`, `freeze`, `valAcc`, etc.

---

## Quick start

End-to-end: load a base model, fine-tune a 3-class classifier, save it, run it.

```swift
import RWKVTrain
import Foundation

// 1. Load the pretrained base model (folder with config/weights/tokenizer).
let baseURL = URL(fileURLWithPath: "/path/to/base-model")
let base = try RWKVModel(directory: baseURL)

// 2. Build a dataset. Here we tokenize with the model's own tokenizer.
func ex(_ text: String, _ label: Int) -> Example {
    Example(ids: base.tokenizer.encode(text), label: label)
}
let data = InMemoryDataProvider(
    train: [
        ex("Отличный сервис, очень доволен!", 2),
        ex("Ужасно, никому не советую.",       0),
        ex("Нормально, ничего особенного.",    1),
        // … hundreds–thousands more …
    ],
    validation: [
        ex("Прекрасная работа, спасибо!", 2),
        ex("Полное разочарование.",       0),
    ],
    classes: ["negative", "neutral", "positive"]
)

// 3. Train (blocking — run off the main thread).
let trainer = Trainer(model: base, classes: data.classes, logger: OSLogLogger())
let result = try trainer.train(
    data: data,
    config: TrainingConfig(freeze: 14, contextSize: 128, epochs: 5)
)
print("val accuracy:", result.validationAccuracy)

// 4. Save as a new self-contained model package.
let outURL = URL(fileURLWithPath: "/path/to/my-trained-model")
try trainer.save(result, to: outURL)

// 5. Inference.
let model = try RWKVModel(directory: outURL)
let clf = try Classifier(model: model)
let p = clf.classify("Сделано на отлично!")
print(p.label, p.probabilities)   // "positive", [(positive, 0.91), (neutral, 0.07), …]
```

Run training on a background queue and marshal progress back to the UI:

```swift
let token = CancellationToken()
DispatchQueue.global(qos: .userInitiated).async {
    do {
        let result = try trainer.train(data: data,
                                       config: TrainingConfig(freeze: 14),
                                       cancellation: token)
        DispatchQueue.main.async { /* update UI, save, … */ }
    } catch {
        DispatchQueue.main.async { /* show error */ }
    }
}
// later, to abort:
token.cancel()
```

---

## Providing data

Data flows in through the `DataProvider` protocol — **no `Bundle.main`
assumptions, no fixed languages or class counts**. Your provider is responsible
for tokenization and returns ready-to-use `Example` values.

```swift
public struct Example: Sendable {
    public let ids: [Int]   // token ids, no padding (the pipeline pads/truncates)
    public let label: Int   // class index 0 ..< numClasses
}

public protocol DataProvider: Sendable {
    func trainExamples() throws -> [Example]
    func validationExamples() throws -> [Example]   // may be empty
    var classes: [String] { get }                    // class names in label order
}
```

Two ready-made providers ship with the module.

### In-memory

When you already have texts/labels (or tokenized them yourself):

```swift
let data = InMemoryDataProvider(
    train: trainExamples,
    validation: valExamples,         // optional
    classes: ["spam", "ham"]
)
```

### JSONL files

For `*.jsonl` files with one `{"text": "...", "label": 0}` object per line:

```swift
let data = JSONLDataProvider(
    trainURL: URL(fileURLWithPath: "train.jsonl"),
    validationURL: URL(fileURLWithPath: "val.jsonl"),   // optional
    tokenizer: base.tokenizer,
    maxLen: 128,                       // truncate token sequences
    classes: ["negative", "neutral", "positive"]
)
```

Notes:
- Labels are **0-based** and must match the order of `classes`.
- Empty texts are encoded as `[0]` to avoid empty inputs.
- Rows longer than `maxLen` tokens are truncated.

### Custom provider

Implement the protocol to read from Core Data, a server, etc.:

```swift
struct MyProvider: DataProvider {
    let classes = ["a", "b", "c"]
    func trainExamples() throws -> [Example] { /* … */ }
    func validationExamples() throws -> [Example] { [] }
}
```

---

## `ModelConfig`

Describes a model package; serialized to `config.json`.

```swift
public struct ModelConfig: Codable, Sendable {
    // Architecture (required)
    public var arch: String          // "rwkv7"
    public var nLayer: Int
    public var nEmbd: Int
    public var headSize: Int          // must equal the kernel HEAD_SIZE (64)
    public var vocab: Int
    public var contextSize: Int       // must be divisible by the kernel CHUNK (32)

    // Metadata (optional)
    public var language: String?
    public var tokenizer: String?

    // Task / training (filled in after fine-tuning)
    public var task: RWKVTask?              // .classification | .featureExtraction
    public var pooling: RWKVPooling?        // .mean | .last
    public var numClasses: Int?
    public var classes: [String]?
    public var freeze: Int?
    public var parent: String?
    public var valAcc: Float?
    public var createdAt: Double?
}
```

`ModelConfig.validate()` (called automatically on load and before training)
enforces:

- `nEmbd % headSize == 0`
- `headSize == 64` (the kernel's `HEAD_SIZE`)
- `contextSize % 32 == 0` (the kernel's `CHUNK`)

A mismatch throws `RWKVError.invalidGeometry`.

---

## `TrainingConfig`

```swift
public struct TrainingConfig: Sendable {
    public var freeze: Int            // number of frozen bottom layers
    public var contextSize: Int = 128 // padded/truncated length; multiple of 32
    public var epochs: Int = 5
    public var batchSize: Int = 8
    public var learningRate: Float = 1e-4
    public var featureBatch: Int = 5  // texts per forward during feature extraction
    public var pooling: RWKVPooling = .mean
}
```

### Choosing `freeze`

`freeze` is the number of **frozen bottom layers**; the remaining
`nLayer - freeze` layers are trained together with the head.

| `freeze` (for an 18-layer model) | Trained layers | Trade-off |
|---|---|---|
| `nLayer - 1` (e.g. 17) | head + 1 layer | fastest, lowest memory, least capacity |
| `~0.75 · nLayer` (e.g. 14) | head + 4 layers | good default |
| small (e.g. 6) | many layers | most capacity, slowest, most memory |

Must satisfy `0 <= freeze < nLayer`, otherwise `RWKVError.invalidGeometry`.

### Pooling

How the `[B, T, D]` sequence is reduced to a `[B, D]` vector for the head:

- `.mean` — masked mean over real (non-padding) tokens. **Default, robust.**
- `.last` — vector of the last real token (classic RWKV).

Use `.mean` unless you have a reason not to: it is stable against right-padding.

### Suggested presets

```swift
// Fast / phone-friendly
TrainingConfig(freeze: 16, contextSize: 128, epochs: 4,
               batchSize: 8, learningRate: 1e-4, featureBatch: 5)

// Balanced (default-ish)
TrainingConfig(freeze: 14, contextSize: 128, epochs: 5,
               batchSize: 8, learningRate: 1e-4, featureBatch: 8)

// Higher capacity (Mac / iPad Pro)
TrainingConfig(freeze: 10, contextSize: 256, epochs: 6,
               batchSize: 16, learningRate: 8e-5, featureBatch: 12)
```

---

## Logging & metrics

All output goes through the `RWKVLogger` protocol — no scattered `print`s.
Implementations must be `Sendable` (training runs on a background thread).

```swift
public protocol RWKVLogger: Sendable {
    func log(_ level: RWKVLogLevel, _ message: @autoclosure () -> String)
    func metric(_ metric: RWKVMetric, value: Double, step: Int)
}
```

Built-in loggers:

- `NoopLogger()` — discards everything (default).
- `OSLogLogger(subsystem:category:minLevel:)` — routes to `os.Logger`
  (visible in Console.app / Instruments).

Metrics emitted during training (`RWKVMetric`):

| Metric | When | Meaning |
|---|---|---|
| `.extractionProgress` | feature extraction | fraction 0…1 |
| `.tokensPerSecond` | extraction | throughput |
| `.peakMemoryMB` | extraction + training | resident footprint (MB) |
| `.loss` | each training step | cross-entropy |
| `.epoch` | end of epoch | epoch index |
| `.valAccuracy` | end of epoch | validation accuracy 0…1 |

A custom logger that drives a SwiftUI chart:

```swift
final class ChartLogger: RWKVLogger, @unchecked Sendable {
    let onMetric: @Sendable (RWKVMetric, Double, Int) -> Void
    init(_ onMetric: @escaping @Sendable (RWKVMetric, Double, Int) -> Void) {
        self.onMetric = onMetric
    }
    func log(_ level: RWKVLogLevel, _ message: @autoclosure () -> String) {}
    func metric(_ metric: RWKVMetric, value: Double, step: Int) {
        onMetric(metric, value, step)
    }
}

let logger = ChartLogger { metric, value, step in
    DispatchQueue.main.async {
        if metric == .loss { lossHistory.append(value) }
        if metric == .valAccuracy { valAcc = value }
    }
}
```

---

## Cancellation

`train(...)` is a long, blocking call. Pass a `CancellationToken` and call
`cancel()` from any thread. The pipeline checks the flag between batches/epochs
and throws `RWKVError.cancelled`.

```swift
let token = CancellationToken()
// background: try trainer.train(data: data, config: cfg, cancellation: token)
// UI button:  token.cancel()
```

On cancellation (or any error) the on-disk feature cache and the Metal buffer
cache are cleaned up automatically.

---

## Saving & loading trained models

`Trainer.save(_:to:)` writes a **new, self-contained model package**:

```swift
try trainer.save(result, to: outURL)
// outURL/
//   config.json          ← trained ModelConfig (task, classes, freeze, valAcc, …)
//   model.safetensors     ← base weights + trained layers + head (bf16)
//   tokenizer.json         ← copied from the base model
```

Load it back like any model:

```swift
let model = try RWKVModel(directory: outURL)
print(model.config.classes ?? [])
print(model.hasClassifierHead)   // true
```

Trained tensors are stored in **bf16** to match the base weights (smaller on
disk). Training itself runs in fp32 internally for stability.

---

## Inference

### Classification

```swift
let model = try RWKVModel(directory: outURL)
let clf = try Classifier(model: model)          // throws if no head/classes

let pred = clf.classify("some text")
pred.label                                       // winning class name
pred.probabilities                               // [(label, probability)], sorted desc
```

`Classifier.init` throws `RWKVError.missingWeight("head.weight")` if the model
has no trained head, or `RWKVError.invalidConfig` if it has no `classes`.

### Feature extraction (no head)

Get a sentence/document embedding (`[nEmbd]`) from any backbone:

```swift
let extractor = FeatureExtractor(model: model)
let vector: [Float] = extractor.features("some text")   // length == nEmbd
```

Useful for similarity search, clustering, or training your own head elsewhere.

---

## How it works

Partial fine-tuning in two phases:

1. **Boundary feature extraction.** The frozen bottom layers `[0, freeze)` are
   run once over every example. The boundary activation `x` at the input of
   layer `freeze` is written to a memory-mapped file on disk in bf16. Only `x`
   is stored; `vFirst` is recomputed from token ids (layer 0) and `xPrev` is the
   last time-step of `x`. This is the slow, GPU-bound phase.

2. **Top-layer training.** The trainable layers `[freeze, nLayer)` plus the head
   are trained over the cached features with a manual Adam optimizer and a
   gradient-checkpointed, differentiable WKV-7 kernel. Because the frozen part
   is never re-run, this phase is fast and memory-light.

The WKV-7 recurrence (the part that does not map onto standard ops) is a custom
Metal kernel:

- `wkv7Forward` — full-sequence forward, no autodiff (used for the frozen pass).
- `wkv7Train` — full-sequence forward + a hand-written checkpointed backward,
  wrapped in an MLX `CustomFunction` so autodiff picks it up (used for the
  trainable layers).

Constants: `HEAD_SIZE = 64`, `CHUNK = 32`. The hidden state is checkpointed
every 32 tokens so the backward pass reconstructs each chunk stably.

---

## Memory & performance

- **Peak memory** is dominated by activations during feature extraction, scaling
  with `featureBatch × contextSize`. Lower `featureBatch` if you hit memory
  pressure on a phone; raise it on a Mac to keep the GPU busy.
- `featureBatch = 1` underutilizes the GPU (slow, and the device runs hot).
  Values of `5–12` are typically much faster.
- The module caps the MLX Metal buffer cache (`GPU.cacheLimit = 64 MB`) during
  training and clears it on exit to avoid a 2× footprint on a second run.
- Feature extraction is GPU-bound; top-layer training is fast because it works on
  the small cached features.
- Track `RWKVMetric.peakMemoryMB` and `.tokensPerSecond` via your logger to tune
  `featureBatch` / `batchSize` for a given device.

---

## Error handling

All recoverable failures throw `RWKVError` (instead of `precondition`/`fatalError`
in the public path):

```swift
public enum RWKVError: Error {
    case missingFile(file: String, directory: String)
    case invalidConfig(reason: String)
    case weightsLoadFailed(reason: String)
    case missingWeight(key: String)
    case tokenizerLoadFailed(reason: String)
    case invalidGeometry(reason: String)
    case invalidDataset(reason: String)
    case cancelled
}
```

```swift
do {
    let result = try trainer.train(data: data, config: cfg, cancellation: token)
    try trainer.save(result, to: outURL)
} catch RWKVError.cancelled {
    // user aborted
} catch let RWKVError.invalidGeometry(reason) {
    // e.g. contextSize not divisible by 32
    print("geometry:", reason)
} catch {
    print("training failed:", error)
}
```

---

## Kernel parity

The WKV-7 Metal kernel is verified **bit-for-bit** against the Python reference
from [`rwkv-metal`](https://github.com/RafaelUI/SwiftRWKV). The test loads a
fixture (`r,w,k,v,a,b,h_in` inputs + reference `out,h_out,sa_out` outputs for one
chunk) and checks the three outputs:

```
$ swift test
[parity] chunk forward:        out Δ=0.00e+00  h_out Δ=0.00e+00  sa_out Δ=0.00e+00
[parity] full vs chunk:        Δ=0.00e+00
[parity] train-forward:        Δ=0.00e+00
Executed 3 tests, with 0 failures
```

This guarantees the kernel was ported into the framework without regression.

---

## FAQ / gotchas

**Does it run on the iOS Simulator?**
The kernel needs a real Metal GPU. Use a physical device or macOS.

**`contextSize` / `headSize` constraints?**
`contextSize` must be a multiple of `32` (CHUNK); `headSize` must be `64`
(HEAD_SIZE). Otherwise you get `RWKVError.invalidGeometry`.

**Why bf16 on disk but fp32 in training?**
bf16 keeps model files small and matches the base weights. Training keeps
trainable params/optimizer state in fp32, because bf16 Adam updates lose
precision on rounding.

**Can I train without a validation set?**
Yes — return `[]` from `validationExamples()`. Then `validationAccuracy`
reflects whatever the pipeline computes on an empty set (treat it as undefined);
prefer providing at least a small val split.

**Is `Trainer.train` async?**
No, it's synchronous and blocking. Call it from a background queue and report
progress through your `RWKVLogger`. MLX is not thread-safe, so do not run two
trainings concurrently — serialize them on one queue.

**How do classes map to labels?**
`classes[i]` is the name for label `i`. Keep the order consistent between your
`DataProvider.classes` and the `classes` you pass to `Trainer`.

---

*SwiftRWKV — ImpulseLeap / Alexei Goncharov · <https://www.impulseleap.com> ·
<https://github.com/RafaelUI/SwiftRWKV>*
