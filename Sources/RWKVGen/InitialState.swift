import Foundation
import CryptoKit
import MLX
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Начальное состояние как подключаемый объект: обученное (state-tuning),
//  свёрнутый промпт или одно поверх другого. Порт
//  rwkv_metal/model/initial_state.py и rwkv_metal/lora/state_tune.py.
//
//  RWKV — RNN: вместо нулей последовательность может стартовать с любого
//  состояния. Обученное состояние и состояние после промпта — ОДИН И ТОТ ЖЕ
//  объект, `RWKVBatchState` с батчем 1:
//
//      wkv        [L, 1, H, S, S]  fp32
//      tmixShift  [L, 1, 1, D]
//      cmixShift  [L, 1, 1, D]
//
//  У обученного по умолчанию обучается только матрица WKV (как в RWKV-PEFT),
//  сдвиги остаются нулями; у состояния после промпта заполнены все три. В
//  файле всегда все три — потребителю не нужно знать, откуда объект.
//
//  Как подключается. `backbone.initialState = state` раскладывает состояние по
//  слоям; блок берёт из него h_in и сдвиги ТОЛЬКО там, где вызывающий своих не
//  передал (`rwkvBlockForward`, `rwkvBlockForwardWithState`). Поэтому:
//
//      body(ids)                     старт с подключённого состояния
//      states(prompt) / prefill      подключённое → прогон промпта → кэш
//      bodyWithState(ids, state: s)  явное состояние, подключённое не участвует
//
//  то есть склейка «обученное → системный промпт → кэш» — обычное
//  продолжение, отдельного кода у неё нет. Подключённое нулевое состояние
//  счёта не меняет: выход побитно тот же, что без него.
//
//  Обучение. Состояние слоя — параметр СВОЕГО блока (`blocks.N.state0.wkv`),
//  и обратный проход по блокам (`SequentialBody`) подхватывает его как любой
//  параметр блока: в памяти один блок. Нижний блок с параметрами — нулевой,
//  так что градиент идёт через всю глубину: шаг стоит как адаптеры на всех
//  слоях. [1, …] транслируется на батч внутри блока — сумму градиента по
//  строкам даёт vjp блока.
//
//  Файл. Отдельный safetensors (`StateFile`): три тензора в fp32, контракт
//  (слоёв, D, голов, размер головы), что обучалось, и ОТПЕЧАТОК БАЗЫ.
//  Состояние имеет смысл только на базе, на которой получено. Чужая
//  архитектура — отказ всегда; чужая база — отказ, пока вызывающий явно не
//  разрешил. Формат и отпечаток — те же, что в rwkv-metal, бит в бит.
// ───────────────────────────────────────────────────────────────────────

/// Начальное состояние одного слоя, батч 1.
public struct InitialLayerState {
    public var wkv: MLXArray          // [H, S, S] fp32 — всегда
    public var tmixShift: MLXArray    // [1, 1, D]
    public var cmixShift: MLXArray    // [1, 1, D]

    /// То же на `batch` строк (трансляцией, без копии): wkv [B, H, S, S],
    /// сдвиги [B, 1, D]. Сдвиги к типу потока приводит token-shift.
    func broadcast(batch B: Int) -> InitialLayerState {
        InitialLayerState(
            wkv: MLX.broadcast(wkv.asType(.float32).expandedDimensions(axis: 0), to: [B] + wkv.shape),
            tmixShift: MLX.broadcast(tmixShift, to: [B, 1, tmixShift.shape[2]]),
            cmixShift: MLX.broadcast(cmixShift, to: [B, 1, cmixShift.shape[2]]))
    }
}

public extension X070Backbone {

    /// Нулевое начальное состояние этой модели (батч 1, всё в fp32).
    func zeroInitialState() -> RWKVBatchState {
        RWKVBatchState(cfg: cfg, batch: 1, dtype: .float32)
    }

    /// Подключённое начальное состояние (батч 1) либо nil. Присваивание
    /// подключает: модель стартует с него всюду, где вызывающий не передал
    /// своё. nil — отключить.
    var initialState: RWKVBatchState? {
        get {
            guard let layers = initialLayers else { return nil }
            return RWKVBatchState.stacked(
                wkv: layers.map { $0.wkv.expandedDimensions(axis: 0) },
                tmix: layers.map { $0.tmixShift }, cmix: layers.map { $0.cmixShift }).evaluated()
        }
        set {
            guard let state = newValue else { initialLayers = nil; return }
            if let why = StateFile.mismatch(state, cfg) { preconditionFailure(why) }
            let layers = (0 ..< cfg.nLayer).map { l in
                InitialLayerState(wkv: state.layerWKV(l)[0].asType(.float32),
                                  tmixShift: state.layerTmixShift(l).asType(.float32),
                                  cmixShift: state.layerCmixShift(l).asType(.float32))
            }
            eval(layers.flatMap { [$0.wkv, $0.tmixShift, $0.cmixShift] })
            initialLayers = layers
        }
    }

    /// Состояние для ПОШАГОВОГО декода с самого начала: подключённое либо
    /// нули. (`prefill` стартует с подключённого сам — он идёт параллельным
    /// проходом; это нужно тому, кто декодирует `step` с пустого места.)
    func startState() -> RWKVState {
        let dtype = weight("ln0.weight").dtype
        guard let s = initialState else { return RWKVState(cfg: cfg, dtype: dtype) }
        return RWKVState(s.asType(dtype))
    }

    /// Свернуть промпт в состояние (батч 1). Старт — `state`, а без него
    /// подключённое состояние (или нули): «обученное → промпт → кэш».
    func foldPrompt(_ ids: [Int], from state: RWKVBatchState? = nil) -> RWKVBatchState {
        precondition(!ids.isEmpty, "foldPrompt: пустой промпт")
        let x = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        return withoutGradientPath { states(x, state: state) }.stopGradient().evaluated()
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Обучаемое множество
// ───────────────────────────────────────────────────────────────────────

/// Начальное состояние как обучаемое множество. База не трогается: её
/// заморозка — в том, что в множестве её нет.
///
/// Создание ПОДКЛЮЧАЕТ состояние к модели (с `initial`, а без него — с уже
/// подключённого либо с нулей) и переводит все слои на дифференцируемое ядро
/// (`trainLayers`): градиент состояния идёт через всю глубину.
public final class StateTrainableSet: TrainableSet {

    public enum Part: String, CaseIterable, Sendable {
        case wkv, tmixShift = "tmix_shift", cmixShift = "cmix_shift"
    }

    private let bb: X070Backbone
    public let parts: [Part]
    private let scale: Float?

    /// - train: что обучается; по умолчанию только матрица WKV.
    /// - learningRateScale: множитель шага для состояния относительно шага
    ///   тренера (nil — единица) — когда рядом в одном множестве обучаются веса.
    public init(_ bb: X070Backbone, train: [Part] = [.wkv], initial: RWKVBatchState? = nil,
                learningRateScale: Float? = nil) {
        precondition(!train.isEmpty, "обучаемое состояние без единой части")
        self.bb = bb
        self.parts = Part.allCases.filter(train.contains)
        self.scale = learningRateScale
        if let initial { bb.initialState = initial }
        else if bb.initialLayers == nil { bb.initialState = bb.zeroInitialState() }
        bb.trainLayers = Set(0 ..< bb.cfg.nLayer)
    }

    public var parameterNames: [String] {
        (0 ..< bb.cfg.nLayer).flatMap { l in parts.map { "blocks.\(l).state0.\($0.rawValue)" } }
    }

    /// Чисел в обучаемом состоянии.
    public var parameterCount: Int { initialParameters().reduce(0) { $0 + $1.size } }

    public var learningRateScales: [Float]? {
        scale.map { [Float](repeating: $0, count: bb.cfg.nLayer * parts.count) }
    }

    public func initialParameters() -> [MLXArray] {
        let ps = bb.initialLayers!.flatMap { layer in
            parts.map { part -> MLXArray in
                switch part {
                case .wkv: return layer.wkv.asType(.float32)
                case .tmixShift: return layer.tmixShift.asType(.float32)
                case .cmixShift: return layer.cmixShift.asType(.float32)
                }
            }
        }
        eval(ps)
        return ps
    }

    public func inject(_ ps: [MLXArray]) {
        var layers = bb.initialLayers!
        precondition(ps.count == layers.count * parts.count,
                     "состоянию передано \(ps.count) параметров, ожидалось \(layers.count * parts.count)")
        var i = 0
        for l in layers.indices {
            for part in parts {
                switch part {
                case .wkv: layers[l].wkv = ps[i]
                case .tmixShift: layers[l].tmixShift = ps[i]
                case .cmixShift: layers[l].cmixShift = ps[i]
                }
                i += 1
            }
        }
        bb.initialLayers = layers
    }

    public func commit(_ ps: [MLXArray]) {
        inject(ps)
        eval(ps)
    }
}

// ───────────────────────────────────────────────────────────────────────
//  State-tuning языковой модели
// ───────────────────────────────────────────────────────────────────────

public struct StateTuneConfig: Sendable {
    /// Состояние — не веса: оно стартует с нуля, и шаг Adam ему нужен на
    /// порядки крупнее, чем адаптерам (RWKV-PEFT: 1 → 0.01).
    public var lr: Float = 1.0
    public var lrMin: Float = 0.01
    public var schedule: LRSchedule = .cosine
    public var warmupSteps = 10
    public var maxSteps = 500
    public var gradClip: Float = 0          // 0 — без отсечения
    public var beta1: Float = 0.9
    public var beta2: Float = 0.99
    public var adamEps: Float = 1e-8
    public var train: [StateTrainableSet.Part] = [.wkv]
    public var sequentialBackward = true
    public var lossChunkTokens = 256
    public var cacheLimitGB = 0.25
    public var logEvery = 20

    public init() {}

    var training: TrainingConfig {
        TrainingConfig(lr: lr, lrMin: lrMin, schedule: schedule, warmupSteps: warmupSteps,
                       gradClip: gradClip, weightDecay: 0, beta1: beta1, beta2: beta2,
                       adamEps: adamEps, maxSteps: maxSteps, gradAccum: 1,
                       cacheLimitGB: cacheLimitGB, logEvery: logEvery)
    }
}

public enum StateTuning {

    /// Сколько раз градиент посчитан обратным проходом по блокам (для тестов).
    nonisolated(unsafe) static var sequentialGradientCalls = 0

    /// Обучить начальное состояние языковой модели на парах (x, y) [B, T]
    /// (T кратна `WKV7_CHUNK`). База заморожена целиком — в том числе
    /// адаптеры, если они есть. `initial` — стартовое состояние (например,
    /// после промпта); nil — подключённое либо нули. Возвращает обученное
    /// состояние (батч 1); оно остаётся подключённым к модели.
    @discardableResult
    public static func run(
        _ bb: X070Backbone,
        nextBatch: () -> LoRABatch,
        config cfg: StateTuneConfig = StateTuneConfig(),
        initial: RWKVBatchState? = nil,
        isCancelled: () -> Bool = { false },
        onStep: (_ step: Int, _ loss: Float, _ gradNorm: Float, _ peakMB: Double) -> Void = { _,_,_,_ in }
    ) -> (result: LoRATrainResult, state: RWKVBatchState) {
        let savedLayers = bb.trainLayers, savedCheckpoint = bb.useBlockCheckpoint
        defer { bb.trainLayers = savedLayers; bb.useBlockCheckpoint = savedCheckpoint }
        bb.useBlockCheckpoint = false
        let set = StateTrainableSet(bb, train: cfg.train, initial: initial)
        var checked = false
        let res = withoutActuallyEscaping(nextBatch) { escapingNext -> TrainingResult in
            let source: () -> LoRABatch = {
                let b = escapingNext()
                if !checked {
                    precondition(b.x.shape[1] % WKV7_CHUNK == 0,
                        "T (\(b.x.shape[1])) должна делиться на CHUNK \(WKV7_CHUNK)")
                    checked = true
                }
                return b
            }
            let provider: GradientProvider<LoRABatch>? = cfg.sequentialBackward
                ? { ps, batch in
                    sequentialGradientCalls += 1
                    return LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: batch,
                                                            lossChunk: cfg.lossChunkTokens)
                }
                : nil
            let trainer = Trainer<LoRABatch>(
                trainable: set,
                objective: { LoRAFinetune.languageModelLoss(bb, $0, chunk: cfg.lossChunkTokens) },
                nextBatch: source, config: cfg.training, gradient: provider)
            return trainer.run(isCancelled: isCancelled) { s in
                onStep(s.step, s.loss, s.gradNorm, s.peakMemoryMB)
            }
        }
        return (LoRATrainResult(finalLoss: res.finalLoss, steps: res.steps), bb.initialState!)
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Файл и отпечаток базы
// ───────────────────────────────────────────────────────────────────────

public enum StateFile {

    public static let format = "rwkv-state"
    public static let version = "1"

    public enum Kind: String, Sendable {
        case tuned, prompt, tunedPrompt = "tuned+prompt"
    }

    /// Метаданные файла.
    public struct Info: Equatable, Sendable {
        public var kind: String
        public var trained: [String]
        public var note: String
        public var baseFingerprint: String
        public var nLayer: Int, nEmbd: Int, nHead: Int, headSize: Int
    }

    public enum Failure: Error, CustomStringConvertible, Equatable {
        case notAStateFile(String)
        case unknownVersion(String)
        case missingTensor(String)
        /// Состояние от другой архитектуры — отказ всегда.
        case otherArchitecture(String)
        /// Состояние получено на другой базе — отказ, пока не `allowOtherBase`.
        case otherBase(file: String, model: String, note: String)

        public var description: String {
            switch self {
            case .notAStateFile(let f): return "не файл состояния (format=\(f))"
            case .unknownVersion(let v): return "версия формата \(v), знаем \(StateFile.version)"
            case .missingTensor(let t): return "в файле состояния нет тензора \(t)"
            case .otherArchitecture(let why): return "состояние от другой архитектуры: \(why)"
            case .otherBase(let f, let m, let note):
                return "состояние получено на другой базе (отпечаток \(f.prefix(12))…, у модели "
                    + "\(m.prefix(12))…; примечание файла: \(note)). На чужой базе оно даёт другой "
                    + "результат; allowOtherBase: true — подключить всё равно."
            }
        }
    }

    /// nil — состояние подходит модели; иначе — чем не подходит.
    static func mismatch(_ s: RWKVBatchState, _ cfg: X070Config) -> String? {
        let L = cfg.nLayer, H = cfg.nHead, S = cfg.headSize, D = cfg.nEmbd
        let want: [(String, [Int], [Int])] = [("wkv", s.wkv.shape, [L, 1, H, S, S]),
                                              ("tmix_shift", s.tmixShift.shape, [L, 1, 1, D]),
                                              ("cmix_shift", s.cmixShift.shape, [L, 1, 1, D])]
        for (name, got, shape) in want where got != shape {
            return "\(name) \(got), у модели \(shape) (слоёв \(L), D \(D), голов \(H), размер головы \(S); "
                + "батч обязан быть 1)"
        }
        return nil
    }

    // Малые тензоры, по которым считается отпечаток. Порядок — часть формата:
    // тот же список в rwkv_metal/model/initial_state.py.
    static let fingerprintBlock = ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                                   "tmix.x_r", "tmix.x_w", "tmix.x_k", "tmix.x_v", "tmix.x_a", "tmix.x_g",
                                   "tmix.k_k", "tmix.k_a", "tmix.r_k", "tmix.ln_x.weight", "tmix.ln_x.bias",
                                   "cmix.x_k"]
    static let fingerprintStem = ["ln0.weight", "ln0.bias", "ln_out.weight", "ln_out.bias"]

    /// sha256 (hex) по контракту, малым тензорам базы и строке 0 плотного веса
    /// двух проекций (первой и последней), всё через bf16 (строки сжатых
    /// проекций .rwkvq — через fp16, их решётку). Одна база в bf16 и в fp32 —
    /// один отпечаток; плотная модель, её REDUCTION и её COMPRESSION —
    /// три разные базы (малые тензоры у двух пресетов общие, их различают
    /// проекции). Отпечаток — базы: обучаемая подмена, адаптеры и подключённое
    /// состояние в него не входят.
    public static func baseFingerprint(_ bb: X070Backbone) -> String {
        let cfg = bb.cfg
        var h = SHA256()
        func feed(_ name: String, _ a: MLXArray, lattice: DType = .bfloat16) {
            let canon = a.asType(lattice).asType(.float32).reshaped([-1])
            eval(canon)
            h.update(data: Data(name.utf8) + Data([0]))
            canon.asArray(Float.self).withUnsafeBytes { h.update(bufferPointer: $0) }
        }
        h.update(data: Data("rwkv-state-base-v1\n\(cfg.nLayer),\(cfg.nEmbd),\(cfg.nHead),\(cfg.headSize)\n".utf8))
        var names = fingerprintStem
        for l in 0 ..< cfg.nLayer { names += fingerprintBlock.map { "blocks.\(l).\($0)" } }
        for name in names { feed(name, bb.weight(name)) }
        // Решётка строки — по роду слоя. Плотный вес идёт через bf16, как малые
        // тензоры. Вес из .rwkvq живёт на решётке fp16 (разворачивается прямо в
        // fp16, в том числе после `unpackTopLayers`; rwkv-metal разворачивает
        // его в fp32 и округляет так же): через fp16, иначе два округления
        // подряд разошлись бы с одним.
        for name in ["blocks.0.tmix.r_proj", "blocks.\(cfg.nLayer - 1).cmix.value"] {
            let row = bb.weight(name + ".weight")[0]
            feed(name, row, lattice: row.dtype == .float16 ? .float16 : .bfloat16)
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Записать состояние отдельным файлом. `base` — модель, на которой оно
    /// получено: из неё берутся контракт и отпечаток.
    @discardableResult
    public static func save(_ state: RWKVBatchState, base bb: X070Backbone, to url: URL,
                            kind: Kind = .tuned, trained: [StateTrainableSet.Part] = [.wkv],
                            note: String = "") throws -> Info {
        if let why = mismatch(state, bb.cfg) { throw Failure.otherArchitecture(why) }
        let info = Info(kind: kind.rawValue, trained: trained.map(\.rawValue), note: note,
                        baseFingerprint: baseFingerprint(bb), nLayer: bb.cfg.nLayer,
                        nEmbd: bb.cfg.nEmbd, nHead: bb.cfg.nHead, headSize: bb.cfg.headSize)
        let md = ["format": format, "version": version, "kind": info.kind,
                  "trained": info.trained.joined(separator: ","),
                  "base_fingerprint": info.baseFingerprint, "note": note,
                  "n_layer": String(info.nLayer), "n_embd": String(info.nEmbd),
                  "n_head": String(info.nHead), "head_size": String(info.headSize)]
        let arrays = ["wkv": state.wkv.asType(.float32),
                      "tmix_shift": state.tmixShift.asType(.float32),
                      "cmix_shift": state.cmixShift.asType(.float32)]
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, metadata: md, url: url)
        return info
    }

    /// Прочитать файл состояния. С `base` проверяются контракт (несовпадение —
    /// отказ всегда) и отпечаток базы (несовпадение — `Failure.otherBase`, если
    /// не `allowOtherBase`).
    public static func load(_ url: URL, base bb: X070Backbone? = nil,
                            allowOtherBase: Bool = false) throws -> (state: RWKVBatchState, info: Info) {
        let (arrays, md) = try loadArraysAndMetadata(url: url)
        guard md["format"] == format else { throw Failure.notAStateFile(md["format"] ?? "нет") }
        guard md["version"] == version else { throw Failure.unknownVersion(md["version"] ?? "нет") }
        for t in ["wkv", "tmix_shift", "cmix_shift"] where arrays[t] == nil { throw Failure.missingTensor(t) }
        let state = RWKVBatchState(wkv: arrays["wkv"]!, tmixShift: arrays["tmix_shift"]!,
                                   cmixShift: arrays["cmix_shift"]!)
        let info = Info(kind: md["kind"] ?? "", trained: (md["trained"] ?? "").split(separator: ",").map(String.init),
                        note: md["note"] ?? "", baseFingerprint: md["base_fingerprint"] ?? "",
                        nLayer: Int(md["n_layer"] ?? "") ?? -1, nEmbd: Int(md["n_embd"] ?? "") ?? -1,
                        nHead: Int(md["n_head"] ?? "") ?? -1, headSize: Int(md["head_size"] ?? "") ?? -1)
        if let bb {
            let c = bb.cfg
            for (name, file, model) in [("n_layer", info.nLayer, c.nLayer), ("n_embd", info.nEmbd, c.nEmbd),
                                        ("n_head", info.nHead, c.nHead), ("head_size", info.headSize, c.headSize)]
            where file != model {
                throw Failure.otherArchitecture("\(name)=\(file) в файле, \(model) у модели")
            }
            if let why = mismatch(state, c) { throw Failure.otherArchitecture(why) }
            let fp = baseFingerprint(bb)
            if info.baseFingerprint != fp && !allowOtherBase {
                throw Failure.otherBase(file: info.baseFingerprint, model: fp, note: info.note)
            }
        }
        return (state, info)
    }
}
