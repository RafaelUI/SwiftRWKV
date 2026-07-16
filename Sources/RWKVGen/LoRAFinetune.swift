import Foundation
import MLX
import MLXNN          // crossEntropy

// ───────────────────────────────────────────────────────────────────────
//  Высокоуровневый LoRA/QLoRA-файнтюн — порт rwkv_metal/lora/finetune.py.
//
//  Рецепт (валидирован в Python на World-1.5B):
//   • diff-множество = только loraA/loraB (функциональный аналог
//     nn.value_and_grad: mx.value_and_grad дифференцировал бы всё дерево);
//   • fp32-мастер адаптеров (bf16-апдейты теряются на округлении), в forward
//     инжектится bf16-копия — каст внутри замыкания сохраняет grad-цепь;
//   • большой эффективный батч через grad-accumulation, eval между микрошагами
//     (иначе ленивый граф разрастается);
//   • global-norm grad clip = 1.0;  lr=1e-4, alpha=16 для предобученных баз
//     (агрессивный lr расходится мгновенно);
//   • mx.set_cache_limit — ограничить рост буферного кэша Metal.
//
//  Gradient checkpointing двухуровневый:
//   (1) внутри WKV — wkv7Train чекпойнтит h/sa по чанкам (всегда);
//   (2) на уровне блока — cfg.useBlockCheckpoint ⇒ blockCheckpointed/checkpointed
//       пересчитывают тело блока в backward (опционально).
//  Уровень (2): −пик памяти на активациях блока ценой +1 forward.
//
//  ВАЖНО: T (длина контекста батча) обязана делиться на WKV7_CHUNK (16),
//  т.к. обучаемые слои идут через wkv7Train.
// ───────────────────────────────────────────────────────────────────────

import RWKVKernel    // WKV7_CHUNK

public typealias LoRABatch = (x: MLXArray, y: MLXArray)   // ids [B,T] int32

public struct LoRAConfig: Sendable {
    public var lr: Float
    public var gradClip: Float
    public var weightDecay: Float
    public var beta1: Float
    public var beta2: Float
    public var adamEps: Float
    public var maxSteps: Int
    public var gradAccum: Int
    public var warmupSteps: Int
    public var cacheLimitGB: Double      // <=0 ⇒ не ограничивать
    public var logEvery: Int
    /// true ⇒ блочный gradient checkpoint (−~45% пик памяти / +~22% времени).
    public var useBlockCheckpoint: Bool

    public init(lr: Float = 1e-4, gradClip: Float = 1.0, weightDecay: Float = 0.0,
                beta1: Float = 0.9, beta2: Float = 0.95, adamEps: Float = 1e-8,
                maxSteps: Int = 1000, gradAccum: Int = 1, warmupSteps: Int = 0,
                cacheLimitGB: Double = 1.5, logEvery: Int = 10,
                useBlockCheckpoint: Bool = false) {
        self.lr = lr; self.gradClip = gradClip; self.weightDecay = weightDecay
        self.beta1 = beta1; self.beta2 = beta2; self.adamEps = adamEps
        self.maxSteps = maxSteps; self.gradAccum = gradAccum; self.warmupSteps = warmupSteps
        self.cacheLimitGB = cacheLimitGB; self.logEvery = logEvery
        self.useBlockCheckpoint = useBlockCheckpoint
    }
}

public struct LoRATrainResult: Sendable {
    public let finalLoss: Float
    public let steps: Int
}

// Резидентная память процесса, МБ (phys_footprint — то, по чему считает jetsam).
func residentMemoryMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                       / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / (1024 * 1024) : 0
}

public enum LoRAFinetune {

    /// Обучить навешенные LoRA-адаптеры. `nextBatch` отдаёт следующий (x,y);
    /// циклирование маленького датасета — на стороне вызывающего.
    @discardableResult
    public static func run(
        _ bb: X070Backbone,
        nextBatch: () -> LoRABatch,
        config cfg: LoRAConfig,
        isCancelled: () -> Bool = { false },
        onStep: (_ step: Int, _ loss: Float, _ gradNorm: Float, _ peakMB: Double) -> Void = { _,_,_,_ in }
    ) -> LoRATrainResult {

        if cfg.cacheLimitGB > 0 {
            MLX.GPU.set(cacheLimit: Int(cfg.cacheLimitGB * 1e9))
        }
        bb.useBlockCheckpoint = cfg.useBlockCheckpoint
        precondition(!bb.loraA.isEmpty, "нет адаптеров — сначала LoRA.add(...)")

        // Упорядоченные мастер-параметры (fp32): по таргетам [A, B].
        let targets = bb.loraA.keys.sorted()
        var params: [MLXArray] = []
        var slot: [(target: String, isA: Bool)] = []
        for t in targets {
            params.append(bb.loraA[t]!.asType(.float32)); slot.append((t, true))
            params.append(bb.loraB[t]!.asType(.float32)); slot.append((t, false))
        }
        eval(params)

        // Инжект ps (fp32) → bb.lora* (bf16, каст внутри ⇒ grad течёт в fp32-мастер).
        func inject(_ ps: [MLXArray]) {
            for (i, s) in slot.enumerated() {
                if s.isA { bb.loraA[s.target] = ps[i].asType(.bfloat16) }
                else      { bb.loraB[s.target] = ps[i].asType(.bfloat16) }
            }
        }

        func lossOf(_ ps: [MLXArray], _ x: MLXArray, _ y: MLXArray) -> [MLXArray] {
            inject(ps)
            let logits = bb(x)                                  // [B,T,V]
            let B = logits.shape[0], T = logits.shape[1], V = logits.shape[2]
            let ce = crossEntropy(logits: logits.reshaped([B * T, V]),
                                  targets: y.reshaped([B * T]).asType(.int32),
                                  reduction: .mean)
            return [ce.asType(.float32)]
        }
        let vg = valueAndGrad({ (ps: [MLXArray]) in lossOf(ps, _xCur, _yCur) },
                              argumentNumbers: Array(params.indices))

        // ── ручной AdamW (decoupled WD) ──
        var m = params.map { MLXArray.zeros($0.shape, dtype: .float32) }
        var v = params.map { MLXArray.zeros($0.shape, dtype: .float32) }
        var t = 0
        var lastLoss: Float = .nan

        func lrAt(_ step: Int) -> Float {
            (cfg.warmupSteps > 0 && step < cfg.warmupSteps)
                ? cfg.lr * Float(step + 1) / Float(cfg.warmupSteps) : cfg.lr
        }

        // global-norm grad clip; возвращает (clippedGrads, norm).
        func clip(_ grads: [MLXArray]) -> ([MLXArray], MLXArray) {
            var sq = MLXArray(Float(0))
            for g in grads { sq = sq + (g * g).sum() }
            let norm = sqrt(sq)
            let s = minimum(MLXArray(Float(1)), MLXArray(cfg.gradClip) / (norm + 1e-6))
            return (grads.map { $0 * s }, norm)
        }

        var step = 0
        var firstChecked = false
        while step < cfg.maxSteps {
            if isCancelled() { break }
            let lr = lrAt(step)

            // grad-accumulation
            var grads: [MLXArray]
            var lossVal: MLXArray
            do {
                let batch = nextBatch()
                if !firstChecked {
                    precondition(batch.x.shape[1] % WKV7_CHUNK == 0,
                        "T (\(batch.x.shape[1])) должна делиться на CHUNK \(WKV7_CHUNK)")
                    firstChecked = true
                }
                _xCur = batch.x; _yCur = batch.y
                let (vals, gs) = vg(params)
                eval(vals + gs)
                lossVal = vals[0]; grads = gs
            }
            for _ in 1 ..< cfg.gradAccum {
                let batch = nextBatch()
                _xCur = batch.x; _yCur = batch.y
                let (vals, gs) = vg(params)
                eval(vals + gs)
                lossVal = lossVal + vals[0]
                for i in grads.indices { grads[i] = grads[i] + gs[i] }
                eval(grads)
            }
            if cfg.gradAccum > 1 {
                let inv = 1.0 / Float(cfg.gradAccum)
                grads = grads.map { $0 * inv }
                lossVal = lossVal * inv
            }

            let (cg, norm) = clip(grads)

            t += 1
            let c1 = 1 - Float(pow(Double(cfg.beta1), Double(t)))
            let c2 = 1 - Float(pow(Double(cfg.beta2), Double(t)))
            for i in params.indices {
                m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * cg[i]
                v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * (cg[i] * cg[i])
                let mhat = m[i] / c1, vhat = v[i] / c2
                var upd = mhat / (sqrt(vhat) + cfg.adamEps)
                if cfg.weightDecay > 0 { upd = upd + cfg.weightDecay * params[i] }   // decoupled
                params[i] = params[i] - lr * upd
            }
            eval(params + m + v)

            lastLoss = lossVal.item(Float.self)
            step += 1
            if cfg.logEvery > 0 && (step % cfg.logEvery == 0 || step == cfg.maxSteps) {
                onStep(step, lastLoss, norm.item(Float.self), residentMemoryMB())
            }
        }

        // финальный инжект обученных весов (bf16) в backbone для инференса
        inject(params); eval(Array(bb.loraA.values) + Array(bb.loraB.values))
        return LoRATrainResult(finalLoss: lastLoss, steps: step)
    }
}

// Текущий батч для замыкания valueAndGrad (без захвата inout).
private var _xCur = MLXArray.zeros([1, 1])
private var _yCur = MLXArray.zeros([1, 1])
