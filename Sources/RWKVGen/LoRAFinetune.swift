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
    /// Голова и лосс по кускам из стольких позиций (0 — целиком); см.
    /// `LoRAFinetune.languageModelLoss`. Память длинной последовательности.
    public var lossChunkTokens: Int = 0
    /// Обратный проход по блокам с вычислением после каждого (см.
    /// SequentialBackward.swift): память одного блока вместо нескольких.
    /// Градиент тот же; `useBlockCheckpoint` при этом не действует.
    public var sequentialBackward: Bool = false
    /// Спад LR после warmup. Исторический дефолт — .constant (плоско).
    public var schedule: LRSchedule
    public var lrMin: Float

    public init(lr: Float = 1e-4, gradClip: Float = 1.0, weightDecay: Float = 0.0,
                beta1: Float = 0.9, beta2: Float = 0.95, adamEps: Float = 1e-8,
                maxSteps: Int = 1000, gradAccum: Int = 1, warmupSteps: Int = 0,
                cacheLimitGB: Double = 1.5, logEvery: Int = 10,
                useBlockCheckpoint: Bool = false,
                schedule: LRSchedule = .constant, lrMin: Float = 0) {
        self.lr = lr; self.gradClip = gradClip; self.weightDecay = weightDecay
        self.beta1 = beta1; self.beta2 = beta2; self.adamEps = adamEps
        self.maxSteps = maxSteps; self.gradAccum = gradAccum; self.warmupSteps = warmupSteps
        self.cacheLimitGB = cacheLimitGB; self.logEvery = logEvery
        self.useBlockCheckpoint = useBlockCheckpoint
        self.schedule = schedule; self.lrMin = lrMin
    }

    /// Проекция в общий TrainingConfig.
    var training: TrainingConfig {
        TrainingConfig(lr: lr, lrMin: lrMin, schedule: schedule,
                       warmupSteps: warmupSteps, gradClip: gradClip,
                       weightDecay: weightDecay, beta1: beta1, beta2: beta2,
                       adamEps: adamEps, maxSteps: maxSteps, gradAccum: gradAccum,
                       cacheLimitGB: cacheLimitGB, logEvery: logEvery)
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

    /// LM-лосс: cross-entropy по next-token на всём [B,T,vocab].
    ///
    /// `chunk` > 0 — голова и кросс-энтропия по кускам из `chunk` позиций, каждый
    /// под чекпоинтом (`lossChunkTokens` в `LoRAConfig`). Зачем: логиты
    /// [B·T, 65536] в fp32 — 268 МБ на 1024 токена, а лосс держит их в трёх-
    /// четырёх копиях (логиты, экспонента, softmax для градиента, котангент):
    /// на 1.5B при T = 2048 это ~1.6 ГБ пика из 6.3. По кускам одновременно
    /// живёт один кусок; цена — голова считается дважды (второй раз в обратном
    /// проходе). Градиент тот же с точностью до порядка сложения.
    /// Действует, только когда голова — константа (нет ни адаптера, ни
    /// обучаемой подмены): чекпоинт дифференцирует по явным входам, и до
    /// обучаемой головы (претрейн) градиент через него не дошёл бы. Короткая
    /// последовательность (T <= chunk) идёт прежним путём побитно.
    static func languageModelLoss(_ bb: X070Backbone, _ batch: LoRABatch, chunk: Int = 0) -> MLXArray {
        let T0 = batch.x.shape[1]
        if chunk > 0, T0 > chunk, bb.headIsConstant {
            let h = bb.body(batch.x)                        // [B,T,D]
            let B = h.shape[0], V = bb.cfg.vocab
            var total = MLXArray(Float(0))
            var from = 0
            while from < T0 {
                let to = Swift.min(from + chunk, T0)
                let targets = batch.y[0..., from ..< to].reshaped([-1]).asType(.int32)
                let piece = checkpointed { (ins: [MLXArray]) -> [MLXArray] in
                    let lg = bb.projectForBlock(ins[0], "head.weight", lora: nil)
                    return [crossEntropy(logits: lg.reshaped([-1, V]), targets: targets,
                                         reduction: .sum).asType(.float32)]
                }
                total = total + piece([h[0..., from ..< to]])[0]
                from = to
            }
            return total / Float(B * T0)
        }
        let logits = bb(batch.x)                            // [B,T,V]
        let B = logits.shape[0], T = logits.shape[1], V = logits.shape[2]
        return crossEntropy(logits: logits.reshaped([B * T, V]),
                            targets: batch.y.reshaped([B * T]).asType(.int32),
                            reduction: .mean)
    }

    /// Обучить навешенные LoRA-адаптеры. `nextBatch` отдаёт следующий (x,y);
    /// циклирование маленького датасета — на стороне вызывающего.
    ///
    /// Тонкая обёртка над общим Trainer: здесь остаётся только специфика
    /// LoRA — какое множество обучается (LoRATrainableSet), какой лосс (LM CE)
    /// и проверка кратности T размеру чанка. Оптимизатор, расписание,
    /// накопление и клип — общие.
    @discardableResult
    public static func run(
        _ bb: X070Backbone,
        nextBatch: () -> LoRABatch,
        config cfg: LoRAConfig,
        isCancelled: () -> Bool = { false },
        onStep: (_ step: Int, _ loss: Float, _ gradNorm: Float, _ peakMB: Double) -> Void = { _,_,_,_ in }
    ) -> LoRATrainResult {

        bb.useBlockCheckpoint = cfg.useBlockCheckpoint && !cfg.sequentialBackward
        precondition(!bb.loraA.isEmpty, "нет адаптеров — сначала LoRA.add(...)")

        // withoutActuallyEscaping, а не @escaping в сигнатуре: публичный API
        // обязан остаться прежним (в этом весь смысл рефакторинга), а время
        // жизни тренера заведомо ограничено этим вызовом.
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
            let set = LoRATrainableSet(bb)
            let provider: GradientProvider<LoRABatch>? = cfg.sequentialBackward
                ? { ps, batch in
                    sequentialGradients(bb, set: set, parameters: ps, batch: batch,
                                        lossChunk: cfg.lossChunkTokens)
                }
                : nil
            let trainer = Trainer<LoRABatch>(
                trainable: set,
                objective: { languageModelLoss(bb, $0, chunk: cfg.lossChunkTokens) },
                nextBatch: source,
                config: cfg.training,
                gradient: provider)
            return trainer.run(isCancelled: isCancelled) { s in
                onStep(s.step, s.loss, s.gradNorm, s.peakMemoryMB)
            }
        }
        return LoRATrainResult(finalLoss: res.finalLoss, steps: res.steps)
    }
}
