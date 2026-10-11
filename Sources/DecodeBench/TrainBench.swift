import Foundation
import MLX
import MLXRandom
import RWKVGen
import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  --train-bench: шаг QLoRA над базой из .rwkvq — время и память по длинам.
//
//  Протокол общий с эталоном обучения (rwkv-metal,
//  tests/bench_qlora_rank.py): батч 1 × T случайных токенов, адаптеры ранга
//  16 (alpha 32) на проекциях tmix всех слоёв, AdamW lr 1e-4, потолок кэша
//  MLX 1.5 ГБ; шаг прогрева, сброс пиков, затем замеренные шаги.
//
//  Память — ДВУМЯ приборами, и они отвечают на разные вопросы:
//    • пик MLX (`GPU.peakMemory`) — сколько одновременно держал аллокатор
//      MLX; сравним с `mx.get_peak_memory()` эталона;
//    • пик footprint — `phys_footprint` процесса (то, что показывает
//      Мониторинг системы и по чему считает jetsam), опросом каждые 2 мс.
//      В нём и кэш MLX, и страницы, которые Metal ещё не вернул системе.
//  Пик footprint за ВЕСЬ процесс даёт системная команда:
//      /usr/bin/time -l decode-bench --train-bench …  («peak memory footprint»)
// ───────────────────────────────────────────────────────────────────────

final class FootprintSampler {
    private var peak = 0.0
    private let lock = NSLock()
    /// Предохранитель: footprint выше порога — процесс завершается сам, не
    /// дожидаясь, пока машина уйдёт в своп (замер под свопом всё равно негоден).
    var guardGB = 10.0
    init(intervalMs: Int = 2) {
        let t = Thread { [self] in   // сэмплер живёт до конца процесса
            while true {
                let f = processFootprintGB()
                if f > self.guardGB {
                    print(String(format: "ПРЕДОХРАНИТЕЛЬ: footprint %.1f ГБ выше порога %.1f — выход", f, self.guardGB))
                    fflush(stdout); exit(3)
                }
                self.lock.lock(); if f > self.peak { self.peak = f }; self.lock.unlock()
                usleep(UInt32(intervalMs * 1000))
            }
        }
        t.stackSize = 1 << 16
        t.start()
    }
    func reset() { lock.lock(); peak = processFootprintGB(); lock.unlock() }
    func take() -> Double { lock.lock(); defer { lock.unlock() }; return peak }
}

func swapUsedMB() -> Double {
    var u = xsw_usage()
    var size = MemoryLayout<xsw_usage>.size
    sysctlbyname("vm.swapusage", &u, &size, nil, 0)
    return Double(u.xsu_used) / 1e6
}

struct TrainBenchOptions {
    var lengths = [512, 1024, 2048]
    var rank = 16
    var checkpoint = "on"          // on | off | both
    var steps = 3
    var batch = 1
    var cacheLimitGB = 1.5
    var layersFrom = 0             // адаптеры на слоях layersFrom ..< L
    var lossChunk = 0              // голова и лосс по кускам (LoRAConfig.lossChunkTokens)
    var forward = true             // мерить ли отдельно прямой проход
    var sequential = false         // обратный проход по блокам (LoRAConfig.sequentialBackward)
    var guardGB = 10.0             // предохранитель по footprint
    var profile = false            // разложение шага по статьям (RWKVSegmentProfile)
}

func trainBench(_ path: String, _ o: TrainBenchOptions) throws {
    let sampler = FootprintSampler()
    sampler.guardGB = o.guardGB
    setvbuf(stdout, nil, _IOLBF, 0)
    let f0 = processFootprintGB()
    var t = Date()
    let bb = try X070Backbone.load(fromRwkvq: expand(path))
    let L = bb.cfg.nLayer
    let info = LoRA.add(to: bb, spec: LoRASpec(rank: o.rank, alpha: Float(2 * o.rank),
                                               layers: o.layersFrom ..< L))
    print(String(format: "модель L=%d D=%d; загрузка %.1f с; адаптеров %d (%.2f М параметров); "
                 + "MLX активная %.2f ГБ, footprint +%.2f ГБ",
                 L, bb.cfg.nEmbd, Date().timeIntervalSince(t), info.numAdapters,
                 Double(info.trainableParams) / 1e6, Double(GPU.activeMemory) / 1e9,
                 processFootprintGB() - f0))
    print("ранг \(o.rank), слои \(o.layersFrom)..<\(L), батч \(o.batch), шагов \(o.steps), "
          + "потолок кэша MLX \(o.cacheLimitGB) ГБ, лосс кусками по \(o.lossChunk)\(o.sequential ? ", обратный проход по блокам" : "")")

    MLXRandom.seed(5)
    let arms: [Bool] = o.checkpoint == "both" ? [true, false] : [o.checkpoint != "off"]
    for T in o.lengths {
        precondition(T % WKV7_CHUNK == 0, "T обязана делиться на \(WKV7_CHUNK)")
        let x = MLXRandom.randInt(1 ..< 60000, [o.batch, T]).asType(.int32)
        let y = MLXRandom.randInt(1 ..< 60000, [o.batch, T]).asType(.int32)
        eval(x, y)
        for ckpt in arms {
            // ── шаг целиком, как его делает библиотека ──
            var stamps: [Date] = []
            var loss: Float = .nan
            var sw0 = 0.0
            var cfg = LoRAConfig(maxSteps: o.steps + 1, cacheLimitGB: o.cacheLimitGB, logEvery: 1,
                                 useBlockCheckpoint: ckpt)
            cfg.lossChunkTokens = o.lossChunk
            cfg.sequentialBackward = o.sequential
            LoRAFinetune.run(bb, nextBatch: { (x: x, y: y) }, config: cfg) { step, l, _, _ in
                if step == 1 {
                    GPU.clearCache(); GPU.resetPeakMemory(); sampler.reset(); sw0 = swapUsedMB()
                }
                stamps.append(Date()); loss = l
            }
            let per = zip(stamps.dropFirst(), stamps).map { $0.timeIntervalSince($1) * 1e3 }
            let peakMLX = Double(GPU.peakMemory) / 1e9, peakFP = sampler.take()
            let active = Double(GPU.activeMemory) / 1e9
            let ms = median(per)

            // ── только прямой проход тем же путём (лосс без градиента) ──
            bb.useBlockCheckpoint = false
            func forward(_ body: () -> MLXArray) -> Double {
                var ts: [Double] = []
                for i in 0 ..< 3 {
                    t = Date(); eval(body())
                    if i > 0 { ts.append(Date().timeIntervalSince(t) * 1e3) }
                }
                return median(ts)
            }
            GPU.clearCache(); GPU.resetPeakMemory()
            // на длинной последовательности прямой проход с логитами целиком
            // сам пикует выше шага — тогда не меряем (--tb-no-forward)
            let fwdTrain = o.forward ? forward { bb(x).sum() } : 0
            let fwdPeak = Double(GPU.peakMemory) / 1e9
            let fwdFrozen = o.forward ? forward { bb.withoutGradientPath { bb(x).sum() } } : 0
            GPU.clearCache()

            if o.profile {
                // вычитающая абляция: шаг с обнулённым обратным проходом статьи
                func ablated(_ wkv: Bool, _ proj: Bool) -> Double {
                    RWKVAblation.skipWKVBackward = wkv
                    RWKVAblation.skipProjectionBackward = proj
                    defer { RWKVAblation.skipWKVBackward = false; RWKVAblation.skipProjectionBackward = false }
                    var st: [Date] = []
                    let c = LoRAConfig(lr: 0, maxSteps: 4, cacheLimitGB: o.cacheLimitGB, logEvery: 1,
                                       useBlockCheckpoint: ckpt)
                    LoRAFinetune.run(bb, nextBatch: { (x: x, y: y) }, config: c) { _, _, _, _ in st.append(Date()) }
                    return median(zip(st.dropFirst(), st).map { $0.timeIntervalSince($1) * 1e3 })
                }
                // ядра обучаемого WKV: три плеча чередованием в одном процессе
                func arm(_ v2: Bool, _ keep: Bool) -> Double {
                    wkv7BackwardV2 = v2; wkv7KeepForwardOutputs = keep
                    defer { wkv7BackwardV2 = true; wkv7KeepForwardOutputs = true }
                    return ablated(false, false)
                }
                var arms: [String: [Double]] = [:]
                for _ in 0 ..< 2 {
                    arms["прежнее ядро, пересчёт", default: []].append(arm(false, false))
                    arms["новое ядро, пересчёт", default: []].append(arm(true, false))
                    arms["новое ядро, выходы держатся", default: []].append(arm(true, true))
                }
                print("  ядра WKV (T=\(T), чекпоинт \(ckpt ? "да" : "нет")): " + arms.sorted { $0.key > $1.key }.map {
                    "\($0.key) " + $0.value.map { String(format: "%.0f", $0) }.joined(separator: " / ") + " мс"
                }.joined(separator: " | "))
                let full = ablated(false, false), noW = ablated(true, false), noP = ablated(false, true)
                let none = ablated(true, true), full2 = ablated(false, false)
                print(String(format: "  абляция обратного прохода (T=%d, чекпоинт %@): шаг %.0f / %.0f мс | без обратного WKV %.0f "
                             + "(цена %.0f) | без обратного проекций %.0f (цена %.0f) | без обоих %.0f",
                             T, ckpt ? "да" : "нет", full, full2, noW, full - noW, noP, full - noP, none))
            }
            print(String(format: "T=%d чекпоинт %@: шаг %.0f мс (%.0f ток/с; шаги %@) | "
                         + "прямой %.0f мс, прямой замороженным путём %.0f | пик MLX %.2f ГБ "
                         + "(прямой %.2f), активная %.2f | пик footprint %.2f ГБ (+%.2f к старту) | "
                         + "своп %+.0f МБ | лосс %.3f",
                         T, ckpt ? "да" : "нет", ms, Double(o.batch * T) / ms * 1e3,
                         per.map { String(format: "%.0f", $0) }.joined(separator: "/"),
                         fwdTrain, fwdFrozen, peakMLX, fwdPeak, active, peakFP, peakFP - f0,
                         swapUsedMB() - sw0, loss))
            fflush(stdout)
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  --tb-micro T: из чего складывается шаг — ядро WKV (прямое, обучаемое с
//  обратным) и матмулы проекций по типам, на формах 1.5B (D=2048, H=32),
//  по 24 вызова под одним eval (как в шаге: 24 слоя).
// ───────────────────────────────────────────────────────────────────────
func trainMicro(_ T: Int, D: Int = 2048, layers: Int = 24) {
    let S = WKV7_HEAD_SIZE, H = D / S
    MLXRandom.seed(3)
    func rnd(_ s: [Int], _ scale: Float = 0.1) -> MLXArray { MLXRandom.normal(s) * scale }
    let r = rnd([1, T, H, S]), k = rnd([1, T, H, S]), v = rnd([1, T, H, S])
    let a = rnd([1, T, H, S]), b = rnd([1, T, H, S])
    let w = exp(-0.606531 * sigmoid(rnd([1, T, H, S], 1)))
    eval(r, k, v, a, b, w)
    func time(_ n: Int = 5, _ f: () -> [MLXArray]) -> Double {
        eval(f()); var ts: [Double] = []
        for _ in 0 ..< n { let t = Date(); eval(f()); ts.append(Date().timeIntervalSince(t) * 1e3) }
        return median(ts)
    }
    // слои независимы: вход сдвигается на номер слоя, чтобы MLX не свёл вызовы
    let seq = time { (0 ..< layers).map { wkv7Forward(r + Float($0) * 1e-6, w, k, v, a, b) } }
    let trF = time { (0 ..< layers).map { wkv7Train(r + Float($0) * 1e-6, w, k, v, a, b) } }
    let vg = valueAndGrad({ (ins: [MLXArray]) -> [MLXArray] in
        [wkv7Train(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5]).sum()]
    }, argumentNumbers: [0, 1, 2, 3, 4, 5])
    func both(_ v2: Bool, _ keep: Bool) -> Double {
        wkv7BackwardV2 = v2; wkv7KeepForwardOutputs = keep
        defer { wkv7BackwardV2 = true; wkv7KeepForwardOutputs = true }
        return time { (0 ..< layers).flatMap { vg([r + Float($0) * 1e-6, w, k, v, a, b]).1 } }
    }
    let old = both(false, false), v2 = both(true, false), keep = both(true, true), old2 = both(false, false)
    print(String(format: "T=%d WKV x%d: прямое замороженное %.0f мс | обучаемое прямое %.0f | прямое + обратное: "
                 + "прежнее ядро с пересчётом %.0f / %.0f, новое ядро с пересчётом %.0f, новое без пересчёта %.0f",
                 T, layers, seq, trF, old, old2, v2, keep))

    let n = 6 * layers
    for (name, dt) in [("fp16", DType.float16), ("bf16", .bfloat16), ("fp32", .float32)] {
        let x = rnd([T, D], 1).asType(dt), W = rnd([D, D], 0.02).asType(dt)
        eval(x, W)
        let m = time { (0 ..< n).map { matmul(x + MLXArray(Float($0) * 1e-3).asType(dt), W) } }
        print(String(format: "T=%d матмул [T,%d]x[%d,%d] %@ x%d: %.0f мс", T, D, D, D, name, n, m))
    }
    fflush(stdout)
}

