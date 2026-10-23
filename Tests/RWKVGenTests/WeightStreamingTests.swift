//
//  WeightStreamingTests.swift
//  Подкачка весов с диска (`X070Backbone.loadStreaming`, 23.10): числа всех путей —
//  те же до бита, что у бэкбона из того же файла в памяти; в обратном проходе по
//  блокам прочитанные веса блока отпускаются после блока, таблица эмбеддингов —
//  после основания, голова — после лосса.
//
//  Файл — крошечная модель в fp32, бэкбон — в bf16: приведение при перечитывании
//  обязано совпасть с приведением при сборке. Память — по `GPU.activeMemory` /
//  `GPU.peakMemory` (живые массивы MLX), на модели, где веса блока заметны.
//
import XCTest
import MLX
@testable import RWKVGen

final class WeightStreamingTests: XCTestCase {

    static var file: URL!
    static var cfg: X070Config!

    override class func setUp() {
        super.setUp()
        cfg = TinyBackbone.config(nLayer: 6, nEmbd: 512, vocab: 4096)   // голова ~ блоку: её отпускание видно
        let w = TinyBackbone.weights(cfg, seed: 42).mapValues { $0.asType(.float32) }
        file = FileManager.default.temporaryDirectory.appendingPathComponent("swr_stream_\(ProcessInfo.processInfo.processIdentifier).safetensors")
        try! save(arrays: w, url: file)
    }

    override class func tearDown() {
        try? FileManager.default.removeItem(at: file)
        super.tearDown()
    }

    func resident() throws -> X070Backbone {
        X070Backbone(weights: try loadArrays(url: Self.file), cfg: Self.cfg, computeDType: .bfloat16)
    }

    func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && allClose(a.asType(.float32), b.asType(.float32), rtol: 0, atol: 0).item(Bool.self)
    }

    func batch(B: Int = 2, T: Int = 32, seed: UInt64 = 5) -> LoRABatch {
        let ids = TinyBackbone.ids(B, T + 1, vocab: Self.cfg.vocab, seed: seed)
        return (x: ids[0..., 0 ..< T], y: ids[0..., 1 ... T])
    }

    /// Байты весов одного блока в типе вычислений.
    func blockBytes(_ bb: X070Backbone) -> Int {
        bb.weightKeys.filter { $0.hasPrefix("blocks.1.") }.reduce(0) { $0 + bb.weight($1).nbytes }
    }

    func testSameNumbersAsResident() throws {
        let s = try X070Backbone.loadStreaming(safetensors: Self.file, computeDType: .bfloat16)
        let r = try resident()
        XCTAssertTrue(s.isWeightStreaming); XCTAssertFalse(r.isWeightStreaming)
        XCTAssertEqual(s.cfg.nLayer, Self.cfg.nLayer); XCTAssertEqual(s.cfg.nEmbd, Self.cfg.nEmbd)
        XCTAssertEqual(s.cfg.vocab, Self.cfg.vocab)
        let b = batch()
        XCTAssertTrue(same(s(b.x), r(b.x)), "логиты разошлись")
        s.releaseAllStreamedWeights()
        // градиент состояния по блокам — дважды подряд (второй раз веса перечитываются)
        for (bb, tag) in [(s, "подкачка"), (r, "в памяти")] { _ = tag; bb.initialState = nil }
        let ss = StateTrainableSet(s), rs = StateTrainableSet(r)
        for _ in 0 ..< 2 {
            let gs = LoRAFinetune.sequentialGradients(s, set: ss, parameters: ss.initialParameters(), batch: b, lossChunk: 0)
            let gr = LoRAFinetune.sequentialGradients(r, set: rs, parameters: rs.initialParameters(), batch: b, lossChunk: 0)
            eval([gs.loss, gr.loss] + gs.gradients + gr.gradients)
            XCTAssertTrue(same(gs.loss, gr.loss), "лосс разошёлся")
            for (a, c) in zip(gs.gradients, gr.gradients) { XCTAssertTrue(same(a, c), "градиент состояния разошёлся") }
        }
        // обучение: то же состояние
        var c = StateTuneConfig()
        c.lr = 0.05; c.lrMin = 0.01; c.warmupSteps = 0; c.maxSteps = 3; c.logEvery = 0; c.cacheLimitGB = 0; c.lossChunkTokens = 0
        let (_, st) = StateTuning.run(s, nextBatch: { b }, config: c)
        let (_, rt) = StateTuning.run(r, nextBatch: { b }, config: c)
        XCTAssertTrue(same(st.wkv, rt.wkv), "обученное состояние разошлось")
    }

    /// Во время шага по блокам в памяти живут веса не больше двух блоков (пик), после
    /// шага — ни одного; у бэкбона в памяти — все.
    func testOnlyOneBlockResidentDuringStep() throws {
        let b = batch(B: 1, T: 32)
        MLX.Memory.cacheLimit = 0
        defer { MLX.Memory.cacheLimit = MLX.Memory.memoryLimit }
        func measure(_ bb: X070Backbone) -> (peakAbove: Int, keptAfter: Int) {
            let set = StateTrainableSet(bb)
            let ps = set.initialParameters(); eval(ps)
            // разогрев: ядра собраны, временные буферы отпущены
            _ = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 0)
            MLX.Memory.clearCache()
            let before = GPU.activeMemory
            GPU.resetPeakMemory()
            let g = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 0)
            eval([g.loss] + g.gradients)
            let peak = GPU.peakMemory
            MLX.Memory.clearCache()
            return (peak - before, GPU.activeMemory - before)
        }
        let r = try resident()
        for k in r.weightKeys { eval(r.weight(k)) }
        let block = blockBytes(r)
        let all = r.weightKeys.reduce(0) { $0 + r.weight($1).nbytes }
        let mr = measure(r)
        let s = try X070Backbone.loadStreaming(safetensors: Self.file, computeDType: .bfloat16)
        let ms = measure(s)
        print("ПОДКАЧКА: блок \(block / 1000) КБ, все веса \(all / 1000) КБ; пик сверх начала: в памяти \(mr.peakAbove / 1000) КБ, "
              + "подкачка \(ms.peakAbove / 1000) КБ; осталось после шага: \(mr.keptAfter / 1000) / \(ms.keptAfter / 1000) КБ")
        let head = r.weight("head.weight").nbytes
        XCTAssertLessThan(ms.keptAfter, Swift.min(block, head) / 2, "после шага остались прочитанные веса (блок или голова)")
        // и ничего прочитанного не держится: отпустить всё разом — освобождать нечего
        MLX.Memory.clearCache()
        let held = GPU.activeMemory
        s.releaseAllStreamedWeights()
        MLX.Memory.clearCache()
        let freed = held - GPU.activeMemory
        XCTAssertLessThan(freed, Swift.min(block, head) / 2, "после шага держались прочитанные веса (\(freed) байт)")
        // пик подкачки: пик весов в памяти + читаемый блок дважды (в файле fp32 — вдвое больше
        // блока в bf16 — и после приведения) + запас; без отпускания в прямом проходе сверху
        // легли бы веса всех шести блоков
        XCTAssertLessThan(ms.peakAbove, mr.peakAbove + 3 * block + all / 8,
                          "пик подкачки выше, чем у весов в памяти плюс читаемый блок")
        XCTAssertGreaterThan(ms.peakAbove, 0)
    }

    /// Запись входов с шагом (`stride`): блоки, которые обратный проход считает заново
    /// (`Trace.restoreInput`), тоже отпускают веса; градиент — тот же, что в памяти.
    func testStridedTraceReleasesRecomputedBlocks() throws {
        let b = batch(B: 1, T: 32)
        func run(_ bb: X070Backbone) -> [MLXArray] {
            let set = StateTrainableSet(bb)
            let ps = set.initialParameters()
            set.inject(ps)
            let plan = SequentialBody.Plan(names: set.parameterNames, nLayer: bb.cfg.nLayer)
            let trace = SequentialBody.forward(bb, ids: b.x, plan: plan, stride: 3)
            var grads = [MLXArray?](repeating: nil, count: ps.count)
            SequentialBody.backward(bb, trace: trace, dTop: MLXArray.ones(trace.top.shape, dtype: trace.top.dtype),
                                    parameters: ps, plan: plan, inject: { set.inject($0) }, grads: &grads)
            let g = grads.map { $0! }
            eval(g)
            return g
        }
        let r = try resident()
        let s = try X070Backbone.loadStreaming(safetensors: Self.file, computeDType: .bfloat16)
        let gr = run(r)
        MLX.Memory.cacheLimit = 0
        defer { MLX.Memory.cacheLimit = MLX.Memory.memoryLimit }
        MLX.Memory.clearCache()
        let before = GPU.activeMemory
        let gs = run(s)
        MLX.Memory.clearCache()
        let kept = GPU.activeMemory - before - gs.reduce(0) { $0 + $1.nbytes }
        for (a, c) in zip(gs, gr) { XCTAssertTrue(same(a, c), "градиент с записью через блок разошёлся") }

        // пересчёт отрезка сам по себе (Trace.restoreInput): посчитанные заново блоки отпускают веса
        let set = StateTrainableSet(s)
        set.inject(set.initialParameters())
        let plan = SequentialBody.Plan(names: set.parameterNames, nLayer: s.cfg.nLayer)
        let trace = SequentialBody.forward(s, ids: b.x, plan: plan, stride: 3)
        eval(trace.top)
        MLX.Memory.clearCache()
        let a0 = GPU.activeMemory
        trace.restoreInput(5, s)                 // блоки 3 и 4 считаются заново
        MLX.Memory.clearCache()
        let grown = GPU.activeMemory - a0
        XCTAssertLessThan(grown, blockBytes(r) / 2, "пересчитанные блоки не отпустили веса (\(grown) байт)")
        XCTAssertLessThan(kept, blockBytes(r) / 2, "после прохода с пересчётом остались прочитанные веса (\(kept) байт)")
    }

    func testRefusesNonModelFile() throws {
        let junk = FileManager.default.temporaryDirectory.appendingPathComponent("swr_junk_\(ProcessInfo.processInfo.processIdentifier).safetensors")
        try save(arrays: ["x": MLXArray.zeros([2])], url: junk)
        defer { try? FileManager.default.removeItem(at: junk) }
        XCTAssertThrowsError(try X070Backbone.loadStreaming(safetensors: junk))
    }
}
