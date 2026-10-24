//
//  RwkvqWeightStreamingTests.swift
//  Подкачка весов уже собранной модели (`enableWeightStreaming(cache:)`, 24.10):
//  модель из .rwkvq (0.1B REDUCTION; без файла тест пропускается) и плотная.
//  Числа всех путей — те же до бита, что у модели в памяти; после выгрузки в
//  памяти остаются малые нормы; во время шага по блокам — веса одного-двух
//  блоков; отказы.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

final class RwkvqWeightStreamingTests: XCTestCase {

    static let tag = ProcessInfo.processInfo.processIdentifier
    static func cache(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("swr_spill_\(tag)_\(name).safetensors")
    }
    override class func tearDown() {
        for n in ["same", "mem", "twice", "native", "dense"] { try? FileManager.default.removeItem(at: cache(n)) }
        super.tearDown()
    }

    func load() throws -> X070Backbone {
        try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("reduction"))
    }
    func streaming(_ name: String) throws -> X070Backbone {
        let bb = try load()
        try bb.enableWeightStreaming(cache: Self.cache(name))
        return bb
    }
    func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype && RwkvqLoadTests.bad(a, b) == 0
    }
    func batch(B: Int = 1, T: Int = 32, seed: UInt64 = 5) -> LoRABatch {
        let ids = TinyBackbone.ids(B, T + 1, vocab: 65536, seed: seed)
        return (x: ids[0..., 0 ..< T], y: ids[0..., 1 ... T])
    }
    /// Байты блока `l` в памяти модели: плотные веса блока и буферы базы (сжатые).
    func blockBytes(_ bb: X070Backbone, _ l: Int = 1) -> Int {
        let p = "blocks.\(l)."
        let dense = bb.w.filter { $0.key.hasPrefix(p) }.values.reduce(0) { $0 + $1.nbytes }
        let base = (bb.rwkvqSidecar?.spillBuffers ?? [:]).filter {
            $0.key.hasPrefix(p) || $0.key.hasPrefix("fused::" + p)
        }.values.reduce(0) { $0 + $1.nbytes }
        return dense + base
    }

    func testSameNumbersAsResident() throws {
        let r = try load(), s = try streaming("same")
        XCTAssertTrue(s.isWeightStreaming); XCTAssertFalse(r.isWeightStreaming)
        // всё, что видит приложение: параллельный проход, префилл, шаги декода, состояние
        let a = RwkvqLoadTests.outputs(r), b = RwkvqLoadTests.outputs(s)
        XCTAssertEqual(a.count, b.count)
        for (i, (x, y)) in zip(a, b).enumerated() { XCTAssertTrue(same(x, y), "выход \(i) разошёлся") }
        s.releaseAllStreamedWeights()
        // градиент состояния по блокам — дважды (второй раз всё перечитывается)
        let bt = batch()
        let ss = StateTrainableSet(s), rs = StateTrainableSet(r)
        for _ in 0 ..< 2 {
            let gs = LoRAFinetune.sequentialGradients(s, set: ss, parameters: ss.initialParameters(), batch: bt, lossChunk: 0)
            let gr = LoRAFinetune.sequentialGradients(r, set: rs, parameters: rs.initialParameters(), batch: bt, lossChunk: 0)
            eval([gs.loss, gr.loss] + gs.gradients + gr.gradients)
            XCTAssertTrue(same(gs.loss, gr.loss), "лосс разошёлся")
            for (x, y) in zip(gs.gradients, gr.gradients) { XCTAssertTrue(same(x, y), "градиент состояния разошёлся") }
        }
        // обучение состояния с ранней остановкой
        var c = StateTuneConfig()
        c.lr = 0.05; c.lrMin = 0.01; c.warmupSteps = 0; c.maxSteps = 4; c.logEvery = 0; c.cacheLimitGB = 0
        c.lossChunkTokens = 0; c.evalEvery = 2
        r.initialState = nil; s.initialState = nil
        let held = [batch(seed: 6), batch(B: 2, seed: 7)]
        let (rr, rt) = StateTuning.run(r, nextBatch: { bt }, config: c, heldOut: held)
        let (rs2, st) = StateTuning.run(s, nextBatch: { bt }, config: c, heldOut: held)
        XCTAssertEqual(rr.heldOut, rs2.heldOut)
        XCTAssertTrue(same(st.wkv, rt.wkv), "обученное состояние разошлось")
    }

    /// После выгрузки в памяти — малые нормы; во время шага по блокам — не больше пары
    /// блоков сверх того; после шага прочитанное отпущено.
    func testOnlyOneBlockResident() throws {
        MLX.Memory.cacheLimit = 0
        defer { MLX.Memory.cacheLimit = MLX.Memory.memoryLimit }
        MLX.Memory.clearCache()
        let a0 = GPU.activeMemory
        let r = try load()
        MLX.Memory.clearCache()
        let resident = GPU.activeMemory - a0
        let block = blockBytes(r)
        let a1 = GPU.activeMemory
        let s = try streaming("mem")
        MLX.Memory.clearCache()
        let kept = GPU.activeMemory - a1
        print("ПОДКАЧКА .rwkvq: модель в памяти \(resident / 1000) КБ, после выгрузки \(kept / 1000) КБ, блок \(block / 1000) КБ")
        XCTAssertLessThan(kept, resident / 10, "после выгрузки в памяти осталась модель")
        // в памяти — только нормы основания и выхода: ни одного буфера базы (масштабы sym малы,
        // но их у модели много — порог по ним, а не по модели)
        XCTAssertLessThan(kept, block / 50, "после выгрузки держатся буферы базы (\(kept) байт)")
        // в файле — модель один раз (срезы сведённых групп не дублируются)
        let fileBytes = try FileManager.default.attributesOfItem(atPath: Self.cache("mem").path)[.size] as! Int
        XCTAssertLessThan(fileBytes, resident + resident / 20, "файл выгрузки больше модели в памяти")
        XCTAssertGreaterThan(fileBytes, resident / 2)

        let b = batch()
        func measure(_ bb: X070Backbone) -> (peakAbove: Int, keptAfter: Int) {
            let set = StateTrainableSet(bb)
            let ps = set.initialParameters(); eval(ps)
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
        let mr = measure(r), ms = measure(s)
        print("ПОДКАЧКА .rwkvq: пик шага сверх начала — в памяти \(mr.peakAbove / 1000) КБ, подкачка \(ms.peakAbove / 1000) КБ; "
              + "осталось после шага \(mr.keptAfter / 1000) / \(ms.keptAfter / 1000) КБ")
        XCTAssertLessThan(ms.keptAfter, block / 2, "после шага остались прочитанные веса")
        MLX.Memory.clearCache()
        let held = GPU.activeMemory
        s.releaseAllStreamedWeights()
        MLX.Memory.clearCache()
        XCTAssertLessThan(held - GPU.activeMemory, block / 2, "после шага держались прочитанные веса")
        XCTAssertTrue(s.denseTargetCast.isEmpty, "держится приведённая копия плотной цели")
        // пик: как у модели в памяти плюс читаемые блок-два и таблица эмбеддингов основания
        let emb = r.rwkvqEmbTable.map { $0.packedBytes } ?? 0
        XCTAssertLessThan(ms.peakAbove, mr.peakAbove + 3 * block + emb, "пик подкачки выше блока-двух сверх модели в памяти")
        // пик всего: модель в памяти и шаг над ней — против выгруженной модели и шага с чтением
        XCTAssertLessThan(kept + ms.peakAbove, resident + mr.peakAbove, "с подкачкой пик шага не ниже, чем с моделью в памяти")
    }

    func testRefusals() throws {
        let s = try streaming("twice")
        XCTAssertThrowsError(try s.enableWeightStreaming(cache: Self.cache("twice"))) {
            guard case WeightStreamingError.alreadyStreaming = $0 else { return XCTFail("\($0)") }
        }
        let n = try X070Backbone.load(fromRwkvq: try RwkvqFileTests.sourcePath("compression"), useNativeKernel: true)
        XCTAssertFalse(n.rwkvqNative.isEmpty, "у COMPRESSION sb6 переложился бы в родной контейнер")
        XCTAssertThrowsError(try n.enableWeightStreaming(cache: Self.cache("native"))) {
            guard case WeightStreamingError.nativeKernel = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: Self.cache("native").path))
    }

    /// Плотная модель в памяти: выгрузка и чтение обратно — те же числа.
    func testDenseResidentModel() throws {
        let cfg = TinyBackbone.config(nLayer: 3, nEmbd: 128, vocab: 64)
        let r = X070Backbone(weights: TinyBackbone.weights(cfg), cfg: cfg, computeDType: .bfloat16)
        let s = X070Backbone(weights: TinyBackbone.weights(cfg), cfg: cfg, computeDType: .bfloat16)
        try s.enableWeightStreaming(cache: Self.cache("dense"))
        let ids = TinyBackbone.ids(2, 32, vocab: 64, seed: 3)
        XCTAssertTrue(same(r(ids), s(ids)))
        let bt: LoRABatch = (x: TinyBackbone.ids(1, 32, vocab: 64, seed: 4), y: TinyBackbone.ids(1, 32, vocab: 64, seed: 5))
        let rs = StateTrainableSet(r), ss = StateTrainableSet(s)
        let gr = LoRAFinetune.sequentialGradients(r, set: rs, parameters: rs.initialParameters(), batch: bt, lossChunk: 0)
        let gs = LoRAFinetune.sequentialGradients(s, set: ss, parameters: ss.initialParameters(), batch: bt, lossChunk: 0)
        eval([gr.loss, gs.loss] + gr.gradients + gs.gradients)
        XCTAssertTrue(same(gr.loss, gs.loss))
        for (x, y) in zip(gr.gradients, gs.gradients) { XCTAssertTrue(same(x, y)) }
    }
}
