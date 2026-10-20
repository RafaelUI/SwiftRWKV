//
//  InitialStateBankTests.swift
//  Банк начальных состояний (`InitialStateBank.swift`, 20.10): своё
//  состояние у каждой строки батча. Эталоны — независимые: строка с
//  состоянием банка = та же строка одна с этим состоянием, подключённым как
//  одно; градиент банка = сумма градиентов строк по одному состоянию
//  (автоград); обратный проход по блокам = автоград на ТОМ ЖЕ составе батча.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

final class InitialStateBankTests: XCTestCase {

    typealias Part = StateTrainableSet.Part

    func tiny() -> (X070Backbone, X070Config) { TinyBackbone.make(nLayer: 3, seed: 42) }

    func randomBank(_ bb: X070Backbone, K: Int, seed: UInt64 = 5) -> RWKVBatchState {
        MLXRandom.seed(seed)
        let z = RWKVBatchState(cfg: bb.cfg, batch: K, dtype: .float32)
        return RWKVBatchState(wkv: MLXRandom.normal(z.wkv.shape) * 0.3,
                              tmixShift: MLXRandom.normal(z.tmixShift.shape) * 0.5,
                              cmixShift: MLXRandom.normal(z.cmixShift.shape) * 0.5).evaluated()
    }

    /// Строка k банка как одно состояние (батч 1).
    func row(_ s: RWKVBatchState, _ k: Int) -> RWKVBatchState {
        RWKVBatchState(wkv: s.wkv[0..., k ..< k + 1], tmixShift: s.tmixShift[0..., k ..< k + 1],
                       cmixShift: s.cmixShift[0..., k ..< k + 1]).evaluated()
    }

    func diff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        diff(ref, got) / (MLX.abs(ref.asType(.float32)).max().item(Float.self) + 1e-12)
    }

    static let rows: [Int32] = [2, 0, 2, 1]

    /// Строка с состоянием банка = та же строка одна с этим состоянием,
    /// подключённым как одно; и прямой проход, и состояние после него.
    func testRowsEqualSingleState() {
        let (bb, cfg) = tiny()
        let bank = randomBank(bb, K: 3)
        let ids = TinyBackbone.ids(4, 24, vocab: cfg.vocab, seed: 11)
        let none = bb.body(ids)
        bb.initialStateBank = bank
        XCTAssertEqual(bb.initialBankSize, 3)
        let rows = MLXArray(Self.rows)
        let (out, st) = bb.withInitialRows(rows) { (bb.body(ids), bb.states(ids)) }
        eval(out, st.wkv)
        XCTAssertNil(bb.initialRows, "строки не сняты после прохода")
        bb.initialStateBank = nil
        for (i, k) in Self.rows.enumerated() {
            bb.initialState = row(bank, Int(k))
            let one = bb.body(ids[i ..< i + 1])
            let oneSt = bb.states(ids[i ..< i + 1])
            XCTAssertLessThan(diff(out[i ..< i + 1], one), 1e-5, "строка \(i)")
            XCTAssertLessThan(diff(st.wkv[0..., i ..< i + 1], oneSt.wkv), 1e-5, "состояние строки \(i)")
        }
        XCTAssertGreaterThan(diff(out, none), 1e-2, "банк ни на что не влияет — проверка ничего не стоит")
    }

    /// Все строки на одном состоянии банка = это состояние, подключённое как
    /// одно (трансляция против выборки).
    func testSameRowEverywhereEqualsSingle() {
        let (bb, cfg) = tiny()
        let bank = randomBank(bb, K: 2)
        let ids = TinyBackbone.ids(3, 20, vocab: cfg.vocab, seed: 12)
        bb.initialStateBank = bank
        let viaBank = bb.withInitialRows(MLXArray([Int32](repeating: 1, count: 3))) { bb.body(ids) }
        eval(viaBank)
        bb.initialStateBank = nil
        bb.initialState = row(bank, 1)
        XCTAssertLessThan(diff(viaBank, bb.body(ids)), 1e-6)
    }

    /// `withoutInitialState` выключает и банк: побитно как без состояния.
    func testSuspendedBankIsNoState() {
        let (bb, cfg) = tiny()
        let ids = TinyBackbone.ids(4, 20, vocab: cfg.vocab, seed: 13)
        let none = bb.body(ids)
        bb.initialStateBank = randomBank(bb, K: 3)
        let off = bb.withInitialRows(MLXArray(Self.rows)) { bb.withoutInitialState { bb.body(ids) } }
        XCTAssertEqual(diff(off, none), 0)
    }

    // ── градиент ─────────────────────────────────────────────────────

    func batch(_ cfg: X070Config) -> LoRABatch {
        let ids = TinyBackbone.ids(4, 33, vocab: cfg.vocab, seed: 7)
        return (x: ids[0..., 0 ..< 32], y: ids[0..., 1 ... 32])
    }

    func autograd(_ bb: X070Backbone, _ set: TrainableSet, _ ps: [MLXArray], _ b: LoRABatch) -> (Float, [MLXArray]) {
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            set.inject(p)
            return [LoRAFinetune.languageModelLoss(bb, b).asType(.float32)]
        }, argumentNumbers: Array(ps.indices))
        let (v, g) = vg(ps)
        eval(v + g)
        set.inject(ps)
        return (v[0].item(Float.self), g)
    }

    /// Градиент банка (автоград и обратный проход по блокам) = сумма
    /// градиентов строк, каждая — одна со своим состоянием, деленная на B
    /// (лосс — среднее по батчу); строка банка без строк батча — ровно ноль.
    func testGradientIsSumOfRowGradients() {
        for parts in [[Part.wkv], Part.allCases] {
            let (bb, cfg) = tiny()
            let K = 4                                          // строка 3 в батче не встречается
            let bank = randomBank(bb, K: K)
            let b = batch(cfg)
            let rows = MLXArray(Self.rows)
            let set = StateBankTrainableSet(bb, train: parts, initial: bank)
            XCTAssertEqual(set.parameterNames.first, "blocks.0.state0bank.\(set.parts[0].rawValue)")
            XCTAssertEqual(bb.trainLayers, Set(0 ..< cfg.nLayer))
            let ps = set.initialParameters()
            let (lossA, gA) = bb.withInitialRows(rows) { autograd(bb, set, ps, b) }
            let seq = bb.withInitialRows(rows) {
                LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 0)
            }
            eval([seq.loss] + seq.gradients)

            // эталон: строки по одной, одно состояние, автоград
            bb.initialStateBank = nil
            var ref = ps.map { MLXArray.zeros($0.shape, dtype: .float32) }
            var lossRef: Float = 0
            for (i, k) in Self.rows.enumerated() {
                let single = StateTrainableSet(bb, train: parts, initial: row(bank, Int(k)))
                let sp = single.initialParameters()
                let (l, g) = autograd(bb, single, sp, (x: b.x[i ..< i + 1], y: b.y[i ..< i + 1]))
                lossRef += l / Float(Self.rows.count)
                for j in ps.indices {
                    // параметр одного: [H,S,S] / [1,1,D]; банка: [K,H,S,S] / [K,1,D]
                    let gk = g[j].reshaped([1] + Array(ps[j].shape.dropFirst()))
                    let hot = (MLXArray(0 ..< Int32(K)) .== MLXArray(k)).asType(.float32)
                        .reshaped([K] + [Int](repeating: 1, count: ps[j].ndim - 1))
                    ref[j] = ref[j] + hot * gk / Float(Self.rows.count)
                }
                bb.initialState = nil
            }
            bb.initialStateBank = bank
            eval(ref)
            var worstA: Float = 0, worstS: Float = 0, worstAS: Float = 0
            for j in ps.indices {
                XCTAssertEqual(gA[j].shape, ps[j].shape)
                worstA = max(worstA, rel(ref[j], gA[j]))
                worstS = max(worstS, rel(ref[j], seq.gradients[j]))
                worstAS = max(worstAS, rel(gA[j], seq.gradients[j]))
                XCTAssertEqual(MLX.abs(gA[j][3]).max().item(Float.self), 0, "строка банка без строк батча получила градиент")
                XCTAssertEqual(MLX.abs(seq.gradients[j][3]).max().item(Float.self), 0)
                XCTAssertGreaterThan(MLX.abs(ref[j][1]).max().item(Float.self), 0, "эталон вырожден")
            }
            print(String(format: "БАНК %@: лосс %.6f / %.6f / %.6f; худшая отн. разность: автоград−эталон %.1e, по блокам−эталон %.1e, по блокам−автоград %.1e",
                         parts.map(\.rawValue).joined(separator: "+"), lossA, seq.loss.item(Float.self), lossRef,
                         worstA, worstS, worstAS))
            XCTAssertLessThan(worstA, 1e-4, "автоград банка против строк по одной")
            XCTAssertLessThan(worstS, 1e-4, "по блокам против строк по одной")
            XCTAssertLessThan(worstAS, 1e-6, "по блокам против автограда на том же составе")
            XCTAssertEqual(lossA, lossRef, accuracy: abs(lossRef) * 1e-5)
        }
    }

    /// inject/commit пишут в банк; строки после прохода и при ошибке —
    /// прежние.
    func testTrainableAndRowsRestore() throws {
        let (bb, _) = tiny()
        let set = StateBankTrainableSet(bb, initial: randomBank(bb, K: 2))
        let ps = set.initialParameters()
        set.commit(ps.map { $0 * 2 })
        XCTAssertLessThan(diff(bb.initialStateBank!.wkv[0], ps[0] * 2), 1e-7)
        struct Boom: Error {}
        let outer = MLXArray([Int32(1)])
        try bb.withInitialRows(outer) {
            XCTAssertThrowsError(try bb.withInitialRows(MLXArray([Int32(0)])) { throw Boom() })
            XCTAssertEqual(bb.initialRows!.item(Int32.self), 1, "внутренние строки не сняты при ошибке")
        }
        XCTAssertNil(bb.initialRows)
    }
}
