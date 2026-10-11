import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Начальное состояние как подключаемый объект и его обучение
//  (`InitialState.swift`). Структурные свойства — на крошечной модели
//  (обязаны выполняться при любых весах); согласие с rwkv-metal на настоящих
//  весах — в `StateTuningParityTests` ниже.
// ───────────────────────────────────────────────────────────────────────

final class StateTuningTests: XCTestCase {

    typealias Part = StateTrainableSet.Part

    func tiny(nLayer: Int = 3, seed: UInt64 = 42) -> (X070Backbone, X070Config) {
        TinyBackbone.make(nLayer: nLayer, seed: seed)
    }

    func batch(_ cfg: X070Config, B: Int = 2, T: Int = 48, seed: UInt64 = 7) -> LoRABatch {
        let ids = TinyBackbone.ids(B, T + 1, vocab: cfg.vocab, seed: seed)
        return (x: ids[0..., 0 ..< T], y: ids[0..., 1 ... T])
    }

    /// Ненулевое состояние: на нулях часть градиентов вырождена.
    func randomState(_ bb: X070Backbone, seed: UInt64 = 3, shifts: Bool = true) -> RWKVBatchState {
        MLXRandom.seed(seed)
        let z = bb.zeroInitialState()
        let s = RWKVBatchState(wkv: MLXRandom.normal(z.wkv.shape) * 0.05,
                               tmixShift: MLXRandom.normal(z.tmixShift.shape) * (shifts ? 0.5 : 0),
                               cmixShift: MLXRandom.normal(z.cmixShift.shape) * (shifts ? 0.5 : 0))
        return s.evaluated()
    }

    func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && allClose(a.asType(.float32), b.asType(.float32), rtol: 0, atol: 0).item(Bool.self)
    }

    func diff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        diff(ref, got) / (MLX.abs(ref.asType(.float32)).max().item(Float.self) + 1e-12)
    }

    func sameState(_ a: RWKVBatchState, _ b: RWKVBatchState) -> Bool {
        same(a.wkv, b.wkv) && same(a.tmixShift, b.tmixShift) && same(a.cmixShift, b.cmixShift)
    }

    // ── подключение ──────────────────────────────────────────────────

    /// Нулевое подключённое состояние счёта не меняет — ни на бит.
    func testZeroStateChangesNothing() {
        let (bb, cfg) = tiny()
        let ids = batch(cfg).x
        let plain = bb(ids); eval(plain)
        XCTAssertNil(bb.initialState)
        bb.initialState = bb.zeroInitialState()
        let zero = bb(ids); eval(zero)
        XCTAssertTrue(same(plain, zero), "нулевое состояние изменило логиты на \(diff(plain, zero))")
        bb.initialState = nil
        XCTAssertNil(bb.initialState)
        XCTAssertNil(bb.initialLayers)
    }

    /// Подключённое == явное; явное вытесняет подключённое; каждая часть
    /// состояния действует сама.
    func testAttachedEqualsExplicitAndExplicitWins() {
        let (bb, cfg) = tiny()
        let ids = batch(cfg).x
        let none = bb.body(ids); eval(none)
        let S = randomState(bb)
        bb.initialState = S
        XCTAssertTrue(sameState(bb.initialState!, S), "подключено не то, что подключали")
        let attached = bb.body(ids); eval(attached)
        let viaState = bb.bodyWithState(ids).0; eval(viaState)
        bb.initialState = nil
        let explicit = bb.bodyWithState(ids, state: S.repeated(2)).0; eval(explicit)
        XCTAssertTrue(same(attached, explicit), "подключённое != явное: \(diff(attached, explicit))")
        XCTAssertTrue(same(viaState, explicit), "проход с состоянием не взял подключённое: \(diff(viaState, explicit))")
        XCTAssertGreaterThan(diff(attached, none), 1e-2, "состояние ни на что не влияет")

        bb.initialState = randomState(bb, seed: 5)
        let over = bb.bodyWithState(ids, state: S.repeated(2)).0; eval(over)
        XCTAssertTrue(same(over, explicit), "явное состояние не вытеснило подключённое")

        for part in Part.allCases {
            var z = bb.zeroInitialState()
            switch part {
            case .wkv: z.wkv = S.wkv
            case .tmixShift: z.tmixShift = S.tmixShift
            case .cmixShift: z.cmixShift = S.cmixShift
            }
            bb.initialState = z
            let d = diff(bb.body(ids), none)
            XCTAssertGreaterThan(d, 1e-4, "часть \(part.rawValue) состояния не действует")
            // и слой берёт СВОЮ часть: то же через явное состояние
            let viaExplicit: MLXArray = { bb.initialState = nil; return bb.bodyWithState(ids, state: z.repeated(2)).0 }()
            bb.initialState = z
            XCTAssertTrue(same(bb.body(ids), viaExplicit), "часть \(part.rawValue): подключённое != явное")
        }
    }

    /// Вытеснение — ПО ЧАСТЯМ: явная матрица WKV при неявных сдвигах идёт с
    /// подключёнными сдвигами, и наоборот.
    func testExplicitPartsOverrideAttachedOnesSeparately() {
        let (bb, cfg) = tiny()
        let x = bb.stem(batch(cfg).x)
        let attached = randomState(bb), explicit = randomState(bb, seed: 5).repeated(2)
        let a = attached.repeated(2)
        func block(h: MLXArray?, t: MLXArray?, c: MLXArray?) -> MLXArray {
            let out = bb.blockForwardWithState(x, nil, 0, hIn: h, mask: nil, tmixPrev: t, cmixPrev: c, endIdx: nil).0
            eval(out)
            return out
        }
        bb.initialState = nil
        let wantWKV = block(h: explicit.layerWKV(0), t: a.layerTmixShift(0), c: a.layerCmixShift(0))
        let wantTmix = block(h: a.layerWKV(0), t: explicit.layerTmixShift(0), c: a.layerCmixShift(0))
        let wantCmix = block(h: a.layerWKV(0), t: a.layerTmixShift(0), c: explicit.layerCmixShift(0))
        bb.initialState = attached
        XCTAssertTrue(same(block(h: explicit.layerWKV(0), t: nil, c: nil), wantWKV), "явная матрица WKV")
        XCTAssertTrue(same(block(h: nil, t: explicit.layerTmixShift(0), c: nil), wantTmix), "явный сдвиг tmix")
        XCTAssertTrue(same(block(h: nil, t: nil, c: explicit.layerCmixShift(0)), wantCmix), "явный сдвиг cmix")
        XCTAssertFalse(same(wantWKV, wantTmix))
    }

    /// Состояние чужой архитектуры не подключается (проверка формы).
    func testMismatchIsDetected() {
        let (bb, _) = tiny(nLayer: 3)
        let (other, _) = tiny(nLayer: 2)
        XCTAssertNil(StateFile.mismatch(bb.zeroInitialState(), bb.cfg))
        XCTAssertNotNil(StateFile.mismatch(other.zeroInitialState(), bb.cfg))
        XCTAssertNotNil(StateFile.mismatch(bb.zeroInitialState().repeated(2), bb.cfg), "батч 2 принят")
    }

    /// Склейка: свёрнутый промпт, подключённый как старт, продолжает так же,
    /// как сплошной проход; «обученное → промпт → кэш» — тоже.
    func testFoldedPromptContinuesLikeWholePass() {
        let (bb, cfg) = tiny()
        let P = 32, Q = 32
        let full = TinyBackbone.ids(1, P + Q, vocab: cfg.vocab, seed: 21)
        let prompt = full[0, 0 ..< P].asArray(Int32.self).map(Int.init)
        let tail = full[0..., P...]

        let whole = bb.body(full)[0..., P...]; eval(whole)
        let cached = bb.foldPrompt(prompt)
        XCTAssertEqual(cached.batch, 1)
        bb.initialState = cached
        let cont = bb.body(tail); eval(cont)
        bb.initialState = nil
        XCTAssertLessThan(rel(whole, cont), 1e-3, "свёрнутый промпт как старт расходится со сплошным проходом")

        let tuned = randomState(bb, seed: 9, shifts: false)
        bb.initialState = tuned
        let wholeTuned = bb.body(full)[0..., P...]; eval(wholeTuned)
        let cachedTuned = bb.foldPrompt(prompt)
        let contTuned = bb.bodyWithState(tail, state: cachedTuned).0; eval(contTuned)
        XCTAssertLessThan(rel(wholeTuned, contTuned), 1e-3, "обученное → промпт → кэш расходится со сплошным проходом")
        XCTAssertGreaterThan(diff(cachedTuned.wkv, cached.wkv), 1e-5, "кэш промпта не зависит от подключённого состояния")
        // явный старт вытесняет подключённое и при свёртке
        let fromZero = bb.foldPrompt(prompt, from: bb.zeroInitialState())
        XCTAssertLessThan(rel(cached.wkv, fromZero.wkv), 1e-5)
        XCTAssertTrue(bb.trainLayers.isEmpty)
    }

    /// Генерация: `prefill` стартует с подключённого состояния и согласен с
    /// пошаговым декодом от `startState()`.
    func testPrefillStartsFromAttachedState() {
        let (bb, cfg) = tiny()
        // промпт короткий: на длинном вклад стартового состояния затухает
        let ids = TinyBackbone.ids(1, 6, vocab: cfg.vocab, seed: 4).asArray(Int32.self).map(Int.init)
        var s0 = RWKVState(cfg: cfg, dtype: .float32)
        let plain = bb.prefill(ids, state: &s0); eval(plain)
        bb.initialState = randomState(bb)
        var s1 = RWKVState(cfg: cfg, dtype: .float32)
        let fast = bb.prefill(ids, state: &s1); eval(fast)
        var s2 = bb.startState()
        let slow = bb.prefillRecurrent(ids, state: &s2); eval(slow)
        XCTAssertGreaterThan(diff(fast, plain), 1e-3, "prefill не заметил подключённого состояния")
        XCTAssertLessThan(rel(slow, fast), 1e-3, "параллельный prefill и пошаговый декод стартуют с разного")
        XCTAssertLessThan(rel(s2.wkv.last!, s1.wkv.last!), 1e-3)
        // без подключённого startState — нули
        bb.initialState = nil
        XCTAssertEqual(MLX.abs(bb.startState().wkv[0]).max().item(Float.self), 0)
    }

    // ── градиент ─────────────────────────────────────────────────────

    func whole(_ bb: X070Backbone, _ set: TrainableSet, _ ps: [MLXArray],
               _ b: LoRABatch, chunk: Int = 0) -> (Float, [MLXArray]) {
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            set.inject(p)
            return [LoRAFinetune.languageModelLoss(bb, b, chunk: chunk).asType(.float32)]
        }, argumentNumbers: Array(ps.indices))
        let (v, g) = vg(ps)
        eval(v + g)
        set.inject(ps)
        return (v[0].item(Float.self), g)
    }

    /// Градиент состояния обратным проходом по блокам == автоград по всему графу.
    func testSequentialGradientMatchesAutograd() {
        for (what, parts, B, T, chunk) in [
            ("WKV, одна строка", [Part.wkv], 1, 48, 0),
            ("WKV, батч 2", [.wkv], 2, 48, 0),
            ("WKV и сдвиги, батч 2", Part.allCases, 2, 48, 0),
            ("WKV, лосс кусками", [.wkv], 2, 48, 16),
            ("только сдвиги", [.tmixShift, .cmixShift], 2, 32, 0),
        ] {
            let (bb, cfg) = tiny()
            let set = StateTrainableSet(bb, train: parts, initial: randomState(bb))
            XCTAssertEqual(bb.trainLayers, Set(0 ..< cfg.nLayer))
            XCTAssertEqual(set.parameterNames.count, cfg.nLayer * parts.count)
            XCTAssertEqual(set.parameterNames.first, "blocks.0.state0.\(set.parts[0].rawValue)")
            let ps = set.initialParameters()
            let b = batch(cfg, B: B, T: T)
            let (lossRef, gRef) = whole(bb, set, ps, b)
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: chunk)
            eval([r.loss] + r.gradients)
            var worst: Float = 0
            for (i, name) in set.parameterNames.enumerated() {
                XCTAssertGreaterThan(MLX.abs(gRef[i]).max().item(Float.self), 0, "\(what): эталонный градиент \(name) нулевой")
                XCTAssertEqual(r.gradients[i].shape, ps[i].shape, name)
                XCTAssertEqual(r.gradients[i].dtype, .float32, name)
                worst = Swift.max(worst, rel(gRef[i], r.gradients[i]))
            }
            print(String(format: "СОСТОЯНИЕ ПО БЛОКАМ, %@: лосс %.6f против %.6f; худший градиент %.1e (%d параметров)",
                         what, r.loss.item(Float.self), lossRef, worst, gRef.count))
            XCTAssertLessThan(worst, chunk > 0 ? 1e-4 : 1e-6, what)
            XCTAssertEqual(r.loss.item(Float.self), lossRef, accuracy: Swift.abs(lossRef) * 1e-5, what)
            // в модели остались значения, а не трассеры
            let again = LoRAFinetune.languageModelLoss(bb, b).item(Float.self)
            XCTAssertEqual(again, lossRef, accuracy: Swift.abs(lossRef) * 1e-5, what)
        }
    }

    /// Трансляция [1, …] на батч и сумма по строкам: градиент двух строк —
    /// среднее градиентов по строкам; и конечная разность по слоям (проверка,
    /// не зависящая от автограда).
    func testBatchGradientIsMeanOfRowsAndMatchesFiniteDifference() {
        let (bb, cfg) = tiny()
        let set = StateTrainableSet(bb, initial: randomState(bb, shifts: false))
        let ps = set.initialParameters()
        let b = batch(cfg, B: 2)
        func grad(_ batch: LoRABatch) -> [MLXArray] {
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: batch, lossChunk: 0)
            eval(r.gradients)
            return r.gradients
        }
        let both = grad(b)
        let first = grad((x: b.x[0 ..< 1], y: b.y[0 ..< 1]))
        let second = grad((x: b.x[1 ..< 2], y: b.y[1 ..< 2]))
        for i in both.indices {
            XCTAssertLessThan(rel(both[i], (first[i] + second[i]) * 0.5), 1e-4, set.parameterNames[i])
            XCTAssertGreaterThan(rel(first[i], second[i]), 1e-2, "строки батча дали один градиент — сверять нечего")
        }

        MLXRandom.seed(1)
        let eps: Float = 1e-2
        for l in 0 ..< cfg.nLayer {
            let d = MLXRandom.normal(ps[l].shape)
            let analytic = (both[l] * d).sum().item(Float.self)
            func at(_ e: Float) -> Float {
                var moved = ps
                moved[l] = ps[l] + e * d
                set.inject(moved)
                return LoRAFinetune.languageModelLoss(bb, b).item(Float.self)
            }
            let numeric = (at(eps) - at(-eps)) / (2 * eps)
            set.inject(ps)
            XCTAssertEqual(numeric, analytic, accuracy: 0.03 * Swift.abs(analytic) + 2e-3,
                           "слой \(l): конечная разность \(numeric) против градиента \(analytic)")
        }
    }

    // ── обучение ─────────────────────────────────────────────────────

    func tune(sequential: Bool, parts: [Part] = [.wkv], initial: Bool = false)
        -> (X070Backbone, RWKVBatchState, [Float], Int) {
        let (bb, cfg) = tiny()
        let b = batch(cfg, B: 2, T: 32, seed: 13)
        var c = StateTuneConfig()
        c.lr = 0.05; c.lrMin = 0.01; c.warmupSteps = 0; c.maxSteps = 6; c.logEvery = 1
        c.cacheLimitGB = 0; c.lossChunkTokens = 0; c.train = parts; c.sequentialBackward = sequential
        var losses: [Float] = []
        let calls = StateTuning.sequentialGradientCalls
        let out = StateTuning.run(bb, nextBatch: { b }, config: c,
                                  initial: initial ? randomState(bb, seed: 9) : nil,
                                  onStep: { _, l, _, _ in losses.append(l) })
        XCTAssertEqual(out.result.steps, 6)
        return (bb, out.state, losses, StateTuning.sequentialGradientCalls - calls)
    }

    func testDefaults() {
        let c = StateTuneConfig()
        XCTAssertEqual(c.train, [.wkv])
        XCTAssertTrue(c.sequentialBackward)
        XCTAssertEqual(c.lr, 1.0); XCTAssertEqual(c.lrMin, 0.01); XCTAssertEqual(c.gradClip, 0)
    }

    /// Обучение идёт по блокам, выключатель даёт то же состояние, лосс падает,
    /// база не меняется, обучается только заказанное.
    func testStateTuningRun() {
        let (reference, _) = tiny()
        let (bb, state, losses, calls) = tune(sequential: true)
        XCTAssertEqual(calls, 6, "state-tuning не пошёл обратным проходом по блокам")
        XCTAssertLessThan(losses.last!, losses.first! - 1e-3, "лосс не падает: \(losses)")
        XCTAssertGreaterThan(MLX.abs(state.wkv).max().item(Float.self), 1e-3, "состояние не сдвинулось")
        XCTAssertEqual(MLX.abs(state.tmixShift).max().item(Float.self), 0, "сдвиги обучались, хотя просили только WKV")
        XCTAssertEqual(MLX.abs(state.cmixShift).max().item(Float.self), 0)
        XCTAssertEqual(state.wkv.dtype, .float32)
        XCTAssertTrue(sameState(bb.initialState!, state), "после обучения подключено не то состояние")
        XCTAssertTrue(bb.trainLayers.isEmpty, "trainLayers не возвращён")
        for key in reference.allWeightKeys {
            XCTAssertTrue(same(bb.weight(key), reference.weight(key)), "state-tuning изменил базу: \(key)")
        }
        let (_, ref, _, refCalls) = tune(sequential: false)
        XCTAssertEqual(refCalls, 0, "выключатель обратного прохода по блокам не действует")
        XCTAssertLessThan(rel(state.wkv, ref.wkv), 1e-4, "по блокам и одним графом — разные состояния")

        // со сдвигами — обучаются и они; со стартовым состоянием — старт с него
        let (_, withShifts, _, _) = tune(sequential: true, parts: Part.allCases)
        XCTAssertGreaterThan(MLX.abs(withShifts.tmixShift).max().item(Float.self), 1e-4)
        XCTAssertGreaterThan(MLX.abs(withShifts.cmixShift).max().item(Float.self), 1e-4)
        let (_, fromInit, _, _) = tune(sequential: true, initial: true)
        XCTAssertGreaterThan(MLX.abs(fromInit.tmixShift).max().item(Float.self), 0.1, "стартовое состояние потеряно")
    }

    /// Множитель шага: состояние с множителем k при шаге lr — то же, что без
    /// множителя при шаге k·lr; в композиции множители лежат на своих местах.
    func testLearningRateScale() {
        func run(lr: Float, scale: Float?) -> MLXArray {
            let (bb, cfg) = tiny()
            let b = batch(cfg, B: 1, T: 32)
            let set = StateTrainableSet(bb, learningRateScale: scale)
            let trainer = Trainer<LoRABatch>(
                trainable: set, objective: { LoRAFinetune.languageModelLoss(bb, $0) },
                nextBatch: { b },
                config: TrainingConfig(lr: lr, gradClip: 0, maxSteps: 3, cacheLimitGB: 0, logEvery: 0))
            trainer.run()
            return bb.initialState!.wkv
        }
        let scaled = run(lr: 0.001, scale: 20), plain = run(lr: 0.02, scale: nil), small = run(lr: 0.001, scale: nil)
        XCTAssertLessThan(rel(plain, scaled), 1e-4)
        XCTAssertGreaterThan(rel(plain, small), 0.5, "множитель шага не действует")

        let (bb, cfg) = tiny()
        let (other, _) = tiny(seed: 7)
        let composite = CompositeTrainableSet([StateTrainableSet(bb, learningRateScale: 3), StateTrainableSet(other)])
        XCTAssertEqual(composite.learningRateScales,
                       [Float](repeating: 3, count: cfg.nLayer) + [Float](repeating: 1, count: cfg.nLayer))
        XCTAssertNil(CompositeTrainableSet([StateTrainableSet(other)]).learningRateScales)
    }

    // ── файл и отпечаток ─────────────────────────────────────────────

    func tmp(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("swr_state_\(name).safetensors")
    }

    func testFileRoundTripAndRefusals() throws {
        let (bb, _) = tiny()
        let state = randomState(bb)
        let url = tmp("roundtrip")
        let written = try StateFile.save(state, base: bb, to: url, kind: .tunedPrompt,
                                         trained: [.wkv, .tmixShift], note: "гейт")
        let (loaded, info) = try StateFile.load(url, base: bb)
        XCTAssertTrue(sameState(loaded, state), "файл туда-обратно не побитно")
        XCTAssertEqual(loaded.wkv.dtype, .float32)
        XCTAssertEqual(info, written)
        XCTAssertEqual(info.kind, "tuned+prompt"); XCTAssertEqual(info.trained, ["wkv", "tmix_shift"])
        XCTAssertEqual(info.note, "гейт"); XCTAssertEqual(info.nLayer, 3)
        XCTAssertEqual(info.baseFingerprint, StateFile.baseFingerprint(bb))
        XCTAssertFalse(info.baseFingerprint.isEmpty)
        _ = try StateFile.load(url)                    // без базы — без проверок

        // чужая база: отказ, пока не разрешили
        let (other, _) = tiny(seed: 7)
        XCTAssertThrowsError(try StateFile.load(url, base: other)) { e in
            guard case StateFile.Failure.otherBase = e else { return XCTFail("\(e)") }
        }
        XCTAssertTrue(sameState(try StateFile.load(url, base: other, allowOtherBase: true).state, state))

        // чужая архитектура: отказ всегда
        let (deeper, _) = tiny(nLayer: 4)
        for allow in [false, true] {
            XCTAssertThrowsError(try StateFile.load(url, base: deeper, allowOtherBase: allow)) { e in
                guard case StateFile.Failure.otherArchitecture = e else { return XCTFail("\(e)") }
            }
        }
        XCTAssertThrowsError(try StateFile.save(state.repeated(2), base: bb, to: tmp("bad"))) { e in
            guard case StateFile.Failure.otherArchitecture = e else { return XCTFail("\(e)") }
        }

        // не файл состояния; чужая версия; нет тензора
        let (arrays, md) = try loadArraysAndMetadata(url: url)
        func rewritten(_ change: (inout [String: MLXArray], inout [String: String]) -> Void) throws -> URL {
            var a = arrays, m = md
            change(&a, &m)
            let u = tmp("tampered")
            try MLX.save(arrays: a, metadata: m, url: u)
            return u
        }
        XCTAssertThrowsError(try StateFile.load(rewritten { _, m in m["format"] = "lora" })) { e in
            XCTAssertEqual(e as? StateFile.Failure, .notAStateFile("lora"))
        }
        XCTAssertThrowsError(try StateFile.load(rewritten { _, m in m["version"] = "2" })) { e in
            XCTAssertEqual(e as? StateFile.Failure, .unknownVersion("2"))
        }
        XCTAssertThrowsError(try StateFile.load(rewritten { a, _ in a["cmix_shift"] = nil })) { e in
            XCTAssertEqual(e as? StateFile.Failure, .missingTensor("cmix_shift"))
        }
        // контракт в метаданных проверяется сам, не только формы
        XCTAssertThrowsError(try StateFile.load(rewritten { _, m in m["n_head"] = "5" }, base: bb, allowOtherBase: true)) { e in
            guard case StateFile.Failure.otherArchitecture = e else { return XCTFail("\(e)") }
        }
        // пустой отпечаток в файле — не пропуск
        XCTAssertThrowsError(try StateFile.load(rewritten { _, m in m["base_fingerprint"] = "" }, base: bb)) { e in
            guard case StateFile.Failure.otherBase = e else { return XCTFail("\(e)") }
        }
    }

    /// Отпечаток: один у bf16 и fp32, не видит подключённого состояния и
    /// адаптеров, видит любой слой и проекции.
    func testBaseFingerprint() {
        let cfg = TinyBackbone.config(nLayer: 3)
        let weights = TinyBackbone.weights(cfg)
        func fp(_ w: [String: MLXArray], _ dtype: DType = .float32) -> String {
            StateFile.baseFingerprint(X070Backbone(weights: w, cfg: cfg, computeDType: dtype))
        }
        let onLattice = weights.mapValues { $0.asType(.bfloat16).asType(.float32) }
        let base = fp(onLattice)
        XCTAssertEqual(base.count, 64)
        XCTAssertEqual(fp(onLattice, .bfloat16), base, "отпечаток зависит от типа, в котором загружена база")
        // fp32-источник: числа вне решётки bf16 (сдвиг много меньше её шага) — та же база
        var off = onLattice
        off["blocks.2.tmix.r_k"] = off["blocks.2.tmix.r_k"]! * (1 + 1e-6)
        off["ln0.weight"] = off["ln0.weight"]! * (1 - 1e-6)
        XCTAssertFalse(same(off["ln0.weight"]!, onLattice["ln0.weight"]!))
        XCTAssertEqual(fp(off), base, "fp32-источник и его bf16-загрузка дали разные отпечатки")

        for key in ["ln0.bias", "ln_out.weight", "blocks.0.ln1.weight", "blocks.2.ln2.bias", "blocks.1.tmix.x_g",
                    "blocks.2.tmix.k_a", "blocks.2.tmix.ln_x.bias", "blocks.2.cmix.x_k"] {
            var changed = onLattice
            changed[key] = changed[key]! + 1
            XCTAssertNotEqual(fp(changed), base, "отпечаток не видит \(key)")
        }
        for key in ["blocks.0.tmix.r_proj.weight", "blocks.2.cmix.value.weight"] {
            var changed = onLattice
            changed[key] = changed[key]! * 2
            XCTAssertNotEqual(fp(changed), base, "отпечаток не видит проекцию \(key)")
        }
        XCTAssertNotEqual(StateFile.baseFingerprint(TinyBackbone.make(nLayer: 2).0),
                          StateFile.baseFingerprint(TinyBackbone.make(nLayer: 3).0))

        let bb = X070Backbone(weights: onLattice, cfg: cfg, computeDType: .float32)
        bb.initialState = randomState(bb)
        XCTAssertEqual(StateFile.baseFingerprint(bb), base, "отпечаток зависит от подключённого состояния")
        LoRA.add(to: bb, spec: LoRASpec(rank: 4, alpha: 8))
        XCTAssertEqual(StateFile.baseFingerprint(bb), base, "отпечаток зависит от адаптеров")
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Согласие с rwkv-metal на настоящих весах. Фикстуры:
//      cd ~/Develop/rwkv-metal
//      .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_state_reference.py
//  Без них тесты пропускаются.
// ───────────────────────────────────────────────────────────────────────

final class StateTuningParityTests: XCTestCase {

    static let testdata = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Develop/SwiftRWKV/.testdata")
    static let presets = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Develop/WKV-kvant/files_0110")

    func fixture(_ name: String) throws -> URL {
        let url = Self.testdata.appendingPathComponent(name)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                          "нет \(name): Scripts/dump_state_reference.py")
        return url
    }

    func dense(_ dtype: DType = .bfloat16) throws -> X070Backbone {
        let weights = try loadArrays(url: try fixture("world_0.1b_x070.safetensors"))
        let nLayer = weights.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max()! + 1
        return X070Backbone(weights: weights, cfg: X070Config(
            nLayer: nLayer, nEmbd: weights["ln_out.weight"]!.shape[0],
            headSize: weights["blocks.0.tmix.k_k"]!.shape[1], vocab: weights["head.weight"]!.shape[0]),
            computeDType: dtype)
    }

    func rel(_ ref: MLXArray, _ got: MLXArray) -> Float {
        MLX.abs(ref.asType(.float32) - got.asType(.float32)).max().item(Float.self)
            / (MLX.abs(ref.asType(.float32)).max().item(Float.self) + 1e-12)
    }

    /// Отпечаток базы — тот же, что считает rwkv-metal: плотная, REDUCTION,
    /// COMPRESSION. Три разных.
    func testFingerprintMatchesPython() throws {
        let (_, md) = try loadArraysAndMetadata(url: try fixture("state_ref_out.safetensors"))
        XCTAssertEqual(StateFile.baseFingerprint(try dense()), md["dense"])
        var seen: Set<String> = [md["dense"]!]
        for preset in ["reduction", "compression"] {
            let path = Self.presets.appendingPathComponent("0p1b_\(preset).rwkvq").path
            guard let want = md[preset], FileManager.default.fileExists(atPath: path) else { continue }
            let got = StateFile.baseFingerprint(try X070Backbone.load(fromRwkvq: path))
            XCTAssertEqual(got, want, preset)
            seen.insert(got)
        }
        print("ОТПЕЧАТКИ БАЗЫ: сверено с rwkv-metal \(seen.count)")
        XCTAssertGreaterThanOrEqual(seen.count, 1)
    }

    /// Файл состояния, записанный rwkv-metal, читается, проходит проверку базы
    /// и даёт те же скрытые состояния, лосс и градиент состояния.
    func testPythonStateFileReproducesReference() throws {
        let ref = try loadArrays(url: try fixture("state_ref_out.safetensors"))
        // Эталон снят в fp32: в bf16 две реализации расходятся на решётку, и за
        // ней не видно ошибок мельче. Отпечаток от типа загрузки не зависит —
        // файл, записанный на bf16-базе, проходит проверку на fp32.
        let bb = try dense(.float32)
        let (state, info) = try StateFile.load(try fixture("state_ref.safetensors"), base: bb)
        XCTAssertEqual(info.kind, "tuned"); XCTAssertEqual(info.trained, ["wkv"])
        XCTAssertEqual(info.nLayer, bb.cfg.nLayer)

        let set = StateTrainableSet(bb, initial: state)
        defer { bb.trainLayers = [] }
        let x = ref["ids"]!, y = ref["targets"]!
        let hidden = bb.body(x); eval(hidden)
        let dh = rel(ref["hidden"]!, hidden)
        let ps = set.initialParameters()
        let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: (x: x, y: y), lossChunk: 0)
        eval([r.loss] + r.gradients)
        let lossRef = ref["loss"]!.item(Float.self)
        let g0 = rel(ref["grad.0"]!, r.gradients[0]), gL = rel(ref["grad.last"]!, r.gradients[bb.cfg.nLayer - 1])
        print(String(format: "СОСТОЯНИЕ К ЭТАЛОНУ rwkv-metal: скрытые %.1e, лосс %.5f против %.5f, градиент слоя 0 %.1e, верхнего %.1e",
                     dh, r.loss.item(Float.self), lossRef, g0, gL))
        XCTAssertLessThan(dh, 1e-4, "скрытые состояния с подключённым состоянием разошлись с эталоном")
        XCTAssertEqual(r.loss.item(Float.self), lossRef, accuracy: 1e-4 * lossRef)
        XCTAssertLessThan(g0, 2e-3); XCTAssertLessThan(gL, 2e-3)

        // на REDUCTION тот же файл без разрешения не принимается
        let path = Self.presets.appendingPathComponent("0p1b_reduction.rwkvq").path
        guard FileManager.default.fileExists(atPath: path) else { return }
        let q = try X070Backbone.load(fromRwkvq: path)
        XCTAssertThrowsError(try StateFile.load(try fixture("state_ref.safetensors"), base: q)) { e in
            guard case StateFile.Failure.otherBase = e else { return XCTFail("\(e)") }
        }
        _ = try StateFile.load(try fixture("state_ref.safetensors"), base: q, allowOtherBase: true)
    }

    /// Над .rwkvq: градиент состояния по блокам == автоград по всему графу
    /// (идёт через сжатые проекции их собственным VJP).
    func testGradientOverQuantisedBase() throws {
        for preset in ["reduction", "compression"] {
            let path = Self.presets.appendingPathComponent("0p1b_\(preset).rwkvq").path
            try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "нет \(path)")
            let bb = try X070Backbone.load(fromRwkvq: path)
            let (state, _) = try StateFile.load(try fixture("state_ref.safetensors"), base: bb, allowOtherBase: true)
            let set = StateTrainableSet(bb, initial: state)
            let ps = set.initialParameters()
            let ids = TinyBackbone.ids(2, 33, vocab: 60000, seed: 7)
            let b: LoRABatch = (x: ids[0..., 0 ..< 32], y: ids[0..., 1 ... 32])
            let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
                set.inject(p)
                return [LoRAFinetune.languageModelLoss(bb, b).asType(.float32)]
            }, argumentNumbers: Array(ps.indices))
            let (v, gRef) = vg(ps)
            eval(v + gRef)
            set.inject(ps)
            let r = LoRAFinetune.sequentialGradients(bb, set: set, parameters: ps, batch: b, lossChunk: 0)
            eval([r.loss] + r.gradients)
            var worst: Float = 0
            for i in ps.indices {
                XCTAssertGreaterThan(MLX.abs(gRef[i]).max().item(Float.self), 0)
                worst = Swift.max(worst, rel(gRef[i], r.gradients[i]))
            }
            print(String(format: "СОСТОЯНИЕ НАД %@: лосс %.5f против %.5f; худший градиент %.1e",
                         preset.uppercased(), r.loss.item(Float.self), v[0].item(Float.self), worst))
            XCTAssertLessThan(worst, 1e-2, preset)
            XCTAssertEqual(r.loss.item(Float.self), v[0].item(Float.self), accuracy: 1e-4 * v[0].item(Float.self))
        }
    }
}
