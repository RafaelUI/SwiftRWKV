import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Кросс-языковой паритет с эталонной реализацией (rwkv-metal, Python/MLX)
//  на НАСТОЯЩИХ предобученных весах.
//
//  Зачем, если тестов уже много
//  ────────────────────────────
//  Всё остальное в этом наборе проверяет Swift против САМОГО СЕБЯ:
//  структурные тождества, которые обязаны выполняться при любых весах.
//  Они ловят многое, но принципиально не могут поймать согласованное
//  недопонимание архитектуры — если формула перенесена неверно, но
//  внутренне непротиворечиво, все структурные тесты пройдут.
//
//  Здесь источник истины внешний: эталонные активации сняты с рабочей
//  Python-реализации на реальной 0.1B-модели.
//
//  Как получить фикстуры
//  ─────────────────────
//      cd ~/Develop/rwkv-metal
//      .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_reference.py \
//          --model world_0.1b_x070.safetensors \
//          --out   ~/Develop/SwiftRWKV/.testdata/reference_0.1b.safetensors
//
//  Пути переопределяются переменными окружения RWKV_PARITY_MODEL и
//  RWKV_PARITY_REFERENCE. Без фикстур тест ПРОПУСКАЕТСЯ: модель весит 382 МБ
//  и в репозитории ей не место.
// ───────────────────────────────────────────────────────────────────────

final class X070ParityTests: XCTestCase {

    struct Fixtures {
        let weights: [String: MLXArray]
        let reference: [String: MLXArray]
        let cfg: X070Config
    }

    /// nil ⇒ фикстур нет, тест пропускается.
    func loadFixtures() throws -> Fixtures? {
        let env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser
        let modelPath = env["RWKV_PARITY_MODEL"]
            ?? home.appendingPathComponent("Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let refPath = env["RWKV_PARITY_REFERENCE"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/reference_0.1b.safetensors").path

        let fm = FileManager.default
        guard fm.fileExists(atPath: modelPath), fm.fileExists(atPath: refPath) else {
            return nil
        }
        let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
        let reference = try loadArrays(url: URL(fileURLWithPath: refPath))

        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return Fixtures(weights: weights, reference: reference, cfg: cfg)
    }

    func skipIfMissing(_ f: Fixtures?) throws -> Fixtures {
        try XCTSkipIf(f == nil, """
            Нет фикстур паритета — тест пропущен. Чтобы включить:
              cd ~/Develop/rwkv-metal && .venv/bin/python \
              ~/Develop/SwiftRWKV/Scripts/dump_reference.py \
              --model world_0.1b_x070.safetensors \
              --out ~/Develop/SwiftRWKV/.testdata/reference_0.1b.safetensors
            """)
        return f!
    }

    /// Относительная невязка по максимуму модуля.
    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        let r = ref.asType(.float32), g = got.asType(.float32)
        eval(r, g)
        return MLX.abs(r - g).max().item(Float.self)
             / (MLX.abs(r).max().item(Float.self) + 1e-9)
    }

    // ── Полный проход ────────────────────────────────────────────────

    /// hidden и logits на реальных весах.
    ///
    /// Допуск 2e-2 — не «на глаз»: обе стороны считают в bf16 (10 бит
    /// мантиссы, шаг кванта ~1e-3 относительно), а расхождение копится по
    /// 12 слоям и по последовательности. Порядок операций в Swift и Python
    /// совпадает не побитово (разные редукции MLX), поэтому точного
    /// равенства не будет ни при какой корректности. Ошибка ПЕРЕНОСА —
    /// перепутанный знак, потерянный член, не тот показатель — даёт
    /// расхождение на порядки больше.
    func testFullForwardMatchesReference() throws {
        let f = try skipIfMissing(try loadFixtures())
        let bb = X070Backbone(weights: f.weights, cfg: f.cfg)
        let ids = f.reference["ids"]!

        let hidden = bb.body(ids)
        let logits = bb(ids)
        eval(hidden, logits)

        let dh = relDiff(f.reference["hidden"]!, hidden)
        let dl = relDiff(f.reference["logits"]!, logits)
        print("ПАРИТЕТ hidden: rel=\(dh)   logits: rel=\(dl)")

        XCTAssertLessThan(dh, 2e-2, "hidden разошёлся с эталоном")
        XCTAssertLessThan(dl, 2e-2, "logits разошлись с эталоном")
    }

    /// Предсказания: там, где аргмаксы расходятся, расхождение обязано быть
    /// НИЧЬЁЙ, а не другим ответом.
    ///
    /// Доля совпавших аргмаксов — плохая метрика: на случайных id у модели
    /// сплошь и рядом почти равные топ-кандидаты, и одна ничья на 64 позиции
    /// уронила бы порог 99% при полностью корректном переносе. Содержательный
    /// вопрос другой: НАСКОЛЬКО хуже выбранный Swift токен по мнению эталона.
    /// Если зазор в логитах меньше собственного шума bf16 — модели согласны,
    /// просто разошлись на ничьей. Ошибка переноса дала бы зазор в единицы
    /// логитов.
    func testPredictionsMatchReference() throws {
        let f = try skipIfMissing(try loadFixtures())
        let bb = X070Backbone(weights: f.weights, cfg: f.cfg)
        let logits = bb(f.reference["ids"]!).asType(.float32)
        let refLogits = f.reference["logits"]!.asType(.float32)

        let got = logits.argMax(axis: -1)
        let want = refLogits.argMax(axis: -1)
        eval(got, want, logits, refLogits)

        let agree = (got .== want).asType(.int32).sum().item(Int32.self)
        let total = Int32(got.size)
        print("ПАРИТЕТ argmax: \(agree)/\(total)")

        // Зазор в ЭТАЛОННЫХ логитах между его выбором и выбором Swift.
        let B = logits.shape[0], T = logits.shape[1]
        let flatRef = refLogits.reshaped([B * T, logits.shape[2]])
        let gotFlat = got.reshaped([B * T, 1])
        let wantFlat = want.reshaped([B * T, 1])
        let refBest = takeAlong(flatRef, wantFlat, axis: 1)
        let refForGot = takeAlong(flatRef, gotFlat, axis: 1)
        let gap = (refBest - refForGot)
        eval(gap)
        let maxGap = gap.max().item(Float.self)

        // Масштаб шума: наблюдаемое относительное расхождение логитов,
        // приведённое к их абсолютной величине.
        let logitScale = MLX.abs(refLogits).max().item(Float.self)
        let noise = 2e-2 * logitScale
        print("ПАРИТЕТ argmax gap: max=\(maxGap), шум bf16 ≈ \(noise)")

        XCTAssertGreaterThanOrEqual(maxGap, 0)
        XCTAssertLessThan(maxGap, noise,
                          "Swift выбрал токен, который эталон считает хуже на "
                          + "\(maxGap) логита — это больше шума bf16 (\(noise)), "
                          + "то есть не ничья, а расхождение моделей")
    }

    // ── Состояние ────────────────────────────────────────────────────

    /// Граничное состояние — то, на чём стоит префикс-кэш реранкера.
    /// Здесь особенно важен внешний эталон: структурные тесты проверяли лишь
    /// САМОСОГЛАСОВАННОСТЬ состояния, а не то, что оно вычислено правильно.
    func testStateMatchesReference() throws {
        let f = try skipIfMissing(try loadFixtures())
        let bb = X070Backbone(weights: f.weights, cfg: f.cfg)
        let st = bb.states(f.reference["ids"]!)

        let dw = relDiff(f.reference["state_wkv"]!, st.wkv)
        let dt = relDiff(f.reference["state_tmix"]!, st.tmixShift)
        let dc = relDiff(f.reference["state_cmix"]!, st.cmixShift)
        print("ПАРИТЕТ state: wkv=\(dw) tmix=\(dt) cmix=\(dc)")

        XCTAssertLessThan(dw, 2e-2, "wkv-состояние разошлось с эталоном")
        XCTAssertLessThan(dt, 2e-2, "tmix-сдвиг разошёлся с эталоном")
        XCTAssertLessThan(dc, 2e-2, "cmix-сдвиг разошёлся с эталоном")
    }

    /// Разрезанный проход через состояние — против эталонного разрезанного,
    /// то есть проверяется в точности тот путь, которым пойдёт реранкер.
    func testSplitContinuationMatchesReference() throws {
        let f = try skipIfMissing(try loadFixtures())
        let bb = X070Backbone(weights: f.weights, cfg: f.cfg)
        let ids = f.reference["ids"]!
        let T = ids.shape[1], half = T / 2

        let (h1, s1) = bb.bodyWithState(ids[0..., 0 ..< half])
        let (h2, s2) = bb.bodyWithState(ids[0..., half ..< T], state: s1)
        let joined = concatenated([h1, h2], axis: 1)

        let dh = relDiff(f.reference["split_hidden"]!, joined)
        let dw = relDiff(f.reference["split_state_wkv"]!, s2.wkv)
        print("ПАРИТЕТ split: hidden=\(dh) wkv=\(dw)")

        XCTAssertLessThan(dh, 2e-2, "продолжение разошлось с эталонным")
        XCTAssertLessThan(dw, 2e-2, "состояние после продолжения разошлось")
    }

    /// Right-padding с маской: строки разной длины в одном батче.
    func testMaskedStateMatchesReference() throws {
        let f = try skipIfMissing(try loadFixtures())
        let bb = X070Backbone(weights: f.weights, cfg: f.cfg)
        let ids = f.reference["ids"]!
        let T = ids.shape[1]

        let lensArr = f.reference["mask_lengths"]!
        eval(lensArr)
        let lengths = lensArr.asArray(Int32.self).map { Int($0) }

        let st = bb.states(ids,
                           mask: buildMask(lengths: lengths, total: T),
                           endIdx: lastRealIndex(lengths: lengths))

        let dw = relDiff(f.reference["masked_state_wkv"]!, st.wkv)
        let dt = relDiff(f.reference["masked_state_tmix"]!, st.tmixShift)
        print("ПАРИТЕТ masked: wkv=\(dw) tmix=\(dt)")

        XCTAssertLessThan(dw, 2e-2, "маскированное состояние разошлось с эталоном")
        XCTAssertLessThan(dt, 2e-2, "маскированный tmix-сдвиг разошёлся")
    }

    // ── Рекуррентный путь генерации ──────────────────────────────────

    /// Пошаговая генерация (B=1) обязана давать те же логиты, что
    /// параллельный проход: это ДРУГОЙ код (wkvStep на чистых MLX-операциях
    /// вместо Metal-ядра), и сверять его больше не с чем.
    func testRecurrentDecodeMatchesParallelOnRealWeights() throws {
        let f = try skipIfMissing(try loadFixtures())
        let bb = X070Backbone(weights: f.weights, cfg: f.cfg)
        let ids = f.reference["ids"]!
        eval(ids)
        let row = ids[0 ..< 1]
        let tokens = row.asArray(Int32.self).map { Int($0) }

        // параллельный проход
        let parallel = bb(row)
        eval(parallel)
        let lastParallel = parallel[0, tokens.count - 1]

        // рекуррентный
        var state = RWKVState(cfg: f.cfg)
        let lastStep = bb.prefillRecurrent(tokens, state: &state)
        eval(lastStep)

        let d = relDiff(lastParallel, lastStep)
        print("ПАРИТЕТ decode: rel=\(d)")
        XCTAssertLessThan(d, 5e-2,
                          "рекуррентный декод разошёлся с параллельным проходом")

        // и предсказание то же
        let a = lastParallel.asType(.float32).argMax().item(Int32.self)
        let b = lastStep.asType(.float32).argMax().item(Int32.self)
        XCTAssertEqual(a, b, "рекуррентный декод предсказал другой токен")
    }
}
