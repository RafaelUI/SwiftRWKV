//
//  PrefillTests.swift
//  Параллельный prefill против рекуррентного, на РЕАЛЬНОЙ 0.1B.
//
//  `prefill` идёт параллельным проходом, `prefillRecurrent` — токен за
//  токеном. Это два разных ядра, считающих одно и то же, и расходятся они
//  МОЛЧА: логиты остаются правдоподобными, состояние — правдоподобным,
//  продолжение — связным. Единственный способ заметить — сверить.
//
//  Проверяется не только «логиты похожи». Логиты — это выход последнего
//  токена, а перенос состояния может быть сломан так, что последний токен
//  выйдет правильным, а продолжение — нет: например если потеряны сдвиги
//  token-shift, первый шаг после prefill возьмёт ноль вместо предыдущего
//  входа. Поэтому сверяются три вещи: логиты, само состояние послойно и
//  продолжение на несколько токенов вперёд.
//
//  Пропускается без модели и словаря.
//
import XCTest
import MLX
@testable import RWKVGen

final class PrefillTests: XCTestCase {

    func backbone() throws -> (X070Backbone, WorldTokenizer, X070Config) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let vp = env["RWKV_WORLD_VOCAB"] ?? home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        for p in [mp, vp] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p), "нет фикстуры \(p)")
        }
        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: w["head.weight"]!.shape[0])
        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: vp))
        else { throw XCTSkip("словарь не разобрался") }
        return (X070Backbone(weights: w, cfg: cfg), tok, cfg)
    }

    /// Логиты последнего токена совпадают, и совпадает argmax.
    ///
    /// Допуск 5e-3 относительный — тот же порядок, что у `body` против `step`
    /// вообще (замерено 7.4e-7 на этой модели; граница взята с запасом, потому
    /// что она про ДВА ядра, а не про одно вычисление). argmax обязан совпасть
    /// точно: именно он превращается в текст.
    func testParallelPrefillAgreesWithRecurrentOnLogits() throws {
        let (bb, tok, cfg) = try backbone()
        for prompt in ["The capital of France is",
                       "Пчёлы собирают нектар с цветов и делают мёд, а потом",
                       "a"] {
            let ids = tok.encode(prompt)
            var sp = RWKVState(cfg: cfg), sr = RWKVState(cfg: cfg)
            let par = bb.prefill(ids, state: &sp)
            let rec = bb.prefillRecurrent(ids, state: &sr)
            eval(par, rec)
            let scale = MLX.abs(rec).max().item(Float.self)
            let rel = MLX.abs(par - rec).max().item(Float.self) / scale
            XCTAssertLessThan(rel, 5e-3, "«\(prompt)»: логиты разошлись на \(rel)")
            XCTAssertEqual(par.argMax().item(Int.self), rec.argMax().item(Int.self),
                           "«\(prompt)»: выбран другой токен")
        }
    }

    /// Совпадает САМО СОСТОЯНИЕ — послойно, все три его части.
    ///
    /// Логиты этого не покрывают: они читаются с выхода последнего токена, а
    /// сдвиги token-shift на них не влияют вовсе. Потеряв сдвиги, prefill
    /// вернул бы правильные логиты и сломанное продолжение.
    func testParallelPrefillAgreesWithRecurrentOnState() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("Пчёлы собирают нектар с цветов и делают мёд, а потом")
        XCTAssertGreaterThan(ids.count, 8)

        var sp = RWKVState(cfg: cfg), sr = RWKVState(cfg: cfg)
        _ = bb.prefill(ids, state: &sp)
        _ = bb.prefillRecurrent(ids, state: &sr)

        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            eval(a, b)
            let s = MLX.abs(b.asType(.float32)).max().item(Float.self)
            let d = MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            return s > 0 ? d / s : d
        }

        // Формы сверяются не с «другой стороной», а с ОБЪЯВЛЕННЫМИ. Иначе два
        // пути могли бы одинаково разъехаться и остаться согласными между
        // собой — что и было: рекуррентный путь отращивал [1,1,D] на верхних
        // слоях за счёт бродкаста множителей token-shift, считал при этом
        // правильно, и заметить это можно было только по формам.
        let want = [cfg.nHead, cfg.headSize, cfg.headSize]
        for layer in 0 ..< cfg.nLayer {
            for (name, s) in [("параллельный", sp), ("рекуррентный", sr)] {
                XCTAssertEqual(s.wkv[layer].shape, want,
                               "\(name), слой \(layer): форма wkv")
                XCTAssertEqual(s.tmixPrev[layer].shape, [1, cfg.nEmbd],
                               "\(name), слой \(layer): форма сдвига tmix")
                XCTAssertEqual(s.cmixPrev[layer].shape, [1, cfg.nEmbd],
                               "\(name), слой \(layer): форма сдвига cmix")
            }
            XCTAssertLessThan(rel(sp.wkv[layer], sr.wkv[layer]), 1e-2,
                              "слой \(layer): wkv разошёлся")
            XCTAssertLessThan(rel(sp.tmixPrev[layer], sr.tmixPrev[layer]), 1e-2,
                              "слой \(layer): сдвиг tmix разошёлся")
            XCTAssertLessThan(rel(sp.cmixPrev[layer], sr.cmixPrev[layer]), 1e-2,
                              "слой \(layer): сдвиг cmix разошёлся")
        }
    }

    /// Продолжение после параллельного prefill совпадает с продолжением после
    /// рекуррентного — по ТОКЕНАМ, а не по числам.
    ///
    /// Самая близкая к пользователю проверка: расхождение в переносе состояния
    /// проявляется именно здесь, и проявляется текстом, а не числом.
    func testContinuationAfterEitherPrefillIsTheSameText() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")

        func continueGreedy(_ prefill: ([Int], inout RWKVState) -> MLXArray) -> [Int] {
            var st = RWKVState(cfg: cfg)
            var logits = prefill(ids, &st)
            var out: [Int] = []
            for _ in 0 ..< 16 {
                let id = logits.argMax().item(Int.self)
                out.append(id)
                logits = bb.step(id, state: &st)
            }
            return out
        }

        let afterParallel = continueGreedy { bb.prefill($0, state: &$1) }
        let afterRecurrent = continueGreedy { bb.prefillRecurrent($0, state: &$1) }
        XCTAssertEqual(afterParallel, afterRecurrent,
                       "продолжения разошлись: \(tok.decode(afterParallel)) / "
                       + "\(tok.decode(afterRecurrent))")

        // Различающее: продолжение вообще зависит от промпта. Без этого тест
        // был бы зелёным и при prefill, игнорирующем вход целиком.
        var other = RWKVState(cfg: cfg)
        var l = bb.prefill(tok.encode("Рецепт борща начинается с"), state: &other)
        var alt: [Int] = []
        for _ in 0 ..< 16 { let i = l.argMax().item(Int.self); alt.append(i)
                            l = bb.step(i, state: &other) }
        XCTAssertNotEqual(afterParallel, alt, "продолжение не зависит от промпта")
    }

    /// `vFirst` НЕ переживает границу шага — и потому его отсутствие после
    /// параллельного prefill ничего не стоит.
    ///
    /// Это утверждение, на котором держится мост `RWKVBatchState → RWKVState`:
    /// батчевое состояние `vFirst` не хранит СОЗНАТЕЛЬНО, и если бы декод на
    /// него опирался, переход на параллельный prefill дал бы правдоподобный,
    /// но неверный первый токен. Проверяется прямо: обнуляем `vFirst` в
    /// состоянии, где он заведомо выставлен, и требуем ПОБИТОВОГО совпадения
    /// логитов следующего шага.
    func testVFirstDoesNotCrossTheStepBoundary() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("The capital of France is")

        var st = RWKVState(cfg: cfg)
        _ = bb.prefillRecurrent(ids, state: &st)
        XCTAssertNotNil(st.vFirst, "рекуррентный путь не выставил vFirst — тест пуст")

        var withV = st
        var withoutV = st
        withoutV.vFirst = nil

        let a = bb.step(1, state: &withV)
        let b = bb.step(1, state: &withoutV)
        eval(a, b)
        XCTAssertEqual(MLX.abs(a - b).max().item(Float.self), 0,
                       "vFirst влияет на следующий шаг — мост состояния неверен")
    }

    /// Промпт из одного токена — граничный случай, на котором параллельный
    /// путь легко ошибается на единицу.
    func testSingleTokenPrompt() throws {
        let (bb, _, cfg) = try backbone()
        var sp = RWKVState(cfg: cfg), sr = RWKVState(cfg: cfg)
        let par = bb.prefill([510], state: &sp)
        let rec = bb.prefillRecurrent([510], state: &sr)
        eval(par, rec)
        XCTAssertEqual(par.shape, [cfg.vocab])
        let scale = MLX.abs(rec).max().item(Float.self)
        XCTAssertLessThan(MLX.abs(par - rec).max().item(Float.self) / scale, 5e-3)
        XCTAssertEqual(par.argMax().item(Int.self), rec.argMax().item(Int.self))
    }

    /// Длина, не кратная чанку ядра (16), обрабатывается верно.
    ///
    /// Параллельный путь считает WKV чанками по 16 и добивает хвост no-op
    /// шагами. Если добивка неверна, ломаются ровно те длины, которые не
    /// кратны 16, — а тесты на «удобных» длинах остаются зелёными.
    func testLengthsAroundTheKernelChunkBoundary() throws {
        let (bb, _, cfg) = try backbone()
        for n in [15, 16, 17, 31, 32, 33] {
            let ids = (0 ..< n).map { 100 + $0 }
            var sp = RWKVState(cfg: cfg), sr = RWKVState(cfg: cfg)
            let par = bb.prefill(ids, state: &sp)
            let rec = bb.prefillRecurrent(ids, state: &sr)
            eval(par, rec)
            let scale = MLX.abs(rec).max().item(Float.self)
            let rel = MLX.abs(par - rec).max().item(Float.self) / scale
            XCTAssertLessThan(rel, 5e-3, "длина \(n): расхождение \(rel)")
            XCTAssertEqual(par.argMax().item(Int.self), rec.argMax().item(Int.self),
                           "длина \(n): другой токен")
        }
    }

    /// Мост `RWKVBatchState → RWKVState` берёт ту строку, которую просили.
    ///
    /// Без этого перепутанные оси прошли бы незамеченными: при batch=1 любая
    /// ошибка индексации даёт тот же единственный ряд.
    func testStateBridgePicksTheRequestedRow() throws {
        let (bb, _, cfg) = try backbone()
        let a = (0 ..< 12).map { 200 + $0 }
        let b = (0 ..< 12).map { 900 + $0 }
        let batch = bb.states(MLXArray((a + b).map { Int32($0) }, [2, 12]))
        XCTAssertEqual(batch.batch, 2)

        let s0 = RWKVState(batch, row: 0)
        let s1 = RWKVState(batch, row: 1)
        XCTAssertEqual(s0.wkv.count, cfg.nLayer)
        XCTAssertEqual(s0.wkv[0].shape, [cfg.nHead, cfg.headSize, cfg.headSize])
        XCTAssertEqual(s0.tmixPrev[0].shape, [1, cfg.nEmbd])

        // Строки разные — значит выборка действительно по строке.
        eval(s0.wkv[0], s1.wkv[0])
        XCTAssertGreaterThan(MLX.abs(s0.wkv[0] - s1.wkv[0]).max().item(Float.self), 0,
                             "строки батча совпали — мост берёт не ту ось")

        // И строка 0 совпадает с одиночным проходом той же последовательности.
        var solo = RWKVState(cfg: cfg)
        _ = bb.prefill(a, state: &solo)
        eval(solo.wkv[0])
        let d = MLX.abs(s0.wkv[0] - solo.wkv[0]).max().item(Float.self)
        XCTAssertEqual(d, 0, "строка 0 батча разошлась с одиночным проходом на \(d)")
    }
}
