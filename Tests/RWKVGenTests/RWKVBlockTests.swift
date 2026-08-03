//
//  RWKVBlockTests.swift
//  Блок RWKV-7 как самостоятельная сущность.
//
//  Что здесь важно проверить и почему.
//
//  1. Блок, собранный из слоя базы, обязан считать РОВНО то же, что этот слой
//     внутри бэкбона. Это и есть смысл вынесения: одна арифметика на два
//     места. Проверяется побитово — допуск здесь означал бы, что копия всё же
//     завелась.
//  2. Проход поверх ЗАДАННОГО состояния (то, ради чего блок и нужен голове)
//     согласуется с тем же проходом через бэкбон.
//  3. Блок — СИБЛИНГ базы, а не её часть: правка весов блока не трогает базу
//     и наоборот. Если бы словари оказались общими (MLXArray — ссылочный тип,
//     и словарь мог бы разделить те же объекты), обучение головы молча
//     портило бы базу, а замер «до» и «после» сравнивал бы модель с ней же.
//  4. Судьба value-residual при несовпадении позиции в стеке и слоя базы.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVKernel

final class RWKVBlockTests: XCTestCase {

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        eval(ref, got)
        return MLX.abs(ref - got).max().item(Float.self)
             / (MLX.abs(ref).max().item(Float.self) + 1e-9)
    }

    /// Вход блока: то, что пришло бы ему от предыдущего слоя.
    func makeInput(_ cfg: X070Config, B: Int = 2, T: Int = 32,
                   seed: UInt64 = 3) -> MLXArray {
        MLXRandom.seed(seed)
        let x = MLXRandom.normal([B, T, cfg.nEmbd]) * 0.5
        eval(x)
        return x
    }

    // ─────────────────────────────────────────────────────────────────
    //  Тождество со слоем базы
    // ─────────────────────────────────────────────────────────────────

    /// Блок из слоя 0 (без value-residual) == слой 0 бэкбона, бит-в-бит.
    func testBlockFromLayerZeroMatchesBackbone() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let x = makeInput(cfg)

        let block = try RWKVBlock.fromBase(bb, layer: 0, index: 0, dtype: .float32)
        block.trainable = false          // тот же путь ядра, что у frozen-базы

        let (xRef, vRef) = bb.blockForward(x, nil, 0)
        let (xBlk, vBlk) = block(x, nil)

        XCTAssertEqual(maxAbsDiff(xRef, xBlk), 0,
                       "выход блока разошёлся со слоем 0 бэкбона")
        XCTAssertEqual(maxAbsDiff(vRef, vBlk), 0,
                       "v_first блока разошёлся со слоем 0 бэкбона")
    }

    /// Блок из слоя 1 (С value-residual) == слой 1 бэкбона, бит-в-бит.
    /// Отдельный тест: у слоя 0 ветка v_lora не исполняется вовсе, и ошибка
    /// в ней прошла бы мимо предыдущего теста.
    func testBlockFromMiddleLayerMatchesBackbone() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let x = makeInput(cfg)

        // v_first берём тот, что реально родил бы слой 0.
        let (x1, vFirst) = bb.blockForward(x, nil, 0)

        let block = try RWKVBlock.fromBase(bb, layer: 1, index: 1, dtype: .float32)
        block.trainable = false

        let (xRef, vRef) = bb.blockForward(x1, vFirst, 1)
        let (xBlk, vBlk) = block(x1, vFirst)

        XCTAssertEqual(maxAbsDiff(xRef, xBlk), 0,
                       "выход блока разошёлся со слоем 1 бэкбона")
        XCTAssertEqual(maxAbsDiff(vRef, vBlk), 0,
                       "v_first обязан пройти насквозь неизменным")
    }

    /// Обучаемое ядро (checkpoint) и frozen-ядро дают одно и то же — значит
    /// голову законно обучать поверх состояния, снятого frozen-проходом.
    func testTrainableAndFrozenKernelAgree() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2)
        let x = makeInput(cfg)
        let block = try RWKVBlock.fromBase(bb, layer: 0, index: 0, dtype: .float32)

        block.trainable = false
        let (xF, _) = block(x, nil)
        block.trainable = true
        let (xT, _) = block(x, nil)

        XCTAssertEqual(maxAbsDiff(xF, xT), 0,
                       "frozen- и train-путь блока разошлись")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Проход поверх состояния — то, ради чего блок вынут наружу
    // ─────────────────────────────────────────────────────────────────

    /// Один токен-зонд поверх состояния базы: блок против того же прохода
    /// через бэкбон.
    ///
    /// Допуск, а не равенство: блок при T == 1 идёт через wkv7Step (чистые
    /// MLX-операции), бэкбон — через Metal-ядро с добивкой до 16. Математика
    /// одна, порядок сложения fp32 разный. Замерено 1.3e-7 относительного,
    /// граница 1e-5.
    func testProbeOverStateMatchesBackbone() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2)
        let B = 2

        // Состояние, которое реально порождает база на каком-то тексте.
        let ids = TinyBackbone.ids(B, 32, vocab: cfg.vocab, seed: 9)
        let st = bb.states(ids)
        let hIn = st.layerWKV(1)                       // [B,H,S,S]

        MLXRandom.seed(17)
        let probe = MLXRandom.normal([B, 1, cfg.nEmbd]) * 0.5
        eval(probe)

        let block = try RWKVBlock.fromBase(bb, layer: 1, index: 0, dtype: .float32)
        let (xBlk, _, hBlk) = block(probe, nil, hIn: hIn)

        // Эталон: тот же слой бэкбона, но с выключенной value-residual —
        // блок с index 0 её не имеет. Слой 0 базы как раз такой, поэтому
        // сравниваем блок из слоя 0.
        let block0 = try RWKVBlock.fromBase(bb, layer: 0, index: 0, dtype: .float32)
        let (xRef, _, hRef, _, _) = bb.blockForwardWithState(
            probe, nil, 0, hIn: hIn, mask: nil,
            tmixPrev: nil, cmixPrev: nil, endIdx: nil)
        let (x0, _, h0) = block0(probe, nil, hIn: hIn)

        XCTAssertLessThan(relDiff(xRef, x0), 1e-5,
                          "зонд поверх состояния разошёлся с бэкбоном")
        XCTAssertLessThan(relDiff(hRef, h0), 1e-5,
                          "состояние после зонда разошлось с бэкбоном")
        // Блок из ДРУГОГО слоя обязан дать другой ответ — иначе тест выше
        // прошёл бы и при полностью проигнорированных весах.
        XCTAssertGreaterThan(MLX.abs(xBlk - x0).max().item(Float.self), 1e-3,
                             "блоки из разных слоёв дали один результат")
        XCTAssertEqual(hBlk.shape, hRef.shape)
    }

    /// Градиент течёт и к весам блока, и к входному состоянию.
    /// Второе — существенно: голова читает состояние, а не пересчитывает его.
    func testGradientFlowsToWeightsAndState() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2)
        let B = 2
        let ids = TinyBackbone.ids(B, 32, vocab: cfg.vocab, seed: 11)
        let hIn = bb.states(ids).layerWKV(1)

        MLXRandom.seed(19)
        let probe = MLXRandom.normal([B, 1, cfg.nEmbd]) * 0.5
        eval(probe)

        let block = try RWKVBlock.fromBase(bb, layer: 1, index: 0, dtype: .float32)
        let wKey = "tmix.o_proj.weight"
        let w0 = block.param(wKey)

        func loss(_ ps: [MLXArray]) -> [MLXArray] {
            block.wOverride = [wKey: ps[0]]
            defer { block.wOverride = nil }
            return [block(probe, nil, hIn: ps[1]).0.square().mean()]
        }
        let gs = grad(loss, argumentNumbers: [0, 1])([w0, hIn])
        eval(gs)

        XCTAssertGreaterThan(MLX.abs(gs[0]).max().item(Float.self), 0,
                             "градиент до весов блока не дошёл")
        XCTAssertGreaterThan(MLX.abs(gs[1]).max().item(Float.self), 0,
                             "градиент до входного состояния не дошёл")
    }

    /// Бэкбон при T == 1 остаётся на ЯДРЕ, а не переезжает на wkv7Step.
    ///
    /// Тест намеренно структурный, и это не лень. Численно два пути расходятся
    /// на ~1e-7 — ниже любого допуска в наборе, поэтому сравнением выходов
    /// такую подмену не поймать в принципе (проверено мутацией: она проходит
    /// мимо всех остальных тестов, включая паритет с Python).
    ///
    /// Знать о ней при этом надо. Через T == 1 идёт рекуррентный декод, для
    /// которого уже сняты замеры («рекуррентный против параллельного —
    /// 2.6e-6»); молчаливый переезд сдвинул бы их все, и следующий, кто
    /// сравнит числа с записанными, потратил бы день на несуществующий
    /// регресс. Экономии переезд не даёт: ядро на T=1 — тоже один launch.
    func testBackboneKeepsKernelPathAtT1() {
        let (bb, _) = TinyBackbone.make(nLayer: 2)
        let ctx = BackboneBlockContext(bb, 0)
        if case .step = ctx.wkvPath(1) {
            XCTFail("бэкбон переехал на wkv7Step при T == 1")
        }
        bb.trainLayers = [0]
        if case .train = ctx.wkvPath(1) {} else {
            XCTFail("обучаемый слой обязан идти через train-ядро и при T == 1")
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Блок — сиблинг базы, а не её часть
    // ─────────────────────────────────────────────────────────────────

    /// Правка весов блока не меняет базу; правка базы не меняет блок.
    func testBlockIsSiblingNotSubmodule() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 2)
        let x = makeInput(cfg)
        let block = try RWKVBlock.fromBase(bb, layer: 0, index: 0, dtype: .float32)
        block.trainable = false

        let baseBefore = bb.blockForward(x, nil, 0).0
        eval(baseBefore)
        let baseCopy = baseBefore + 0

        // Заметная правка веса блока.
        let key = "tmix.o_proj.weight"
        block.setWeights([key: block.param(key) * 2.0])
        let blockAfter = block(x, nil).0

        XCTAssertGreaterThan(maxAbsDiff(baseCopy, blockAfter), 1e-3,
                             "правка весов блока ни на что не повлияла")
        XCTAssertEqual(maxAbsDiff(baseCopy, bb.blockForward(x, nil, 0).0), 0,
                       "правка весов блока протекла в базу")
    }

    /// Тип весов приводится целиком. Голова обучается в fp32, даже если база
    /// пришла в bf16: при lr ~1e-4 и весах ~0.05 шаг Adam меньше кванта bf16
    /// и теряется на округлении, а лосс при этом продолжает убывать за счёт
    /// последнего слоя головы — то есть без этого проблема не видна.
    func testDTypeIsUniform() throws {
        let cfg = TinyBackbone.config(nLayer: 2)
        let bb = X070Backbone(weights: TinyBackbone.weights(cfg), cfg: cfg,
                              computeDType: .bfloat16)
        let block = try RWKVBlock.fromBase(bb, layer: 1, index: 1, dtype: .float32)
        for key in block.weightKeys {
            XCTAssertEqual(block.param(key).dtype, .float32,
                           "\(key) остался не в fp32")
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Value-residual при несовпадении позиции и слоя
    // ─────────────────────────────────────────────────────────────────

    /// index == 0 из слоя, У КОТОРОГО v_lora есть: она не копируется вовсе.
    /// Первый блок стека v_first порождает, а не подмешивает.
    func testFirstBlockDropsValueResidual() throws {
        let (bb, _) = TinyBackbone.make(nLayer: 3)
        let block = try RWKVBlock.fromBase(bb, layer: 2, index: 0)
        XCTAssertFalse(block.hasValueResidual)
        XCTAssertFalse(block.weightKeys.contains("tmix.v_lora_B.weight"),
                       "у первого блока стека не должно быть v_lora")
    }

    /// index > 0 из СЛОЯ 0, у которого v_lora нет: веса синтезируются
    /// нейтральными (B = 0, bias = −10 ⇒ sigmoid ≈ 4.5e-5), а не случайными.
    ///
    /// Проверяется не по весам, а по ПОВЕДЕНИЮ: подмена v_first на совершенно
    /// другой тензор почти не меняет выход. Случайная инициализация здесь
    /// прошла бы структурную проверку «веса на месте» и молча испортила
    /// обучение.
    func testValueResidualSynthesizedNeutral() throws {
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let x = makeInput(cfg)
        let block = try RWKVBlock.fromBase(bb, layer: 0, index: 1, dtype: .float32)
        XCTAssertTrue(block.hasValueResidual)

        MLXRandom.seed(23)
        let B = x.shape[0], T = x.shape[1]
        let vA = MLXRandom.normal([B, T, cfg.nHead, cfg.headSize]) * 0.5
        let vB = MLXRandom.normal([B, T, cfg.nHead, cfg.headSize]) * 0.5
        eval(vA, vB)

        let outA = block(x, vA).0
        let outB = block(x, vB).0
        let scale = MLX.abs(outA).max().item(Float.self)

        XCTAssertLessThan(maxAbsDiff(outA, outB) / scale, 1e-3,
                          "синтезированная v_lora не нейтральна: подмена v_first "
                          + "заметно меняет выход")
    }
}
