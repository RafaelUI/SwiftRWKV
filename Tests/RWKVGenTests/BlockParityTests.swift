//
//  BlockParityTests.swift
//  Кросс-языковой паритет фундамента реранкера на РЕАЛЬНОЙ 0.1B.
//
//  Что здесь проверяется сверх уже доказанного паритета бэкбона:
//
//    1. wkv7Step — один шаг рекуррентности мимо Metal-ядра. Против Python на
//       ТЕХ ЖЕ входах (они лежат в эталоне, а не генерируются заново).
//    2. Правила копирования весов слоя базы в блок головы. Их три, и все три
//       молчаливые — ошибка в любом даёт работающую голову, которая учится
//       не тому. Сверяются контрольными суммами весов ДО разбора выхода.
//    3. Проход блока поверх состояния базы, снятого с настоящего текста.
//
//  Как получить фикстуру:
//      cd ~/Develop/rwkv-metal
//      .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_block_reference.py \
//          --model world_0.1b_x070.safetensors \
//          --out   ~/Develop/SwiftRWKV/.testdata/block_ref.safetensors
//
//  Пути переопределяются RWKV_PARITY_MODEL и RWKV_BLOCK_REFERENCE. Без
//  фикстур тест ПРОПУСКАЕТСЯ, а не падает.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVKernel

final class BlockParityTests: XCTestCase {

    struct Fixtures {
        let backbone: X070Backbone
        let ref: [String: MLXArray]
    }

    func loadFixtures() throws -> Fixtures? {
        let env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser
        let modelPath = env["RWKV_PARITY_MODEL"]
            ?? home.appendingPathComponent("Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let refPath = env["RWKV_BLOCK_REFERENCE"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/block_ref.safetensors").path

        let fm = FileManager.default
        guard fm.fileExists(atPath: modelPath), fm.fileExists(atPath: refPath) else {
            return nil
        }
        let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
        let ref = try loadArrays(url: URL(fileURLWithPath: refPath))

        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return Fixtures(backbone: X070Backbone(weights: weights, cfg: cfg), ref: ref)
    }

    func skipIfMissing(_ f: Fixtures?) throws -> Fixtures {
        try XCTSkipIf(f == nil, """
            Нет эталона блока — тест пропущен. Чтобы включить:
              cd ~/Develop/rwkv-metal && .venv/bin/python \
              ~/Develop/SwiftRWKV/Scripts/dump_block_reference.py \
              --model world_0.1b_x070.safetensors \
              --out ~/Develop/SwiftRWKV/.testdata/block_ref.safetensors
            """)
        return f!
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        let r = ref.asType(.float32), g = got.asType(.float32)
        eval(r, g)
        return MLX.abs(r - g).max().item(Float.self)
             / (MLX.abs(r).max().item(Float.self) + 1e-9)
    }

    // ─────────────────────────────────────────────────────────────────
    //  1. wkv7Step
    // ─────────────────────────────────────────────────────────────────

    /// Один шаг на ТЕХ ЖЕ входах, что и в Python. Входы приходят из эталона,
    /// а не генерируются заново: два генератора случайных чисел совпадать не
    /// обязаны, и «проверка» на разных входах ничего бы не значила.
    ///
    /// Совпадение здесь ПОБИТОВОЕ, и это не подгонка допуска, а замер: обе
    /// стороны строят один и тот же граф fp32-операций, а редукции MLX для
    /// одинаковых форм раскладываются одинаково по обе стороны биндинга.
    ///
    /// Строгое равенство держится намеренно. Ослабить его до «1e-5» значило бы
    /// перестать замечать, что что-то в цепочке поменялось — а поменяться она
    /// может только от правки, и тогда об этом надо узнать.
    func testStepMatchesPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        let r = f.ref["step/r"]!, w = f.ref["step/w"]!, k = f.ref["step/k"]!
        let v = f.ref["step/v"]!, a = f.ref["step/a"]!, b = f.ref["step/b"]!
        let hIn = f.ref["step/h_in"]!

        let (out, hOut) = wkv7Step(r, w, k, v, a, b, hIn)

        XCTAssertEqual(relDiff(f.ref["step/out"]!, out), 0,
                       "выход wkv7Step разошёлся с Python")
        XCTAssertEqual(relDiff(f.ref["step/h_out"]!, hOut), 0,
                       "состояние после wkv7Step разошлось с Python")
    }

    // ─────────────────────────────────────────────────────────────────
    //  2. Состояние базы на реальном тексте
    // ─────────────────────────────────────────────────────────────────

    /// Состояние, из которого голова читает. Проверяется отдельно от блока,
    /// чтобы расхождение состояния не выглядело как расхождение блока: это
    /// разные дефекты и чинятся они в разных местах.
    ///
    /// Токены берутся из эталона: расхождение токенизатора здесь тоже надо
    /// отделить, а не смешать с расхождением модели (ровно так в прошлой
    /// сессии нашлись 72 сломанные строки словаря).
    func testPairStateMatchesPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        let idx = f.ref["pair/idx"]!
        let st = f.backbone.states(idx)

        // Замерено (относительное): слой 0 — 6.5e-3, слой 5 — 3.3e-3,
        // слой 11 — 5.6e-4. Это шум bf16, накопленный по слоям; порядок тот
        // же, что у уже задокументированного паритета wkv-состояния (0.085%).
        // Граница 2e-2 — втрое выше худшего.
        for layer in [0, 5, 11] {
            let got = st.layerWKV(layer)
            let want = f.ref["pair/wkv_\(layer)"]!
            XCTAssertLessThan(relDiff(want, got), 2e-2,
                              "состояние слоя \(layer) разошлось с Python")
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  3. Блок головы: правила копирования и проход
    // ─────────────────────────────────────────────────────────────────

    /// Контрольные суммы весов блока. Порядок и состав — как в
    /// `_weight_checksums` дампа; последние две записи это v_lora_B, и NaN
    /// там означает «веса нет вовсе» (в отличие от нуля — так выглядит
    /// НЕЙТРАЛИЗОВАННАЯ v_lora, и спутать эти два случая было бы легко).
    ///
    /// Проверять суммы, а не выход, здесь существенно: ошибка в правилах
    /// копирования даёт полностью рабочий блок с чужими весами, а его выход
    /// отличается ровно настолько, насколько отличаются веса, — по одному
    /// числу не поймёшь, дело в копировании или в арифметике.
    func testWeightCopyRulesMatchPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        let keys = ["ln1.weight", "ln2.weight", "tmix.x_r", "tmix.x_k", "tmix.k_k",
                    "tmix.r_k", "tmix.r_proj.weight", "tmix.o_proj.weight",
                    "tmix.w_lora_B.bias", "tmix.a_lora_B.bias",
                    "tmix.ln_x.weight", "cmix.x_k", "cmix.key.weight",
                    "cmix.value.weight",
                    "tmix.v_lora_B.weight", "tmix.v_lora_B.bias"]

        for (src, index) in [(5, 0), (5, 1), (0, 1), (11, 0)] {
            let block = try RWKVBlock.fromBase(f.backbone, layer: src, index: index)
            let want = f.ref["block/s\(src)_i\(index)/wsum"]!.asType(.float32)
            eval(want)

            for (i, key) in keys.enumerated() {
                let expected = want[i].item(Float.self)
                let present = block.weightKeys.contains(key)

                if expected.isNaN {
                    XCTAssertFalse(present, """
                        s\(src)_i\(index): в Python веса \(key) нет, \
                        а в Swift он есть — правила копирования разошлись
                        """)
                    continue
                }
                XCTAssertTrue(present, """
                    s\(src)_i\(index): в Python вес \(key) есть, \
                    а в Swift его нет
                    """)
                let got = block.param(key).asType(.float32).sum()
                eval(got)
                // Суммы по тензорам до 768×3072: абсолютная величина суммы
                // сильно разная, поэтому сравнение относительное, с полом на
                // случай суммы около нуля (её даёт нейтрализованная v_lora_B).
                let denom = max(abs(expected), 1e-3)
                XCTAssertLessThan(abs(got.item(Float.self) - expected) / denom, 1e-3,
                                  "s\(src)_i\(index): сумма \(key) разошлась")
            }
        }
    }

    /// Проход блока поверх состояния базы — против Python на реальной 0.1B.
    ///
    /// Допуск 1e-5, а не «пара процентов на bf16». Блок считает в fp32, и
    /// состояние ему подаётся ИЗ ЭТАЛОНА, поэтому шум базы сюда не входит
    /// вовсе. Замерено: выход 1.5e-7…3.5e-7, v_first 4.8e-8…9.2e-8,
    /// состояние 3.3e-8…3.3e-7. Граница — примерно тридцатикратный запас.
    ///
    /// Слабый допуск здесь был бы хуже бесполезного: типичная ошибка переноса
    /// (не тот слой, перепутанные оси состояния, забытая v_lora) меняет ответ
    /// на десятки процентов, и «3e-2» её бы поймал, но заодно пропустил бы
    /// всё, что мельче. Насколько именно чужой ответ далёк, проверяет
    /// различающее утверждение ниже.
    func testBlockOverStateMatchesPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        let probe = f.ref["block/probe"]!
        let vFirstIn = f.ref["block/v_first_in"]!
        // Состояние берём ИЗ ЭТАЛОНА, а не пересчитываем: иначе тест мерил бы
        // сумму двух расхождений и не сказал бы, какое из них выросло.
        var outs: [String: MLXArray] = [:]

        for (src, index) in [(5, 0), (5, 1), (0, 1), (11, 0)] {
            let tag = "block/s\(src)_i\(index)"
            let block = try RWKVBlock.fromBase(f.backbone, layer: src, index: index)
            let hIn = f.ref["pair/wkv_\(src)"]!.asType(.float32)
            let vIn: MLXArray? = index == 0 ? nil : vFirstIn

            let (x, vOut, hOut) = block(probe, vIn, hIn: hIn)
            outs[tag] = x

            XCTAssertLessThan(relDiff(f.ref[tag + "/x"]!, x), 1e-5,
                              "\(tag): выход блока разошёлся с Python")
            XCTAssertLessThan(relDiff(f.ref[tag + "/v_first_out"]!, vOut), 1e-5,
                              "\(tag): v_first разошёлся с Python")
            XCTAssertLessThan(relDiff(f.ref[tag + "/h_out"]!, hOut), 1e-5,
                              "\(tag): состояние после блока разошлось с Python")
        }

        // РАЗЛИЧАЮЩАЯ проверка: допуск 3e-2 не должен проходиться «любым
        // блоком». Каждый выход обязан быть ближе к своему эталону, чем к
        // чужому — а чужие эталоны отличаются на порядки, не на проценты.
        for (tag, got) in outs {
            let own = relDiff(f.ref[tag + "/x"]!, got)
            for other in outs.keys where other != tag {
                let cross = relDiff(f.ref[other + "/x"]!, got)
                XCTAssertLessThan(own, cross,
                                  "\(tag) ближе к эталону \(other), чем к своему")
            }
        }
    }
}
