//
//  RerankParityTests.swift
//  Голова реранкера против Python на реальной 0.1B.
//
//  Веса головы берутся ИЗ ЭТАЛОНА и загружаются в Swift как есть. Для головы
//  это необходимо, а не избыточно: у неё есть зонды и MLP, которых в базе нет
//  вовсе, и породить их одинаково двумя генераторами случайных чисел нельзя.
//  (Для блоков было наоборот — они целиком выводятся из базы, и дамп их весов
//  проверял бы копирование против самого себя.)
//
//  score_fc2 в эталоне РАНДОМИЗИРОВАН. Штатная инициализация задаёт ноль, и
//  тогда скор тождественно ноль при любой, в том числе полностью сломанной,
//  реализации всего, что до него: паритет на нулях не значит ничего.
//
//  Фикстура — та же, что у BlockParityTests:
//      cd ~/Develop/rwkv-metal
//      .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_block_reference.py \
//          --model world_0.1b_x070.safetensors \
//          --out   ~/Develop/SwiftRWKV/.testdata/block_ref.safetensors
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVRerank

final class RerankParityTests: XCTestCase {

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
            Нет эталона реранкера — тест пропущен. Чтобы включить:
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

    /// Конфигурации из эталона.
    ///
    /// Вторая существует не для полноты. Голова из ОДНОГО блока с ОДНИМ
    /// зондом не исполняет три ветки: выбор слота состояния (слот всегда 0),
    /// перенос v_first между блоками (переносить некуда) и снятие скора с
    /// ПОСЛЕДНЕГО зонда (он же первый). Мутационная проверка показала, что
    /// все три дефекта проходили мимо тестов, пока эталон был одноблочный.
    static let configs: [(String, RerankerConfig)] = [
        ("h1", RerankerConfig(layerIdx: [5])),
        ("h2", RerankerConfig(layerIdx: [0, 5], nProbe: 2)),
    ]

    /// Голова с ПИТОНОВСКИМИ весами.
    func makeHead(_ f: Fixtures, _ tag: String, _ cfg: RerankerConfig) throws -> Reranker {
        let model = try Reranker(base: f.backbone, cfg: cfg)
        let names = model.head.parameterNames
        let missing = names.filter { f.ref["head/\(tag)/w/" + $0] == nil }
        XCTAssertTrue(missing.isEmpty, """
            \(tag): в эталоне нет параметров головы: \
            \(missing.joined(separator: ", ")). Имена в Swift и Python обязаны \
            совпадать — на этом стоит обмен чекпоинтами
            """)
        model.head.setParameters(names.map {
            f.ref["head/\(tag)/w/" + $0]!.asType(.float32)
        })
        eval(model.head.parameters)
        return model
    }

    // ─────────────────────────────────────────────────────────────────

    /// Имена параметров головы совпадают с питоновскими один в один.
    ///
    /// Проверяется отдельно от чисел, потому что ломается отдельно: при
    /// расхождении имён обмен чекпоинтами между реализациями невозможен, и
    /// узнать об этом лучше прямым сообщением, а не через «скоры не сошлись».
    func testParameterNamesMatchPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        for (tag, cfg) in Self.configs {
            let model = try Reranker(base: f.backbone, cfg: cfg)
            let prefix = "head/\(tag)/w/"
            let mine = Set(model.head.parameterNames)
            let theirs = Set(f.ref.keys.filter { $0.hasPrefix(prefix) }
                                       .map { String($0.dropFirst(prefix.count)) })

            XCTAssertEqual(mine.subtracting(theirs), [],
                           "\(tag): есть в Swift, нет в Python")
            XCTAssertEqual(theirs.subtracting(mine), [],
                           "\(tag): есть в Python, нет в Swift")
        }
    }

    /// Голова читает те же слои, что и Python: `uniqueSources` совпадают.
    /// Сверяется до чисел — при разъехавшихся слотах «скоры не сошлись» не
    /// объясняет, почему.
    func testUniqueSourcesMatchPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        for (tag, cfg) in Self.configs {
            let head = try RerankerHead(base: f.backbone, cfg: cfg)
            let want = f.ref["head/\(tag)/unique_sources"]!
            eval(want)
            XCTAssertEqual(head.uniqueSources,
                           (0 ..< want.size).map { want[$0].item(Int32.self) }
                               .map(Int.init),
                           "\(tag): голова читает не те слои")
        }
    }

    /// Скоры головы на четырёх разных состояниях.
    ///
    /// Четырёх, а не одном: одинаковые входы дали бы одинаковые скоры, и
    /// перепутанная ось батча прошла бы мимо.
    func testHeadScoresMatchPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        var byTag: [String: MLXArray] = [:]

        for (tag, cfg) in Self.configs {
            let model = try makeHead(f, tag, cfg)
            let sel = f.ref["head/\(tag)/selected"]!.asType(.float32)
            let want = f.ref["head/\(tag)/scores"]!

            // Замерено 3.3e-7 относительного — голова целиком в fp32,
            // состояние подаётся из эталона, поэтому шум bf16 сюда не входит
            // вовсе. Граница 1e-5, тридцатикратный запас.
            let got = model.scoreStates(sel)
            eval(got)
            XCTAssertEqual(got.shape, want.shape)
            XCTAssertLessThan(relDiff(want, got), 1e-5,
                              "\(tag): скоры головы разошлись с Python")

            // Различающая проверка: скоры обязаны быть РАЗНЫМИ. Совпадение
            // четырёх одинаковых нулей с четырьмя одинаковыми нулями — не
            // паритет, а именно так выглядела бы голова, игнорирующая вход.
            let spread = got.max().item(Float.self) - got.min().item(Float.self)
            XCTAssertGreaterThan(spread, 0.5,
                                 "\(tag): скоры почти одинаковы — эталон вырожден")
            byTag[tag] = got
        }

        // И две конфигурации обязаны расходиться между собой: иначе тест
        // прошёл бы у головы, которая игнорирует и слои, и зонды.
        XCTAssertGreaterThan(
            MLX.abs(byTag["h1"]! - byTag["h2"]!).max().item(Float.self), 0.1,
            "одноблочная и двублочная головы дали одно и то же")
    }

    /// Лоссы на тех же скорах. Проверяются отдельно от головы, потому что
    /// ломаются отдельно: устойчивая форма BCE, деление на температуру, ось
    /// softmax — всё это живёт мимо головы.
    func testLossesMatchPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        let scores = f.ref["loss/scores"]!.asType(.float32)
        let labels = f.ref["loss/labels"]!

        let cases: [(String, MLXArray, MLXArray)] = [
            ("listwise", f.ref["loss/listwise"]!, listwiseLoss(scores, labels)),
            ("listwise T=0.5", f.ref["loss/listwise_t05"]!,
             listwiseLoss(scores, labels, temperature: 0.5)),
            ("bce", f.ref["loss/bce"]!, bceLoss(scores, labels)),
            ("mixed 0.7", f.ref["loss/mixed07"]!,
             mixedLoss(scores, labels, alpha: 0.7)),
        ]
        for (name, want, got) in cases {
            eval(want, got)
            XCTAssertEqual(got.item(Float.self), want.item(Float.self),
                           accuracy: 1e-5, "лосс \(name) разошёлся с Python")
        }
    }

    /// Сквозной путь: токены пары → состояние базы → голова → скор.
    ///
    /// Отдельно от предыдущего, потому что здесь состояние Swift считает САМ,
    /// а не берёт из эталона, — и в скор входит накопленный по 12 слоям шум
    /// bf16. Допуск поэтому другой, и смешивать эти два теста было бы
    /// ошибкой: расхождение состояния и расхождение головы чинятся в разных
    /// местах.
    func testEndToEndScoreMatchesPython() throws {
        let f = try skipIfMissing(try loadFixtures())
        let model = try makeHead(f, "h1", RerankerConfig(layerIdx: [5]))
        let idx = f.ref["pair/idx"]!

        let selfState = model.select(model.encode(idx))         // [1,1,H,S,S]
        let refState = f.ref["pair/wkv_5"]!.asType(.float32)
                        .expandedDimensions(axis: 1)

        let gotSelf = model.scoreStates(selfState)
        let gotRef = model.scoreStates(refState)
        eval(gotSelf, gotRef)

        // Эталонный скор для этой пары — первый из четырёх (в дампе
        // head/h1/selected[0] это ровно state.wkv[5]).
        let want = f.ref["head/h1/scores"]![0]
        eval(want)
        XCTAssertLessThan(relDiff(want, gotRef), 1e-5,
                          "скор на эталонном состоянии разошёлся")

        // Замерено: 5.4e-3 против 4.2e-7 на эталонном состоянии. Разница в
        // четыре порядка — это целиком шум состояния (3.3e-3 на слое 5, см.
        // BlockParityTests), и голова его НЕ усиливает: 5.4e-3 против 3.3e-3
        // на входе. Полезное само по себе: скор — скаляр, снятый с матрицы
        // 64×64, и он мог бы оказаться чувствительнее источника.
        // Граница 5e-2 — с запасом на другую пару текстов.
        XCTAssertLessThan(relDiff(want, gotSelf), 5e-2,
                          "сквозной скор разошёлся с Python сильнее шума bf16")
    }
}
