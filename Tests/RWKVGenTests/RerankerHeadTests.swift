//
//  RerankerHeadTests.swift
//  Голова реранкера: структурные свойства, которые обязаны выполняться при
//  любых весах.
//
//  Главное из них — стартовый лосс РОВНО ln(C). Это не украшение: голова
//  zero-init по последнему слою, значит до обучения все скоры точно нули, а
//  listwise-лосс на равных скорах равен ln(C) аналитически. Если первый
//  залогированный лосс не ln(C), сломаны данные или проводка головы, и
//  искать надо там, а не в расписании lr. Проверяется строгим равенством:
//  допуск здесь означал бы, что «почти ноль» сойдёт, а он не сойдёт.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen
@testable import RWKVRerank

final class RerankerHeadTests: XCTestCase {

    func makeBase(nLayer: Int = 4) -> (X070Backbone, X070Config) {
        TinyBackbone.make(nLayer: nLayer)
    }

    /// Реальное состояние базы на каких-нибудь токенах.
    func makeState(_ bb: X070Backbone, _ cfg: X070Config,
                   B: Int = 3, T: Int = 32, seed: UInt64 = 4) -> RWKVBatchState {
        bb.states(TinyBackbone.ids(B, T, vocab: cfg.vocab, seed: seed))
    }

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    // ─────────────────────────────────────────────────────────────────
    //  Zero-init и стартовый лосс
    // ─────────────────────────────────────────────────────────────────

    /// До обучения скоры РОВНО нули — при любом состоянии на входе.
    func testUntrainedScoresAreExactlyZero() throws {
        let (bb, cfg) = makeBase()
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        let scores = model.scoreStates(model.select(makeState(bb, cfg)))
        eval(scores)

        XCTAssertEqual(scores.shape, [3])
        XCTAssertEqual(MLX.abs(scores).max().item(Float.self), 0,
                       "zero-init головы нарушен: скоры не нули")
    }

    /// Отсюда — стартовый listwise-лосс ровно ln(C).
    ///
    /// Проверяется на нескольких C: одно совпадение могло бы быть
    /// случайностью подобранного числа кандидатов.
    func testInitialListwiseLossIsLnC() throws {
        let (bb, _) = makeBase()
        let model = try Reranker(base: bb)

        for C in [2, 4, 8, 25] {
            let ids = TinyBackbone.ids(C, 24, vocab: bb.cfg.vocab, seed: UInt64(C))
            let scores = model.scoreStates(model.select(bb.states(ids)))
                              .reshaped([1, C])
            let loss = listwiseLoss(scores, MLXArray([Int32(0)]))
            eval(loss)
            XCTAssertEqual(loss.item(Float.self), log(Float(C)), accuracy: 1e-6,
                           "стартовый лосс при C=\(C) не равен ln(C)")
        }
    }

    /// Разные состояния дают РАЗНЫЕ скоры, как только голова перестала быть
    /// нулевой. Без этой проверки предыдущие две прошли бы и у головы,
    /// которая игнорирует вход целиком.
    func testScoresDependOnState() throws {
        let (bb, cfg) = makeBase()
        let model = try Reranker(base: bb)
        MLXRandom.seed(77)
        model.head.fc2Weight = MLXRandom.normal([1, model.head.hidden]) * 0.05
        eval(model.head.fc2Weight)

        let scores = model.scoreStates(model.select(makeState(bb, cfg)))
        eval(scores)
        let spread = scores.max().item(Float.self) - scores.min().item(Float.self)
        XCTAssertGreaterThan(spread, 1e-4,
                             "скоры одинаковы — голова не читает состояние")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Какие слои голова читает
    // ─────────────────────────────────────────────────────────────────

    /// Отрицательные индексы, дедупликация источников, слоты.
    func testLayerResolutionAndUniqueSources() throws {
        let (bb, _) = makeBase(nLayer: 12)

        let h1 = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        XCTAssertEqual(h1.layerIdx, [11])
        XCTAssertEqual(h1.uniqueSources, [11])

        let h2 = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [0, 5, 11]))
        XCTAssertEqual(h2.uniqueSources, [0, 5, 11])
        XCTAssertEqual(h2.sourceSlot, [0, 1, 2])

        // sharedState: глубина стека прежняя, читается ОДИН слой — и это
        // ровно то, ради чего он есть (углубить голову, не раздувая кэш).
        let h3 = try RerankerHead(base: bb,
                                  cfg: RerankerConfig(layerIdx: [0, 5, 11],
                                                      sharedState: true))
        XCTAssertEqual(h3.blocks.count, 3)
        XCTAssertEqual(h3.uniqueSources, [11], "sharedState обязан читать один слой")
        XCTAssertEqual(h3.sourceSlot, [0, 0, 0])

        XCTAssertThrowsError(try resolveLayerIndices([12], nLayer: 12))
        XCTAssertThrowsError(try resolveLayerIndices([], nLayer: 12))
    }

    /// `select` отдаёт ровно читаемые слои, а не всё состояние.
    ///
    /// Это и есть экономия кэша: 1 слой из 12 вместо всех двенадцати.
    func testSelectKeepsOnlyReadLayers() throws {
        let (bb, cfg) = makeBase(nLayer: 12)
        let head = try RerankerHead(base: bb, cfg: RerankerConfig(layerIdx: [5]))
        let st = makeState(bb, cfg, B: 2)
        let sel = head.select(st)
        eval(sel)

        XCTAssertEqual(sel.shape, [2, 1, cfg.nHead, cfg.headSize, cfg.headSize])
        XCTAssertEqual(maxAbsDiff(sel[0..., 0], st.layerWKV(5)), 0,
                       "select взял не тот слой")
        XCTAssertEqual(st.wkv.shape[0], 12)
        XCTAssertLessThan(sel.size, st.wkv.size / 10,
                          "select не даёт экономии — смысл кэша теряется")
    }

    /// Голова из блока над слоем 5 и над слоем 11 имеют ОДИНАКОВЫЕ формы, но
    /// обязаны считать разное. Это та самая пара, которую легко перепутать
    /// при загрузке чекпоинта.
    func testDifferentSourceLayersGiveDifferentScores() throws {
        let (bb, cfg) = makeBase(nLayer: 12)
        let st = makeState(bb, cfg, B: 2)

        func scores(_ layer: Int) -> MLXArray {
            let m = try! Reranker(base: bb, cfg: RerankerConfig(layerIdx: [layer]))
            MLXRandom.seed(91)
            m.head.fc2Weight = MLXRandom.normal([1, m.head.hidden]) * 0.05
            let s = m.scoreStates(m.select(st))
            eval(s)
            return s
        }
        XCTAssertGreaterThan(maxAbsDiff(scores(5), scores(11)), 1e-4,
                             "головы над разными слоями дали один результат")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Параметры, обучение, чекпоинт
    // ─────────────────────────────────────────────────────────────────

    /// Порядок параметров совпадает с порядком имён и переживает круг
    /// «прочитать → записать». Это условие корректности чекпоинтов и
    /// моментов Adam: перепутанный порядок молча перемешал бы веса.
    func testParameterRoundTrip() throws {
        let (bb, cfg) = makeBase()
        let model = try Reranker(base: bb)
        let head = model.head

        XCTAssertEqual(head.parameterNames.count, head.parameters.count)
        let before = model.scoreStates(model.select(makeState(bb, cfg)))
        let ps = head.parameters.map { $0 + 0 }
        eval(before, ps)

        // Испортить, затем восстановить — если порядок «уплывает», второй
        // проход не совпадёт с первым.
        head.setParameters(ps.map { $0 * 2.0 })
        head.setParameters(ps)
        let after = model.scoreStates(model.select(makeState(bb, cfg)))

        XCTAssertEqual(maxAbsDiff(before, after), 0,
                       "круг чтения-записи параметров изменил результат")
    }

    /// Градиент доходит до ВСЕХ параметров головы, включая веса блоков и
    /// зонды. Голова, у которой учится только MLP, тоже сходится — просто
    /// хуже, и по лоссу это неотличимо.
    func testGradientReachesEveryParameter() throws {
        let (bb, cfg) = makeBase()
        let model = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        // fc2 == 0 обнулил бы градиент почти везде: скор не зависит от того,
        // что было до него. Это верное поведение zero-init, но проверять на
        // нём течение градиента бессмысленно.
        MLXRandom.seed(31)
        model.head.fc2Weight = MLXRandom.normal([1, model.head.hidden]) * 0.05
        eval(model.head.fc2Weight)

        let sel = model.select(makeState(bb, cfg, B: 4)).asType(.float32)
        let labels = MLXArray([Int32(1)])
        let trainable = RerankerHeadTrainableSet(model.head)
        let ps = trainable.initialParameters()

        func loss(_ p: [MLXArray]) -> [MLXArray] {
            let s = model.head.apply(sel, parameters: p).reshaped([1, 4])
            return [listwiseLoss(s, labels)]
        }
        let gs = grad(loss, argumentNumbers: Array(0 ..< ps.count))(ps)
        eval(gs)

        for (name, g) in zip(model.head.parameterNames, gs) {
            XCTAssertGreaterThan(MLX.abs(g).max().item(Float.self), 0,
                                 "градиент не дошёл до \(name)")
        }
    }

    /// Чекпоинт: круг сохранения-загрузки и — главное — отказ загружать
    /// голову от ДРУГОГО слоя. Формы у них совпадают, поэтому без проверки
    /// метаданных загрузка прошла бы молча.
    func testCheckpointRoundTripAndMismatch() throws {
        let (bb, cfg) = makeBase(nLayer: 12)
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rerank_head_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let src = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [5]))
        MLXRandom.seed(13)
        src.head.fc2Weight = MLXRandom.normal([1, src.head.hidden]) * 0.05
        eval(src.head.fc2Weight)
        let st = makeState(bb, cfg, B: 2)
        let want = src.scoreStates(src.select(st))
        eval(want)

        try src.saveHead(to: url, extra: ["instruct": "тестовая инструкция"])

        // Через fromHead конфигурацию помнить не нужно — она в файле.
        let loaded = try Reranker.fromHead(base: bb, url: url)
        XCTAssertEqual(loaded.head.layerIdx, [5])
        XCTAssertEqual(maxAbsDiff(want, loaded.scoreStates(loaded.select(st))), 0,
                       "чекпоинт не воспроизвёл скоры")

        let md = try Reranker.readHeadMetadata(url)
        XCTAssertEqual(md["layer_idx"], "5")
        XCTAssertEqual(md["instruct"], "тестовая инструкция",
                       "текстовый контракт не сохранился")

        // В метаданные идёт НОРМАЛИЗОВАННЫЙ индекс. Иначе голова, собранная
        // как [-1] на 12 слоях, и голова, собранная как [11], записались бы
        // разными строками, и сверка при загрузке отвергла бы совпадающие
        // конфигурации — молча превратив полезную проверку в помеху.
        // Мутационная проверка показала, что без этого утверждения дефект
        // проходит мимо всех остальных тестов.
        let negURL = url.deletingLastPathComponent()
            .appendingPathComponent("rerank_neg_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: negURL) }
        let neg = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [-1]))
        try neg.saveHead(to: negURL)
        XCTAssertEqual(try Reranker.readHeadMetadata(negURL)["layer_idx"], "11",
                       "в метаданные попал ненормализованный индекс слоя")
        // ...и такой чекпоинт обязан приниматься моделью, собранной как [11].
        let asEleven = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [11]))
        XCTAssertNoThrow(try asEleven.loadHead(from: negURL))

        // Голова со слоя 5 в модель, читающую слой 11 — формы совпадают,
        // молча пройти не должно.
        let wrong = try Reranker(base: bb, cfg: RerankerConfig(layerIdx: [11]))
        XCTAssertThrowsError(try wrong.loadHead(from: url)) { err in
            guard case RerankerError.checkpointMismatch(let key, _, _) = err else {
                return XCTFail("ожидалась checkpointMismatch, получено \(err)")
            }
            XCTAssertEqual(key, "layer_idx")
        }
        // ...а с strict: false — пройти должно: это путь для чекпоинтов,
        // сохранённых до появления метаданных.
        XCTAssertNoThrow(try wrong.loadHead(from: url, strict: false))
    }

    /// Голова — сиблинг базы: её обучение не трогает веса базы.
    func testHeadIsSiblingOfBase() throws {
        let (bb, cfg) = makeBase()
        let ids = TinyBackbone.ids(2, 24, vocab: cfg.vocab, seed: 3)
        let baseBefore = bb.body(ids)
        eval(baseBefore)
        let copy = baseBefore + 0

        let model = try Reranker(base: bb)
        let head = model.head
        head.setParameters(head.parameters.map { $0 * 3.0 })
        eval(head.parameters)

        XCTAssertEqual(maxAbsDiff(copy, bb.body(ids)), 0,
                       "правка головы протекла в базу")
        XCTAssertTrue(bb.trainLayers.isEmpty,
                      "база реранкера обязана быть заморожена")
    }

    /// nProbe > 1: скор снимается с ПОСЛЕДНЕГО зонда, и лишний зонд меняет
    /// ответ. Заодно — что этот путь вообще работает: T = nProbe не кратно
    /// CHUNK, и обучаемое ядро на нём просто упало бы.
    func testMultipleProbesRunAndChangeResult() throws {
        let (bb, cfg) = makeBase()
        let st = makeState(bb, cfg, B: 2)

        func scores(_ nProbe: Int) -> MLXArray {
            let m = try! Reranker(base: bb,
                                  cfg: RerankerConfig(layerIdx: [-1], nProbe: nProbe),
                                  seed: 5)
            MLXRandom.seed(41)
            m.head.fc2Weight = MLXRandom.normal([1, m.head.hidden]) * 0.05
            let s = m.scoreStates(m.select(st))
            eval(s)
            return s
        }
        let s1 = scores(1), s2 = scores(2)
        XCTAssertEqual(s1.shape, [2])
        XCTAssertEqual(s2.shape, [2])
        XCTAssertGreaterThan(maxAbsDiff(s1, s2), 1e-5,
                             "второй зонд ни на что не повлиял")
    }
}
