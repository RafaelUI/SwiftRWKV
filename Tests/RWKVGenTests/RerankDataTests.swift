//
//  RerankDataTests.swift
//  Шаблон пары, загрузка обоих форматов, построение кандидатов.
//
//  Что здесь НЕ проверяется и почему. Наборы кандидатов с питоновскими не
//  сверяются: там Mersenne Twister, здесь SplitMix64, и совпадать
//  последовательности не обязаны. Поэтому проверяются СВОЙСТВА — позитив на
//  месте, кандидаты различны, майненные негативы попали внутрь, позиция
//  перемешана, пул дедуплицирован, — то есть ровно то, от чего зависит
//  корректность обучения. Совпадение последовательностей на неё не влияет.
//
import XCTest
@testable import RWKVRerank

final class RerankDataTests: XCTestCase {

    // ─────────────────────────────────────────────────────────────────
    //  Шаблон
    // ─────────────────────────────────────────────────────────────────

    /// Документ идёт ДО запроса, и префикс от запроса не зависит — на этом
    /// стоит весь кэш. Если префикс начнёт зависеть от запроса, кэширование
    /// потеряет смысл, а ошибка будет незаметной: всё продолжит работать,
    /// просто в разы медленнее.
    func testPrefixIsIndependentOfQuery() {
        let t = PairTemplate()
        let a = t.prefix(instruct: "найди", document: "документ")
        let b = t.prefix(instruct: "найди", document: "документ")
        XCTAssertEqual(a, b)
        XCTAssertTrue(a.contains("документ"), "документ обязан быть в префиксе")
        XCTAssertFalse(a.contains("Query"), "запрос попал в кэшируемый префикс")

        let full = t.full(instruct: "найди", document: "документ", query: "запрос")
        XCTAssertEqual(full, "Instruct: найди\nDocument: документ\nQuery: запрос")
        XCTAssertTrue(full.hasPrefix(a), "префикс + суффикс != целое")
    }

    /// Обратный порядок: кэшировать нечего, и это видно прямо в префиксе.
    func testQueryFirstTemplateHasNothingToCache() {
        let t = PairTemplate(docFirst: false)
        let p = t.prefix(instruct: "найди", document: "документ")
        XCTAssertFalse(p.contains("документ"),
                       "при query_first документ не должен быть в префиксе")
        XCTAssertEqual(t.full(instruct: "найди", document: "d", query: "q"),
                       "Instruct: найди\nQuery: q\nDocument: d")
        XCTAssertNotEqual(t.contract, PairTemplate().contract,
                          "контракты двух шаблонов обязаны различаться")
    }

    func testParseAnchor() {
        let (i, q) = RerankDataset.parseAnchor("Instruct: найди пассаж\nQuery: как зимуют пчёлы?")
        XCTAssertEqual(i, "найди пассаж")
        XCTAssertEqual(q, "как зимуют пчёлы?")

        // Без префикса — весь текст запрос, инструкция по умолчанию.
        let (i2, q2) = RerankDataset.parseAnchor("просто запрос")
        XCTAssertEqual(i2, defaultRerankInstruct)
        XCTAssertEqual(q2, "просто запрос")

        // Многострочная инструкция: берётся ПЕРВЫЙ \nQuery:, как в Python.
        let (i3, q3) = RerankDataset.parseAnchor("Instruct: a\nb\nQuery: c\nQuery: d")
        XCTAssertEqual(i3, "a\nb")
        XCTAssertEqual(q3, "c\nQuery: d")
    }

    // ─────────────────────────────────────────────────────────────────
    //  ГПСЧ
    // ─────────────────────────────────────────────────────────────────

    /// Один сид — одна последовательность, разные сиды — разные.
    /// Без первого сравнение двух прогонов ломалось бы из-за данных.
    func testRNGDeterministic() {
        var a = SplitMix64(seed: 7), b = SplitMix64(seed: 7), c = SplitMix64(seed: 8)
        let xs = (0 ..< 50).map { _ in a.below(1000) }
        let ys = (0 ..< 50).map { _ in b.below(1000) }
        let zs = (0 ..< 50).map { _ in c.below(1000) }
        XCTAssertEqual(xs, ys, "один сид дал разные последовательности")
        XCTAssertNotEqual(xs, zs, "разные сиды дали одну последовательность")
    }

    /// `below(n)` не перекошен. Проверяется на маленьком n, где перекос от
    /// наивного `next() % n` был бы заметен.
    func testRNGUniformity() {
        var rng = SplitMix64(seed: 1)
        let n = 7, draws = 70_000
        var counts = [Int](repeating: 0, count: n)
        for _ in 0 ..< draws { counts[rng.below(n)] += 1 }
        let expected = Double(draws) / Double(n)
        for (i, c) in counts.enumerated() {
            XCTAssertLessThan(abs(Double(c) - expected) / expected, 0.05,
                              "корзина \(i): \(c) против ожидаемых \(Int(expected))")
        }
    }

    /// Перемешивание переставляет и ничего не теряет.
    func testShuffleIsPermutation() {
        var rng = SplitMix64(seed: 3)
        let xs = Array(0 ..< 64)
        let ys = rng.shuffled(xs)
        XCTAssertEqual(ys.sorted(), xs, "перемешивание потеряло или добавило элементы")
        XCTAssertNotEqual(ys, xs, "перемешивание ничего не переставило")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Кандидаты
    // ─────────────────────────────────────────────────────────────────

    func makeRows(_ n: Int, negsPerRow: Int = 2) -> [RerankRow] {
        (0 ..< n).map { i in
            RerankRow(instruct: "инструкция", query: "запрос \(i)",
                      positive: "позитив \(i)",
                      negatives: (0 ..< negsPerRow).map { "негатив \(i)_\($0)" })
        }
    }

    /// Основные инварианты набора кандидатов.
    func testCandidateInvariants() throws {
        let rows = makeRows(40, negsPerRow: 2)
        let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 8, seed: 0)

        XCTAssertEqual(samples.count, rows.count)
        // Пул: 40 позитивов + 80 негативов, все различны.
        XCTAssertEqual(pool.count, 120)
        XCTAssertEqual(Set(pool).count, pool.count, "пул не дедуплицирован")

        for (i, s) in samples.enumerated() {
            XCTAssertEqual(s.docIds.count, 8)
            XCTAssertEqual(Set(s.docIds).count, 8, "кандидаты повторяются")
            XCTAssertEqual(pool[s.docIds[s.label]], "позитив \(i)",
                           "label указывает не на позитив")
            XCTAssertEqual(s.hardNegs.count, 2, "майненные негативы потерялись")
            for h in s.hardNegs {
                XCTAssertTrue(pool[s.docIds[h]].hasPrefix("негатив \(i)_"),
                              "hardNegs указывает не на майненный негатив")
            }
            XCTAssertFalse(s.hardNegs.contains(s.label))
        }
    }

    /// Позиция позитива ПЕРЕМЕШАНА.
    ///
    /// При listwise-лоссе фиксированная позиция — ярлык, который голова
    /// выучит вместо задачи, и по лоссу это неотличимо: он будет исправно
    /// падать. Проверяем распределение, а не «не всегда ноль»: позитив,
    /// стоящий на нуле в 90% случаев, тоже ярлык.
    func testPositiveIsShuffled() throws {
        let rows = makeRows(400, negsPerRow: 1)
        let (_, samples) = try RerankCandidates.build(rows, nCandidates: 8, seed: 0)
        var counts = [Int](repeating: 0, count: 8)
        for s in samples { counts[s.label] += 1 }
        let expected = Double(samples.count) / 8.0
        for (pos, c) in counts.enumerated() {
            XCTAssertGreaterThan(Double(c), expected * 0.6,
                                 "позиция \(pos) встречается слишком редко: \(c)")
            XCTAssertLessThan(Double(c), expected * 1.4,
                              "позиция \(pos) встречается слишком часто: \(c)")
        }
    }

    /// Пул набирается ЦЕЛИКОМ до раздачи кандидатов.
    ///
    /// Иначе ранние запросы добирали бы негативы из куцего пула, а поздние —
    /// из полного, и сложность примера зависела бы от номера строки в файле.
    /// Проверяется так: у ПЕРВОГО примера добранные кандидаты обязаны
    /// встречаться и среди документов последних строк.
    func testPoolIsCompleteBeforeSampling() throws {
        let rows = makeRows(50, negsPerRow: 1)
        let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 8, seed: 0)
        let lateDocs = Set((40 ..< 50).flatMap { ["позитив \($0)", "негатив \($0)_0"] })

        let firstFewSampled = samples.prefix(5).flatMap { s in
            s.docIds.enumerated()
                .filter { $0.offset != s.label && !s.hardNegs.contains($0.offset) }
                .map { pool[$0.element] }
        }
        XCTAssertTrue(firstFewSampled.contains { lateDocs.contains($0) }, """
            ранние примеры не добрали ни одного документа из поздних строк — \
            похоже, пул набирается по ходу
            """)
    }

    /// Слишком маленький пул — внятная ошибка, а не молчаливый недобор.
    func testPoolTooSmallThrows() {
        let rows = makeRows(2, negsPerRow: 1)          // пул из 4 документов
        XCTAssertThrowsError(try RerankCandidates.build(rows, nCandidates: 8)) { err in
            guard case RerankDataError.poolTooSmall(let need, let have) = err else {
                return XCTFail("ожидалась poolTooSmall, получено \(err)")
            }
            XCTAssertEqual(need, 8)
            XCTAssertEqual(have, 4)
        }
    }

    /// Майненных негативов больше, чем мест: лишние отбрасываются, но
    /// позитив остаётся, а число кандидатов соблюдается.
    func testMoreMinedNegativesThanSlots() throws {
        let rows = makeRows(20, negsPerRow: 5)
        let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 3, seed: 0)
        for (i, s) in samples.enumerated() {
            XCTAssertEqual(s.docIds.count, 3)
            XCTAssertEqual(pool[s.docIds[s.label]], "позитив \(i)")
            XCTAssertEqual(s.hardNegs.count, 2, "оба свободных места — под майненные")
        }
    }

    /// Разбиение по ЗАПРОСАМ: непересекающееся и полное.
    func testSplitTrainEval() throws {
        let rows = makeRows(60, negsPerRow: 1)
        let (_, samples) = try RerankCandidates.build(rows, nCandidates: 8, seed: 0)
        let (train, ev) = RerankCandidates.splitTrainEval(samples, nEval: 15, seed: 0)

        XCTAssertEqual(train.count, 45)
        XCTAssertEqual(ev.count, 15)
        let trainQ = Set(train.map { $0.query }), evalQ = Set(ev.map { $0.query })
        XCTAssertTrue(trainQ.isDisjoint(with: evalQ), "запросы пересеклись")
        XCTAssertEqual(trainQ.union(evalQ).count, 60, "часть примеров потерялась")
    }

    // ─────────────────────────────────────────────────────────────────
    //  Загрузка
    // ─────────────────────────────────────────────────────────────────

    func writeTemp(_ lines: [String]) throws -> String {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rerank_\(UUID().uuidString).jsonl")
        try lines.joined(separator: "\n").write(to: url, atomically: true,
                                                encoding: .utf8)
        return url.path
    }

    /// Оба формата распознаются по ПОЛЯМ, без флага.
    func testLoadsBothFormats() throws {
        let path = try writeTemp([
            #"{"anchor":"Instruct: найди\nQuery: вопрос A","positive":"пA","negative":"нA","task":"retrieval"}"#,
            #"{"query":"вопрос B","positive":"пB","negatives":["нB1","нB2"],"language":"rus"}"#,
            // мусор и вырожденные строки — пропускаются молча
            "не json",
            #"{"query":"вопрос C","positive":"пC","negatives":[]}"#,
            #"{"anchor":"Instruct: x\nQuery: y","positive":"пD","negative":"нD","task":"sts"}"#,
        ])
        defer { try? FileManager.default.removeItem(atPath: path) }

        let rows = try RerankDataset.loadJSONL(path: path)
        XCTAssertEqual(rows.count, 2, "распознано не два формата, а \(rows.count)")

        let byQuery = Dictionary(uniqueKeysWithValues: rows.map { ($0.query, $0) })
        XCTAssertEqual(byQuery["вопрос A"]?.instruct, "найди")
        XCTAssertEqual(byQuery["вопрос A"]?.negatives, ["нA"])
        XCTAssertEqual(byQuery["вопрос B"]?.negatives, ["нB1", "нB2"])
        XCTAssertEqual(byQuery["вопрос B"]?.language, "rus")
        XCTAssertEqual(byQuery["вопрос B"]?.instruct, defaultRerankInstruct)
    }

    /// Фильтр по языку.
    func testLanguageFilter() throws {
        let path = try writeTemp([
            #"{"query":"q1","positive":"p1","negatives":["n"],"language":"rus"}"#,
            #"{"query":"q2","positive":"p2","negatives":["n"],"language":"eng"}"#,
        ])
        defer { try? FileManager.default.removeItem(atPath: path) }
        let rows = try RerankDataset.loadJSONL(path: path, language: "eng")
        XCTAssertEqual(rows.map { $0.query }, ["q2"])
    }

    /// Резервуарная выборка: ровно `limit` строк, детерминированно по сиду,
    /// и — главное — НЕ первые `limit`.
    ///
    /// Последнее и есть смысл: у собранного по источникам корпуса начало
    /// файла систематически отличается от середины, и «первые N» — не
    /// выборка, а срез одного источника.
    func testReservoirSamplingIsNotFirstN() throws {
        let lines = (0 ..< 500).map {
            #"{"query":"q\#($0)","positive":"p\#($0)","negatives":["n\#($0)"]}"#
        }
        let path = try writeTemp(lines)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let a = try RerankDataset.loadJSONL(path: path, limit: 50, seed: 1)
        let b = try RerankDataset.loadJSONL(path: path, limit: 50, seed: 1)
        let c = try RerankDataset.loadJSONL(path: path, limit: 50, seed: 2)

        XCTAssertEqual(a.count, 50)
        XCTAssertEqual(a.map { $0.query }, b.map { $0.query },
                       "один сид дал разные выборки")
        XCTAssertNotEqual(a.map { $0.query }, c.map { $0.query },
                          "разные сиды дали одну выборку")

        let firstN = Set((0 ..< 50).map { "q\($0)" })
        XCTAssertNotEqual(Set(a.map { $0.query }), firstN,
                          "выборка совпала с первыми 50 — резервуар не работает")
        // И охват должен быть по всему файлу, а не по его началу.
        let indices = a.compactMap { Int($0.query.dropFirst()) }
        XCTAssertGreaterThan(indices.max() ?? 0, 400,
                             "в выборку не попало ничего из конца файла")
    }

    /// Без `limit` резервуар не включается и читается всё.
    func testNoLimitReadsEverything() throws {
        let lines = (0 ..< 30).map {
            #"{"query":"q\#($0)","positive":"p\#($0)","negatives":["n"]}"#
        }
        let path = try writeTemp(lines)
        defer { try? FileManager.default.removeItem(atPath: path) }
        XCTAssertEqual(try RerankDataset.loadJSONL(path: path).count, 30)
    }

    /// Настоящий срез reranker-triples-multi, если он собран.
    ///
    /// Проверяет то, чего синтетика не покажет: что реальный файл (25 языков,
    /// многобайтовые тексты, строка с одним негативом вместо пяти) читается
    /// и превращается в рабочие примеры.
    func testRealSliceLoads() throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = ProcessInfo.processInfo.environment["RWKV_RERANK_SLICE"]
            ?? home.appendingPathComponent(
                "Develop/SwiftRWKV/.testdata/reranker_slice.jsonl").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), """
            Нет среза reranker-triples-multi. Собрать:
              ~/Develop/tests/venv/bin/python Scripts/convert_reranker_triples.py \
              --src ~/Develop/reranker-triples-multi/data \
              --out .testdata/reranker_slice.jsonl --per-language 6 --max-chars 600
            """)

        let rows = try RerankDataset.loadJSONL(path: path)
        XCTAssertGreaterThan(rows.count, 100)
        XCTAssertGreaterThan(Set(rows.compactMap { $0.language }).count, 20,
                             "языков меньше двадцати — срез собран не тот")
        XCTAssertTrue(rows.allSatisfy { !$0.negatives.isEmpty })

        let (pool, samples) = try RerankCandidates.build(rows, nCandidates: 8, seed: 0)
        XCTAssertEqual(samples.count, rows.count)
        XCTAssertTrue(samples.allSatisfy { $0.docIds.count == 8 })
        // Строка с одним майненным негативом вместо пяти в срезе есть, и
        // она обязана пройти — просто с меньшим hardNegs.
        XCTAssertTrue(samples.contains { $0.hardNegs.count < 5 },
                      "в срезе нет строки с неполным набором негативов")
        XCTAssertTrue(samples.allSatisfy { !$0.hardNegs.isEmpty })
        XCTAssertGreaterThan(pool.count, rows.count)
    }
}
