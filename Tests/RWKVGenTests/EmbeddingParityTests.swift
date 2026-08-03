import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVEmbedding

// ───────────────────────────────────────────────────────────────────────
//  Паритет эмбеддинг-пути с Python (rwkv-metal) на реальной 0.1B.
//
//  Паритет бэкбона (X070ParityTests) доказывает, что база перенесена верно,
//  и ничего не говорит о том, что построено ПОВЕРХ неё: пулинг по poolIndex,
//  голова, L2-нормировка, температура, направление контрастного лосса,
//  добивка пула кандидатов маской. Каждое из этих мест переносится
//  внутренне непротиворечиво и при этом неверно — и тогда все структурные
//  тесты остаются зелёными, потому что сравнивать им не с чем.
//
//  Веса головы в эталоне РАНДОМИЗИРОВАНЫ (штатный fc2 = 0 делает голову
//  тождеством, и ошибка в fc1/fc2 прошла бы мимо) и загружаются отсюда,
//  а не генерируются заново: два RNG совпасть не обязаны.
//
//  Фикстура:
//      cd ~/Develop/rwkv-metal
//      .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_embedding_reference.py \
//          --model world_0.1b_x070.safetensors \
//          --slice ~/Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl \
//          --out   ~/Develop/SwiftRWKV/.testdata/embedding_ref.safetensors
//
//  Пути: RWKV_PARITY_MODEL, RWKV_EMBEDDING_REFERENCE, RWKV_LITRETRIEVAL_SLICE,
//  RWKV_WORLD_VOCAB. Без фикстур тест ПРОПУСКАЕТСЯ.
// ───────────────────────────────────────────────────────────────────────

final class EmbeddingParityTests: XCTestCase {

    struct Fixtures {
        let model: EmbeddingModel
        let tokenizer: WorldTokenizer
        let ref: [String: MLXArray]
        let rows: [EmbeddingSample]
    }

    /// Допуск для сквозных величин.
    ///
    /// 1% — та же граница, что у X070ParityTests: база считается в bf16, и
    /// эмбеддинг-путь наследует её шум, ничего к нему принципиально не
    /// добавляя (голова — два матмула и LayerNorm поверх уже огрублённого
    /// выхода). Ужесточать её здесь значило бы требовать от надстройки
    /// точности, которой нет у основания.
    let tolerance: Float = 0.01

    func relativeDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32), y = b.asType(.float32)
        let diff = MLX.abs(x - y).max().item(Float.self)
        let scale = MLX.abs(y).max().item(Float.self)
        return diff / Swift.max(scale, 1e-6)
    }

    func loadFixtures() throws -> Fixtures? {
        let env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser
        let modelPath = env["RWKV_PARITY_MODEL"]
            ?? home.appendingPathComponent("Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let refPath = env["RWKV_EMBEDDING_REFERENCE"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/embedding_ref.safetensors").path
        let slicePath = env["RWKV_LITRETRIEVAL_SLICE"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl").path
        let vocabPath = env["RWKV_WORLD_VOCAB"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path

        let fm = FileManager.default
        for p in [modelPath, refPath, slicePath, vocabPath] where !fm.fileExists(atPath: p) {
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
        let bb = X070Backbone(weights: weights, cfg: cfg)

        // Голова — из эталона, а не сгенерированная заново.
        let head = EmbeddingHead(dim: cfg.nEmbd)
        head.setParameters([ref["head/fc1"]!.asType(.float32),
                            ref["head/fc2"]!.asType(.float32),
                            ref["head/norm.weight"]!.asType(.float32),
                            ref["head/norm.bias"]!.asType(.float32)])
        eval(head.parameters)

        guard let tok = WorldTokenizer(vocabURL: URL(fileURLWithPath: vocabPath)) else {
            return nil
        }
        let rows = try EmbeddingDataset.loadJSONL(path: slicePath)

        return Fixtures(model: EmbeddingModel(backbone: bb, head: head),
                        tokenizer: tok, ref: ref, rows: rows)
    }

    func require() throws -> Fixtures {
        let f = try loadFixtures()
        try XCTSkipIf(f == nil, """
            Нет фикстур паритета эмбеддингов — тест пропущен. Чтобы включить:
              cd ~/Develop/rwkv-metal
              .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_embedding_reference.py \\
                  --model world_0.1b_x070.safetensors \\
                  --slice ~/Develop/SwiftRWKV/.testdata/litretrieval_slice.jsonl \\
                  --out   ~/Develop/SwiftRWKV/.testdata/embedding_ref.safetensors
            """)
        return f!
    }

    static let maxChars = 800
    static let temperature: Float = 0.05

    func cut(_ s: String) -> String { String(s.prefix(Self.maxChars)) }

    // ── Токенизация ──────────────────────────────────────────────────

    /// Отдельно от модели: расхождение токенизатора и расхождение численности —
    /// разные дефекты, чинятся в разных местах, и смешивать их в одном
    /// утверждении значит потом гадать, какое из двух сработало.
    func testTokenizationMatchesReference() throws {
        let f = try require()
        let tri = f.rows.filter { $0.task == .retrieval }.prefix(8)

        for (field, texts) in [("a", tri.map { self.cut($0.anchor) }),
                               ("p", tri.map { self.cut($0.positive) }),
                               ("n", tri.map { self.cut($0.negative) })] {
            let (idx, pool) = encodeBatch(tokenizer: f.tokenizer, texts: texts,
                                          terminator: 0, maxTokens: nil)
            eval(idx, pool)
            let refIdx = f.ref["triplet/\(field)_idx"]!
            let refPool = f.ref["triplet/\(field)_pool"]!
            XCTAssertEqual(idx.shape, refIdx.shape,
                           "форма токенов поля \(field) разошлась")
            XCTAssertEqual(relativeDiff(idx, refIdx), 0,
                           "токенизация поля \(field) разошлась с Python")
            XCTAssertEqual(relativeDiff(pool, refPool), 0,
                           "позиция пулинга поля \(field) разошлась")
        }
    }

    // ── Векторы ──────────────────────────────────────────────────────

    /// Сырой пулинг без головы — отделяет дефект головы от дефекта пулинга.
    ///
    /// Меряется КОСИНУСОМ, а не покомпонентной относительной разницей.
    /// Причина не в мягкости: после L2-нормировки максимальная компонента
    /// вектора размерности 768 — величина порядка 0.1, и деление на неё
    /// раздувает ту же абсолютную ошибку примерно вдесятеро против того же
    /// измерения на ненормированном hidden (там 0.57%). Для эмбеддинга
    /// значимо направление — им и меряем; покомпонентная граница оставлена
    /// вторым, более грубым утверждением.
    func testPooledVectorBeforeHeadMatches() throws {
        let f = try require()
        let idx = f.ref["triplet/a_idx"]!.asType(.int32)
        let pool = f.ref["triplet/a_pool"]!.asType(.int32)
        let got = f.model.pooledOnly(idx, poolIndex: pool).asType(.float32)
        let want = f.ref["triplet/a_pooled_raw"]!.asType(.float32)
        let wantNorm = want / sqrt((want * want).sum(axis: -1, keepDims: true) + 1e-12)
        eval(got, wantNorm)

        // Основное утверждение — РАЗЛИЧАЮЩЕЕ, а не пороговое: каждый вектор
        // обязан быть ближе к СВОЕМУ эталону, чем к любому чужому.
        // Ослаблением допуска такой тест не пройти, поэтому он не зависит от
        // того, насколько удачно угадана граница шума.
        //
        // Задача при этом не тривиальна, и в этом весь смысл. Сырые векторы
        // базы АНИЗОТРОПНЫ: попарные косинусы между разными литературными
        // отрывками здесь 0.99…0.996, то есть все восемь кандидатов лежат в
        // узком конусе. Опознать среди них свой 8 раз из 8 при собственном
        // косинусе 0.9999 — содержательное утверждение; сломанная реализация
        // с таким набором кандидатов промахнулась бы.
        //
        // Попутно это и объясняет, зачем вообще контрастное дообучение:
        // именно этот конус оно и разжимает (ср. метрики sts в эталоне —
        // cos+ 0.939 против cos− 0.933).
        let cross = matmul(got, wantNorm.transposed())          // [B,B]
        eval(cross)
        let B = cross.shape[0]
        let flat = cross.asArray(Float.self)
        var worstGap = Float.infinity
        for i in 0 ..< B {
            let own = flat[i * B + i]
            var bestOther = -Float.infinity
            for j in 0 ..< B where j != i { bestOther = Swift.max(bestOther, flat[i * B + j]) }
            worstGap = Swift.min(worstGap, own - bestOther)
            XCTAssertGreaterThan(own, bestOther,
                                 "строка \(i): свой эталон \(own) проиграл чужому "
                                 + "\(bestOther) — пулинг перепутал строки")
        }
        XCTAssertGreaterThan(worstGap, 0,
                             "ни одна строка не должна опознаваться вничью")

        // Числовая граница — вторым слоем, и она ИЗМЕРЕНА, а не назначена.
        // Замер: худший косинус 0.99985 на последовательностях до 259 токенов.
        // Это согласуется с bf16: паритет hidden даёт 0.57% покомпонентно на
        // 32 токенах, на 259 накапливается около 1.7%, а вектор размерности
        // 768 с независимыми покомпонентными ошибками ε даёт косинус ≈ 1−ε²/2,
        // то есть как раз ~1e-4. Порог 0.999 — на порядок ниже измеренного,
        // чтобы ловить настоящий сдвиг, но не срабатывать на шум bf16.
        let cos = (got * wantNorm).sum(axis: -1)
        eval(cos)
        let worst = Double(cos.min().item(Float.self))
        XCTAssertGreaterThan(worst, 0.999,
                             "направление пулинга разошлось: худший косинус \(worst)")
        XCTAssertLessThan(relativeDiff(got, wantNorm), 0.02)
    }

    func testEmbeddingsMatchReference() throws {
        let f = try require()
        for field in ["a", "p", "n"] {
            let idx = f.ref["triplet/\(field)_idx"]!.asType(.int32)
            let pool = f.ref["triplet/\(field)_pool"]!.asType(.int32)
            let got = f.model.embed(idx, poolIndex: pool)
            eval(got)
            let d = relativeDiff(got, f.ref["triplet/\(field)_emb"]!)
            XCTAssertLessThan(d, tolerance,
                              "векторы поля \(field) разошлись на \(d)")
        }
    }

    // ── Лоссы ────────────────────────────────────────────────────────

    /// Лосс чувствителен к тому, чего векторы не показывают: к температуре,
    /// к тому, что пул отрицательных склеен из положительных И отрицательных,
    /// и к направлению (retrieval односторонний, sts двусторонний).
    func testRetrievalAndSTSLossesMatch() throws {
        let f = try require()
        let batch = TripletBatch(
            anchorIdx: f.ref["triplet/a_idx"]!.asType(.int32),
            anchorPool: f.ref["triplet/a_pool"]!.asType(.int32),
            positiveIdx: f.ref["triplet/p_idx"]!.asType(.int32),
            positivePool: f.ref["triplet/p_pool"]!.asType(.int32),
            negativeIdx: f.ref["triplet/n_idx"]!.asType(.int32),
            negativePool: f.ref["triplet/n_pool"]!.asType(.int32))

        let ret = EmbeddingObjective.retrievalLoss(f.model, batch, temperature: Self.temperature)
        let sts = EmbeddingObjective.stsLoss(f.model, batch, temperature: Self.temperature)
        eval(ret, sts)

        let refRet = f.ref["loss/retrieval"]!.item(Float.self)
        let refSts = f.ref["loss/sts"]!.item(Float.self)
        XCTAssertEqual(ret.item(Float.self), refRet, accuracy: refRet * tolerance,
                       "лосс retrieval разошёлся")
        XCTAssertEqual(sts.item(Float.self), refSts, accuracy: refSts * tolerance,
                       "лосс sts разошёлся")
        XCTAssertNotEqual(refRet, refSts, accuracy: 1e-4,
                          "эталон обязан различать односторонний и двусторонний лосс; "
                          + "если они совпали, тест выше ничего не проверяет")
    }

    /// Классификация: проверяет ещё и маску добивки — при K, различающемся
    /// между строками, пад-позиции обязаны быть выключены до софтмакса.
    func testClassificationLossMatches() throws {
        let f = try require()
        let batch = ClassificationBatch(
            anchorIdx: f.ref["cls/a_idx"]!.asType(.int32),
            anchorPool: f.ref["cls/a_pool"]!.asType(.int32),
            candidateIdx: f.ref["cls/c_idx"]!.asType(.int32),
            candidatePool: f.ref["cls/c_pool"]!.asType(.int32),
            mask: f.ref["cls/mask"]!.asType(.float32),
            targetIndex: f.ref["cls/target"]!.asType(.int32))

        let loss = EmbeddingObjective.classificationLoss(f.model, batch,
                                                         temperature: Self.temperature)
        eval(loss)
        let want = f.ref["loss/classification"]!.item(Float.self)
        XCTAssertEqual(loss.item(Float.self), want, accuracy: want * tolerance,
                       "лосс классификации разошёлся")
    }

    // ── Метрики ──────────────────────────────────────────────────────

    func testRetrievalMetricsMatch() throws {
        let f = try require()
        let rows = f.rows.filter { $0.task == .retrieval }
        let m = EmbeddingMetrics.evaluateRetrieval(
            model: f.model, tokenizer: f.tokenizer, rows: rows,
            maxChars: Self.maxChars, maxTokens: nil)

        let want = f.ref["metric/retrieval"]!.asArray(Float.self)
        XCTAssertEqual(m.count, Int(want[5]))
        XCTAssertEqual(m.mrr, Double(want[0]), accuracy: 0.01, "MRR разошёлся")
        XCTAssertEqual(m.recall[1]!, Double(want[1]), accuracy: 0.05)
        XCTAssertEqual(m.recall[5]!, Double(want[2]), accuracy: 0.05)
        XCTAssertEqual(m.recall[10]!, Double(want[3]), accuracy: 0.05)
        XCTAssertEqual(m.ndcg10, Double(want[4]), accuracy: 0.01, "nDCG@10 разошёлся")
    }

    func testSTSMetricsMatch() throws {
        let f = try require()
        let rows = f.rows.filter { $0.task == .sts }
        let m = EmbeddingMetrics.evaluateSTS(
            model: f.model, tokenizer: f.tokenizer, rows: rows,
            maxChars: Self.maxChars, maxTokens: nil)

        let want = f.ref["metric/sts"]!.asArray(Float.self)
        XCTAssertEqual(m.count, Int(want[3]))
        XCTAssertEqual(m.accuracy, Double(want[0]), accuracy: 0.05,
                       "попарная точность разошлась")
        XCTAssertEqual(m.meanSimilarityPositive, Double(want[1]), accuracy: 0.01)
        XCTAssertEqual(m.meanSimilarityNegative, Double(want[2]), accuracy: 0.01)
    }

    /// Классификация на полном пуле сравнивается ПО ПРЕДСКАЗАНИЯМ, а не по
    /// точности: с необученной головой точность вырождена (в эталоне 0.0), и
    /// такое же число выдаст любая сломанная реализация. Вектор из сорока
    /// дискретных выборов вырожденным не бывает.
    func testClassificationPredictionsMatch() throws {
        let f = try require()
        let rows = f.rows.filter { $0.task == .classification }
        let anchors = rows.compactMap { r -> String? in
            let l = r.positive.trimmingCharacters(in: .whitespacesAndNewlines)
            return ClassificationLabels.pool.contains(l) ? cut(r.anchor) : nil
        }
        let targets = rows.compactMap { r -> Int? in
            ClassificationLabels.pool.firstIndex(
                of: r.positive.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        let a = EmbeddingMetrics.embedAll(model: f.model, tokenizer: f.tokenizer,
                                          texts: anchors, maxTokens: nil).asType(.float32)
        let l = EmbeddingMetrics.embedAll(model: f.model, tokenizer: f.tokenizer,
                                          texts: ClassificationLabels.pool,
                                          maxTokens: nil).asType(.float32)
        let pred = argMax(matmul(a, l.transposed()), axis: -1).asType(.int32)
        eval(pred)

        let refPred = f.ref["metric/classification_predictions"]!.asType(.int32)
        let refTargets = f.ref["metric/classification_targets"]!.asType(.int32)

        XCTAssertEqual(targets.map { Int32($0) }, refTargets.asArray(Int32.self),
                       "разметка разошлась — значит пул меток в Swift и Python разный")

        let got = pred.asArray(Int32.self)
        let want = refPred.asArray(Int32.self)
        XCTAssertEqual(got.count, want.count)
        let agree = zip(got, want).filter { $0 == $1 }.count
        // Не требуем ПОЛНОГО совпадения: argmax по 25 почти одинаковым
        // косинусам — величина неустойчивая, и один-два переворота на
        // границе объясняются шумом bf16, а не расхождением реализаций.
        XCTAssertGreaterThan(Double(agree) / Double(got.count), 0.9,
                             "совпало лишь \(agree) из \(got.count) предсказаний")

        let metrics = EmbeddingMetrics.evaluateClassification(
            model: f.model, tokenizer: f.tokenizer, rows: rows,
            maxChars: Self.maxChars, useFullPool: true, maxTokens: nil)
        let wantAcc = f.ref["metric/classification_full_pool"]!.asArray(Float.self)
        XCTAssertEqual(metrics.candidatesPerRow, Double(wantAcc[2]), accuracy: 1e-9)
        XCTAssertEqual(metrics.count, Int(wantAcc[1]))
        XCTAssertEqual(metrics.accuracy, Double(wantAcc[0]), accuracy: 0.05)
    }
}
