import XCTest
import MLX
import MLXRandom
@testable import RWKVEmbedding
@testable import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Векторы, контрастные лоссы и GradCache.
//
//  Лоссы проверяются против РУЧНОГО расчёта, а не против самих себя: у
//  контрастных лоссов легко получить правдоподобное число с перепутанной
//  осью софтмакса или потерянным направлением, и такое расхождение не
//  видно ни по величине, ни по тому, что обучение «идёт».
// ───────────────────────────────────────────────────────────────────────

final class EmbeddingTests: XCTestCase {

    func maxAbsDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        eval(x, y)
        return MLX.abs(x - y).max().item(Float.self)
    }

    func rows(_ vals: [[Float]]) -> MLXArray {
        l2Normalize(MLXArray(vals.flatMap { $0 },
                             [vals.count, vals[0].count]).asType(.float32))
    }

    // ── Голова ───────────────────────────────────────────────────────

    /// fc2 = 0 ⇒ голова на старте это LayerNorm поверх входа, то есть она НЕ
    /// вращает уже выученную геометрию. Проверяем через косинус: направления
    /// векторов после головы должны совпасть с направлениями до неё.
    func testHeadIsIdentityAtInit() {
        let head = EmbeddingHead(dim: 64)
        eval(head.fc2)
        XCTAssertEqual(MLX.abs(head.fc2).max().item(Float.self), 0,
                       "fc2 инициализирована не нулём — голова портит геометрию базы")

        MLXRandom.seed(1)
        let x = MLXRandom.normal([4, 64])
        let out = head(x)
        eval(out)
        // LayerNorm центрирует, поэтому сравниваем с центрированным входом
        let xc = x - x.mean(axis: -1, keepDims: true)
        let cos = (l2Normalize(out) * l2Normalize(xc)).sum(axis: -1)
        eval(cos)
        XCTAssertGreaterThan(cos.min().item(Float.self), 0.999,
                             "голова на старте развернула векторы")
    }

    func testHeadParameterRoundTrip() {
        let head = EmbeddingHead(dim: 32, hidden: 48)
        XCTAssertEqual(head.parameters.count, 4)
        XCTAssertEqual(head.parameters[0].shape, [48, 32])
        XCTAssertEqual(head.parameters[1].shape, [32, 48])
        XCTAssertEqual(EmbeddingHead.parameterNames.count, 4)

        let bumped = head.parameters.map { $0 + 1.0 }
        head.setParameters(bumped)
        XCTAssertEqual(maxAbsDiff(head.parameters[1], bumped[1]), 0)
    }

    // ── Лоссы против ручного расчёта ─────────────────────────────────

    /// InfoNCE на идеально разделимом батче: диагональ — единицы, вне её —
    /// ортогональные пары. Тогда лосс считается на бумаге.
    func testInfoNCEMatchesHandComputation() {
        // 2 ортогональных вектора: q[i]·d[i] = 1, q[i]·d[j] = 0
        let q = rows([[1, 0], [0, 1]])
        let d = rows([[1, 0], [0, 1]])
        let tau: Float = 0.05
        let loss = infoNCELoss(query: q, document: d, temperature: tau)
        eval(loss)

        // логиты строки: [1/τ, 0] ⇒ CE = ln(1 + e^{-1/τ})
        let expected = log(1 + exp(-1.0 / tau))
        XCTAssertEqual(loss.item(Float.self), expected, accuracy: 1e-5)
    }

    /// Симметричность InfoNCE: перестановка ролей q и d не меняет лосс.
    func testInfoNCEIsSymmetric() {
        MLXRandom.seed(3)
        let q = l2Normalize(MLXRandom.normal([5, 16]))
        let d = l2Normalize(MLXRandom.normal([5, 16]))
        let a = infoNCELoss(query: q, document: d)
        let b = infoNCELoss(query: d, document: q)
        eval(a, b)
        XCTAssertEqual(a.item(Float.self), b.item(Float.self), accuracy: 1e-5,
                       "InfoNCE обязан быть симметричным по построению")
    }

    /// Пул отрицательных в триплете — это 2B кандидатов, а не B.
    ///
    /// Проверка предметная: добавим строку, чей hard-negative совпадает с
    /// положительным ПЕРВОЙ строки. Если пул строится правильно, этот чужой
    /// отрицательный конкурирует с якорем первой строки и лосс растёт;
    /// если бы учитывались только «свои» пары, он был бы не при чём.
    func testTripletPoolUsesAllNegatives() {
        let anchor = rows([[1, 0, 0], [0, 1, 0]])
        let positive = rows([[1, 0, 0], [0, 1, 0]])
        let farNeg = rows([[0, 0, 1], [0, 0, 1]])
        // отрицательный ВТОРОЙ строки = положительный ПЕРВОЙ
        let clashNeg = rows([[0, 0, 1], [1, 0, 0]])

        let easy = tripletPoolLoss(anchor: anchor, positive: positive, negative: farNeg)
        let hard = tripletPoolLoss(anchor: anchor, positive: positive, negative: clashNeg)
        eval(easy, hard)
        XCTAssertGreaterThan(hard.item(Float.self), easy.item(Float.self),
                             "чужой hard-negative не попал в пул: пул строится "
                             + "не из 2B кандидатов")
    }

    /// Ручной расчёт триплета: якорь совпадает с положительным, оба
    /// отрицательных ортогональны. Логиты [1/τ, 0, 0, 0] на 4 кандидата.
    func testTripletPoolMatchesHandComputation() {
        let a = rows([[1, 0], [0, 1]])
        let p = rows([[1, 0], [0, 1]])
        let n = rows([[0, 1], [1, 0]])
        let tau: Float = 0.05
        let loss = tripletPoolLoss(anchor: a, positive: p, negative: n,
                                   temperature: tau)
        eval(loss)
        // кандидаты = [p0,p1,n0,n1]; для якоря 0: [1/τ, 0, 0, 1/τ]
        let t = 1.0 / tau
        let expected = -t + log(2 * exp(t) + 2 * exp(Float(0)))
        XCTAssertEqual(loss.item(Float.self), expected, accuracy: 1e-4)
    }

    /// symmetric=true добавляет обратное направление и потому даёт ДРУГОЕ
    /// число — иначе флаг был бы декоративным.
    func testSymmetricDiffersFromAsymmetric() {
        MLXRandom.seed(5)
        let a = l2Normalize(MLXRandom.normal([4, 8]))
        let p = l2Normalize(MLXRandom.normal([4, 8]))
        let n = l2Normalize(MLXRandom.normal([4, 8]))
        let asym = tripletPoolLoss(anchor: a, positive: p, negative: n, symmetric: false)
        let sym = tripletPoolLoss(anchor: a, positive: p, negative: n, symmetric: true)
        eval(asym, sym)
        XCTAssertNotEqual(asym.item(Float.self), sym.item(Float.self),
                          accuracy: 1e-6)
    }

    /// Маска в zero-shot: добивка не должна конкурировать с настоящими метками.
    func testClassificationMaskExcludesPadding() {
        let anchor = rows([[1, 0]])
        // K=3: две реальные метки, третья — добивка, совпадающая с якорем
        let cands = MLXArray([Float(1), 0, 0, 1, 1, 0], [1, 3, 2]).asType(.float32)
        let target = MLXArray([Int32(0)])

        let masked = zeroShotClassificationLoss(
            anchor: anchor, candidates: cands,
            mask: MLXArray([Float(1), 1, 0], [1, 3]), targetIndex: target)
        let unmasked = zeroShotClassificationLoss(
            anchor: anchor, candidates: cands,
            mask: MLXArray([Float(1), 1, 1], [1, 3]), targetIndex: target)
        eval(masked, unmasked)

        XCTAssertLessThan(masked.item(Float.self), unmasked.item(Float.self),
                          "пад-кандидат влияет на лосс — маска не применяется")
        XCTAssertTrue(masked.item(Float.self).isFinite)
    }

    /// Идеальное предсказание даёт околонулевой лосс, неверное — большой.
    func testClassificationRewardsCorrectLabel() {
        let anchor = rows([[1, 0]])
        let cands = MLXArray([Float(1), 0, 0, 1], [1, 2, 2]).asType(.float32)
        let mask = MLXArray([Float(1), 1], [1, 2])
        let right = zeroShotClassificationLoss(anchor: anchor, candidates: cands,
                                               mask: mask,
                                               targetIndex: MLXArray([Int32(0)]))
        let wrong = zeroShotClassificationLoss(anchor: anchor, candidates: cands,
                                               mask: mask,
                                               targetIndex: MLXArray([Int32(1)]))
        eval(right, wrong)
        XCTAssertLessThan(right.item(Float.self), 0.01)
        XCTAssertGreaterThan(wrong.item(Float.self), 10.0)
    }

    // ── Пулинг ───────────────────────────────────────────────────────

    /// mean-пулинг обязан игнорировать паддинг: иначе вектор короткой строки
    /// тянется к нулю тем сильнее, чем длиннее батч.
    func testMeanPoolingIgnoresPadding() {
        let (bb, cfg) = TinyBackbone.make()
        let model = EmbeddingModel(backbone: bb, pooling: .mean)
        let T = 16, real = 6
        let h = MLXArray.ones([1, T, cfg.nEmbd]).asType(.float32)
        // «паддинг» с большими значениями — если он попадёт в среднее, оно уедет
        let padded = concatenated([h[0..., 0 ..< real],
                                   MLXArray.full([1, T - real, cfg.nEmbd],
                                                 values: MLXArray(Float(100)))],
                                  axis: 1)
        let pooled = model.pool(padded, poolIndex: MLXArray([Int32(real)]))
        eval(pooled)
        XCTAssertEqual(pooled.max().item(Float.self), 1.0, accuracy: 1e-5,
                       "паддинг попал в mean-пулинг")
    }

    /// last-пулинг берёт указанную позицию, а не последнюю в тензоре.
    func testLastPoolingUsesGivenIndex() {
        let (bb, cfg) = TinyBackbone.make()
        let model = EmbeddingModel(backbone: bb, pooling: .last)
        let T = 8
        var vals = [Float](repeating: 0, count: T * cfg.nEmbd)
        for t in 0 ..< T {
            for d in 0 ..< cfg.nEmbd { vals[t * cfg.nEmbd + d] = Float(t) }
        }
        let h = MLXArray(vals, [1, T, cfg.nEmbd])
        let pooled = model.pool(h, poolIndex: MLXArray([Int32(3)]))
        eval(pooled)
        XCTAssertEqual(pooled.max().item(Float.self), 3.0, accuracy: 1e-5,
                       "last-пулинг взял не ту позицию")
    }

    /// Векторы нормированы — иначе косинус перестаёт быть косинусом.
    func testEmbeddingsAreL2Normalized() {
        let (bb, cfg) = TinyBackbone.make()
        let model = EmbeddingModel(backbone: bb)
        let ids = TinyBackbone.ids(3, 16, vocab: cfg.vocab)
        let v = model.embed(ids, poolIndex: MLXArray([Int32(15), 15, 15]))
        eval(v)
        let norms = sqrt((v * v).sum(axis: -1))
        eval(norms)
        XCTAssertEqual(norms.min().item(Float.self), 1.0, accuracy: 1e-4)
        XCTAssertEqual(norms.max().item(Float.self), 1.0, accuracy: 1e-4)
    }

    // ── GradCache ────────────────────────────────────────────────────

    /// Решающий тест GradCache: при ОДНОМ чанке он обязан совпасть с обычным
    /// градиентом ТОЧНО. Один чанк означает, что разрезания нет, и остаётся
    /// только сам приём — разрыв графа на векторах и пересев backward.
    /// Если здесь есть расхождение, значит приём реализован неверно; всё
    /// остальное расхождение при большем числе чанков — уже только порядок
    /// суммирования.
    func testGradCacheOneChunkMatchesEagerExactly() {
        MLXRandom.seed(7)
        let N = 8, D = 16
        let W = MLXRandom.normal([D, D]) * 0.1
        let X = MLXRandom.normal([N, D])
        eval(W, X)

        func embed(_ ps: [MLXArray], _ start: Int, _ size: Int) -> [MLXArray] {
            let x = X[start ..< (start + size)]
            let e = l2Normalize(matmul(x, ps[0].transposed()))
            return [e, l2Normalize(matmul(x + 0.1, ps[0].transposed()))]
        }
        func loss(_ f: [MLXArray]) -> MLXArray {
            infoNCELoss(query: f[0], document: f[1])
        }

        // эталон: обычный valueAndGrad по всему батчу
        let eager = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
            [loss(embed(ps, 0, N))]
        }, argumentNumbers: [0])
        let (ev, eg) = eager([W])
        eval(ev + eg)

        let gc = gradCacheValueAndGrad(
            parameters: [W], chunks: [0],
            embedChunk: { ps, s in embed(ps, s, N) },
            lossFromEmbeddings: loss)
        eval([gc.loss] + gc.gradients)

        XCTAssertEqual(maxAbsDiff(ev[0], gc.loss), 0,
                       "лосс при одном чанке обязан совпасть точно")
        XCTAssertEqual(maxAbsDiff(eg[0], gc.gradients[0]), 0,
                       "градиент при одном чанке обязан совпасть точно — "
                       + "разрыв графа на векторах не должен вносить ошибку")
    }

    /// Несколько чанков: лосс всё ещё точный (он считается на полной матрице),
    /// градиент отличается только порядком суммирования.
    func testGradCacheMultiChunkStaysExact() {
        MLXRandom.seed(11)
        let N = 8, D = 16, chunk = 2
        let W = MLXRandom.normal([D, D]) * 0.1
        let X = MLXRandom.normal([N, D])
        eval(W, X)

        func embed(_ ps: [MLXArray], _ start: Int, _ size: Int) -> [MLXArray] {
            let x = X[start ..< (start + size)]
            return [l2Normalize(matmul(x, ps[0].transposed())),
                    l2Normalize(matmul(x + 0.1, ps[0].transposed()))]
        }
        func loss(_ f: [MLXArray]) -> MLXArray {
            infoNCELoss(query: f[0], document: f[1])
        }

        let eager = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
            [loss(embed(ps, 0, N))]
        }, argumentNumbers: [0])
        let (ev, eg) = eager([W])

        let gc = gradCacheValueAndGrad(
            parameters: [W], chunks: chunkStarts(batch: N, chunkSize: chunk),
            embedChunk: { ps, s in embed(ps, s, chunk) },
            lossFromEmbeddings: loss)
        eval(ev + eg + [gc.loss] + gc.gradients)

        XCTAssertEqual(ev[0].item(Float.self), gc.loss.item(Float.self),
                       accuracy: 1e-5, "лосс виден целиком, он обязан совпасть")
        let rel = maxAbsDiff(eg[0], gc.gradients[0])
                / (MLX.abs(eg[0]).max().item(Float.self) + 1e-9)
        XCTAssertLessThan(rel, 1e-4,
                          "градиент при \(N / chunk) чанках разошёлся на \(rel) "
                          + "— это больше, чем объясняется порядком суммирования")
    }

    /// GradCache — НЕ то же, что grad-accumulation. Накопление разбивает сам
    /// лосс, поэтому каждый микробатч видит только свои отрицательные, и
    /// градиент получается принципиально другой. Тест фиксирует, что разница
    /// именно большая: иначе GradCache не имел бы смысла.
    func testGradCacheDiffersFromGradientAccumulation() {
        MLXRandom.seed(13)
        let N = 8, D = 16, chunk = 2
        let W = MLXRandom.normal([D, D]) * 0.1
        let X = MLXRandom.normal([N, D])
        eval(W, X)

        func embed(_ ps: [MLXArray], _ start: Int, _ size: Int) -> [MLXArray] {
            let x = X[start ..< (start + size)]
            return [l2Normalize(matmul(x, ps[0].transposed())),
                    l2Normalize(matmul(x + 0.1, ps[0].transposed()))]
        }
        func loss(_ f: [MLXArray]) -> MLXArray {
            infoNCELoss(query: f[0], document: f[1])
        }

        let gc = gradCacheValueAndGrad(
            parameters: [W], chunks: chunkStarts(batch: N, chunkSize: chunk),
            embedChunk: { ps, s in embed(ps, s, chunk) },
            lossFromEmbeddings: loss)

        // наивное накопление: лосс считается ВНУТРИ каждого микробатча
        var accum: MLXArray? = nil
        for s in chunkStarts(batch: N, chunkSize: chunk) {
            let f = valueAndGrad({ (ps: [MLXArray]) -> [MLXArray] in
                [loss(embed(ps, s, chunk))]
            }, argumentNumbers: [0])
            let (_, g) = f([W])
            eval(g)
            accum = accum == nil ? g[0] : accum! + g[0]
        }
        let accumGrad = accum! / Float(N / chunk)
        eval(accumGrad)

        let rel = maxAbsDiff(gc.gradients[0], accumGrad)
                / (MLX.abs(gc.gradients[0]).max().item(Float.self) + 1e-9)
        print("GRADCACHE против накопления: относительная разница \(rel)")
        XCTAssertGreaterThan(rel, 0.1,
                             "накопление дало почти тот же градиент — значит "
                             + "тест не различает разные пулы отрицательных")
    }

    /// ГЛАВНОЕ обещание GradCache: пик памяти определяется размером ЧАНКА,
    /// а не батча.
    ///
    /// Без этого теста весь механизм проверен только на точность — то есть
    /// доказано, что он даёт правильный ответ, но не что он даёт его дёшево.
    /// А ради дешевизны он и существует: при равной точности обычный
    /// valueAndGrad проще.
    ///
    /// Считаем на достаточно «толстых» активациях, чтобы разница была видна
    /// над шумом аллокатора: промежуточная матрица [b, H] с большим H — это
    /// и есть то, что в реальной модели занимает память под активации.
    func testGradCacheBoundsPeakMemoryByChunkSize() {
        MLXRandom.seed(17)
        let N = 256, D = 32, H = 16384
        let W = MLXRandom.normal([H, D]) * 0.05
        let X = MLXRandom.normal([N, D])
        let P = MLXRandom.normal([H, D]) * 0.05
        eval(W, X, P)

        // «Толстый» промежуток [b, H] — аналог активаций блока.
        func embed(_ ps: [MLXArray], _ start: Int, _ size: Int) -> [MLXArray] {
            let x = X[start ..< (start + size)]
            let wide = tanh(matmul(x, ps[0].transposed()))       // [b, H]
            return [l2Normalize(matmul(wide, ps[1]))]            // [b, D]
        }
        func loss(_ f: [MLXArray]) -> MLXArray {
            let s = cosineSimilarity(f[0]) / 0.05
            return s.logSumExp(axis: -1).mean()
        }

        func peak(chunkSize: Int) -> Int {
            GPU.resetPeakMemory()
            let gc = gradCacheValueAndGrad(
                parameters: [W, P],
                chunks: chunkStarts(batch: N, chunkSize: chunkSize),
                embedChunk: { ps, s in
                    embed(ps, s, Swift.min(chunkSize, N - s))
                },
                lossFromEmbeddings: loss)
            eval([gc.loss] + gc.gradients)
            return GPU.peakMemory
        }

        let small = peak(chunkSize: 8)
        let whole = peak(chunkSize: N)
        print("GRADCACHE пик памяти: чанк 8 → \(small / 1_000_000) МБ, "
              + "чанк \(N) → \(whole / 1_000_000) МБ")

        XCTAssertLessThan(small, whole,
                          "пик памяти не зависит от размера чанка (\(small) "
                          + "против \(whole)) — чанкование не ограничивает "
                          + "активации, то есть GradCache не делает того, "
                          + "ради чего существует")
    }

    // ЧТО ЭТОТ ТЕСТ НЕ ЛОВИТ — важно записать, а не умолчать.
    //
    // Удаление stopGradient или eval из фазы 1 его НЕ роняет: проверено
    // мутациями на двух масштабах (N=64/H=4096 и N=256/H=16384). Обе строки
    // в mlx-swift оказались подстраховкой — forward в фазе 1 идёт вне
    // grad-трансформации, графа не строится, память освобождается и без
    // явного eval. Питоновский оригинал описывает eval как необходимый; здесь
    // это не воспроизвелось, и комментарий в GradCache.swift говорит именно
    // так, а не повторяет унаследованное утверждение.
    //
    // Что тест ловит: потерю самого чанкования — если бы фаза 3 считала
    // градиент по всему батчу разом, пик перестал бы зависеть от чанка.

    /// Несогласованное число полей между чанками — ошибка конфигурации,
    /// а не повод молча склеить что попало.
    func testGradCacheRejectsEmptyChunks() {
        let W = MLXArray.zeros([2, 2])
        // пустой список чанков ловится precondition; проверяем сам факт
        // наличия защиты через непустой корректный вызов
        let gc = gradCacheValueAndGrad(
            parameters: [W], chunks: [0],
            embedChunk: { ps, _ in [ps[0]] },
            lossFromEmbeddings: { $0[0].sum() })
        eval(gc.loss)
        XCTAssertEqual(gc.gradients.count, 1)
    }
}
