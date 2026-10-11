import Foundation
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  ЧЕМ меряется — вторая ось обучения (первая, «что обучается», в
//  EmbeddingTrainable, третья, «откуда данные», в EmbeddingDataset).
//
//  Порт rwkv_metal/embedding/tasks.py: склейка между `EmbeddingModel.embed`
//  и чистыми лоссами из ContrastiveLoss под формы, которые отдают батчеры.
//
//  Три задачи различаются НЕ силой, а видом отношения:
//    retrieval — запрос и отвечающий документ, отношение несимметрично;
//    sts       — два текста в симметричном отношении похожести;
//    classification — метка эмбеддится как обычный текст, пул кандидатов
//                     свой у каждой строки.
//  Отсюда symmetric=false / true и отдельный лосс у третьей.
// ───────────────────────────────────────────────────────────────────────

public enum EmbeddingObjective {

    // ── Eager-путь ───────────────────────────────────────────────────

    /// Retrieval: якорь → кандидат, пул = все положительные и отрицательные
    /// батча (2B на якорь).
    public static func retrievalLoss(_ model: EmbeddingModel, _ batch: TripletBatch,
                                     temperature: Float = 0.05) -> MLXArray {
        tripletLoss(model, batch, temperature: temperature, symmetric: false)
    }

    /// STS: то же, но в обе стороны — обе стороны пары равноправны.
    public static func stsLoss(_ model: EmbeddingModel, _ batch: TripletBatch,
                               temperature: Float = 0.05) -> MLXArray {
        tripletLoss(model, batch, temperature: temperature, symmetric: true)
    }

    static func tripletLoss(_ model: EmbeddingModel, _ batch: TripletBatch,
                            temperature: Float, symmetric: Bool) -> MLXArray {
        let a = model.embed(batch.anchorIdx, poolIndex: batch.anchorPool)
        let p = model.embed(batch.positiveIdx, poolIndex: batch.positivePool)
        let n = model.embed(batch.negativeIdx, poolIndex: batch.negativePool)
        return tripletPoolLoss(anchor: a, positive: p, negative: n,
                               temperature: temperature, symmetric: symmetric)
    }

    /// Zero-shot классификация: кандидаты прогоняются той же моделью, что и
    /// якорь. Никакой обученной классификационной головы — поэтому набор
    /// меток не обязан быть фиксированным, и каждая строка предъявляет свой.
    public static func classificationLoss(_ model: EmbeddingModel,
                                          _ batch: ClassificationBatch,
                                          temperature: Float = 0.05) -> MLXArray {
        let anchor = model.embed(batch.anchorIdx, poolIndex: batch.anchorPool)  // [B,D]
        let B = batch.candidateIdx.shape[0]
        let K = batch.candidateIdx.shape[1]
        let T = batch.candidateIdx.shape[2]
        let flat = model.embed(batch.candidateIdx.reshaped([B * K, T]),
                               poolIndex: batch.candidatePool.reshaped([B * K]))
        let candidates = flat.reshaped([B, K, -1])                              // [B,K,D]
        return zeroShotClassificationLoss(anchor: anchor, candidates: candidates,
                                          mask: batch.mask,
                                          targetIndex: batch.targetIndex,
                                          temperature: temperature)
    }

    // ── Обратный проход по блокам ────────────────────────────────────
    //
    // Тот же градиент, что у eager-пути, при памяти одного блока одной пачки
    // (см. RWKVGen/Training/SequentialBackward.swift). Автоград по всему графу
    // держит активации всех слоёв всех пачек до конца обратного прохода: на
    // 0.1B при 8 триплетах по 256 токенов это 7.8 ГБ пика. Здесь прямой проход
    // без градиента хранит только входы блоков, лосс считается по выходам
    // последнего блока, а блоки идут сверху вниз с вычислением после каждого.
    // Цена — каждый блок считается дважды.

    /// Сколько раз градиент посчитан этим путём (счётчик для тестов).
    nonisolated(unsafe) static var sequentialCalls = 0

    /// (лосс, градиенты в порядке `trainable.parameterNames`). `groups` —
    /// пачки, которые модель векторизует порознь (якоря, положительные, …);
    /// `loss` получает их векторы в том же порядке.
    static func sequentialValueAndGrad(
        model: EmbeddingModel, trainable: TrainableSet, parameters: [MLXArray],
        groups: [(idx: MLXArray, pool: MLXArray)],
        loss: @escaping ([MLXArray]) -> MLXArray
    ) -> (loss: MLXArray, gradients: [MLXArray]) {
        sequentialCalls += 1
        let bb = model.backbone
        let plan = SequentialBody.Plan(names: trainable.parameterNames, nLayer: bb.cfg.nLayer)

        // 1. прямой проход каждой пачки без градиента
        trainable.inject(parameters)
        let traces = groups.map { SequentialBody.forward(bb, ids: $0.idx, plan: plan) }

        // 2. всё, что над телом: ln_out, пулинг, голова, лосс — по выходам
        //    последнего блока и по параметрам «над телом» (голова, ln_out)
        let above = plan.above
        let n = groups.count
        let (value, g) = vjp({ ins in
            var current = parameters
            for (j, i) in above.enumerated() { current[i] = ins[n + j] }
            trainable.inject(current)
            let vectors = (0 ..< n).map { model.embedFromTop(ins[$0], poolIndex: groups[$0].pool) }
            return [loss(vectors).asType(.float32)]
        }, primals: traces.map { $0.top } + above.map { parameters[$0] },
           cotangents: [MLXArray(Float(1))])
        trainable.inject(parameters)
        eval(value + g)
        var grads = [MLXArray?](repeating: nil, count: parameters.count)
        for (j, i) in above.enumerated() { grads[i] = g[n + j] }

        // 3. блоки сверху вниз, пачка за пачкой
        for (k, trace) in traces.enumerated() {
            SequentialBody.backward(bb, trace: trace, dTop: g[k], parameters: parameters, plan: plan,
                                    inject: { trainable.inject($0) }, grads: &grads)
        }
        return (value[0], grads.enumerated().map { i, g in
            g ?? MLXArray.zeros(parameters[i].shape, dtype: parameters[i].dtype)
        })
    }

    /// Провайдер градиента триплетных задач обратным проходом по блокам.
    public static func sequentialProvider(
        model: EmbeddingModel, trainable: TrainableSet,
        temperature: Float = 0.05, symmetric: Bool
    ) -> GradientProvider<TripletBatch> {
        { params, batch in
            sequentialValueAndGrad(
                model: model, trainable: trainable, parameters: params,
                groups: [(batch.anchorIdx, batch.anchorPool), (batch.positiveIdx, batch.positivePool),
                         (batch.negativeIdx, batch.negativePool)],
                loss: { v in
                    tripletPoolLoss(anchor: v[0], positive: v[1], negative: v[2],
                                    temperature: temperature, symmetric: symmetric)
                })
        }
    }

    /// То же для zero-shot классификации: якоря и кандидаты — две пачки.
    public static func sequentialClassificationProvider(
        model: EmbeddingModel, trainable: TrainableSet, temperature: Float = 0.05
    ) -> GradientProvider<ClassificationBatch> {
        { params, batch in
            let B = batch.candidateIdx.shape[0], K = batch.candidateIdx.shape[1]
            let T = batch.candidateIdx.shape[2]
            return sequentialValueAndGrad(
                model: model, trainable: trainable, parameters: params,
                groups: [(batch.anchorIdx, batch.anchorPool),
                         (batch.candidateIdx.reshaped([B * K, T]), batch.candidatePool.reshaped([B * K]))],
                loss: { v in
                    zeroShotClassificationLoss(anchor: v[0], candidates: v[1].reshaped([B, K, -1]),
                                               mask: batch.mask, targetIndex: batch.targetIndex,
                                               temperature: temperature)
                })
        }
    }

    // ── GradCache-путь ───────────────────────────────────────────────
    //
    // Отдельный путь есть ТОЛЬКО у триплетных задач, и это не пробел.
    // Смысл GradCache — растить пул отрицательных, не платя памятью. У
    // классификации пул СВОЙ у каждой строки, он не общий на батч, поэтому
    // больший батч не добавляет ей ни одного отрицательного: разменивать
    // ради этого время на лишний forward не на что.

    /// Провайдер градиента для триплетных задач под GradCache.
    ///
    /// Возвращает замыкание в форме, которую понимает `Trainer`. Лосс
    /// по-прежнему видит ВЕСЬ батч (в этом вся суть), а активации
    /// ограничены `chunkSize` строк.
    ///
    /// - trainable: то же множество, что отдано тренеру. Нужно здесь,
    ///   потому что подстановка обязана происходить ВНУТРИ каждой фазы:
    ///   фаза 1 считает векторы, фаза 3 пересчитывает их уже под
    ///   grad-трансформацией, и оба раза модель должна видеть текущие
    ///   параметры.
    public static func gradCacheProvider(
        model: EmbeddingModel,
        trainable: TrainableSet,
        chunkSize: Int,
        temperature: Float = 0.05,
        symmetric: Bool
    ) -> GradientProvider<TripletBatch> {

        precondition(chunkSize > 0, "chunkSize должен быть положительным")

        return { params, batch in
            let B = batch.batchSize
            let starts = chunkStarts(batch: B, chunkSize: chunkSize)

            let res = gradCacheValueAndGrad(
                parameters: params,
                chunks: starts,
                embedChunk: { ps, start in
                    trainable.inject(ps)
                    let end = Swift.min(start + chunkSize, B)
                    return [
                        model.embed(batch.anchorIdx[start ..< end],
                                    poolIndex: batch.anchorPool[start ..< end]),
                        model.embed(batch.positiveIdx[start ..< end],
                                    poolIndex: batch.positivePool[start ..< end]),
                        model.embed(batch.negativeIdx[start ..< end],
                                    poolIndex: batch.negativePool[start ..< end]),
                    ]
                },
                lossFromEmbeddings: { fields in
                    tripletPoolLoss(anchor: fields[0], positive: fields[1],
                                    negative: fields[2], temperature: temperature,
                                    symmetric: symmetric)
                })

            return (loss: res.loss, gradients: res.gradients)
        }
    }
}
