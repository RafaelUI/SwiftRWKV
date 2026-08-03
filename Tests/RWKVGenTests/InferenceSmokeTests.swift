//
//  InferenceSmokeTests.swift
//  Инференс на РЕАЛЬНОЙ 0.1B: рекуррентный декод и квантованная база.
//
//  Проверяется публичный API так, как его увидит пользователь: загрузить
//  веса, свернуть промпт, шагать по токенам, подключить сайдкар .rwkvq и
//  сделать то же самое. Модульные тесты проверяют куски и паритет с Python;
//  здесь — что связка целиком делает то, что обещает.
//
//  Пропускается без модели и сайдкара.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

final class InferenceSmokeTests: XCTestCase {

    func backbone() throws -> (X070Backbone, WorldTokenizer, X070Config) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let vp = env["RWKV_WORLD_VOCAB"] ?? home.appendingPathComponent(
            "Develop/SwiftRWKV/.testdata/rwkv_vocab_v20230424.txt").path
        for p in [mp, vp] {
            try XCTSkipUnless(FileManager.default.fileExists(atPath: p),
                              "нет фикстуры \(p)")
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

    func sidecar() throws -> RwkvqSidecar {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let p = ProcessInfo.processInfo.environment["RWKV_RWKVQ_SIDECAR"]
            ?? home.appendingPathComponent(
                "Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx").path
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: p + ".safetensors"),
            "нет сайдкара .rwkvq")
        return try RwkvqSidecar(path: p)
    }

    /// Жадное продолжение промпта: шаги идут, логиты конечны, вывод
    /// ДЕТЕРМИНИРОВАН.
    ///
    /// Семантику текста утверждать нельзя, а вот воспроизводимость — можно и
    /// нужно: рекуррентный декод несёт состояние между шагами, и любая
    /// протечка между прогонами проявится именно здесь.
    func testGreedyDecodeIsDeterministic() throws {
        let (bb, tok, cfg) = try backbone()

        func greedy(_ prompt: String, _ n: Int) -> [Int] {
            var st = RWKVState(cfg: cfg)
            var logits = bb.prefill(tok.encode(prompt), state: &st)
            var out: [Int] = []
            for _ in 0 ..< n {
                eval(logits)
                XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite },
                              "логиты не конечны")
                let id = logits.argMax().item(Int.self)
                out.append(id)
                logits = bb.step(id, state: &st)
            }
            return out
        }

        let a = greedy("The capital of France is", 12)
        let b = greedy("The capital of France is", 12)
        XCTAssertEqual(a, b, "жадный декод невоспроизводим — состояние течёт")
        XCTAssertEqual(a.count, 12)
        XCTAssertTrue(a.allSatisfy { $0 >= 0 && $0 < cfg.vocab })

        // Различающее утверждение: ДРУГОЙ промпт даёт другое продолжение.
        // Без него тест был бы зелёным и при полностью проигнорированном
        // промпте.
        let c = greedy("Рецепт борща начинается с", 12)
        XCTAssertNotEqual(a, c, "продолжение не зависит от промпта")
    }

    /// Рекуррентный декод согласован с параллельным проходом.
    ///
    /// Это главный инвариант декода: `step` по одному токену обязан давать
    /// то же, что `body` целиком. Расходятся они молча — логиты остаются
    /// правдоподобными, просто это логиты другой модели.
    ///
    /// Рекуррентная сторона берётся ЯВНО через `prefillRecurrent`. Раньше
    /// здесь стоял `prefill`, и это было безопасно ровно до тех пор, пока он
    /// сам был рекуррентным; после перевода `prefill` на параллельный проход
    /// тест сравнивал бы параллельный путь сам с собой и остался бы зелёным
    /// при любом дефекте в `step`.
    func testRecurrentDecodeAgreesWithParallelPass() throws {
        let (bb, tok, cfg) = try backbone()
        let ids = tok.encode("Пчёлы собирают нектар с цветов и делают мёд")
        XCTAssertGreaterThan(ids.count, 4)

        var st = RWKVState(cfg: cfg)
        let recurrent = bb.prefillRecurrent(ids, state: &st)

        let parallel = bb.body(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        let last = parallel[0, ids.count - 1]
        let ref = matmul(last.reshaped([1, cfg.nEmbd]),
                         bb.weight("head.weight").asType(.float32).transposed())
            .reshaped([cfg.vocab])
        eval(recurrent, ref)

        // Допуск, а не ноль: пути считают одно и то же разными ядрами.
        // Замерено ~1e-2 на логитах величиной порядка десятков — сравнивать
        // надо ОТНОСИТЕЛЬНО, иначе граница ничего не значит.
        let scale = MLX.abs(ref).max().item(Float.self)
        let rel = MLX.abs(recurrent - ref).max().item(Float.self) / scale
        XCTAssertLessThan(rel, 5e-3, "рекуррентный декод разошёлся с параллельным")

        // И, главное, argmax обязан совпасть: именно он превращается в текст.
        XCTAssertEqual(recurrent.argMax().item(Int.self),
                       ref.argMax().item(Int.self), "выбран другой токен")
    }

    /// Сайдкар читается, имена сходятся, деквантизация даёт правильные формы.
    func testSidecarLoadsAndDequantizes() throws {
        let sc = try sidecar()
        XCTAssertGreaterThan(sc.keys.count, 0)
        XCTAssertGreaterThan(sc.packedBytes, 0)
        XCTAssertEqual(sc.nEmbd, 768)

        // Имя x070 → имя World: без этой таблицы сайдкар не находит ничего,
        // и attach молча возвращает нули подключённых.
        let key = "blocks.0.tmix.k_proj.weight"
        guard let world = RwkvqNaming.worldKey(forX070: key) else {
            return XCTFail("нет отображения имени для \(key)")
        }
        XCTAssertTrue(sc.contains(world), "сайдкар не содержит \(world)")
        let w = try sc.dequantize(world)
        XCTAssertEqual(w.shape, [sc.tensors[world]!.outFeatures,
                                 sc.tensors[world]!.inFeatures])
        eval(w)
        XCTAssertTrue(MLX.abs(w).max().item(Float.self).isFinite)
    }

    /// Подключение квантованной базы: веса освобождаются, экономия реальна.
    ///
    /// Про ДЕКОД тут ничего не утверждается: он на квантованной базе не
    /// работает, см. `testDecodeHasNoQuantisedPath`.
    func testQuantizedAttachFreesMemory() throws {
        let (quant, _, _) = try backbone()
        let sc = try sidecar()

        let info = quant.attachRwkvq(sc)
        XCTAssertGreaterThan(info.attached, 0, "не подключено ни одного веса")
        XCTAssertEqual(info.missing, [], "часть весов не нашлась в сайдкаре")
        XCTAssertGreaterThan(info.freedDenseBytes, 0, "плотные копии не выброшены")
        XCTAssertGreaterThan(info.packedBytes, 0)
        XCTAssertLessThan(info.packedBytes, info.freedDenseBytes,
                          "упакованное не меньше плотного — экономии нет")
        XCTAssertTrue(quant.isRwkvqBacked("blocks.0.tmix.k_proj.weight"))
        print("RWKVQ: подключено \(info.attached), освобождено "
              + "\(info.freedDenseBytes / 1_000_000) МБ, упаковано "
              + "\(info.packedBytes / 1_000_000) МБ")
    }

    /// Отключение возвращает плотный путь.
    func testDetachRestoresDensePath() throws {
        let (bb, _, _) = try backbone()
        let sc = try sidecar()
        _ = bb.attachRwkvq(sc, options: {
            var o = X070Backbone.RwkvqAttachOptions()
            o.dropDenseWeights = false   // иначе возвращать будет нечего
            return o
        }())
        XCTAssertFalse(bb.rwkvqBackedKeys.isEmpty)
        bb.detachRwkvq()
        XCTAssertTrue(bb.rwkvqBackedKeys.isEmpty)
        XCTAssertFalse(bb.isRwkvqBacked("blocks.0.tmix.k_proj.weight"))
    }
    /// Декод ПРИМЕНЯЕТ LoRA-адаптеры — как и параллельный путь.
    ///
    /// Раньше не применял: `proj` прибавляет `scale·(x·Aᵀ)·Bᵀ`, а декод звал
    /// `linear_(x, gg(key))` напрямую. Дообученная модель генерировала так,
    /// будто её не дообучали, — без ошибки, правдоподобным текстом.
    /// Замерено: расхождение путей было больше 10%, стало 1.4e-4 при
    /// зафиксированном сиде (см. ниже, почему число изменилось).
    func testDecodeAppliesLoRAAdapters() throws {
        let (bb, tok, cfg) = try backbone()
        // B ненулевой: при штатной инициализации B = 0, то есть адаптер —
        // тождество, и «применяет или нет» неразличимо в принципе.
        var spec = LoRASpec()
        spec.rank = 8
        spec.alpha = 512
        _ = LoRA.add(to: bb, spec: spec)
        // Сид ОБЯЗАТЕЛЕН. Без него адаптеры каждый прогон разные, а граница
        // ниже — допуск на расхождение двух путей, который от величины
        // адаптеров зависит напрямую. Тест был зелёным до тех пор, пока
        // случайные B не оказались покрупнее: 1.9e-4 против границы 1e-4.
        // Дефекта не было — была неповторяемая граница, то есть красное,
        // отложенное до неудачного дня. Записанное когда-то «1.1e-5» — это
        // просто удачный бросок, а не свойство кода.
        MLXRandom.seed(20260803)
        for t in bb.loraTargets {
            bb.loraB[t] = MLXRandom.normal(bb.loraB[t]!.shape) * 0.05
        }
        eval(bb.loraTargets.compactMap { bb.loraB[$0] })
        XCTAssertFalse(bb.loraTargets.isEmpty, "адаптеры не навесились")
        // `add` помечает слои обучаемыми, а обучаемый путь идёт через ядро,
        // требующее T кратного 16. Здесь мы ИНФЕРИРУЕМ, значит помету надо
        // снять — на применение адаптеров она не влияет.
        bb.trainLayers = []

        let ids = tok.encode("Пчёлы собирают нектар")
        func recurrentLogits() -> MLXArray {
            var st = RWKVState(cfg: cfg)
            return bb.prefillRecurrent(ids, state: &st)
        }
        let h = bb.body(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        let ref = matmul(h[0, ids.count - 1].reshaped([1, cfg.nEmbd]),
                         bb.weight("head.weight").asType(.float32).transposed())
            .reshaped([cfg.vocab])
        let withLoRA = recurrentLogits()
        eval(withLoRA, ref)
        let scale = MLX.abs(ref).max().item(Float.self)
        // Граница взята ОТ ЗАМЕРА при этом сиде (1.39e-4), а не от круглого
        // числа. Она заведомо шире, чем расхождение путей без адаптеров
        // (7.4e-7): адаптер с alpha=512 и rank=8 умножает вклад на 64, и
        // разница двух ядер растёт вместе с ним. Различающее утверждение
        // ниже — вот что делает тест непроходимым при игнорировании адаптеров.
        XCTAssertLessThan(MLX.abs(withLoRA - ref).max().item(Float.self) / scale,
                          5e-4, "декод разошёлся с параллельным путём")

        // Различающее утверждение: БЕЗ адаптеров логиты другие. Иначе
        // совпадение выше выполнялось бы и при полностью проигнорированных
        // адаптерах с ОБЕИХ сторон.
        bb.loraA = [:]; bb.loraB = [:]; bb.loraScale = [:]
        let without = recurrentLogits()
        eval(without)
        XCTAssertGreaterThan(
            MLX.abs(withLoRA - without).max().item(Float.self) / scale, 0.05,
            "адаптеры ни на что не влияют — тест пуст")
    }
    /// Декод на квантованной базе так же точен, как параллельный путь.
    ///
    /// Раньше он просто падал: `gg(key)` читал плотный словарь, а
    /// `attachRwkvq` плотные копии выбрасывает — force-unwrap без сообщения.
    ///
    /// Сверяется рекуррент с ПАРАЛЛЕЛЬНЫМ путём ТОЙ ЖЕ квантованной модели,
    /// а не с плотной. Это принципиально: квантование меняет числа само по
    /// себе (замерено 0.62 относительного на логитах и другой top-1 на
    /// коротком промпте), и требовать совпадения с плотной значило бы
    /// проверять качество квантования, а не верность декода.
    func testQuantizedDecodeMatchesQuantizedParallel() throws {
        let (quant, tok, cfg) = try backbone()
        let info = quant.attachRwkvq(try sidecar())
        XCTAssertGreaterThan(info.attached, 0)
        XCTAssertFalse(quant.hasWeight("blocks.0.tmix.k_proj.weight"),
                       "плотный вес не выброшен — тест не про квантованный путь")

        let ids = tok.encode("The capital of France is")
        var st = RWKVState(cfg: cfg)
        let recurrent = quant.prefillRecurrent(ids, state: &st)
        let h = quant.body(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        // Через общий путь: плотного head.weight на квантованной базе нет.
        let parallel = quant.projectForBlock(
            h[0, ids.count - 1].reshaped([1, cfg.nEmbd]),
            "head.weight", lora: "head").reshaped([cfg.vocab])
        eval(recurrent, parallel)

        let scale = MLX.abs(parallel).max().item(Float.self)
        let rel = MLX.abs(recurrent - parallel).max().item(Float.self) / scale
        // Замерено 8.0e-7 — тот же порядок, что у плотной базы (7.4e-7).
        XCTAssertLessThan(rel, 1e-5, "декод разошёлся с параллельным путём")
        XCTAssertEqual(recurrent.argMax().item(Int.self),
                       parallel.argMax().item(Int.self), "выбран другой токен")

        // И жадный декод идёт, а не падает — ради этого всё и делалось.
        var st2 = RWKVState(cfg: cfg)
        var logits = quant.prefillRecurrent(ids, state: &st2)
        for _ in 0 ..< 8 {
            let id = logits.argMax().item(Int.self)
            XCTAssertTrue(id >= 0 && id < cfg.vocab)
            logits = quant.step(id, state: &st2)
        }
    }



    /// Декод работает и при КВАНТОВАННОЙ таблице эмбеддингов.
    ///
    /// По умолчанию она не квантуется (формат sb6 не умеет выбирать строки,
    /// и таблицу пришлось бы разворачивать целиком на каждом проходе), но
    /// подключить её можно, и тогда декод обязан ходить тем же путём, что
    /// параллельный. Без этого теста мутация «читать emb.weight напрямую» не
    /// ловилась ничем: при умолчании плотная таблица на месте, и оба пути
    /// совпадают сами собой.
    func testDecodeWorksWithQuantisedEmbedding() throws {
        let (bb, tok, cfg) = try backbone()
        var opts = X070Backbone.RwkvqAttachOptions()
        opts.quantizeEmbedding = true
        let info = bb.attachRwkvq(try sidecar(), options: opts)
        try XCTSkipUnless(bb.isRwkvqBacked("emb.weight"),
                          "в сайдкаре нет emb.weight — проверять нечего")
        XCTAssertGreaterThan(info.attached, 0)
        XCTAssertFalse(bb.hasWeight("emb.weight"),
                       "плотная таблица на месте — тест вырожден")

        let ids = tok.encode("The capital of France is")
        var st = RWKVState(cfg: cfg)
        let recurrent = bb.prefillRecurrent(ids, state: &st)
        let h = bb.body(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        let parallel = bb.projectForBlock(
            h[0, ids.count - 1].reshaped([1, cfg.nEmbd]),
            "head.weight", lora: "head").reshaped([cfg.vocab])
        eval(recurrent, parallel)
        let scale = MLX.abs(parallel).max().item(Float.self)
        XCTAssertLessThan(MLX.abs(recurrent - parallel).max().item(Float.self) / scale,
                          1e-5, "декод разошёлся с параллельным путём")
    }

    /// Эмбеддинг по умолчанию НЕ квантуется.
    ///
    /// Формат sb6 не поддерживает выборку строк, и таблицу пришлось бы
    /// разворачивать целиком на каждом проходе — то есть платить памятью за
    /// экономию памяти.
    func testEmbeddingNotQuantizedByDefault() throws {
        let (bb, _, _) = try backbone()
        _ = bb.attachRwkvq(try sidecar())
        XCTAssertFalse(bb.isRwkvqBacked("emb.weight"),
                       "таблица эмбеддингов квантована по умолчанию")
    }
}
