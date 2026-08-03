//
//  RwkvqDTypeTests.swift
//  Тип ХРАНЕНИЯ результата деквантизации против типа АРИФМЕТИКИ.
//
//  Ядро считает комбайн `code·s + m` в float всегда; `dtype` определяет
//  только то, чем результат записывается. Утверждение, ради которого этот
//  файл существует: `dequantize(dtype: .bfloat16)` даёт РОВНО ТО ЖЕ, что
//  `dequantize().asType(.bfloat16)` — до бита, на каждом тензоре сайдкара.
//
//  Это не «достаточно близко». Приведение к bf16 происходило и раньше, просто
//  на шаг позже: модель считает в bf16, и `baseProj` приводил fp32-транзиент
//  сразу после деквантизации. Значит округление ровно одно и ровно то же,
//  а переход на bf16-выход обязан быть переносом БЕЗ изменения чисел. Если
//  бы он таковым не был, экономия памяти покупалась бы тихим сдвигом
//  относительно калибровки пресета — то есть тем самым, от чего
//  предостерегает шапка RwkvqDequant.swift.
//
//  Пропускается без сайдкара.
//
import XCTest
import MLX
@testable import RWKVQuant
@testable import RWKVGen

final class RwkvqDTypeTests: XCTestCase {

    func loadSidecar() throws -> RwkvqSidecar {
        let p = ProcessInfo.processInfo.environment["RWKV_RWKVQ_SIDECAR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx")
                .path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: p + ".safetensors"),
                          "нет сайдкара .rwkvq")
        return try RwkvqSidecar(path: p)
    }

    /// bf16-выход совпадает с fp32-выходом, приведённым к bf16, БИТ-В-БИТ —
    /// на КАЖДОМ тензоре сайдкара, а не на одном показательном.
    func testBFloat16OutputEqualsFloat32Rounded() throws {
        let sc = try loadSidecar()
        XCTAssertFalse(sc.keys.isEmpty, "сайдкар пуст — тест ничего не проверил")
        for key in sc.keys {
            let f32 = try sc.dequantize(key)
            let b16 = try sc.dequantize(key, dtype: .bfloat16)
            XCTAssertEqual(b16.dtype, .bfloat16, "\(key): тип выхода не тот")
            XCTAssertEqual(f32.dtype, .float32, "\(key): умолчание сдвинулось")
            XCTAssertEqual(b16.shape, f32.shape, "\(key): форма разъехалась")

            let want = f32.asType(.bfloat16)
            eval(b16, want)
            // Сравнение в fp32 после расширения: вычитать bf16 из bf16 значит
            // сравнивать уже округлённое с уже округлённым, и любое
            // расхождение младшего бита утонуло бы.
            let d = MLX.abs(b16.asType(.float32) - want.asType(.float32))
                .max().item(Float.self)
            XCTAssertEqual(d, 0, "\(key): bf16-выход разошёлся с fp32→bf16 на \(d)")
        }
    }

    /// fp16-выход тоже совпадает со своим округлением.
    ///
    /// Не потому, что fp16 где-то используется, а потому, что параметр типа
    /// либо работает для всех объявленных значений, либо это не параметр.
    func testFloat16OutputEqualsFloat32Rounded() throws {
        let sc = try loadSidecar()
        let key = try XCTUnwrap(sc.keys.first)
        let f32 = try sc.dequantize(key)
        let f16 = try sc.dequantize(key, dtype: .float16)
        XCTAssertEqual(f16.dtype, .float16)
        let want = f32.asType(.float16)
        eval(f16, want)
        XCTAssertEqual(MLX.abs(f16.asType(.float32) - want.asType(.float32))
            .max().item(Float.self), 0)
    }

    /// bf16 РЕАЛЬНО теряет разряды относительно fp32 — то есть предыдущие
    /// тесты сравнивают не два одинаковых представления.
    ///
    /// Различающее утверждение. Без него всё выше прошло бы и на реализации,
    /// которая параметр `dtype` игнорирует и всегда возвращает fp32:
    /// `f32.asType(.bfloat16)` тогда тоже совпало бы сам с собой.
    func testBFloat16ActuallyLosesPrecision() throws {
        let sc = try loadSidecar()
        let key = try XCTUnwrap(sc.keys.first)
        let f32 = try sc.dequantize(key)
        let b16 = try sc.dequantize(key, dtype: .bfloat16)
        eval(f32, b16)
        let d = MLX.abs(f32 - b16.asType(.float32)).max().item(Float.self)
        XCTAssertGreaterThan(d, 0,
                             "bf16 совпал с fp32 разряд в разряд — dtype не применился")
    }

    /// Транзиент вдвое меньше: это и есть вся причина изменения.
    func testBFloat16TransientIsHalfTheSize() throws {
        let sc = try loadSidecar()
        let key = try XCTUnwrap(sc.keys.first)
        let f32 = try sc.dequantize(key)
        let b16 = try sc.dequantize(key, dtype: .bfloat16)
        XCTAssertEqual(f32.dtype.size, 4)
        XCTAssertEqual(b16.dtype.size, 2)
        XCTAssertEqual(b16.size, f32.size)
    }

    /// Проход по квантованной базе возвращает ТОТ ЖЕ ТИП, что по плотной.
    ///
    /// Инвариант, ради которого тип протаскивается в `rwkvqWeight`: чем бы ни
    /// считала модель, квантованная база не должна этот тип менять.
    ///
    /// **И заодно фиксирует неприятный факт: тип этот — float32, а не
    /// bfloat16.** Веса модели bf16, вход тоже bf16, но WKV-шаг считает
    /// рекуррентность в fp32 (иначе нельзя) и возвращает fp32 в остаточный
    /// поток. Начиная с tmix СЛОЯ 0 вся дальнейшая активация — fp32; видно по
    /// `cmixPrev[0]`, он уже float32, тогда как `tmixPrev[0]` ещё bfloat16.
    ///
    /// Из этого следует то, что стоит знать, прежде чем радоваться bf16-выходу
    /// деквантизации: в бою `x.dtype` почти везде fp32, и вес разворачивается
    /// в fp32 ровно как раньше. Выигрыш от bf16 (замерено ×1.8 на 0.1B и ×2.6
    /// на 2.9B на синтетике с bf16-входом) НЕ реализуется, пока течёт тип.
    /// Замер того, чего стоит сама течь, — в NEXT_SESSION.
    func testQuantizedPassKeepsTheComputeDType() throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let mp = env["RWKV_PARITY_MODEL"] ?? home.appendingPathComponent(
            "Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mp), "нет модели")
        let sc = try loadSidecar()

        let w = try loadArrays(url: URL(fileURLWithPath: mp))
        let nL = w.keys.compactMap { k -> Int? in
            k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: w["head.weight"]!.shape[0])
        let ids = MLXArray([Int32(510), 610, 710], [1, 3])

        let dense = X070Backbone(weights: w, cfg: cfg)
        let denseOut = dense.body(ids)
        eval(denseOut)

        let quant = X070Backbone(weights: w, cfg: cfg)
        XCTAssertGreaterThan(quant.attachRwkvq(sc).attached, 0, "сайдкар не подключился")
        let quantOut = quant.body(ids)
        eval(quantOut)

        XCTAssertEqual(quantOut.dtype, denseOut.dtype,
                       "квантованный проход вернул \(quantOut.dtype) вместо \(denseOut.dtype)")
        // Утверждается ТО, ЧТО ЕСТЬ, а не то, чего хотелось бы. Когда течь
        // типа закроют, этот тест упадёт — и это ровно то, чего от него ждут:
        // он сторожит границу, а не желание.
        XCTAssertEqual(quantOut.dtype, .float32,
                       "тип прохода изменился — перечитать комментарий выше "
                       + "и пересмотреть bf16-путь деквантизации")
        // А вход в самый первый tmix ещё честно bf16 — значит течь именно
        // внутри блока, а не на входе модели.
        var st = RWKVState(cfg: cfg)
        _ = dense.step(510, state: &st)
        XCTAssertEqual(st.tmixPrev[0].dtype, .bfloat16, "вход слоя 0 уже не bf16")
        XCTAssertEqual(st.cmixPrev[0].dtype, .float32, "течь fp32 сдвинулась с tmix слоя 0")
    }

    /// Умолчание осталось fp32.
    ///
    /// На умолчании стоят эталоны бит-в-бит из Python (`RwkvqTests`). Смена
    /// умолчания не сломала бы их — они бы просто начали сравнивать
    /// округлённое с полным и упали бы. Тест здесь затем, чтобы причина была
    /// видна сразу, а не через чужой упавший набор.
    func testDefaultOutputIsStillFloat32() throws {
        let sc = try loadSidecar()
        let key = try XCTUnwrap(sc.keys.first)
        XCTAssertEqual(try sc.dequantize(key).dtype, .float32)
    }
}
