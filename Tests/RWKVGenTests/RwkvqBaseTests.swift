import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVQuant

// ───────────────────────────────────────────────────────────────────────
//  Квантованная база .rwkvq, подключённая к модели.
//
//  Отдельно от RwkvqTests намеренно: там проверялась ДЕКВАНТИЗАЦИЯ, здесь —
//  ПРОВОДКА. Это разные вещи, и вторая не следует из первой: если бы
//  k_proj сопоставился сайдкарному att.value вместо att.key, каждый тензор
//  всё равно деквантовался бы идеально, модель считала бы правдоподобные
//  числа — просто не те. Ловится это только сквозным эталоном.
//
//  Фикстура:
//      cd ~/Develop/rwkv-metal && .venv/bin/python \
//        ~/Develop/SwiftRWKV/Scripts/dump_rwkvq_model_reference.py \
//        --model world_0.1b_x070.safetensors \
//        --sidecar ~/Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx \
//        --out ~/Develop/SwiftRWKV/.testdata/rwkvq_model_ref.safetensors
// ───────────────────────────────────────────────────────────────────────

final class RwkvqBaseTests: XCTestCase {

    struct Fixtures {
        let backbone: X070Backbone
        let sidecar: RwkvqSidecar
        let reference: [String: MLXArray]
        let cfg: X070Config
    }

    func loadFixtures() throws -> Fixtures? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["RWKV_PARITY_MODEL"]
            ?? home.appendingPathComponent("Develop/rwkv-metal/world_0.1b_x070.safetensors").path
        let sidecarPath = env["RWKV_RWKVQ_SIDECAR"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/0.1B_reduction.rwkvq_mlx").path
        let refPath = env["RWKV_RWKVQ_MODEL_REFERENCE"]
            ?? home.appendingPathComponent("Develop/SwiftRWKV/.testdata/rwkvq_model_ref.safetensors").path

        let fm = FileManager.default
        guard fm.fileExists(atPath: modelPath),
              fm.fileExists(atPath: sidecarPath + ".safetensors"),
              fm.fileExists(atPath: refPath) else { return nil }

        let weights = try loadArrays(url: URL(fileURLWithPath: modelPath))
        let nLayer = weights.keys.compactMap { k -> Int? in
            guard k.hasPrefix("blocks.") else { return nil }
            return Int(k.split(separator: ".")[1])
        }.max().map { $0 + 1 } ?? 0
        let cfg = X070Config(nLayer: nLayer,
                             nEmbd: weights["ln_out.weight"]!.shape[0],
                             headSize: weights["blocks.0.tmix.k_k"]!.shape[1],
                             vocab: weights["head.weight"]!.shape[0])
        return Fixtures(backbone: X070Backbone(weights: weights, cfg: cfg),
                        sidecar: try RwkvqSidecar(path: sidecarPath),
                        reference: try loadArrays(url: URL(fileURLWithPath: refPath)),
                        cfg: cfg)
    }

    func require() throws -> Fixtures {
        let f = try loadFixtures()
        try XCTSkipIf(f == nil, "нет фикстур .rwkvq — тест пропущен (см. шапку файла)")
        return f!
    }

    func relDiff(_ ref: MLXArray, _ got: MLXArray) -> Float {
        let r = ref.asType(.float32), g = got.asType(.float32)
        eval(r, g)
        return MLX.abs(r - g).max().item(Float.self)
             / (MLX.abs(r).max().item(Float.self) + 1e-9)
    }

    // ── Проводка ─────────────────────────────────────────────────────

    /// Сквозной паритет: модель с квантованной базой обязана совпасть с
    /// эталоном, где те же тензоры подставлены в Python.
    ///
    /// Допуск 2e-2 — уровень bf16, как в X070ParityTests. Ошибка ПРОВОДКИ
    /// (перепутанные проекции, неверный transpose) даёт расхождение на
    /// порядки больше и здесь не пройдёт.
    func testQuantizedModelMatchesReference() throws {
        let f = try require()
        let info = f.backbone.attachRwkvq(f.sidecar)
        XCTAssertGreaterThan(info.attached, 0, "ничего не подключилось")
        XCTAssertTrue(info.missing.isEmpty || info.missing.allSatisfy {
            $0.contains("emb.weight")
        }, "не нашлись в сайдкаре: \(info.missing)")

        let ids = f.reference["ids"]!
        let hidden = f.backbone.body(ids)
        let logits = f.backbone(ids)
        eval(hidden, logits)

        let dh = relDiff(f.reference["hidden"]!, hidden)
        let dl = relDiff(f.reference["logits"]!, logits)
        print("RWKVQ ПАРИТЕТ hidden=\(dh) logits=\(dl), подключено \(info.attached)")

        XCTAssertLessThan(dh, 2e-2, "hidden с квантованной базой разошёлся")
        XCTAssertLessThan(dl, 2e-2, "logits с квантованной базой разошлись")
    }

    /// Квантование действительно МЕНЯЕТ выход — иначе тест паритета проходил
    /// бы и при полностью проигнорированном сайдкаре.
    ///
    /// Верхней границы здесь намеренно НЕТ. Соблазн написать «изменение не
    /// больше X%» велик, но любое такое X было бы взято с потолка: на этой
    /// модели квантование меняет логиты на ~70% по максимальной
    /// относительной невязке, и это НОРМАЛЬНО — метрика «максимум по всем
    /// элементам» жёсткая, 0.1B к квантованию куда чувствительнее 1.5B,
    /// на котором мерили пресет, а без act_stats режим asym_sb6_aw
    /// вырождается в невзвешенный. Корректность проводки проверяется
    /// сравнением с ЭТАЛОНОМ (testQuantizedModelMatchesReference), а не
    /// догадкой о допустимой величине шума.
    func testQuantizedOutputDiffersFromDense() throws {
        let f = try require()
        let ids = f.reference["ids"]!
        let dense = f.backbone(ids)
        eval(dense)

        f.backbone.attachRwkvq(f.sidecar)
        let quantized = f.backbone(ids)
        eval(quantized)

        let d = relDiff(dense, quantized)
        print("RWKVQ отличие от плотной базы: \(d)")
        XCTAssertGreaterThan(d, 1e-4,
                             "выход не изменился — сайдкар не подключился")
    }

    /// Перепутанная проводка обязана быть ОТЛИЧИМА от шума квантования.
    ///
    /// Это ответ на вопрос «а поймал бы паритетный тест подмену тензоров?».
    /// Меняем местами k_proj и v_proj в отображении имён и смотрим, насколько
    /// сильнее уезжает результат. Если разница между «правильно» и
    /// «перепутано» невелика, паритетный тест бесполезен, каким бы ни был
    /// его допуск.
    func testMiswiringIsDistinguishableFromQuantizationNoise() throws {
        let f = try require()
        let ids = f.reference["ids"]!
        let refLogits = f.reference["logits"]!

        // правильная проводка
        f.backbone.attachRwkvq(f.sidecar)
        let correct = relDiff(refLogits, f.backbone(ids))

        // перепутанная: k_proj ↔ v_proj во всех слоях
        for layer in 0 ..< f.cfg.nLayer {
            let k = "blocks.\(layer).tmix.k_proj.weight"
            let v = "blocks.\(layer).tmix.v_proj.weight"
            guard let kw = f.backbone.rwkvqKeys[k],
                  let vw = f.backbone.rwkvqKeys[v] else { continue }
            f.backbone.rwkvqKeys[k] = vw
            f.backbone.rwkvqKeys[v] = kw
        }
        let swapped = relDiff(refLogits, f.backbone(ids))

        print("RWKVQ проводка: правильная=\(correct) перепутанная=\(swapped)")
        XCTAssertGreaterThan(swapped, correct * 20,
                             "подмена k_proj↔v_proj почти не изменила результат "
                             + "(\(swapped) против \(correct)) — паритетный тест "
                             + "не отличил бы её от шума квантования")
        XCTAssertGreaterThan(swapped, 2e-2,
                             "перепутанная проводка проходит порог паритета")
    }

    // ── Память ───────────────────────────────────────────────────────

    /// Ради чего всё: плотные копии выбрасываются, база остаётся сжатой.
    func testDenseWeightsAreFreed() throws {
        let f = try require()
        let before = f.backbone.w.count
        let info = f.backbone.attachRwkvq(f.sidecar)

        XCTAssertLessThan(f.backbone.w.count, before,
                          "плотные веса не выброшены")
        XCTAssertGreaterThan(info.freedDenseBytes, 100_000_000,
                             "освобождено подозрительно мало: \(info.freedDenseBytes)")
        for key in f.backbone.rwkvqBackedKeys {
            XCTAssertNil(f.backbone.w[key], "\(key) остался плотным")
        }
        print("RWKVQ память: освобождено \(info.freedDenseBytes / 1_000_000) МБ, "
              + "упаковано \(info.packedBytes / 1_000_000) МБ")
    }

    /// dropDenseWeights: false оставляет плотные веса на месте — это режим
    /// для сравнения путей, а не для экономии.
    func testKeepDenseOption() throws {
        let f = try require()
        var opts = X070Backbone.RwkvqAttachOptions()
        opts.dropDenseWeights = false
        f.backbone.attachRwkvq(f.sidecar, options: opts)
        for key in f.backbone.rwkvqBackedKeys {
            XCTAssertNotNil(f.backbone.w[key],
                            "\(key) выброшен, хотя dropDenseWeights=false")
        }
    }

    // ── Выбор подключаемого ──────────────────────────────────────────

    /// Эмбеддинг по умолчанию НЕ квантуется: формат sb6 не умеет выбирать
    /// строки, и таблицу пришлось бы разворачивать целиком на каждом проходе.
    func testEmbeddingNotQuantizedByDefault() throws {
        let f = try require()
        f.backbone.attachRwkvq(f.sidecar)
        XCTAssertFalse(f.backbone.isRwkvqBacked("emb.weight"),
                       "emb не должен подключаться по умолчанию")
        XCTAssertNotNil(f.backbone.w["emb.weight"], "emb должен остаться плотным")
        XCTAssertTrue(f.backbone.isRwkvqBacked("head.weight"),
                      "head по умолчанию должен подключаться")
    }

    /// Ограничение по слоям и по таргетам.
    func testSelectiveAttach() throws {
        let f = try require()
        var opts = X070Backbone.RwkvqAttachOptions()
        opts.layers = 0 ..< 2
        opts.tmixTargets = ["k_proj"]
        opts.quantizeCmix = false
        opts.quantizeHead = false
        let info = f.backbone.attachRwkvq(f.sidecar, options: opts)

        XCTAssertEqual(info.attached, 2, "ожидались ровно два веса")
        XCTAssertTrue(f.backbone.isRwkvqBacked("blocks.0.tmix.k_proj.weight"))
        XCTAssertTrue(f.backbone.isRwkvqBacked("blocks.1.tmix.k_proj.weight"))
        XCTAssertFalse(f.backbone.isRwkvqBacked("blocks.2.tmix.k_proj.weight"))
        XCTAssertFalse(f.backbone.isRwkvqBacked("blocks.0.tmix.r_proj.weight"))
        XCTAssertFalse(f.backbone.isRwkvqBacked("head.weight"))

        // модель обязана остаться работоспособной при частичном подключении
        let out = f.backbone(f.reference["ids"]!)
        eval(out)
        XCTAssertTrue(MLX.abs(out).max().item(Float.self).isFinite)
    }

    /// Сайдкар от другой геометрии не должен молча подключаться: несовпавшая
    /// форма проявилась бы мусором на выходе, а не ошибкой.
    func testShapeMismatchIsRejected() throws {
        let f = try require()
        let wrongCfg = X070Config(nLayer: f.cfg.nLayer, nEmbd: 512,
                                  headSize: 64, vocab: f.cfg.vocab)
        var weights: [String: MLXArray] = [:]
        for key in f.sidecar.tensors.keys.compactMap({ RwkvqNaming.x070Key(forWorld: $0) }) {
            weights[key] = MLXArray.zeros([512, 512])
        }
        let bb = X070Backbone(weights: weights, cfg: wrongCfg)
        let info = bb.attachRwkvq(f.sidecar)
        XCTAssertEqual(info.attached, 0,
                       "сайдкар с другой геометрией подключился молча")
        XCTAssertFalse(info.missing.isEmpty)
    }

    /// Состояние тоже должно считаться через квантованную базу — реранкер
    /// будет ходить именно этим путём.
    func testStateWorksWithQuantizedBase() throws {
        let f = try require()
        f.backbone.attachRwkvq(f.sidecar)
        let ids = f.reference["ids"]!
        let T = ids.shape[1], half = T / 2

        let (_, full) = f.backbone.bodyWithState(ids)
        let (_, s1) = f.backbone.bodyWithState(ids[0..., 0 ..< half])
        let (_, s2) = f.backbone.bodyWithState(ids[0..., half ..< T], state: s1)

        XCTAssertLessThan(relDiff(full.wkv, s2.wkv), 1e-5,
                          "продолжение на квантованной базе разошлось со сплошным")
    }
}
