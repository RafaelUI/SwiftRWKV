import XCTest
import MLX
import MLXRandom
@testable import RWKVKernel

// ───────────────────────────────────────────────────────────────────────
//  Обратное ядро WKV: два дефекта, найденные 10.10 на срывах GPU при
//  обучении настоящих моделей (NaN в градиентах нижних слоёв, ошибки
//  командного буфера Hang / PageFault / InnocentVictim).
//
//  1. ТИПЫ ВХОДОВ. MLX строит исходник пользовательского ядра из типов
//     входов, а кеширует библиотеку по имени. Тот же вызов с другими типами
//     для него — «исходник сменился»: прежняя библиотека и конвейер
//     освобождаются сразу, хотя закодированные, но не исполненные запуски
//     ещё на них ссылаются. Обратное ядро получало примелы как есть: слой 0 —
//     в типе параметров (bf16 / fp16), слои выше — в fp32, и в одном графе
//     одно выбивало конвейер из-под другого.
//  2. ПАМЯТЬ threadgroup. Ядро объявляло 37376 байт при пределе Metal 32768.
//
//  Первый дефект недетерминирован по природе (гонка с исполнением буфера),
//  поэтому тест проверяет свойство, из которого он следует: результат на
//  смешанных типах в ОДНОМ графе обязан побитно совпасть с результатом тех
//  же вызовов по отдельности, и так несколько раундов подряд.
// ───────────────────────────────────────────────────────────────────────

final class WKV7KernelHazardTests: XCTestCase {

    static func inputs(_ T: Int, H: Int, seed: UInt64, dtype: DType) -> [MLXArray] {
        MLXRandom.seed(seed)
        func rnd(_ s: Float) -> MLXArray { MLXRandom.normal([1, T, H, 64]) * s }
        let r = rnd(0.5), k = rnd(0.5), v = rnd(0.5), a = rnd(0.1), b = rnd(0.1)
        let w = exp(-0.606531 * sigmoid(rnd(1.0)))
        let arr = [r, w, k, v, a, b].map { $0.asType(dtype) }
        eval(arr)
        return arr
    }

    static func grads(_ ins: [MLXArray], _ c: MLXArray) -> [MLXArray] {
        let fn: ([MLXArray]) -> [MLXArray] = { a in
            [(wkv7Train(a[0], a[1], a[2], a[3], a[4], a[5]) * c).sum()]
        }
        return grad(fn, argumentNumbers: [0, 1, 2, 3, 4, 5])(ins).map { $0.asType(.float32) }
    }

    func testBackwardSurvivesMixedInputTypesInOneGraph() {
        let T = 48, H = 12
        MLXRandom.seed(1)
        let c = MLXRandom.normal([1, T, H, 64]); eval(c)
        // «слои» с разными типами примелов, как в настоящей модели
        let layers: [[MLXArray]] = [
            Self.inputs(T, H: H, seed: 10, dtype: .bfloat16),
            Self.inputs(T, H: H, seed: 11, dtype: .float32),
            Self.inputs(T, H: H, seed: 12, dtype: .float16),
            Self.inputs(T, H: H, seed: 13, dtype: .float32),
        ]
        // по отдельности: каждый вызов вычислен до следующего
        let alone: [MLXArray] = layers.map { ins in
            let g = concatenated(Self.grads(ins, c).map { $0.reshaped([-1]) }, axis: 0)
            eval(g)
            return g
        }
        for a in alone {
            XCTAssertEqual((a .!= a).asType(.int32).sum().item(Int.self), 0, "NaN в одиночном вызове")
        }
        // вместе: один ленивый граф, типы чередуются — и так несколько раундов
        for round in 0 ..< 6 {
            var order = Array(layers.indices)
            if round % 2 == 1 { order.reverse() }
            let lazy = order.map { i in
                concatenated(Self.grads(layers[i], c).map { $0.reshaped([-1]) }, axis: 0)
            }
            eval(lazy)
            for (pos, i) in order.enumerated() {
                let bad = (lazy[pos] .!= alone[i]).asType(.int32).sum().item(Int.self)
                XCTAssertEqual(bad, 0, "раунд \(round), вход \(i): \(bad) элементов градиента "
                               + "разошлись с одиночным вызовом — ядро выбито из-под запуска")
            }
        }
    }

    /// Тип примела не меняет счёт: ядро считает в fp32, bf16-значения в fp32
    /// представимы точно. Градиент возвращается в типе примела, поэтому
    /// сравнение — после того же округления.
    func testGradientDoesNotDependOnHowTheInputIsStored() {
        let T = 32, H = 3
        MLXRandom.seed(2)
        let c = MLXRandom.normal([1, T, H, 64]); eval(c)
        let bf = Self.inputs(T, H: H, seed: 20, dtype: .bfloat16)
        let wide = bf.map { $0.asType(.float32) }
        let a = Self.grads(bf, c), b = Self.grads(wide, c)
        for i in 0 ..< 6 {
            let want = b[i].asType(.bfloat16).asType(.float32)
            XCTAssertEqual((a[i] .!= want).asType(.int32).sum().item(Int.self), 0, "вход \(i)")
        }
    }
}
