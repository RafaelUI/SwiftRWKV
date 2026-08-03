//
//  RerankLossTests.swift
//  Лоссы реранкера против РУЧНОГО расчёта, а не против самих себя.
//
//  Числа в ожиданиях посчитаны отдельно (см. комментарии у каждого теста), а
//  не сняты с реализации. Тест, чьи ожидания сняты с проверяемого кода,
//  фиксирует поведение, но ничего не доказывает.
//
import XCTest
import MLX
@testable import RWKVRerank

final class RerankLossTests: XCTestCase {

    // ─────────────────────────────────────────────────────────────────
    //  Listwise
    // ─────────────────────────────────────────────────────────────────

    /// Равные скоры ⇒ ровно ln(C), при любом уровне этих скоров.
    ///
    /// Второе важнее первого: softmax инвариантен к сдвигу, и если лосс
    /// поехал от прибавления константы — потерян вычет максимума либо
    /// перепутана ось редукции.
    func testEqualScoresGiveLnC() {
        for C in [2, 3, 8, 25] {
            for level in [Float(0), 5, -5, 100] {
                let s = MLXArray.full([1, C], values: MLXArray(level))
                let loss = listwiseLoss(s, MLXArray([Int32(0)]))
                eval(loss)
                XCTAssertEqual(loss.item(Float.self), log(Float(C)), accuracy: 1e-5,
                               "C=\(C), уровень \(level): не ln(C)")
            }
        }
    }

    /// Ручной расчёт. scores = [1, 2, 3], label = 2.
    ///   logsumexp = 3 + ln(e⁻² + e⁻¹ + 1) = 3 + ln(1.503214...) = 3.407606
    ///   loss = 3.407606 − 3 = 0.407606
    func testListwiseAgainstHandComputation() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let loss = listwiseLoss(s, MLXArray([Int32(2)]))
        eval(loss)
        XCTAssertEqual(loss.item(Float.self), 0.407606, accuracy: 1e-5)
    }

    /// Температура делит логиты. При T = 0.5 те же [1,2,3] превращаются в
    /// [2,4,6]:
    ///   logsumexp = 6 + ln(e⁻⁴ + e⁻² + 1) = 6 + ln(1.15365092) = 6.14293163
    ///   loss = 0.14293163
    ///
    /// Первая редакция этого теста ждала 0.143038 — я ошибся в арифметике,
    /// и упал именно тест, а не код. Полезное напоминание, зачем ожидания
    /// считаются отдельно: ошибку видно сразу, а не через месяц.
    func testTemperatureSharpensDistribution() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let loss = listwiseLoss(s, MLXArray([Int32(2)]), temperature: 0.5)
        eval(loss)
        XCTAssertEqual(loss.item(Float.self), 0.14293163, accuracy: 1e-6)
        // И строго меньше, чем при T = 1: температура ниже единицы обостряет
        // распределение, то есть уверенный правильный ответ штрафуется слабее.
        let base = listwiseLoss(s, MLXArray([Int32(2)]))
        eval(base)
        XCTAssertLessThan(loss.item(Float.self), base.item(Float.self))
    }

    /// Усреднение по строкам батча, а не суммирование: иначе величина лосса
    /// зависела бы от размера батча и расписание lr пришлось бы подбирать
    /// заново при каждой его смене.
    func testMeanOverBatchNotSum() {
        let s = MLXArray([Float(1), 2, 3, 1, 2, 3]).reshaped([2, 3])
        let one = listwiseLoss(s[0..<1], MLXArray([Int32(2)]))
        let two = listwiseLoss(s, MLXArray([Int32(2), Int32(2)]))
        eval(one, two)
        XCTAssertEqual(two.item(Float.self), one.item(Float.self), accuracy: 1e-6)
    }

    /// Разъехавшиеся скоры не дают inf/nan. Именно к этому обучение и ведёт:
    /// чем лучше голова, тем дальше позитив от негативов.
    func testListwiseStableOnLargeScores() {
        let s = MLXArray([Float(800), -800, 0]).reshaped([1, 3])
        let loss = listwiseLoss(s, MLXArray([Int32(0)]))
        eval(loss)
        XCTAssertTrue(loss.item(Float.self).isFinite, "listwise переполнился")
        XCTAssertEqual(loss.item(Float.self), 0, accuracy: 1e-6,
                       "правильный кандидат далеко впереди — лосс должен быть ~0")
    }

    // ─────────────────────────────────────────────────────────────────
    //  BCE
    // ─────────────────────────────────────────────────────────────────

    /// Ручной расчёт. scores = [1, 2, 3], label = 2, цели [0, 0, 1].
    ///   −ln(1−σ(1)) = ln(1+e¹)   = 1.313262
    ///   −ln(1−σ(2)) = ln(1+e²)   = 2.126928
    ///   −ln(σ(3))   = ln(1+e⁻³)  = 0.048587
    ///   среднее по трём = 1.162926
    func testBCEAgainstHandComputation() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let loss = bceLoss(s, MLXArray([Int32(2)]))
        eval(loss)
        XCTAssertEqual(loss.item(Float.self), 1.162926, accuracy: 1e-5)
    }

    /// Нулевые скоры ⇒ ровно ln 2 на каждого кандидата.
    func testBCEAtZeroIsLn2() {
        let s = MLXArray.zeros([1, 8])
        let loss = bceLoss(s, MLXArray([Int32(3)]))
        eval(loss)
        XCTAssertEqual(loss.item(Float.self), log(Float(2)), accuracy: 1e-6)
    }

    /// Устойчивая форма: наивная `−t·ln σ(x) − (1−t)·ln(1−σ(x))` здесь
    /// вернула бы inf.
    func testBCEStableOnLargeScores() {
        let s = MLXArray([Float(800), -800]).reshaped([1, 2])
        for label in [Int32(0), Int32(1)] {
            let loss = bceLoss(s, MLXArray([label]))
            eval(loss)
            XCTAssertTrue(loss.item(Float.self).isFinite,
                          "BCE переполнился при label=\(label)")
        }
    }

    /// Цель ставится в правильную позицию. Тест различающий: при перепутанном
    /// индексе лосс был бы БОЛЬШЕ (позитив стоит последним и он же самый
    /// высокий), и величина этой разницы посчитана вручную:
    ///   label=2 → 1.162926 (см. выше), label=0 → ln(1+e⁻¹)+ln(1+e²)+ln(1+e³)
    ///                                          = 0.313262+2.126928+3.048587
    ///                                          → среднее 1.829592
    func testBCETargetPosition() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let atEnd = bceLoss(s, MLXArray([Int32(2)]))
        let atStart = bceLoss(s, MLXArray([Int32(0)]))
        eval(atEnd, atStart)
        XCTAssertEqual(atStart.item(Float.self), 1.829592, accuracy: 1e-5)
        XCTAssertLessThan(atEnd.item(Float.self), atStart.item(Float.self))
    }

    // ─────────────────────────────────────────────────────────────────
    //  Смесь
    // ─────────────────────────────────────────────────────────────────

    func testMixedIsConvexCombination() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let lbl = MLXArray([Int32(2)])
        let lw = listwiseLoss(s, lbl), bc = bceLoss(s, lbl)
        let mixed = mixedLoss(s, lbl, alpha: 0.7)
        eval(lw, bc, mixed)
        XCTAssertEqual(mixed.item(Float.self),
                       0.7 * lw.item(Float.self) + 0.3 * bc.item(Float.self),
                       accuracy: 1e-6)
    }

    /// Крайние alpha вырождаются в чистые лоссы РОВНО, а не приблизительно.
    func testMixedDegeneratesAtEndpoints() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let lbl = MLXArray([Int32(2)])
        let lw = listwiseLoss(s, lbl), bc = bceLoss(s, lbl)
        let a1 = mixedLoss(s, lbl, alpha: 1.0), a0 = mixedLoss(s, lbl, alpha: 0.0)
        eval(lw, bc, a1, a0)
        XCTAssertEqual(a1.item(Float.self), lw.item(Float.self))
        XCTAssertEqual(a0.item(Float.self), bc.item(Float.self))
    }

    /// Обёртка-перечисление зовёт то же, что и функции.
    func testEnumDispatchMatchesFunctions() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let lbl = MLXArray([Int32(2)])
        for (loss, want) in [(RerankLoss.listwise, listwiseLoss(s, lbl)),
                             (RerankLoss.bce, bceLoss(s, lbl)),
                             (RerankLoss.mixed(0.7), mixedLoss(s, lbl, alpha: 0.7))] {
            let got = loss(s, lbl)
            eval(got, want)
            XCTAssertEqual(got.item(Float.self), want.item(Float.self), accuracy: 1e-7)
        }
    }

    /// Градиент по скорам ненулевой и направлен в нужную сторону: у
    /// правильного кандидата он отрицательный (лосс падает, если поднять
    /// его скор), у остальных — положительный.
    func testListwiseGradientDirection() {
        let s = MLXArray([Float(1), 2, 3]).reshaped([1, 3])
        let lbl = MLXArray([Int32(0)])
        func loss(_ i: [MLXArray]) -> [MLXArray] { [listwiseLoss(i[0], lbl)] }
        let g = grad(loss, argumentNumbers: [0])([s])[0]
        eval(g)
        XCTAssertLessThan(g[0, 0].item(Float.self), 0,
                          "градиент у правильного кандидата обязан быть < 0")
        XCTAssertGreaterThan(g[0, 1].item(Float.self), 0)
        XCTAssertGreaterThan(g[0, 2].item(Float.self), 0)
    }
}
