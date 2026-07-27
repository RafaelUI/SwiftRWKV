import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  ЧТО дифференцируется — одна из трёх ортогональных осей обучения.
//
//  В mlx-swift градиент берётся по ПЛОСКОМУ массиву тензоров с явными
//  argumentNumbers, а не по дереву модуля с freeze()-семантикой, как в Python.
//  Поэтому «какие веса обучаются» выражается упаковкой в упорядоченный массив
//  и обратной подстановкой в модель — ровно то, что оба существующих
//  тренировочных пути делают вручную, каждый со своей бухгалтерией.
//
//  Реализации: адаптеры LoRA/QLoRA, верхние N слоёв целиком, все веса
//  (претрейн), голова поверх замороженной базы.
// ───────────────────────────────────────────────────────────────────────

public protocol TrainableSet: AnyObject {

    /// Стартовые параметры в детерминированном порядке — fp32-мастер.
    ///
    /// Мастер именно fp32: апдейты Adam в bf16 теряются на округлении, и
    /// обучение тихо стоит на месте при формально убывающем лоссе.
    func initialParameters() -> [MLXArray]

    /// Подставить параметры в модель ПЕРЕД forward.
    ///
    /// Вызывается ВНУТРИ grad-замыкания, поэтому приведение типа обязано
    /// происходить здесь: каст внутри замыкания сохраняет grad-цепь до
    /// fp32-мастера, каст снаружи её разорвал бы.
    func inject(_ ps: [MLXArray])

    /// Зафиксировать финальные параметры в модели после обучения, чтобы она
    /// была сразу готова к инференсу.
    func commit(_ ps: [MLXArray])

    /// Человекочитаемые имена параметров — для чекпоинтов и диагностики.
    var parameterNames: [String] { get }
}

public extension TrainableSet {
    func commit(_ ps: [MLXArray]) { inject(ps) }
}

// ───────────────────────────────────────────────────────────────────────
//  Композиция
// ───────────────────────────────────────────────────────────────────────

/// Несколько независимых множеств как одно: параметры склеиваются встык в
/// порядке перечисления, inject/commit расходятся обратно по своим срезам.
///
/// Ради этого и введено: «база + голова» — не одна сущность, а две, и они
/// живут в РАЗНЫХ местах (веса backbone или адаптеры внутри модели, голова —
/// снаружи). Складывать их в одну реализацию через словарь `extra` можно
/// ровно один раз, для одной пары; композиция же покрывает все четыре судьбы
/// базы (full-FT, верхние N, LoRA, QLoRA, а также замороженную — тогда
/// в композиции просто одна голова) без единой новой ветки.
///
/// Порядок частей значим: от него зависит соответствие имён, моментов Adam и
/// чекпоинтов. Он фиксируется массивом, а не сортировкой имён — сортировка
/// перемешала бы «голову» и «базу» между собой при переименовании.
public final class CompositeTrainableSet: TrainableSet {

    private let parts: [TrainableSet]
    private var counts: [Int] = []

    public init(_ parts: [TrainableSet]) {
        precondition(!parts.isEmpty, "композиция без частей не имеет смысла")
        self.parts = parts
    }

    public convenience init(_ parts: TrainableSet...) {
        self.init(parts)
    }

    public var parameterNames: [String] { parts.flatMap { $0.parameterNames } }

    public func initialParameters() -> [MLXArray] {
        var out: [MLXArray] = []
        counts = []
        for p in parts {
            let ps = p.initialParameters()
            counts.append(ps.count)
            out.append(contentsOf: ps)
        }
        return out
    }

    /// Границы срезов. Считаются из `parameterNames`, а НЕ кэшируются из
    /// initialParameters: после `loadCheckpoint` тренер вызывает inject, ни
    /// разу не позвав initialParameters, и кэш был бы пуст.
    private var slices: [Int] {
        counts.isEmpty ? parts.map { $0.parameterNames.count } : counts
    }

    public func inject(_ ps: [MLXArray]) { distribute(ps) { $0.inject($1) } }
    public func commit(_ ps: [MLXArray]) { distribute(ps) { $0.commit($1) } }

    private func distribute(_ ps: [MLXArray], _ body: (TrainableSet, [MLXArray]) -> Void) {
        let sizes = slices
        precondition(ps.count == sizes.reduce(0, +),
                     "композиции передано \(ps.count) параметров, ожидалось \(sizes.reduce(0, +))")
        var offset = 0
        for (part, n) in zip(parts, sizes) {
            body(part, Array(ps[offset ..< offset + n]))
            offset += n
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Адаптеры LoRA / QLoRA
// ───────────────────────────────────────────────────────────────────────

/// Обучаются только loraA/loraB; база (в т.ч. квантованная) не трогается.
public final class LoRATrainableSet: TrainableSet {

    private let bb: X070Backbone
    private let slots: [(target: String, isA: Bool)]

    /// dtype подстановки. bf16 — исторический выбор LoRA-пути: адаптеры живут
    /// в модели в bf16, мастер-копия остаётся fp32. Оставлено как есть, чтобы
    /// численность совпадала с дорефакторной.
    private let injectDType: DType

    public init(_ bb: X070Backbone, injectDType: DType = .bfloat16) {
        self.bb = bb
        self.injectDType = injectDType
        // Сортировка по имени таргета — детерминированный порядок упаковки:
        // от него зависит соответствие параметров, моментов и чекпоинтов.
        var s: [(String, Bool)] = []
        for t in bb.loraA.keys.sorted() {
            s.append((t, true))
            s.append((t, false))
        }
        self.slots = s.map { (target: $0.0, isA: $0.1) }
    }

    public var parameterNames: [String] {
        slots.map { $0.target + ($0.isA ? ".lora_a" : ".lora_b") }
    }

    public func initialParameters() -> [MLXArray] {
        let ps = slots.map { s -> MLXArray in
            (s.isA ? bb.loraA[s.target]! : bb.loraB[s.target]!).asType(.float32)
        }
        eval(ps)
        return ps
    }

    public func inject(_ ps: [MLXArray]) {
        for (i, s) in slots.enumerated() {
            if s.isA { bb.loraA[s.target] = ps[i].asType(injectDType) }
            else     { bb.loraB[s.target] = ps[i].asType(injectDType) }
        }
    }

    public func commit(_ ps: [MLXArray]) {
        inject(ps)
        eval(Array(bb.loraA.values) + Array(bb.loraB.values))
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Веса backbone (верхние N слоёв, либо все — претрейн)
// ───────────────────────────────────────────────────────────────────────

/// Обучаются перечисленные веса backbone целиком, через `wOverride`.
///
/// `trainLayers` определяет, какие слои идут через дифференцируемое WKV-ядро;
/// его выставляет вызывающий (для верхних N — Set(freeze..<nLayer), для
/// претрейна — все слои).
public final class BackboneWeightsTrainableSet: TrainableSet {

    private let bb: X070Backbone
    public let keys: [String]
    private let extra: [String: MLXArray]     // веса вне backbone (напр. голова)
    private let extraKeys: [String]

    /// - keys:  ключи весов backbone (порядок нормализуется сортировкой).
    /// - extra: дополнительные обучаемые тензоры, которых нет в backbone —
    ///          например классификационная голова. Инжектятся не в wOverride,
    ///          а отдаются вызывающему через `currentExtra`.
    public init(_ bb: X070Backbone, keys: [String], extra: [String: MLXArray] = [:]) {
        self.bb = bb
        self.keys = keys.sorted()
        self.extra = extra
        self.extraKeys = extra.keys.sorted()
    }

    public var parameterNames: [String] { keys + extraKeys }

    /// Последние подставленные значения `extra` — objective читает их отсюда,
    /// т.к. в backbone им места нет.
    public private(set) var currentExtra: [String: MLXArray] = [:]

    public func initialParameters() -> [MLXArray] {
        let ps = keys.map { bb.w[$0]!.asType(.float32) }
                 + extraKeys.map { extra[$0]!.asType(.float32) }
        eval(ps)
        return ps
    }

    public func inject(_ ps: [MLXArray]) {
        var ov: [String: MLXArray] = [:]
        for (i, k) in keys.enumerated() { ov[k] = ps[i] }
        bb.wOverride = ov
        var ex: [String: MLXArray] = [:]
        for (j, k) in extraKeys.enumerated() { ex[k] = ps[keys.count + j] }
        currentExtra = ex
    }

    /// Записать обученные веса в саму модель и снять подмену — после этого
    /// backbone готов к инференсу без обучающих hooks.
    public func commit(_ ps: [MLXArray]) {
        let dt = bb.w[keys.first ?? ""]?.dtype ?? .bfloat16
        for (i, k) in keys.enumerated() { bb.w[k] = ps[i].asType(dt) }
        var ex: [String: MLXArray] = [:]
        for (j, k) in extraKeys.enumerated() { ex[k] = ps[keys.count + j] }
        currentExtra = ex
        bb.wOverride = nil
        eval(Array(bb.w.values))
    }

    /// Ключи весов слоёв [from..<nLayer] плюс ln_out — типовой набор для
    /// обучения верхних N слоёв.
    public static func topLayerKeys(_ bb: X070Backbone, from: Int) -> [String] {
        var keys = bb.w.keys.filter { k in
            (from ..< bb.cfg.nLayer).contains { k.hasPrefix("blocks.\($0).") }
        }
        keys.append("ln_out.weight")
        keys.append("ln_out.bias")
        return keys.sorted()
    }
}
