import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Банк начальных состояний: своё состояние у каждой строки батча.
//
//  Подключённое состояние (`initialState`) одно на модель. Эмбеддеру с
//  инструкциями нужно своё на каждую инструкцию — и чтобы батч смешивал
//  инструкции так же, как без состояний (отрицательные внутри батча не
//  меняются). Банк — K состояний, батч K; `initialRows` [B] — номер
//  состояния каждой строки. Блок выбирает строки банка САМ (`take` внутри
//  блока), поэтому градиент банка — сумма по строкам с этим номером, и
//  обратный проход по блокам подхватывает банк как параметр блока
//  (`blocks.N.state0bank.*`), так же как одно состояние.
//
//  Банк и одно подключённое состояние не складываются: при заданных строках
//  берётся банк. Банк без строк — ошибка (молча стартовать с нуля или с
//  одного состояния значило бы считать не то). `withoutInitialState`
//  выключает и банк: документы эмбеддера с состоянием только для запросов.
//  Отпечаток подключённого (`initialStateFingerprint`) банк не видит.
// ───────────────────────────────────────────────────────────────────────

extension InitialLayerState {
    /// Строки банка (wkv [K, H, S, S]) для батча: [B, H, S, S].
    func selecting(_ rows: MLXArray) -> InitialLayerState {
        InitialLayerState(wkv: wkv.take(rows, axis: 0), tmixShift: tmixShift.take(rows, axis: 0),
                          cmixShift: cmixShift.take(rows, axis: 0), perRow: true)
    }
}

public extension X070Backbone {

    /// Подключённый банк (батч K) либо nil. Присваивание подключает; строки
    /// батча задаёт `withInitialRows`.
    var initialStateBank: RWKVBatchState? {
        get {
            guard let layers = initialBank else { return nil }
            return RWKVBatchState.stacked(wkv: layers.map { $0.wkv }, tmix: layers.map { $0.tmixShift },
                                          cmix: layers.map { $0.cmixShift }).evaluated()
        }
        set {
            guard let bank = newValue else { initialBank = nil; return }
            precondition(bank.nLayer == cfg.nLayer, "банк на \(bank.nLayer) слоёв, модель на \(cfg.nLayer)")
            let layers = (0 ..< cfg.nLayer).map { l in
                InitialLayerState(wkv: bank.layerWKV(l).asType(.float32),
                                  tmixShift: bank.layerTmixShift(l).asType(.float32),
                                  cmixShift: bank.layerCmixShift(l).asType(.float32))
            }
            precondition(layers[0].wkv.shape == [bank.batch, cfg.nHead, cfg.headSize, cfg.headSize],
                         "банк: wkv слоя \(layers[0].wkv.shape)")
            eval(layers.flatMap { [$0.wkv, $0.tmixShift, $0.cmixShift] })
            initialBank = layers
        }
    }

    /// Состояний в банке (0 — банка нет).
    var initialBankSize: Int { initialBank?[0].wkv.shape[0] ?? 0 }

    /// Проходы, построенные внутри `body`, стартуют строка за строкой с
    /// состояния банка `rows[i]` ([B] int32). После — прежние строки (в том
    /// числе при ошибке). Граф, построенный внутри, держит выбор сам.
    func withInitialRows<T>(_ rows: MLXArray?, _ body: () throws -> T) rethrows -> T {
        let saved = initialRows
        defer { initialRows = saved }
        initialRows = rows.map { $0.asType(.int32) }
        return try body()
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Банк как обучаемое множество
// ───────────────────────────────────────────────────────────────────────

/// Банк состояний как обучаемое множество: параметр слоя — [K, …] банка этого
/// слоя (`blocks.N.state0bank.wkv`). Строки, которых нет ни в одном батче,
/// градиента не получают и у Adam без затухания весов не двигаются.
///
/// Создание ПОДКЛЮЧАЕТ банк (с `initial`, иначе — уже подключённый) и
/// переводит все слои на дифференцируемое ядро.
public final class StateBankTrainableSet: TrainableSet {

    private let bb: X070Backbone
    public let parts: [StateTrainableSet.Part]

    public init(_ bb: X070Backbone, train: [StateTrainableSet.Part] = [.wkv], initial: RWKVBatchState? = nil) {
        precondition(!train.isEmpty, "обучаемый банк без единой части")
        self.bb = bb
        self.parts = StateTrainableSet.Part.allCases.filter(train.contains)
        if let initial { bb.initialStateBank = initial }
        precondition(bb.initialBank != nil, "банк состояний не подключён и не передан")
        bb.trainLayers = Set(0 ..< bb.cfg.nLayer)
    }

    public var parameterNames: [String] {
        (0 ..< bb.cfg.nLayer).flatMap { l in parts.map { "blocks.\(l).state0bank.\($0.rawValue)" } }
    }

    public var learningRateScales: [Float]? { nil }

    public func initialParameters() -> [MLXArray] {
        let ps = bb.initialBank!.flatMap { layer in
            parts.map { part -> MLXArray in
                switch part {
                case .wkv: return layer.wkv.asType(.float32)
                case .tmixShift: return layer.tmixShift.asType(.float32)
                case .cmixShift: return layer.cmixShift.asType(.float32)
                }
            }
        }
        eval(ps)
        return ps
    }

    public func inject(_ ps: [MLXArray]) {
        var layers = bb.initialBank!
        precondition(ps.count == layers.count * parts.count,
                     "банку передано \(ps.count) параметров, ожидалось \(layers.count * parts.count)")
        var i = 0
        for l in layers.indices {
            for part in parts {
                switch part {
                case .wkv: layers[l].wkv = ps[i]
                case .tmixShift: layers[l].tmixShift = ps[i]
                case .cmixShift: layers[l].cmixShift = ps[i]
                }
                i += 1
            }
        }
        bb.initialBank = layers
    }

    public func commit(_ ps: [MLXArray]) {
        inject(ps)
        eval(ps)
    }
}
