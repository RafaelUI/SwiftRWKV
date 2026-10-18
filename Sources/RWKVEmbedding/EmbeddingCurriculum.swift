import Foundation
import CryptoKit
import MLX
import RWKVGen

// ───────────────────────────────────────────────────────────────────────
//  Прогонщик curriculum эмбеддера: стадии подряд над одной моделью, отчёт
//  на диске после КАЖДОЙ стадии и возобновление после обрыва.
//
//  `EmbeddingFinetune.run` гоняет стадии в памяти: прогон, убитый на третьей
//  из пяти, не оставляет ни чисел первых двух, ни обученных параметров. Здесь
//  в папке прогона после каждой стадии лежат:
//
//      curriculum.json            отчёт: отпечаток плана, завершённые стадии
//                                 (метрики до / после, лосс, шаги, время)
//      stage<i>.params.safetensors   параметры обучаемого множества после
//                                 стадии i (fp32-мастер, по именам)
//
//  Возобновление — по СТАДИЯМ. Повторный запуск с тем же планом подставляет
//  параметры последней завершённой стадии и продолжает со следующей; стадия,
//  оборванная посередине, начинается заново. Так числа возобновлённого
//  прогона побитно те же, что у непрерывного: у стадии свой сид порядка
//  данных, оптимизатор у каждой стадии свой (`EmbeddingFinetune.run` тоже
//  создаёт его заново), а состояние на входе стадии — ровно сохранённый
//  мастер. Другой план (стадии, данные, шаги, режим) в ту же папку — отказ:
//  смешанный отчёт был бы правдоподобным и неверным.
//
//  Модель вызывающий собирает так же, как для первого запуска (та же база,
//  те же развёрнутые слои, то же подключённое состояние): прогонщик хранит
//  обучаемое множество, а не базу.
// ───────────────────────────────────────────────────────────────────────

public enum EmbeddingCurriculumError: Error, CustomStringConvertible {
    case planMismatch(directory: String)
    case corrupt(String)

    public var description: String {
        switch self {
        case .planMismatch(let d):
            return "\(d): в папке прогон другого плана (стадии, данные, шаги или режим не те); "
                + "продолжать его этим планом значило бы смешать два прогона — укажи другую папку"
        case .corrupt(let s): return "папка прогона повреждена: \(s)"
        }
    }
}

/// Отчёт прогона — то, что лежит в `curriculum.json`.
public struct EmbeddingCurriculumReport: Codable, Equatable, Sendable {
    public struct Stage: Codable, Equatable, Sendable {
        public var index: Int
        public var task: String
        public var steps: Int
        public var finalLoss: Double
        public var seconds: Double
        public var before: Double?
        public var after: Double?
        public var beforeText: String?
        public var afterText: String?
    }
    public var version = 1
    /// Отпечаток плана (`EmbeddingCurriculum.planFingerprint`).
    public var plan: String
    public var totalStages: Int
    /// Завершённые стадии, по порядку.
    public var stages: [Stage]
    public var finished: Bool { stages.count == totalStages }
}

public enum EmbeddingCurriculum {

    public static let reportName = "curriculum.json"

    static func paramsURL(_ dir: URL, _ i: Int) -> URL {
        dir.appendingPathComponent("stage\(i).params.safetensors")
    }

    /// Отпечаток плана: всё, от чего зависят числа стадий, — режим, область
    /// состояния, терминатор и по каждой стадии задача, данные (тексты обеих
    /// выборок), шаг, расписание, шагов, батч, оптимизатор, обрезки, сид, пути
    /// счёта. Не входит то, что чисел не меняет: частота логов, потолки кэша,
    /// конвейер.
    public static func planFingerprint(mode: BaseTrainingMode, stateScope: InitialStateScope,
                                       stages: [EmbeddingStage], terminator: Int) -> String {
        var h = SHA256()
        func put(_ s: String) { h.update(data: Data(s.utf8) + Data([0])) }
        put("embedding-curriculum-v1"); put("\(mode)"); put(stateScope.rawValue); put("\(terminator)")
        for st in stages {
            let c = st.config
            put("\(st.task.rawValue)|\(c.lr)|\(c.lrMin)|\(c.schedule.rawValue)|\(c.warmupSteps)|\(c.maxSteps)|"
                + "\(c.gradClip)|\(c.weightDecay)|\(c.beta1)|\(c.beta2)|\(c.adamEps)|\(c.gradAccum)|"
                + "\(st.batchSize)|\(st.gradCacheChunk)|\(st.sequentialBackward)|\(st.bodyChunkTokens)|"
                + "\(st.traceStride)|\(st.temperature)|\(st.maxTokens.map(String.init) ?? "none")|"
                + "\(st.maxChars)|\(st.seed)|\(st.evaluateOnFullPool)"
                + (st.dedupeEvaluationPool ? "|dedupe" : ""))
            for (name, rows) in [("train", st.train), ("held", st.heldOut)] {
                put("\(name):\(rows.count)")
                for r in rows { put(r.task.rawValue); put(r.anchor); put(r.positive); put(r.negative) }
            }
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Прочитать отчёт папки прогона (nil — прогона там нет).
    public static func report(in directory: URL) throws -> EmbeddingCurriculumReport? {
        let url = directory.appendingPathComponent(reportName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            return try JSONDecoder().decode(EmbeddingCurriculumReport.self, from: Data(contentsOf: url))
        } catch {
            throw EmbeddingCurriculumError.corrupt("\(reportName) не разобрался: \(error)")
        }
    }

    /// Прогнать стадии в папке `directory`, продолжая с того места, где
    /// прошлый прогон того же плана остановился.
    ///
    /// - onStage: после записи стадии на диск (индекс стадии и результат).
    /// - Returns: отчёт; `finished == false`, если прогон отменён.
    @discardableResult
    public static func run(model: EmbeddingModel,
                           tokenizer: WorldTokenizer,
                           mode: BaseTrainingMode,
                           stages: [EmbeddingStage],
                           directory: URL,
                           terminator: Int = 0,
                           isCancelled: () -> Bool = { false },
                           onResume: (Int) -> Void = { _ in },
                           onStage: (Int, StageResult) -> Void = { _, _ in },
                           onStep: (Int, TrainingStep) -> Void = { _, _ in }) throws -> EmbeddingCurriculumReport {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let plan = planFingerprint(mode: mode, stateScope: model.stateScope, stages: stages,
                                   terminator: terminator)
        // Множество — одно на весь прогон, как в `EmbeddingFinetune.run`.
        let trainable = EmbeddingTrainable.make(model: model, mode: mode)

        var report = EmbeddingCurriculumReport(plan: plan, totalStages: stages.count, stages: [])
        if let old = try Self.report(in: directory) {
            guard old.plan == plan, old.totalStages == stages.count else {
                throw EmbeddingCurriculumError.planMismatch(directory: directory.path)
            }
            report = old
            if let last = old.stages.last {
                try restore(trainable, from: paramsURL(directory, last.index))
            }
            onResume(old.stages.count)
        }

        for i in report.stages.count ..< stages.count {
            if isCancelled() { break }
            let stage = stages[i]
            let r = try EmbeddingFinetune.runStage(model: model, tokenizer: tokenizer, trainable: trainable,
                                                   stage: stage, terminator: terminator,
                                                   isCancelled: isCancelled,
                                                   onStep: { onStep(i, $0) })
            // Стадия, оборванная отменой, не записывается: продолжение начнёт
            // её заново с параметров предыдущей.
            guard r.steps >= stage.config.maxSteps else { break }
            try save(trainable, to: paramsURL(directory, i))
            report.stages.append(.init(index: i, task: stage.task.rawValue, steps: r.steps,
                                       finalLoss: Double(r.finalLoss), seconds: r.seconds,
                                       before: r.before?.headline, after: r.after?.headline,
                                       beforeText: r.before?.description, afterText: r.after?.description))
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(report).write(to: directory.appendingPathComponent(reportName), options: .atomic)
            onStage(i, r)
        }
        return report
    }

    /// fp32-мастер обучаемого множества — по именам, через временный файл
    /// (обрыв во время записи не оставит половину файла под настоящим именем).
    static func save(_ trainable: TrainableSet, to url: URL) throws {
        let ps = trainable.initialParameters()
        var d: [String: MLXArray] = [:]
        for (name, p) in zip(trainable.parameterNames, ps) { d[name] = p.asType(.float32) }
        eval(Array(d.values))
        let tmp = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp.safetensors")
        try MLX.save(arrays: d, url: tmp)
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
        } else {
            try FileManager.default.moveItem(at: tmp, to: url)
        }
    }

    static func restore(_ trainable: TrainableSet, from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EmbeddingCurriculumError.corrupt("нет \(url.lastPathComponent)")
        }
        let d = try loadArrays(url: url)
        let names = trainable.parameterNames
        let ps = try names.map { name -> MLXArray in
            guard let p = d[name] else {
                throw EmbeddingCurriculumError.corrupt("\(url.lastPathComponent): нет параметра \(name)")
            }
            return p
        }
        for (name, (p, want)) in zip(names, zip(ps, trainable.initialParameters())) where p.shape != want.shape {
            throw EmbeddingCurriculumError.corrupt("\(url.lastPathComponent): \(name) формы \(p.shape), у множества \(want.shape)")
        }
        guard d.count == names.count else {
            throw EmbeddingCurriculumError.corrupt("\(url.lastPathComponent): параметров \(d.count), у множества \(names.count)")
        }
        trainable.commit(ps)
    }
}
