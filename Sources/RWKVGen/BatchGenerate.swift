import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Батчевый декод (25.10) — ОПЦИЯ, а не путь `generate`.
//
//  Несколько независимых генераций идут одним проходом на токен: префилл
//  всех промптов — параллельным проходом с правым паддингом и маской, шаг —
//  тот же параллельный проход на T = 1 с батчевым состоянием
//  (`bodyWithState`). Своей арифметики нет: это ровно путь префилла и
//  кэша префиксов реранкера.
//
//  Почему не умолчание. Память растёт с батчем: состояние строки — L·H·S·S
//  чисел fp32 (0.1B — 2.4 МБ, 1.5B — 12.6 МБ) плюс активации префилла
//  B × max(длина промпта). И числа другие: шаг идёт параллельным ядром, а не
//  рекуррентным, поэтому логиты отличаются от `generate` в пределах того, что
//  разделяет эти два пути вообще (≈1e-6 в fp32; с приведением выхода WKV в
//  bf16 — ≈1e-2), и жадный выбор при почти равных кандидатах может уйти в
//  другую сторону.
//
//  Строка, остановившаяся раньше других, из батча убирается (`gather`):
//  дальше считаются только живые.
// ───────────────────────────────────────────────────────────────────────

extension X070Backbone {

    /// Логиты по скрытому состоянию [B, D] → [B, vocab] — путём шага
    /// генерации (квантованная база — ядрами GEMV, адаптеры — с прибавкой).
    func batchHead(_ x: MLXArray) -> MLXArray {
        projectForBlock(x, "head.weight", lora: "head", decode: true)
    }

    /// Префилл нескольких промптов одним проходом: правый паддинг, маска,
    /// конец каждой строки — её последний настоящий токен. Логиты [B, vocab]
    /// — по последнему токену строки; состояние — для `stepBatch`.
    public func prefillBatch(_ prompts: [[Int]]) -> (logits: MLXArray, state: RWKVBatchState) {
        precondition(!prompts.isEmpty, "prefillBatch: нет промптов")
        precondition(prompts.allSatisfy { !$0.isEmpty }, "prefillBatch: пустой промпт")
        let lengths = prompts.map(\.count)
        let T = lengths.max()!
        var flat: [Int32] = []
        flat.reserveCapacity(prompts.count * T)
        for p in prompts { flat += p.map { Int32($0) } + [Int32](repeating: 0, count: T - p.count) }
        let ids = MLXArray(flat, [prompts.count, T])
        let mask = prompts.count > 1 && lengths.contains(where: { $0 != T })
            ? buildMask(lengths: lengths, total: T) : nil
        let endIdx = lastRealIndex(lengths: lengths)
        let (lnOut, state) = bodyWithState(ids, mask: mask, endIdx: endIdx)
        let last = gatherLast(lnOut, endIdx).reshaped([prompts.count, cfg.nEmbd])
        return (batchHead(last), state)
    }

    /// Один шаг для всех строк: токен на строку, логиты [B, vocab].
    public func stepBatch(_ ids: [Int], state: inout RWKVBatchState) -> MLXArray {
        precondition(ids.count == state.batch, "stepBatch: \(ids.count) токенов на батч \(state.batch)")
        let x = MLXArray(ids.map { Int32($0) }, [ids.count, 1])
        let (lnOut, next) = bodyWithState(x, state: state)
        state = next
        return batchHead(lnOut.reshaped([ids.count, cfg.nEmbd]))
    }

    /// Сгенерировать продолжения нескольких промптов одним батчем.
    ///
    /// Остановка — как у `generate` (лимит, стоп-токены, стоп-строки), по
    /// строкам независимо. Выбор токена — CPU-сэмплер `Sampler`, у строки `i`
    /// сид `sampling.seed &+ i`: жадный выбор от сида не зависит, а с
    /// температурой строки не делят поток случайных чисел. Потока по кускам и
    /// отмены нет — это путь пакетной обработки.
    public func generateBatch(prompts: [[Int]], config: GenerationConfig,
                              tokenizer: any RWKVTokenizer) -> [GenerationResult] {
        let B = prompts.count
        guard B > 0 else { return [] }
        var samplers = (0 ..< B).map { i -> Sampler in
            var c = config.sampling
            c.seed = c.seed &+ UInt64(i)
            var s = Sampler(config: c)
            if config.penalizePrompt { for id in prompts[i] { s.observe(id) } }
            return s
        }
        var produced = [[Int]](repeating: [], count: B)
        var bytes = [[UInt8]](repeating: [], count: B)
        var reasons = [StopReason?](repeating: nil, count: B)
        let holdBack = config.stopStrings.map { $0.utf8.count }.max().map { $0 - 1 } ?? 0

        var (logits, state) = prefillBatch(prompts)
        var alive = Array(0 ..< B)              // строки батча → номера промптов
        for _ in 0 ..< config.maxTokens {
            eval(logits)
            var next: [Int] = []
            var keep: [Int] = []
            for (row, i) in alive.enumerated() {
                let id = samplers[i].next(logits[row])
                produced[i].append(id)
                if config.stopTokens.contains(id) { reasons[i] = .stopToken(id); continue }
                let grewFrom = bytes[i].count
                bytes[i].append(contentsOf: tokenizer.rawBytes(id) ?? [])
                if let hit = firstStopHit(bytes[i], config.stopStrings, from: max(0, grewFrom - holdBack)) {
                    bytes[i].removeSubrange(hit.start ..< bytes[i].count)
                    reasons[i] = .stopString(hit.needle)
                    continue
                }
                next.append(id); keep.append(row)
            }
            if keep.isEmpty { break }
            if keep.count < alive.count {
                state = state.gather(keep)
                alive = keep.map { alive[$0] }
            }
            if produced[alive[0]].count >= config.maxTokens { break }
            logits = stepBatch(next, state: &state)
        }
        return (0 ..< B).map {
            GenerationResult(tokens: produced[$0], text: String(decoding: bytes[$0], as: UTF8.self),
                             stopReason: reasons[$0] ?? .maxTokens)
        }
    }

    /// То же, промпты — строки.
    public func generateBatch(prompts: [String], config: GenerationConfig,
                              tokenizer: any RWKVTokenizer) -> [GenerationResult] {
        generateBatch(prompts: prompts.map { tokenizer.encode($0) }, config: config, tokenizer: tokenizer)
    }
}
