#!/bin/bash
# Мутационная проверка (17.10): возобновление обучения реранкера
# (RerankTraining.train(checkpoint:isCancelled:)) и чекпоинт тренера словарём.
#
#   ./Scripts/mutations_rerank_resume.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
RT=Sources/RWKVRerank/RerankTraining.swift
TR=Sources/RWKVGen/Training/Trainer.swift

FILTER='RerankTrainingTests'
run() {
    local desc="$1"; shift
    printf '%-60s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── продолжение ──"
run "порядок данных не прокручивается" \
    "$RT" 'for _ in 0 ..< saved.stepsDone { _ = nextBatch() }' 'for _ in 0 ..< 0 { _ = nextBatch() }'
run "тензоры тренера не восстанавливаются" \
    "$RT" '            try trainer.restore(fromCheckpoint: savedArrays)' '            _ = savedArrays'
run "отчёты эпох не восстанавливаются" \
    "$RT" '            reports = saved.reports' '            reports = []'
run "счётчик шагов не восстанавливается" \
    "$RT" '            stepsDone = saved.stepsDone' '            stepsDone = 0'
run "лучшие веса не восстанавливаются" \
    "$RT" '            if saved.bestScore != nil {' '            if false {'
run "лучшая оценка не записывается" \
    "$RT" 'bestScore: bestParams == nil ? nil : bestScore,' 'bestScore: nil,'
run "оценка до обучения не из чекпоинта" \
    "$RT" '            before = saved.before' '            before = nil'
run "лучшая эпоха не восстанавливается" \
    "$RT" '            bestEpoch = saved.bestEpoch' '            bestEpoch = 0'
echo "── отказ и отмена ──"
run "ключ обучения не сверяется" \
    "$RT" '            guard st.key == resumeKey else { throw RerankResumeError.otherRun(checkpoint.path) }' '            _ = st.key'
run "шаг обучения не в ключе" \
    "$RT" '        var parts = ["rerank-resume-v1", "\(c.lr)", ' '        var parts = ["rerank-resume-v1", '
run "отложенный кэш не в ключе" \
    "$RT" '                     "eval \(evalCache.map { "\($0.nSamples)x\($0.nCandidates) \($0.shape)" } ?? "none")",' ''
run "отмена не останавливает" \
    "$RT" '            if epoch < config.epochs, isCancelled?() == true { cancelled = true; break }' '            _ = isCancelled'
run "чекпоинт не пишется" \
    "$RT" '                try Self.writeAtomically(arrays, state: st, to: checkpoint)' '                _ = (arrays, st)'
echo "── тренер ──"
run "моменты Adam не восстанавливаются" \
    "$TR" 'ms.append(d["m/" + name]?.asType(.float32)' 'ms.append(d["m_/" + name]?.asType(.float32)'
run "счётчик Adam не восстанавливается" \
    "$TR" '            t = Int(meta[1].item(Int32.self))' '            t = 0'

echo "── пачка больше кэша (25.10) ──"
run "пачка не ограничена кэшем" \
    "$RT" 'let bs = Swift.min(config.batchSize, n)' 'let bs = config.batchSize'
run "пачка короче кэша при большом batchSize" \
    "$RT" 'let bs = Swift.min(config.batchSize, n)' 'let bs = config.batchSize > n ? Swift.max(1, n - 1) : config.batchSize'
