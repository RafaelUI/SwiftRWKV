#!/bin/bash
# Мутационная проверка (17.10): X070PartialFinetune — воспроизводимость по сиду
# и (после переноса) общий Trainer.
#
#   ./Scripts/mutations_partial.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
PF=Sources/RWKVGen/X070PartialFinetune.swift

FILTER='PartialFinetuneCharacterizationTests'
run() {
    local desc="$1"; shift
    printf '%-56s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── сид ──"
run "голова не от сида" \
    "$PF" 'let headInit = seed.map { MLXRandom.normal([numClasses, dim], key: MLXRandom.key($0)) }' 'let headInit = seed.map { _ in MLXRandom.normal([numClasses, dim]) }'
run "генератор порядка не от сида" \
    "$PF" 'var rng = seed.map { SplitMix64(seed: $0) }' 'var rng: SplitMix64? = nil'
run "перемешивание мимо генератора" \
    "$PF" 'if rng != nil { perm.shuffle(using: &rng!) } else { perm.shuffle() }' 'perm.shuffle()'
run "run не передаёт сид" \
    "$PF" 'batchSize: batchSize, lr: lr, seed: seed, onStep: onStep, onEpoch: onEpoch)' 'batchSize: batchSize, lr: lr, seed: nil, onStep: onStep, onEpoch: onEpoch)'

echo "── общий Trainer (эталон — прежний цикл, побитно) ──"
run "β₂ не 0.999" \
    "$PF" 'beta1: 0.9, beta2: 0.999, adamEps: 1e-8,' 'beta1: 0.9, beta2: 0.99, adamEps: 1e-8,'
run "отсечение градиента включено" \
    "$PF" 'gradClip: 0, weightDecay: 0,' 'gradClip: 0.1, weightDecay: 0,'
run "затухание весов включено" \
    "$PF" 'gradClip: 0, weightDecay: 0,' 'gradClip: 0, weightDecay: 0.01,'
run "косинусное расписание вместо постоянного" \
    "$PF" 'lrMin: lr, schedule: .constant,' 'lrMin: 0, schedule: .cosine,'
run "неполный последний батч выброшен" \
    "$PF" 'let stepsPerEpoch = (N + batchSize - 1) / batchSize' 'let stepsPerEpoch = N / batchSize'
run "перемешивание на каждом батче" \
    "$PF" '            if cursor >= perm.count {' '            if true {'
run "эпоха не останавливает тренер" \
    "$PF" 'trainer.run(isCancelled: { isCancelled() || step >= epochEnd },' 'trainer.run(isCancelled: { isCancelled() },'
run "веса после эпохи не забираются" \
    "$PF" '            params = trainer.currentParameters' '            _ = trainer.currentParameters'
