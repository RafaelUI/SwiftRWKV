#!/bin/bash
# Мутационная проверка (14.10): публичный выключатель конвейера
# (TrainingConfig.pipelined и его проброс) и потолок кэша на время оценки
# (EmbeddingStage.evaluationCacheLimitGB).
#
#   ./Scripts/mutations_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
TR=Sources/RWKVGen/Training/Trainer.swift
TC=Sources/RWKVGen/Training/TrainingConfig.swift
SB=Sources/RWKVGen/Training/SequentialBackward.swift
LF=Sources/RWKVGen/LoRAFinetune.swift
IS=Sources/RWKVGen/InitialState.swift
EF=Sources/RWKVEmbedding/EmbeddingFinetune.swift

FILTER='EmbeddingSequentialTests'
run() {
    local desc="$1"; shift
    printf '%-62s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── конвейер ──"
run "запуск не берёт конвейер из своей конфигурации" \
    "$TR" '        SequentialBody.pipelined = cfg.pipelined' '        _ = cfg.pipelined'
run "запуск не возвращает процессную настройку" \
    "$TR" 'defer { SequentialBody.pipelined = savedPipelined }' 'defer { _ = savedPipelined }'
run "обратный проход не смотрит на конвейер" \
    "$SB" 'if pipelined { asyncEvalCalls += 1; asyncEval(arrays) }' 'if false { asyncEvalCalls += 1; asyncEval(arrays) }'
run "конвейер включён по умолчанию" \
    "$TC" '                pipelined: Bool = false) {' '                pipelined: Bool = true) {'
run "LoRAConfig не передаёт конвейер обучению" \
    "$LF" 'logEvery: logEvery, pipelined: pipelined)' 'logEvery: logEvery, pipelined: false)'
run "StateTuneConfig не передаёт конвейер обучению" \
    "$IS" 'logEvery: logEvery, pipelined: pipelined)' 'logEvery: logEvery, pipelined: false)'

echo "── потолок кэша оценки ──"
run "потолок оценки не ставится" \
    "$EF" '        MLX.Memory.cacheLimit = gb.isInfinite ? MLX.Memory.memoryLimit : Int(gb * 1e9)' '        _ = gb'
run "без потолка — ноль вместо предела памяти" \
    "$EF" 'gb.isInfinite ? MLX.Memory.memoryLimit : Int(gb * 1e9)' 'gb.isInfinite ? 0 : Int(gb * 1e9)'
run "после оценки не возвращается потолок обучения" \
    "$EF" '        defer { MLX.Memory.cacheLimit = saved }' '        defer { _ = saved }'
run "оценка после обучения идёт мимо потолка" \
    "$EF" '''        let after = stage.heldOut.isEmpty ? nil
            : evaluateUnderCacheLimit(''' '''        let after = stage.heldOut.isEmpty ? nil
            : evaluate('''

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
