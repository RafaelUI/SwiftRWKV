#!/bin/bash
# Мутационная проверка (23.10): ранняя остановка обучения состояния по отложенному лоссу.
#   ./Scripts/mutations_early_stopping.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
IS=Sources/RWKVGen/InitialState.swift
SB=Sources/RWKVGen/Training/SequentialBackward.swift
FILTER='StateEarlyStopping|WeightStreaming'
run() { local d="$1"; shift; printf '%-58s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "отложенный лосс без маски вопроса" \
    "$SB" '            let masked = ignoresTargets(batch.y)' '            let masked = false'
run "пачки отложенного лосса поровну, а не по токенам" \
    "$SB" $'            weighted += Double(loss.item(Float.self)) * Double(n)\n            tokens += n' \
          $'            weighted += Double(loss.item(Float.self))\n            tokens += 1'
run "отложенный лосс не отпускает голову (подкачка)" \
    "$SB" $'            eval(loss)\n            bb.releaseStreamedHead()' $'            eval(loss)'
run "minDelta не учитывается" \
    "$IS" '                let better = loss < best.loss - cfg.minDelta' '                let better = loss < best.loss'
run "улучшение — рост лосса" \
    "$IS" '                let better = loss < best.loss - cfg.minDelta' '                let better = loss > best.loss - cfg.minDelta'
run "возвращается последнее состояние, а не лучшее" \
    "$IS" '            set.commit(best.params)' '            _ = best.params'
run "лучшее запоминается без своих параметров" \
    "$IS" 'best = (done, loss, trainer.currentParameters)' 'best = (done, loss, best.params)'
run "терпение на проверку длиннее" \
    "$IS" 'if since >= cfg.patience && done < cfg.maxSteps' 'if since > cfg.patience && done < cfg.maxSteps'
run "остановка терпением не отмечается" \
    "$IS" '{ stopped = true; break }' '{ break }'
run "нет проверки до обучения" \
    "$IS" '            var history = [HeldOutReport.Point(step: 0, loss: best.loss)]' '            var history: [HeldOutReport.Point] = []'
run "отрезок на шаг длиннее evalEvery" \
    "$IS" 'return calls > length || isCancelled()' 'return calls > length + 1 || isCancelled()'
run "evalEvery не делит обучение на отрезки" \
    "$IS" '                let length = Swift.min(cfg.evalEvery, cfg.maxSteps - done)' '                let length = cfg.maxSteps - done'
run "отрезок, прерванный отменой, не проверяется" \
    "$IS" '                if last.steps == done { break }' '                if last.steps < done + length { break }'
