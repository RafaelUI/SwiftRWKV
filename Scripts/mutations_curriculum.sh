#!/bin/bash
# Мутационная проверка (17.10): прогонщик curriculum эмбеддера
# (EmbeddingCurriculum): отчёт после каждой стадии, возобновление, отказ на
# чужом плане.
#
#   ./Scripts/mutations_curriculum.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
EC=Sources/RWKVEmbedding/EmbeddingCurriculum.swift

FILTER='EmbeddingCurriculumTests'
run() {
    local desc="$1"; shift
    printf '%-56s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

run "параметры прошлой стадии не подставляются" \
    "$EC" '                try restore(trainable, from: paramsURL(directory, last.index))' '                _ = last'
run "подставляются параметры первой стадии" \
    "$EC" '                try restore(trainable, from: paramsURL(directory, last.index))' '                try restore(trainable, from: paramsURL(directory, 0))'
run "продолжение начинает с первой стадии" \
    "$EC" '        for i in report.stages.count ..< stages.count {' '        for i in 0 ..< stages.count {'
run "оборванная стадия записывается" \
    "$EC" '            guard r.steps >= stage.config.maxSteps else { break }' '            _ = r.steps'
run "параметры стадии не сохраняются" \
    "$EC" '            try save(trainable, to: paramsURL(directory, i))' '            _ = i'
run "отчёт не пишется после стадии" \
    "$EC" '            try enc.encode(report).write(to: directory.appendingPathComponent(reportName), options: .atomic)' '            _ = enc'
run "план не сверяется" \
    "$EC" '            guard old.plan == plan, old.totalStages == stages.count else {' '            guard true else {'
run "данные не в отпечатке плана" \
    "$EC" '                for r in rows { put(r.task.rawValue); put(r.anchor); put(r.positive); put(r.negative) }' '                _ = rows'
run "число шагов не в отпечатке плана" \
    "$EC" '|\(c.warmupSteps)|\(c.maxSteps)|' '|\(c.warmupSteps)|'
run "режим не в отпечатке плана" \
    "$EC" 'put("embedding-curriculum-v1"); put("\(mode)"); ' 'put("embedding-curriculum-v1"); '
