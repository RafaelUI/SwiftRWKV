#!/bin/bash
# Мутационная проверка (17.10): состояние только для запросов —
# X070Backbone.withoutInitialState, роль текста в эмбеддере (обучение по
# блокам, куски, GradCache, оценка, выдача), область в контракте чекпоинта.
#
#   ./Scripts/mutations_state_scope.sh
#
# Не включено: снятие `didSet { stepVersion &+= 1 }` у initialStateSuspended —
# равносильно на нынешнем коде (скомпилированный шаг декода получает состояние
# явно, подключённое его не касается); бамп оставлен для единообразия с
# initialLayers.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
RB=Sources/RWKVGen/RWKVBlock.swift
IS=Sources/RWKVGen/InitialState.swift
EM=Sources/RWKVEmbedding/Embedder.swift
MT=Sources/RWKVEmbedding/EmbeddingMetrics.swift
OB=Sources/RWKVEmbedding/EmbeddingObjective.swift
CK=Sources/RWKVEmbedding/EmbeddingCheckpoint.swift

FILTER='EmbeddingStateScopeTests|EmbeddingSequentialTests'
run() {
    local desc="$1"; shift
    printf '%-66s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── выключение подключённого состояния ──"
run "блок не смотрит на выключение" \
    "$RB" 'backbone.initialStateSuspended ? nil : backbone.initialLayers?[layer]' 'backbone.initialLayers?[layer]'
run "обёртка не выключает" \
    "$IS" '        initialStateSuspended = true' '        initialStateSuspended = false'
run "обёртка не возвращает прежнее" \
    "$IS" 'defer { initialStateSuspended = saved }' 'defer { initialStateSuspended = false }'

echo "── роль в модели и выдаче ──"
run "роль не действует" \
    "$EM" 'return role == .document ? try backbone.withoutInitialState(body) : try body()' 'return try body()'
run "роли перепутаны" \
    "$EM" 'return role == .document ? try backbone.withoutInitialState(body) : try body()' 'return role == .query ? try backbone.withoutInitialState(body) : try body()'
run "embed без роли" \
    "$EM" 'let h = building(role) { backbone.body(idx) }' 'let h = backbone.body(idx)'
run "pooledOnly без роли" \
    "$EM" 'l2Normalize(pool(building(role) { backbone.body(idx) }, poolIndex: poolIndex))' 'l2Normalize(pool(backbone.body(idx), poolIndex: poolIndex))'
run "выдача не передаёт роль" \
    "$EM" 'model.embed(idx, poolIndex: poolIdx, role: role)[0]' 'model.embed(idx, poolIndex: poolIdx, role: .query)[0]'
run "область не попадает в контракт выдачи" \
    "$EM" '        c.stateScope = model.stateScope
' ''

echo "── оценка ──"
# Две крышки ниже первым кругом (17.10) НЕ были пойманы: тест сверял MRR и
# точность, а ранги к сдвигу векторов кандидатов глухи. Тест теперь сверяет
# саму матрицу похожести (`EmbeddingMetrics.similarityObserver`).
run "embedAll без роли" \
    "$MT" 'let top = model.building(role) { SequentialBody.forward(model.backbone, ids: idx, plan: frozen).top }' 'let top = SequentialBody.forward(model.backbone, ids: idx, plan: frozen).top'
run "кандидаты retrieval — запросы" \
    "$MT" 'texts: positives + negatives,
                         terminator: terminator, maxTokens: maxTokens, role: .document)' 'texts: positives + negatives,
                         terminator: terminator, maxTokens: maxTokens, role: .query)'
run "sts: положительные — документы" \
    "$MT" 'texts: rows.map { String($0.positive.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens, role: .query)' 'texts: rows.map { String($0.positive.prefix(maxChars)) },
                         terminator: terminator, maxTokens: maxTokens, role: .document)'
run "метки классов — запросы" \
    "$MT" 'terminator: terminator, maxTokens: maxTokens, role: .document)
            .asType(.float32)' 'terminator: terminator, maxTokens: maxTokens, role: .query)
            .asType(.float32)'

echo "── обучение ──"
run "retrieval: положительные — запросы" \
    "$OB" 'symmetric ? [.query, .query, .query] : [.query, .document, .document]' 'symmetric ? [.query, .query, .query] : [.query, .query, .document]'
run "классификация eager: кандидаты — запросы" \
    "$OB" 'poolIndex: batch.candidatePool.reshaped([B * K]), role: .document)' 'poolIndex: batch.candidatePool.reshaped([B * K]), role: .query)'
run "классификация по блокам: кандидаты — запросы" \
    "$OB" 'roles: [.query, .document],' 'roles: [.query, .query],'
run "прямой проход куска без роли пачки" \
    "$OB" 'model.building(role(c)) { SequentialBody.forward(' 'model.building(.query) { SequentialBody.forward('
run "пересчёт блоков куска без роли пачки" \
    "$OB" '            model.building(role(k)) {' '            model.building(.query) {'
run "роль по номеру куска, а не пачки" \
    "$OB" 'func role(_ part: Int) -> TextRole? { roles?[parts[part].group] }' 'func role(_ part: Int) -> TextRole? { roles?[Swift.min(part, roles!.count - 1)] }'
run "GradCache: положительные с ролью якоря" \
    "$OB" 'poolIndex: batch.positivePool[start ..< end], role: roles[1]),' 'poolIndex: batch.positivePool[start ..< end], role: roles[0]),'

echo "── контракт чекпоинта ──"
run "область не пишется в метаданные" \
    "$CK" '"initial_state_scope": stateScope.rawValue]' '"initial_state_scope": "all"]'
run "область не читается из метаданных" \
    "$CK" 'if let v = md["initial_state_scope"], let s = InitialStateScope(rawValue: v) { stateScope = s }' ''
run "голова сохраняется без области модели" \
    "$CK" 'EmbeddingContract(pooling: pooling, stateScope: stateScope)' 'EmbeddingContract(pooling: pooling)'
run "загрузка не сверяет область" \
    "$CK" '            if want.stateScope != stateScope {' '            if false {'
run "fromHead не берёт область из файла" \
    "$CK" 'pooling: contract.pooling, stateScope: contract.stateScope)' 'pooling: contract.pooling)'
