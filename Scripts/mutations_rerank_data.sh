#!/bin/bash
# Мутационная проверка данных, кэша состояний и метрик реранкера.
#
#   ./Scripts/mutations_rerank_data.sh [фильтр тестов]
#
# Правдоподобные дефекты: перепутанное смещение строки в кэше, оптимистичное
# разрешение ничьих, потерянная дедупликация префиксов, ярлык позиции
# позитива.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
DATA=Sources/RWKVRerank/RerankData.swift
CACHE=Sources/RWKVRerank/StateCache.swift
ENC=Sources/RWKVRerank/RerankEncode.swift
MET=Sources/RWKVRerank/RerankMetrics.swift

FILTER="${1:-RerankDataTests|StateCacheTests|RerankMetricsTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Данные ──"

# Ярлык позиции: при listwise-лоссе голова выучит «позитив всегда нулевой»
# вместо задачи, и по лоссу это неотличимо — он будет исправно падать.
run "позиция позитива не перемешивается" \
    "$DATA" 'let shuffled = rng.shuffled(cand)' 'let shuffled = cand'

run "майненные негативы не попадают в кандидатов" \
    "$DATA" 'for n in negIds where cand.count < nCandidates {' \
            'for n in [Int]() where cand.count < nCandidates {'

run "hardNegs указывают на позиции ДО перемешивания" \
    "$DATA" 'hardNegs: mined.compactMap { position[$0] }.sorted()))' \
            'hardNegs: mined.sorted()))'

run "label берётся до перемешивания" \
    "$DATA" 'docIds: shuffled, label: position[posId]!,' \
            'docIds: shuffled, label: 0,'

run "пул не дедуплицируется" \
    "$DATA" 'if let i = docToId[doc] { return i }' 'if false, let i = docToId[doc] { return i }'

run "разбиение train/eval пересекается" \
    "$DATA" 'return (idx.dropFirst(nEval).map { samples[$0] },' \
            'return (idx.map { samples[$0] },'

run "резервуар вырождается в «первые N»" \
    "$DATA" 'let j = rng.below(seen)' 'let j = seen'

# НЕ ЛОВИТСЯ, и поймать нельзя. Перекос от `next() % n` имеет порядок
# n / 2^64 — для любого мыслимого числа кандидатов это на десятки порядков
# ниже статистического шума выборки, так что различить две реализации
# экспериментом невозможно в принципе. Отбрасывание хвоста здесь —
# дисциплина, а не исправление измеримого дефекта; оставлено потому, что
# стоит ноль, а не потому, что что-то чинит.
run "below(n) через остаток (безвредна, неразличима)" \
    "$DATA" 'var r = next()
        while r > limit { r = next() }
        return Int(r % bound)' 'return Int(next() % bound)'

run "формат triples-multi не распознаётся" \
    "$DATA" 'if let query = obj["query"] as? String, !query.isEmpty {' \
            'if false, let query = obj["query"] as? String, !query.isEmpty {'

run "строка без негативов проходит в выборку" \
    "$DATA" 'guard !negs.isEmpty else { return nil }
                return RerankRow(query: query, positive: positive,' \
            'guard true else { return nil }
                return RerankRow(query: query, positive: positive,'

echo
echo "── Шаблон ──"

run "запрос попадает в кэшируемый префикс" \
    "$DATA" 'docFirst ? "Instruct: \(instruct)\nDocument: \(document)\n"' \
            'docFirst ? "Instruct: \(instruct)\nDocument: \(document)\nQuery: "'

run "контракт шаблона одинаков для обоих порядков" \
    "$DATA" 'public var contract: String { docFirst ? "doc_first" : "query_first" }' \
            'public var contract: String { "doc_first" }'

echo
echo "── Кэш ──"

run "строка пишется по номеру в пачке, а не по своему смещению" \
    "$CACHE" 'let chunk = Data(bytes: src.baseAddress!.advanced(by: i * rowBytes),' \
             'let chunk = Data(bytes: src.baseAddress!.advanced(by: 0),'

run "gather читает всегда строку 0" \
    "$CACHE" 'let from = src.baseAddress!.advanced(by: r * rb)' \
             'let from = src.baseAddress!.advanced(by: 0)'

run "переполнение fp16 не проверяется" \
    "$CACHE" 'if dtype == .float16 && maxAbs > 60000 {' 'if false {'

run "несовместимый кэш принимается" \
    "$CACHE" 'guard shape[1] == wantSrc else {' 'guard true else {'

run "контракт кэша не сверяется" \
    "$CACHE" 'if let have = index.contract[key], have != want {' \
             'if let have = index.contract[key], false {'

echo
echo "── Кодирование ──"

run "префиксы не дедуплицируются" \
    "$ENC" 'var pi = prefixKey[key]' 'var pi: Int? = nil'

run "хвост запроса стартует с нуля, а не с состояния префикса" \
    "$ENC" 'let pairState = model.encode(qidx, mask: qmask, endIdx: qend,
                                             state: sub)' \
           'let pairState = model.encode(qidx, mask: qmask, endIdx: qend,
                                             state: nil)'

run "обрезка документа по символам вместо токенов" \
    "$ENC" 'let body = Array(tok.encode(document).prefix(cfg.maxDocTokens))' \
           'let body = tok.encode(String(document.prefix(cfg.maxDocTokens)))'

run "чужой токенизатор не проверяется" \
    "$ENC" 'guard maxId < model.base.cfg.vocab else {' 'guard true else {'

echo
echo "── Метрики ──"

# Оптимистичное разрешение ничьих: у необученной головы все скоры равны, и
# MRR стал бы 1.0 — колонка «до обучения» превратилась бы в ложь.
run "ничьи разрешаются оптимистично" \
    "$MET" 'ranks[i] = 1.0 + Double(greater) + Double(ties - 1) / 2.0' \
           'ranks[i] = 1.0 + Double(greater)'

run "ничьи разрешаются пессимистично" \
    "$MET" 'ranks[i] = 1.0 + Double(greater) + Double(ties - 1) / 2.0' \
           'ranks[i] = 1.0 + Double(greater) + Double(ties - 1)'

# Мутация «убрать округление ранга вверх» не ловилась — и разбор показал,
# что округление было ТОЖДЕСТВЕННЫМ: ceil(x) ≤ k ⟺ x ≤ k при целом k.
# Строка убрана из кода вместе с мутацией; поведение закреплено тестом
# testSharedFirstPlaceIsNotRecallAt1.

run "майненные негативы считаются как добранные" \
    "$MET" 'let hard = Set(hardNegs.isEmpty ? [] : hardNegs[i])' \
           'let hard = Set<Int>()'

run "nDCG не обрезается на десятке" \
    "$MET" '$0 + ($1 <= 10 ? 1.0 / log2($1 + 1.0) : 0.0)' \
           '$0 + 1.0 / log2($1 + 1.0)'

run "пол случайного угадывания посчитан как 1/C" \
    "$MET" '2.0 / Double(C + 1)' '1.0 / Double(C)'
