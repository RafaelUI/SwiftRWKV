#!/bin/bash
# Мутационная проверка выдачи реранкера: прямой путь, индекс префиксов,
# контракт подачи текста.
#
#   ./Scripts/mutations_rerank_serve.sh [фильтр тестов]
#
# Дефекты здесь того же рода, что и во всём реранкере: почти ни один не
# падает. Индекс с чужой инструкцией, хвост, склеенный не с тем префиксом,
# порядок по возрастанию скора, контракт, взятый из умолчаний вместо файла, —
# всё это выдаёт осмысленно выглядящий список, просто не тот.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
INF=Sources/RWKVRerank/RerankInference.swift

FILTER="${1:-RerankInferenceTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Прямой путь ──"

run "хвост запроса не дописывается к префиксу" \
    "$INF" '+ RerankEncoder.suffixIds(tokenizer, enc, document: d, query: query)' \
           '+ [Int]()'

run "префикс берётся от первого документа для всех" \
    "$INF" 'instruct: ins, document: d)' \
           'instruct: ins, document: docs[0])'

run "проверка словаря снята" \
    "$INF" 'if start == 0 { try RerankEncoder.checkVocab(model, Array(seqs)) }
            let (idx, mask, endIdx) = RerankEncoder.batchIds(
                Array(seqs), bucket: enc.lengthBucket)
            let s = model(idx' \
           'let (idx, mask, endIdx) = RerankEncoder.batchIds(
                Array(seqs), bucket: enc.lengthBucket)
            let s = model(idx'

echo
echo "── Индекс ──"

# Если продолжать не с состояния префикса, а с нуля, скор считается по
# ОДНОМУ запросу без документа. Числа выходят правдоподобные и одинаковые
# для всех кандидатов — то есть ранжирование становится случайным.
run "хвост считается без состояния префикса" \
    "$INF" 'let s = model(idx, mask: mask, endIdx: endIdx, state: sub)' \
           'let s = model(idx, mask: mask, endIdx: endIdx)'

run "подмножество документов игнорируется, берутся первые" \
    "$INF" 'let sub = index.state.gather(part).asType(.float32)' \
           'let sub = index.state[0 ..< part.count].asType(.float32)'

run "в индекс кладётся состояние ПАРЫ, а не префикса" \
    "$INF" 'RerankEncoder.prefixIds(tokenizer, enc, instruct: ins, document: $0)' \
           'RerankEncoder.prefixIds(tokenizer, enc, instruct: ins, document: $0)
                + RerankEncoder.suffixIds(tokenizer, enc, document: $0, query: "")'

run "индекс строится с умолчательной инструкцией" \
    "$INF" 'let ins = instruct ?? config.instruct
        let enc = config.encode
        if enc.cacheLimitGB > 0 {' \
           'let ins = defaultRerankInstruct
        let enc = config.encode
        if enc.cacheLimitGB > 0 {'

run "пачки индекса склеиваются в обратном порядке" \
    "$INF" 'let state = parts.count == 1 ? parts[0]
                                     : RWKVBatchState.concatenated(parts)' \
           'let state = parts.count == 1 ? parts[0]
                                     : RWKVBatchState.concatenated(parts.reversed())'

echo
echo "── Контракт ──"

run "контракт индекса не проверяется вовсе" \
    "$INF" 'try index.check(cfg)' 'if false { try index.check(cfg) }'

run "проверка контракта пропускает несовпадение" \
    "$INF" 'guard have == want else {' 'guard true else {'

run "обрезка документа не входит в контракт префикса" \
    "$INF" 'public static let prefixKeys = ["template", "instruct", "max_doc_tokens"]' \
           'public static let prefixKeys = ["template", "instruct"]'

run "инструкция не входит в контракт префикса" \
    "$INF" 'public static let prefixKeys = ["template", "instruct", "max_doc_tokens"]' \
           'public static let prefixKeys = ["template", "max_doc_tokens"]'

run "индекс строится и при query_first" \
    "$INF" 'guard config.encode.template.docFirst else {
            throw RerankServeError.indexRequiresDocFirst
        }' \
           'if false { throw RerankServeError.indexRequiresDocFirst }'

echo
echo "── Чекпоинт ──"

run "обрезка документа берётся из умолчаний, а не из файла" \
    "$INF" 'if let v = md["max_doc_tokens"], let n = Int(v) { enc.maxDocTokens = n }' \
           ''

run "инструкция берётся из умолчаний, а не из файла" \
    "$INF" 'self.instruct = md["instruct"] ?? defaultRerankInstruct' \
           'self.instruct = defaultRerankInstruct'

run "терминатор из файла не разбирается" \
    "$INF" 'if let v = md["terminator"] { enc.terminator = (v == "none" || v.isEmpty) ? nil : Int(v) }' \
           ''

run "поправки применяются ДО чтения файла" \
    "$INF" 'var cfg = RerankServingConfig(metadata: md)
        overrides?(&cfg)' \
           'var cfg = RerankServingConfig(metadata: [:])
        overrides?(&cfg)'

echo
echo "── Порядок ──"

run "сортировка по возрастанию скора" \
    "$INF" 'a.score != b.score ? a.score > b.score : a.rank < b.rank' \
           'a.score != b.score ? a.score < b.score : a.rank < b.rank'

run "ничьи разрешаются в обратном порядке" \
    "$INF" 'a.score != b.score ? a.score > b.score : a.rank < b.rank' \
           'a.score != b.score ? a.score > b.score : a.rank > b.rank'

run "возвращается позиция в массиве, а не индекс кандидата" \
    "$INF" 'return pairs.prefix(take).map { (index: $0.id, score: $0.score) }' \
           'return pairs.prefix(take).map { (index: $0.rank, score: $0.score) }'

echo
echo "── Индекс на диске ──"

run "тексты документов не пишутся" \
    "$INF" 'md["docs"] = String(data: try JSONEncoder().encode(docs),
                            encoding: .utf8) ?? "[]"' \
           'md["docs"] = String(data: try JSONEncoder().encode([String]()),
                            encoding: .utf8) ?? "[]"'

run "контракт не пишется в файл индекса" \
    "$INF" 'var md = contract
        md["format"] = Self.format' \
           'var md = [String: String]()
        md["format"] = Self.format'

echo
echo "── Потоковый индекс ──"

run "склейка по слоям, а не по документам" \
    "$INF" '"wkv": concatenated(wkv, axis: 1),' \
           '"wkv": concatenated(wkv, axis: 0),'

run "контракт в потоковый индекс не пишется" \
    "$INF" 'var md = contract
        md["format"] = DocIndex.format
        md["n_docs"] = String(docs.count)' \
           'var md = [String: String]()
        md["format"] = DocIndex.format
        md["n_docs"] = String(docs.count)'

echo
echo "── Пакетная выдача ──"

# Плоский список работ режется по границам ПАЧКИ, а не запросов, поэтому
# разложить результат обратно надо по (запрос, документ). Ошибка здесь даёт
# полный набор правдоподобных чисел, приписанных не тем запросам.
run "результат раскладывается по порядку в пачке" \
    "$INF" 'out[part[k].q][position[part[k].d]!] = v' \
           'out[part[k].q][k % ids.count] = v'

run "берётся состояние не того документа" \
    "$INF" 'let sub = index.state.gather(part.map { $0.d }).asType(.float32)' \
           'let sub = index.state.gather(part.map { _ in ids[0] }).asType(.float32)'

run "берётся хвост не того запроса" \
    "$INF" 'part.map { qIds[$0.q] }, bucket: enc.lengthBucket,' \
           'part.map { _ in qIds[0] }, bucket: enc.lengthBucket,'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше. Каждая — либо пробел в наборе"
echo "тестов, либо утверждение о том, что мутация безвредна; второе"
echo "нужно обосновать разбором, а не оставлять как есть."
