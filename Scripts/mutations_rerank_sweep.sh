#!/bin/bash
# Мутационная проверка кэша НАДМНОЖЕСТВА слоёв и развёртки по сидам.
#
#   ./Scripts/mutations_rerank_sweep.sh [фильтр тестов]
#
# Правдоподобные дефекты здесь особого рода: почти все они дают код, который
# работает. Срез не того слоя, слот не той ширины, переиспользованная голова
# между сидами, разброс, посчитанный как ноль, — ничто из этого не падает и
# не меняет форм. Отличить можно только по числам, и именно поэтому набор
# тестов обязан ловить каждую.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
CACHE=Sources/RWKVRerank/StateCache.swift
ENC=Sources/RWKVRerank/RerankEncode.swift
SWEEP=Sources/RWKVRerank/RerankSweep.swift
TRAIN=Sources/RWKVRerank/RerankTraining.swift
HEAD=Sources/RWKVRerank/RerankerHead.swift

FILTER="${1:-RerankSweepTests|StateCacheTests|RerankTrainingTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Разрешение слотов ──"

# Самое опасное место во всей правке: слои есть, но берутся не те. Формы
# сходятся, обучение идёт, числа правдоподобны.
# Самая правдоподобная ошибка: считать, что слот и есть номер слоя. При
# кэше «все слои подряд» это даже верно — и потому тихо доживает до кэша,
# собранного не с нулевого слоя.
run "номер слоя используется как номер слота" \
    "$CACHE" 'guard let slot = have.firstIndex(of: src) else {' \
             'guard let slot = Optional(src), src < have.count else {'

run "отсутствующий слой молча заменяется нулевым" \
    "$CACHE" 'throw StateCacheError.layersMissing(want: head.uniqueSources,
                                                    have: have)' \
             'out.append(0); continue'

run "порядок слотов не следует uniqueSources" \
    "$CACHE" 'for src in head.uniqueSources {' \
             'for src in head.uniqueSources.reversed() {'

run "старый кэш принимается при ЛЮБОМ числе слотов" \
    "$CACHE" 'guard nSources == head.uniqueSources.count else {' \
             'guard true else {'

echo
echo "── Копирование по срезу ──"

run "срез берёт слоты с нуля, игнорируя список" \
    "$CACHE" 'src.baseAddress!.advanced(by: r * rb + sl * sb),' \
             'src.baseAddress!.advanced(by: r * rb + j * sb),'

run "слоты кладутся в приёмник поверх друг друга" \
    "$CACHE" 'memcpy(to.advanced(by: j * sb),' \
             'memcpy(to.advanced(by: 0 * sb),'

run "непрерывный путь копирует срез вместо всей строки" \
    "$CACHE" 'let whole = slots == nil || slots! == Array(0 ..< nSources)' \
             'let whole = false'

run "slotBytes считается от всей строки" \
    "$CACHE" 'index.shape.dropFirst(2).reduce(1, *) * index.dtype.itemSize' \
             'index.shape.dropFirst(1).reduce(1, *) * index.dtype.itemSize'

echo
echo "── Кодирование надмножества ──"

run "состав слоёв в индекс не пишется" \
    "$ENC" 'contract: config.contract,
                                 sources: srcs)' \
           'contract: config.contract)'

run "кодируются слои головы, а не заказанные" \
    "$ENC" 'RerankerHead.select(pairState, sources: srcs))' \
           'model.select(pairState))'

run "список слоёв не сортируется" \
    "$ENC" 'return Array(Set(out)).sorted()' \
           'return Array(Set(out))'

run "повторы в списке слоёв не схлопываются" \
    "$ENC" 'return Array(Set(out)).sorted()' \
           'return out.sorted()'

run "отрицательный индекс слоя не нормализуется" \
    "$ENC" 'let a = i < 0 ? nLayer + i : i' \
           'let a = i'

echo
echo "── Обучение на срезе ──"

run "обучение читает всю строку кэша" \
    "$TRAIN" 'trainCache.batch(rows, slots: trainSlots)' \
             'trainCache.batch(rows)'

run "оценка читает срез обучения, а не свой" \
    "$TRAIN" 'evalSlots = try evalCache.slots(for: model.head)' \
             'evalSlots = trainSlots'

echo
echo "── Развёртка ──"

# Если голова не пересоздаётся, второй сид стартует с обученных весов, и
# «разброс» превращается в кривую дообучения.
run "голова переиспользуется между сидами" \
    "$SWEEP" 'let model = try Reranker(base: base, cfg: cfg,
                                     headDType: headDType, seed: seed)' \
             'let model = try Reranker(base: base, cfg: cfg,
                                     headDType: headDType, seed: 0)'

run "сид не доходит до перемешивания батчей" \
    "$SWEEP" 'c.seed = seed' 'c.seed = 0'

run "разброс по одному прогону объявлен нулём" \
    "$SWEEP" 'guard values.count > 1 else { return .nan }' \
             'guard values.count > 1 else { return 0 }'

run "разброс популяционный, а не выборочный" \
    "$SWEEP" 'return (ss / Double(values.count - 1)).squareRoot()' \
             'return (ss / Double(values.count)).squareRoot()'

echo
echo "── Отбор слоёв головой ──"

run "select по явным слоям берёт слои головы" \
    "$HEAD" 'stacked(sources.map { state.layerWKV($0) }, axis: 1)' \
            'stacked(sources.map { _ in state.layerWKV(0) }, axis: 1)'


echo
echo "── Выравнивание длины ──"

run "порог игнорируется, округляется всё подряд" \
    "$ENC" 'if bucket > 1 && T >= minT { T = ((T + bucket - 1) / bucket) * bucket }' \
           'if bucket > 1 { T = ((T + bucket - 1) / bucket) * bucket }'

run "порог отсекает наоборот: округляются короткие" \
    "$ENC" 'if bucket > 1 && T >= minT { T = ((T + bucket - 1) / bucket) * bucket }' \
           'if bucket > 1 && T <= minT { T = ((T + bucket - 1) / bucket) * bucket }'

run "порог не выводится из размера корзины" \
    "$ENC" 'self.lengthBucketMinTokens = lengthBucketMinTokens ?? (4 * lengthBucket)' \
           'self.lengthBucketMinTokens = lengthBucketMinTokens ?? 0'

# Обе ветки проверяются ПОРОЗНЬ. В первой редакции теста формы префиксов и
# хвостов складывались в один список, и мутация по префиксам проходила мимо:
# утверждение выполнялось за счёт хвостов.
run "порог не передан батчеру префиксов" \
    "$ENC" 'let (idx, mask, endIdx) = batchIds(seqs, bucket: config.lengthBucket,
                                               minT: config.lengthBucketMinTokens)' \
           'let (idx, mask, endIdx) = batchIds(seqs, bucket: config.lengthBucket)'

run "порог не передан батчеру хвостов" \
    "$ENC" 'let (qidx, qmask, qend) = batchIds(qseqs, bucket: config.lengthBucket,
                                                   minT: config.lengthBucketMinTokens)' \
           'let (qidx, qmask, qend) = batchIds(qseqs, bucket: config.lengthBucket)'

run "выравнивание попало в контракт" \
    "$ENC" '"terminator": terminator.map(String.init) ?? "none"]' \
           '"terminator": terminator.map(String.init) ?? "none",
         "length_bucket": String(lengthBucket)]'

echo
echo "── Контракт при обучении ──"

# Раньше сюда передавался пустой словарь: сверялась только ФОРМА состояния,
# а шаблон, обрезки и терминатор проходили молча. Кэш с maxDocTokens = 128
# обучал голову, которую потом применяли с 384, — ошибок формы при этом не
# возникает нигде.
run "контракт при обучении не сверяется" \
    "$TRAIN" 'let want = contract ?? trainCache.contract' \
             'let want: [String: String] = [:]'

run "явное ожидание вызывающего игнорируется" \
    "$TRAIN" 'let want = contract ?? trainCache.contract' \
             'let want = trainCache.contract'

run "отложенный кэш не проверяется" \
    "$TRAIN" 'try evalCache.checkCompatible(head: model.head, contract: want)' \
             'try evalCache.checkCompatible(head: model.head, contract: [:])'

run "результат не несёт контракт обучения" \
    "$TRAIN" 'firstLoss: firstLoss, expectedFirstLoss: log(Float(C)),
            contract: want)' \
             'firstLoss: firstLoss, expectedFirstLoss: log(Float(C)),
            contract: [:])'

run "инструкция пишется в контракт всегда" \
    "$ENC" 'if instructs.count == 1 { contract["instruct"] = instructs.first! }' \
           'contract["instruct"] = instructs.first!'

echo
echo "── Слияние кэшей и оценка без обучения ──"

run "слой берётся из чужого кэша по своему слоту" \
    "$CACHE" 'parts.append(gather(rows, slots: [slot]))' \
             'parts.append(other.gather(rows, slots: [slot]))'

run "слои после слияния не сортируются" \
    "$CACHE" 'let union = Array(Set(mine).union(theirs)).sorted()' \
             'let union = mine + theirs.filter { !mine.contains($0) }'

# Единственная проверка, ловящая слияние кэшей от РАЗНЫХ баз: модель в
# контракте не записана, состояния двух моделей неотличимы ни по форме, ни
# по контракту.
run "общие слои численно не сверяются" \
    "$CACHE" 'guard d / scale <= tolerance else {' \
             'guard true else {'

run "совпадение пар не проверяется" \
    "$CACHE" 'guard nPairs == other.nPairs, index.pairIndex == other.index.pairIndex,' \
             'guard true, true,'

run "оценка без обучения не сверяет контракт" \
    "$TRAIN" 'try cache.checkCompatible(head: model.head, contract: want)' \
             'try cache.checkCompatible(head: model.head, contract: [:])'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше. Каждая — либо пробел в наборе"
echo "тестов, либо утверждение о том, что мутация безвредна; второе"
echo "нужно обосновать разбором, а не оставлять как есть."
