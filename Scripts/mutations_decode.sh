#!/bin/bash
# Мутационная проверка рекуррентного декода.
#
#   ./Scripts/mutations_decode.sh [фильтр тестов]
#
# Декод — ВТОРАЯ реализация той же арифметики, и цена этого уже была
# заплачена: он читал веса в обход общего пути, отчего падал на квантованной
# базе и МОЛЧА игнорировал LoRA-адаптеры. Второе хуже: дообученная модель
# генерировала так, будто её не дообучали, — правдоподобным текстом.
#
# Поэтому мутации здесь про одно: не отвязался ли декод от общей проекции
# снова.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
GEN=Sources/RWKVGen/X070Generation.swift

FILTER="${1:-InferenceSmokeTests|X070ParityTests|PrefillTests}"
run() {
    local desc="$1"; shift
    printf '%-52s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Проекции ──"

run "проекция минует общий путь" \
    "$GEN" 'return projectForBlock(x, wKey, lora: String(wKey.dropLast(7)))' \
           'return matmul(x, weightForBlock(wKey).transposed())'

run "LoRA-цель проекции не передаётся" \
    "$GEN" 'lora: String(wKey.dropLast(7)))' \
           'lora: nil)'

# Ловится только тестом с КВАНТОВАННОЙ таблицей: по умолчанию она плотная,
# и оба пути совпадают сами собой.
run "таблица эмбеддингов читается напрямую" \
    "$GEN" 'let emb = embedForBlock(MLXArray([Int32(id)])).reshaped([1, D])' \
           'let emb = gg("emb.weight")[id].reshaped([1, D])'

echo
echo "── Параллельный prefill и мост состояния ──"
#
# Здесь дефекты особенно тихие: логиты последнего токена читаются с ВЫХОДА
# сети и от переноса состояния не зависят вовсе. Потеряв сдвиги или взяв не
# ту ось, prefill вернёт правильный первый токен и сломает всё, что дальше.

run "сдвиги token-shift не переносятся" \
    "$GEN" 'tmixPrev = (0 ..< L).map { batch.layerTmixShift($0)[row] }    // [1,D]' \
           'tmixPrev = (0 ..< L).map { batch.layerTmixShift($0)[row] * 0 }'

run "сдвиг cmix берётся от tmix" \
    "$GEN" 'cmixPrev = (0 ..< L).map { batch.layerCmixShift($0)[row] }    // [1,D]' \
           'cmixPrev = (0 ..< L).map { batch.layerTmixShift($0)[row] }'

run "мост берёт строку 0 всегда" \
    "$GEN" 'wkv = (0 ..< L).map { batch.layerWKV($0)[row] }               // [H,S,S]' \
           'wkv = (0 ..< L).map { batch.layerWKV($0)[0] }'

run "слои моста идут в обратном порядке" \
    "$GEN" 'wkv = (0 ..< L).map { batch.layerWKV($0)[row] }               // [H,S,S]' \
           'wkv = (0 ..< L).reversed().map { batch.layerWKV($0)[row] }'

run "логиты берутся не с последнего токена" \
    "$GEN" 'let last = lnOut[0, ids.count - 1].reshaped([1, cfg.nEmbd])' \
           'let last = lnOut[0, 0].reshaped([1, cfg.nEmbd])'

run "prefill не перезаписывает состояние" \
    "$GEN" 'state = RWKVState(batch)' \
           'state = RWKVState(cfg: cfg)'

# Две нормировки формы на остаточном потоке. Каждая проверяется ОТДЕЛЬНО, и
# это условие: пока их было четыре (две на записи в состояние и две на
# выходах блоков), они друг друга подменяли — снятие любой одной чинилось
# остальными, и НИ ОДНА мутация не ловилась. Набор из взаимно резервирующих
# защит неотличим от набора из ни одной.
run "первая остаточная связь не нормируется" \
    "$GEN" 'x = (x + h).reshaped([1, D])' \
           'x = x + h'

run "вторая остаточная связь не нормируется" \
    "$GEN" '                                         gg("blocks.\(layer).ln2.bias")), layer, &state))
                .reshaped([1, D])' \
           '                                         gg("blocks.\(layer).ln2.bias")), layer, &state))'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше. Каждая — либо пробел в наборе"
echo "тестов, либо утверждение о том, что мутация безвредна; второе"
echo "нужно обосновать разбором, а не оставлять как есть."
