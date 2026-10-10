#!/bin/bash
# Мутационная проверка загрузки .rwkvq без всплеска памяти (12.10): чтение по
# частям, сведение r/k/v при чтении, emb строками сжатой таблицы.
#
#   ./Scripts/mutations_load.sh [фильтр тестов]
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
RS=Sources/RWKVQuant/RwkvqSymRows.swift
RF=Sources/RWKVQuant/RwkvqFile.swift
SG=Sources/RWKVQuant/RwkvqSymGEMV.swift
LD=Sources/RWKVGen/RwkvqStandaloneLoad.swift
BB=Sources/RWKVGen/X070Backbone.swift

FILTER="${1:-RwkvqLoadTests}"
run() {
    local desc="$1"; shift
    printf '%-60s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── строки сжатой таблицы эмбеддинга ──"
run "масштаб блока не округлён до half" \
    "$RS" 'let scale = (q * dd).asType(.float16).asType(.float32)' 'let scale = q * dd'
run "коды прочитаны беззнаковыми" \
    "$RS" 'let c = RwkvqSymRowTable.signed(codes.take(ids, axis: 0)).asType(.float32)' \
          'let c = codes.take(ids, axis: 0).asType(.float32)'
run "масштабы блоков прочитаны беззнаковыми" \
    "$RS" 'let q = RwkvqSymRowTable.signed(qs.take(ids, axis: 0)).asType(.float32)' \
          'let q = qs.take(ids, axis: 0).asType(.float32)'
run "таблица строк выдаётся и при шести битах" \
    "$RS" 'info.kind == .sym, info.bits == 8,' 'info.kind == .sym,'

echo
echo "── чтение по частям ──"
run "фильтр чтения не действует" \
    "$RF" 'let load = want?(key) ?? true' 'let load = true'
run "порядок чтения от мелких к крупным" \
    "$RF" 'return a != b ? a > b : $0 < $1' 'return a != b ? a < b : $0 < $1'
run "сведённая группа сводится повторно" \
    "$SG" 'for (id, keys) in groups where fused[id] == nil {' 'for (id, keys) in groups {'

run "чтение вперёд просится для всего файла" \
    "$RF" 'return want?(String(name[..<r.lowerBound])) ?? true' 'return true'
run "чтение вперёд не просится" \
    "$RF" 'guard RwkvqContainer.readAheadEnabled else { return }' 'guard false else { return }'

echo
echo "── порядок загрузки ──"
run "r/k/v при чтении не сводятся (копия при подключении)" \
    "$LD" 'only: { wanted.contains($0) }, fusingSym: groups)' 'only: { wanted.contains($0) }, fusingSym: [:])'
run "плотная фаза поднимает всю базу" \
    "$LD" 'let small = try RwkvqSidecar(rwkvq: p, only: { !wanted.contains($0) })' \
          'let small = try RwkvqSidecar(rwkvq: p, only: nil)'
run "сжатый emb у не-fp16 модели" \
    "$LD" 'if !denseEmbedding, computeDType == .float16, let t' 'if !denseEmbedding, let t'
run "denseEmbedding не читается" \
    "$LD" 'if !denseEmbedding, computeDType == .float16, let t' 'if computeDType == .float16, let t'

echo
echo "── сжатая таблица в модели ──"
run "подмена таблицы не имеет приоритета над сжатой" \
    "$BB" 'if wOverride?["emb.weight"] == nil, w["emb.weight"] == nil, let table = rwkvqEmbTable {' \
          'if w["emb.weight"] == nil, let table = rwkvqEmbTable {'
run "emb со сжатой таблицей не числится среди весов" \
    "$BB" '.union(rwkvqEmbTable == nil ? [] : ["emb.weight"])).sorted()' '.union([String]())).sorted()'
run "hasAnyWeight не видит сжатую таблицу" \
    "$BB" '|| (key == "emb.weight" && rwkvqEmbTable != nil)' ''

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
