#!/bin/bash
# Мутационная проверка чтения .rwkvq напрямую и раскладки sym.
#
#   ./Scripts/mutations_rwkvq_file.sh [фильтр тестов]
#
# Свойства, которые здесь стерегутся, не падают и не выглядят дефектом:
# сдвинутый код, чужой масштаб или переставленные колонки дают веса, которые
# остаются весами. Поэтому эталон — codec бит-в-бит, а мутации проверяют,
# что набор это равенство действительно требует.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
DQ=Sources/RWKVQuant/RwkvqDequant.swift
RF=Sources/RWKVQuant/RwkvqFile.swift
SC=Sources/RWKVQuant/RwkvqSidecar.swift

FILTER="${1:-RwkvqFileTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── sym: ядро ──"
run "порядок регистров пары (колонки не на своих местах)" \
    "$DQ" '["l0", "l1", "h0", "h1", "l2", "l3", "h2", "h3"]' \
          '["l0", "l1", "l2", "l3", "h0", "h1", "h2", "h3"]'
run "сдвиг +32 снят неверно (fp32-ветка)" \
    "$DQ" '.\(comp) - 32.0f) * (float)\(scale);\n"
            }' \
          '.\(comp) - 31.0f) * (float)\(scale);\n"
            }'
run "сдвиг +32 снят неверно (fp16-ветка)" \
    "$DQ" '- (half)32.0h)' '- (half)31.0h)'
run "масштаб блока без half-роундтрипа (8 бит)" \
    "$DQ" 'half s = (half)((float)sc[row*NB + b] * (float)d[row*NSB + b/16]);' \
          'float s = ((float)sc[row*NB + b] * (float)d[row*NSB + b/16]);'
run "чужой суперблок у масштаба (8 бит)" \
    "$DQ" '(float)d[row*NSB + b/16]);' '(float)d[row*NSB + b/8]);'
run "второй блок пары берёт масштаб первого (6 бит)" \
    "$DQ" 'half s1 = (half)((float)sc[row*NB + 2*p + 1] * dsb);' \
          'half s1 = (half)((float)sc[row*NB + 2*p] * dsb);'
run "чужой суперблок у пары (6 бит)" \
    "$DQ" 'float dsb = (float)d[row*NSB + p/8];' 'float dsb = (float)d[row*NSB + p/4];'
run "битплоскость бита 5 на месте бита 4" \
    "$DQ" 'symPlanePair("hb2", shift: 5)' 'symPlanePair("hb2", shift: 4)'
run "тип выхода sym не входит в ключ кэша ядер" \
    "$DQ" 'SymDequantKey(inFeatures: IN, outFeatures: OUT, bits: bits, dtype: T)' \
          'SymDequantKey(inFeatures: IN, outFeatures: OUT, bits: bits, dtype: "float")'

echo
echo "── sym: цепочка операций (эталон ядра и формы вне суперблока) ──"
run "знаковый код прочитан как беззнаковый" \
    "$DQ" 'q = signedByte(qblk.reshaped([OUT, IN])).asType(.float32)' \
          'q = qblk.reshaped([OUT, IN]).asType(.float32)'

echo
echo "── перекладка sb6 в K3 ──"
run "сдвиг qm не тот" \
    "$RF" 'let qmByte = (qm - 31 + 256) % 256' 'let qmByte = (qm - 32 + 256) % 256'
run "6-битный битстрим: второе значение" \
    "$RF" 'let v1 = ((x0 >> 6) | (x1 << 2)) & 0x3F' 'let v1 = ((x0 >> 6) | (x1 << 1)) & 0x3F'
run "6-битный битстрим: четвёртое значение" \
    "$RF" 'let v3 = (x2 >> 2) & 0x3F' 'let v3 = (x2 >> 3) & 0x3F'
run "d и dm переставлены" \
    "$RF" 'let ddm = stacked([d, dm], axis: -1)' 'let ddm = stacked([dm, d], axis: -1)'
run "битплоскости переставлены" \
    "$RF" 'qh: box.array("\(key)::gw_qh"), qh2: box.array("\(key)::gw_qh2"),' \
          'qh: box.array("\(key)::gw_qh2"), qh2: box.array("\(key)::gw_qh"),'

echo
echo "── манифест и контейнер ──"
run "ориентация: поле манифеста игнорируется (.rwkvq)" \
    "$RF" 'let transposed = (m["transposed"] as? Bool)
                ?? (naming == "world" && RwkvqNaming.isRawLoRAWorld(key))' \
          'let transposed = (naming == "world" && RwkvqNaming.isRawLoRAWorld(key))'
run "ориентация: поле манифеста игнорируется (сайдкар)" \
    "$SC" 'let transposed = (d["transposed"] as? Bool)
                ?? (namingValue == "world" && RwkvqNaming.isRawLoRAWorld(key))' \
          'let transposed = (namingValue == "world" && RwkvqNaming.isRawLoRAWorld(key))'
run "pickle-контейнер не распознан" \
    "$RF" 'if data.prefix(4).elementsEqual([0x50, 0x4B, 0x03, 0x04]) {' \
          'if data.prefix(4).elementsEqual([0x50, 0x4B, 0x03, 0x05]) {'
run "смещения буфера не сверяются с формой" \
    "$RF" 'guard off[0] >= 0, off[1] >= off[0], base + off[1] <= data.count,
                  off[1] - off[0] == count * dt.size else {' \
          'guard off[0] >= 0, off[1] >= off[0],
                  off[1] - off[0] == count * dt.size else {'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше."
