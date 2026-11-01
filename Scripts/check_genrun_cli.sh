#!/bin/bash
# Свойства ключей gen-run --distill --train (01.11): все проверки — до учителя.
#   ./Scripts/check_genrun_cli.sh [путь к gen-run]
# Модель и словарь — GENRUN_MODEL / GENRUN_VOCAB (умолчание — g1d 0.4B из swr_bench_keep).
# Пустой файл вопросов: прошедший проверки запуск доходит до «нечему учить» (выход 1) без генерации,
# отвергнутый — выход 2 до учителя; в обоих случаях кэш учителя не создаётся.
# Выход 0 — все свойства выполнены, 1 — нарушено хотя бы одно.
set -uo pipefail
cd "$(dirname "$0")/.."
G="${1:-.build/release/gen-run}"
M="${GENRUN_MODEL:-$HOME/Develop/swr_bench_keep/w1014/g1d_0.4b.safetensors}"
V="${GENRUN_VOCAB:-$HOME/Develop/swr_bench_keep/w1014/rwkv_vocab_v20230424.txt}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
: > "$T/empty.txt"
fail=0
prop() {  # prop <описание> <ожидаемый выход> <ожидаемая подстрока> <ключи…>
    local desc="$1" want="$2" msg="$3"; shift 3
    rm -f "$T/cache.json"
    local out; out=$("$G" --model "$M" --vocab "$V" --system "S" --distill --train-questions "$T/empty.txt" \
        --teacher-cache "$T/cache.json" "$@" 2>&1); local rc=$?
    if [ $rc -eq "$want" ] && grep -qF -- "$msg" <<< "$out" && [ ! -e "$T/cache.json" ]; then
        printf '  ок      %s\n' "$desc"
    else
        printf '  НАРУШЕНО %s (выход %d, ждали %d «%s»): %s\n' "$desc" $rc "$want" "$msg" "$(tail -1 <<< "$out")"; fail=1
    fi
}
prop "top без --state-out — проходит"            1 "нечему учить"            --train top  --weights-out "$T/w.safetensors"
prop "lora без --state-out — проходит"           1 "нечему учить"            --train lora --weights-out "$T/w.safetensors"
prop "state с --state-out — проходит"            1 "нечему учить"            --state-out "$T/s.safetensors"
prop "state без --state-out — отказ"             2 "нужен --state-out"
prop "top без --weights-out — отказ"             2 "нужен --weights-out"     --train top
prop "неизвестный --train — отказ"               2 "--train: state | top | lora" --train foo --weights-out "$T/w.safetensors"
prop "top с --held-out — отказ до учителя"       2 "не поддержаны"           --train top --weights-out "$T/w.safetensors" --held-out 3
prop "lora с --eval-states — отказ до учителя"   2 "не поддержаны"           --train lora --weights-out "$T/w.safetensors" --eval-states zero
prop "top с --init fold — отказ до учителя"      2 "--init не применим"      --train top --weights-out "$T/w.safetensors" --init fold
exit $fail
