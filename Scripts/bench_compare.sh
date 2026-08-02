#!/bin/bash
# Замер кодирования: Swift против Python на ОДНОМ файле входа.
#
#   ./Scripts/bench_compare.sh [файл входа] [повторов]
#
# Стороны гоняются ПО ОЧЕРЕДИ, а не параллельно: GPU один, и одновременный
# запуск мерил бы конкуренцию за него, а не реализации.
#
# Повторов по умолчанию два. Одиночный прогон здесь ничем не лучше, чем в
# развёртке по сидам: машина не изолирована, и разница в проценты между
# первым и вторым прогоном — это шум, а не результат.
#
# Память снимается СНАРУЖИ, системными командами (`ps`, `vmmap`), а не
# счётчиками MLX: буферы Metal живут в IOAccelerator и в RSS попадают
# неполностью, а про своп MLX не знает вовсе — и если процесс свопится, врут
# все замеры времени.
set -uo pipefail
cd "$(dirname "$0")/.."

# Убить драйвер = убить всё, что он запустил. Без этого скрипт-цикл на месте
# убитого ребёнка тут же поднимает следующего, и снаружи это выглядит как
# бессмертный процесс. `kill 0` бьёт по группе целиком.
trap 'trap - EXIT INT TERM; kill 0 2>/dev/null' EXIT INT TERM
echo "драйвер: pid $$ (убить целиком: kill -9 -$$  или  pkill -9 -f bench_compare.sh)"

INPUT="${1:-/tmp/bench_input.json}"
REPEATS="${2:-2}"
OUT=/tmp/bench
MODEL=~/Develop/rwkv-metal/world_0.1b_x070.safetensors
VOCAB=.testdata/rwkv_vocab_v20230424.txt
PYREPO=~/Develop/rwkv-metal
WARMUP=3
# Потолок буферного кэша — ОДИН для обеих сторон. Без него формы батчей
# плавают, кэш Metal растёт линейно по числу пачек, и обе стороны уходят в
# своп: замерено 11.6 ГБ у Python и 10.2 ГБ у Swift на восьмистах префиксах.
CACHE_LIMIT=2.0

mkdir -p "$OUT"

# Снимать память процесса, пока он жив. Пишем ряд, а не одно число: пик без
# кривой не отличает «выросло и держится» от «выросло на один батч».
sample_mem() {
    local pid="$1" log="$2"
    echo "t_s rss_gb footprint_gb" > "$log"
    local t0=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        local rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        local fp=$(vmmap --summary "$pid" 2>/dev/null \
                   | awk '/Physical footprint:/ && !/Peak/ {print $3; exit}')
        [ -n "$rss" ] && echo "$(( $(date +%s) - t0 )) \
$(echo "scale=2; $rss/1048576" | bc) ${fp:-?}" >> "$log"
        sleep 2
    done
}

run_one() {
    local tag="$1"; shift
    echo "── $tag ──"
    "$@" > "$OUT/$tag.log" 2>&1 &
    local pid=$!
    sample_mem "$pid" "$OUT/$tag.mem" &
    local mpid=$!
    wait "$pid"; local rc=$?
    kill "$mpid" 2>/dev/null
    wait "$mpid" 2>/dev/null
    if [ $rc -ne 0 ]; then
        echo "  УПАЛО (код $rc), см. $OUT/$tag.log"
        tail -3 "$OUT/$tag.log"
        return
    fi
    python3 - "$OUT/$tag.json" "$OUT/$tag.mem" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as e:
    print("  нет JSON:", e); sys.exit()
rss = fp = 0.0
try:
    for line in open(sys.argv[2]).read().splitlines()[1:]:
        p = line.split()
        if len(p) >= 2:
            rss = max(rss, float(p[1]))
        if len(p) >= 3 and p[2].endswith("G"):
            fp = max(fp, float(p[2].rstrip("G")))
except Exception:
    pass
print(f"  {d['ms_per_prefix']:.1f} мс/префикс, {d['ms_per_tail']:.2f} мс/хвост, "
      f"стена {d['wall_s']:.1f} с, форм {d['distinct_prefix_shapes']}"
      f"/{d['distinct_tail_shapes']}, RSS пик {rss:.2f} ГБ, "
      f"footprint пик {fp:.1f} ГБ")
if not d.get("valid", True):
    print(f"  ↑ НЕДЕЙСТВИТЕЛЬНО: своп {d.get('swapouts')} страниц")
PY
}

echo "вход: $INPUT, повторов: $REPEATS, прогрев: $WARMUP пачек"
echo

for r in $(seq 1 "$REPEATS"); do
    echo "═══ повтор $r ═══"

    run_one "py_$r" env RWKV_METAL_REPO="$PYREPO" \
        "$PYREPO/.venv/bin/python" Scripts/bench_encode.py \
        --model "$MODEL" --input "$INPUT" --warmup "$WARMUP" \
        --cache-limit "$CACHE_LIMIT" --out "$OUT/py_$r.json"

    # bucket 0 — как в Python: длина батча не округляется, форм больше.
    run_one "sw0_$r" .build/release/rerank-run \
        --model "$MODEL" --vocab "$VOCAB" --bench "$INPUT" \
        --bench-warmup "$WARMUP" --bench-bucket 0 \
        --bench-cache-limit "$CACHE_LIMIT" --bench-out "$OUT/sw0_$r.json"

    # bucket 64 — умолчание SwiftRWKV. Прямо отвечает на вопрос, помогает
    # округление длины или мешает.
    run_one "sw64_$r" .build/release/rerank-run \
        --model "$MODEL" --vocab "$VOCAB" --bench "$INPUT" \
        --bench-warmup "$WARMUP" --bench-bucket 64 \
        --bench-cache-limit "$CACHE_LIMIT" --bench-out "$OUT/sw64_$r.json"
    echo
done

echo "готово, сырые числа в $OUT"
