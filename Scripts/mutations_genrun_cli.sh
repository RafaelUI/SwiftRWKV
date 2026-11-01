#!/bin/bash
# Мутационная проверка (01.11): ключи gen-run --distill --train ловятся Scripts/check_genrun_cli.sh.
# Каждая мутация: правка main.swift → сборка gen-run → свойства должны быть нарушены → возврат.
#   ./Scripts/mutations_genrun_cli.sh
set -uo pipefail
cd "$(dirname "$0")/.."
F=Sources/GenRun/main.swift
BACKUP=$(mktemp); cp "$F" "$BACKUP"; trap 'cp "$BACKUP" "$F"; rm -f "$BACKUP"' EXIT
caught=0; total=0
run() {  # run <описание> <что> <на что>
    local desc="$1" from="$2" to="$3"; total=$((total + 1))
    cp "$BACKUP" "$F"
    python3 - "$F" "$from" "$to" <<'PY' || { printf '%-52s НЕ ПРИМЕНЕНА (подстрока не уникальна)\n' "$desc"; return; }
import sys
p, a, b = sys.argv[1:]
s = open(p, encoding='utf-8').read()
if s.count(a) != 1: sys.exit(1)
open(p, 'w', encoding='utf-8').write(s.replace(a, b))
PY
    if ! swift build -c release --product gen-run > /dev/null 2>&1; then printf '%-52s НЕ СОБРАЛАСЬ\n' "$desc"; return; fi
    if ./Scripts/check_genrun_cli.sh > /dev/null 2>&1; then printf '%-52s НЕ ПОЙМАНА\n' "$desc"
    else printf '%-52s поймана\n' "$desc"; caught=$((caught + 1)); fi
}
run "state-out требуется и при top/lora (как до 01.11)" \
    'if args.trainWhat == "state" {
        guard !args.stateOut.isEmpty' 'if true {
        guard !args.stateOut.isEmpty'
run "weights-out не требуется" \
    'guard !args.weightsOut.isEmpty else { print("--train \(args.trainWhat): нужен --weights-out"); exit(2) }' ''
run "неизвестный --train пропускается" \
    'guard ["state", "top", "lora"].contains(args.trainWhat) else' 'guard true else'
run "held-out с top/lora пропускается" \
    'guard args.heldOut == 0, args.evalStates.isEmpty else' 'guard args.evalStates.isEmpty else'
run "init с top/lora пропускается" \
    'guard args.initMode == "zero" else { print("--train \(args.trainWhat): --init' 'guard true else { print("--train \(args.trainWhat): --init'
cp "$BACKUP" "$F"; swift build -c release --product gen-run > /dev/null 2>&1
echo "поймано $caught из $total"
[ $caught -eq $total ]
