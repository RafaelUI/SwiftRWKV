#!/bin/bash
# Мутационная проверка (19.10): снимок состояния генерации.
#   ./Scripts/mutations_snapshot.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
IS=Sources/RWKVGen/InitialState.swift
XG=Sources/RWKVGen/X070Generation.swift
FILTER='StateSnapshotTests'
run() { local d="$1"; shift; printf '%-52s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "в батч уходит сдвиг tmix вместо cmix" \
    "$XG" '                               cmix: cmixPrev.map { $0.expandedDimensions(axis: 0) })' '                               cmix: tmixPrev.map { $0.expandedDimensions(axis: 0) })'
run "тип сдвигов не записывается" \
    "$IS" '"note": note, "shift_dtype": name,' '"note": note,'
run "тип сдвигов не восстанавливается" \
    "$IS" '        s.tmixPrev = s.tmixPrev.map { $0.asType(shiftType) }' '        s.tmixPrev = s.tmixPrev.map { $0.asType(.float32) }'
run "сдвиги пишутся через bf16" \
    "$IS" '                      "tmix_shift": batch.tmixShift.asType(.float32),' '                      "tmix_shift": batch.tmixShift.asType(.bfloat16).asType(.float32),'
run "матрица WKV не того слоя" \
    "$XG" '        RWKVBatchState.stacked(wkv: wkv.map { $0.expandedDimensions(axis: 0) },' '        RWKVBatchState.stacked(wkv: wkv.map { _ in wkv[0].expandedDimensions(axis: 0) },'
run "чужая база принимается" \
    "$IS" '            if info.baseFingerprint != fp && !allowOtherBase {' '            if false {'
run "снимок помечен как обученное состояние" \
    "$IS" '        let info = Info(kind: Kind.snapshot.rawValue, trained: [], note: note,' '        let info = Info(kind: Kind.tuned.rawValue, trained: [], note: note,'
