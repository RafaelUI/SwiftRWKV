#!/bin/bash
# Мутационная проверка (23.10): состояние в формате RWKV-PEFT (.pth) — запись, чтение, отказы.
#   ./Scripts/mutations_peft_state.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
P=Sources/RWKVGen/PEFTStateFile.swift
FILTER='PEFTStateFileTests'
run() { local d="$1"; shift; printf '%-58s ' "$d"; "$M" "$@" --filter "$FILTER"; }
run "запись транспонирует матрицу" \
    "$P" '        let wf = w.asType(.float32)' '        let wf = w.asType(.float32).swappedAxes(-1, -2)'
run "чтение транспонирует матрицу" \
    "$P" '.map { layers[$0]!.asType(.float32) })' '.map { layers[$0]!.asType(.float32).swappedAxes(-1, -2) })'
run "сдвиги не проверяются" \
    "$P" '        if !dropShifts {' '        if false && !dropShifts {'
run "dropShifts не действует" \
    "$P" '        if !dropShifts {' '        if true {'
run "проверяется только сдвиг tmix" \
    "$P" '            for s in [state.tmixShift, state.cmixShift] {' '            for s in [state.tmixShift] {'
run "хранилище пишется как fp16" \
    "$P" 'p.global("torch", "FloatStorage")' 'p.global("torch", "HalfStorage")'
run "CRC не пишется" \
    "$P" '            let crc = TorchPth.crc32(payload)' '            let crc: UInt32 = 0'
run "CRC не проверяется" \
    "$P" '            guard TorchPth.crc32(d) == e.crc else' '            guard true || TorchPth.crc32(d) == e.crc else'
run "данные не выровнены на 64" \
    "$P" '            let pad = (align - bare % align) % align' '            let pad = 0'
run "смещение в хранилище не учитывается" \
    "$P" 'flat[offset ..< (offset + n)]' 'flat[0 ..< n]'
run "страйды не учитываются" \
    "$P" '            } else if st == cStride(size) {' '            } else if true || st == cStride(size) {'
run "bf16-хранилище читается как fp16" \
    "$P" '"BFloat16Storage": .bfloat16' '"BFloat16Storage": .float16'
run "BUILD (_metadata) не принимается" \
    "$P" '                    guard case .dict = stack.last else { throw Failure.unsupported("pickle: BUILD не у словаря") }' \
         '                    throw Failure.unsupported("pickle: BUILD")'
run "nn.Parameter не разворачивается" \
    "$P" '                return first' '                return .none'
run "чужой глобал не отвергается" \
    "$P" '                throw Failure.unsupported("глобал \(m).\(n) не разрешён")' '                return .none'
run "любой ключ att.* принимается за слой" \
    "$P" 'p[2] == "att", p[3] == "time_state" {' 'p[2] == "att" {'
run "архитектура не сверяется с базой" \
    "$P" '            if L != c.nLayer || H != c.nHead || S != c.headSize {' '            if false {'
run "дыра в номерах слоёв принимается" \
    "$P" '        guard have == Array(0 ..< L) else {' '        guard have.count == L else {'
run "формы слоёв не сверяются" \
    "$P" 'where layers[l]!.shape != s0 {' 'where false && layers[l]!.shape != s0 {'
run "условие порядка байт перевёрнуто" \
    "$P" '            guard order == "little" else' '            guard order != "little" else'
