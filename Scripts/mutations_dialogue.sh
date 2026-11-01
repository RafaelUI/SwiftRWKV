#!/bin/bash
# Мутационная проверка (01.11): Dialogue.run — сборка многоходового диалога для gen-run --dialogues.
#   ./Scripts/mutations_dialogue.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M=Scripts/mutate.sh
F=Sources/RWKVGen/Dialogue.swift
FILTER='DialogueTests'
run() { local desc="$1"; shift; printf '%-56s ' "$desc"; "$M" "$@" --filter "$FILTER"; }
run "история не копится (каждый ход — только свой вопрос)" "$F" 'history = prompt + r + separator' 'history = prefix'
run "ответ в истории обрезан" "$F" 'history = prompt + r + separator' 'history = prompt + r.trimmingCharacters(in: .whitespacesAndNewlines) + separator'
run "без разделителя" "$F" 'public static let separator = "\n\n"' 'public static let separator = "\n"'
run "префикс повторяется на каждом ходе" "$F" 'let prompt = history + userPart(q)' 'let prompt = history + (k > 0 ? prefix : "") + userPart(q)'
run "номер хода сдвинут" "$F" 'let r = reply(prompt, k)' 'let r = reply(prompt, k + 1)'
run "ответ не тот, что выдала генерация" "$F" 'out.append(Turn(prompt: prompt, reply: r))' 'out.append(Turn(prompt: prompt, reply: String(r.dropFirst())))'
run "последний ход теряется" "$F" 'for (k, q) in turns.enumerated() {' 'for (k, q) in turns.dropLast().enumerated() {'
