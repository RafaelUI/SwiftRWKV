#!/bin/bash
# Мутационная проверка (25.10): BPE-токенизатор HuggingFace против эталона
# `tokenizers` (BPETokenizerTests).
#
#   ./Scripts/mutations_bpe.sh
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
B=Sources/RWKVGen/BPETokenizer.swift
FILTER='BPETokenizerTests'
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── предобработка ──"
run "нет пробела в начале куска" \
    "$B" 'if addPrefixSpace, b[0] != 0x20 { b.insert(0x20, at: 0) }' 'if false { b.insert(0x20, at: 0) }'
run "пробел не ставится и перед переводом строки" \
    "$B" 'if addPrefixSpace, b[0] != 0x20 { b.insert(0x20, at: 0) }' 'if addPrefixSpace, b[0] != 0x20, b[0] != 0x0A { b.insert(0x20, at: 0) }'
run "NFC вместо NFKC" \
    "$B" 'case .nfkc: r = s.precomposedStringWithCompatibilityMapping' 'case .nfkc: r = s.precomposedStringWithCanonicalMapping'
run "добавленные — короткие первыми" \
    "$B" 'rawAdded = added.filter { !$0.normalized }.sorted { $0.bytes.count > $1.bytes.count }' 'rawAdded = added.filter { !$0.normalized }.sorted { $0.bytes.count < $1.bytes.count }'
echo "── шаблон GPT-2 ──"
run "нет сокращения 's" \
    "$B" 'if a == "s" || a == "t" || a == "m" || a == "d" { len = 2 }' 'if a == "t" || a == "m" || a == "d" { len = 2 }'
run "сокращения без 'll" \
    "$B" '(a == "v" && b == "e") || (a == "l" && b == "l")' '(a == "v" && b == "e")'
run "пробел присоединяется только к буквам" \
    "$B" '(c.u == " " && i + 1 < n && scal[i + 1].c != .space) ? i + 1 : i' '(c.u == " " && i + 1 < n && scal[i + 1].c == .letter) ? i + 1 : i'
run "серия пробелов без отката (?!\\S)" \
    "$B" 'if j == n || j - i == 1 {' 'if true {'
run "неразрывный пробел — не \\s" \
    "$B" '        if p.isWhitespace { return .space }' '        if false { return .space }'
run "ASCII: перевод строки — не \\s" \
    "$B" 'if v == 0x20 || (v >= 0x09 && v <= 0x0D) { return .space }' 'if v == 0x20 || v == 0x09 { return .space }'
echo "── BPE и декодирование ──"
run "при равном ранге — правее" \
    "$B" 'if let m = merges[key], best == nil || m.rank < best!.rank { best = (m.rank, m.id, p) }' 'if let m = merges[key], best == nil || m.rank <= best!.rank { best = (m.rank, m.id, p) }'
run "ранги слияний не по порядку файла" \
    "$B" 'if mm[key] == nil { mm[key] = (Int32(rank), Int32(to)) }' 'if mm[key] == nil { mm[key] = (Int32(ms.count - rank), Int32(to)) }'
run "кэш кусков отдаёт чужое" \
    "$B" 'if let hit = cache[key] { out.append(contentsOf: hit); continue }' 'if let hit = cache.values.first, !cache.isEmpty { out.append(contentsOf: hit); continue }'
run "специальные не пропускаются" \
    "$B" 'if skipSpecial && specialIds.contains(t) { continue }' 'if false && specialIds.contains(t) { continue }'
run "post_processor не отвергается" \
    "$B" '        if let pp = root["post_processor"], !(pp is NSNull) {' '        if let pp = root["post_processor"], false, !(pp is NSNull) {'
run "rawBytes специальных — их текст" \
    "$B" '        return specialIds.contains(id) ? [] : idBytes[id]' '        return idBytes[id]'
