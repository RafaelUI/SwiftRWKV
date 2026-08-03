#!/bin/bash
# Собрать примеры из документации.
#
#   ./Scripts/check_docs.sh [файл.md ...]     # по умолчанию docs/*.md
#
# Зачем. Пример в документации — код, который никто не компилирует, и
# протухает он МОЛЧА: сигнатура поменялась, текст остался, читатель узнаёт об
# этом сам.
#
# Как. Блоки ```swift вырезаются в ОДИН файл тестовой цели
# (Tests/RWKVGenTests/DocExamples.generated.swift), и дальше их собирает
# обычный `swift build --build-tests`. Через SwiftPM, а не своим вызовом
# swiftc, намеренно: пути к модулям зависимостей (Cmlx, _NumericsShims и их
# заголовки)SwiftPM знает сам, а собранные вручную флаги ломаются при каждой
# смене тулчейна и версии зависимостей — то есть ровно тем способом, от
# которого этот скрипт и защищает.
#
# Побочная выгода: сгенерированный файл ОСТАЁТСЯ в тестовой цели, поэтому
# примеры компилируются при каждом `swift test`, а не только по запуску
# скрипта.
#
# Что НЕ проверяется: что примеры дают заявленные числа. Для этого нужна
# реальная модель и минуты счёта — такие проверки живут в тестах и в
# прогонщике. Числа в документации помечены тем, на чём измерены.
#
# Пометки на блоках:
#   ```swift         — обычный пример, кладётся в СВОЮ функцию
#   ```swift-helper  — общий кусок, кладётся на уровень файла и виден всем
#   ```swift-skip    — не проверять (псевдокод, нарочно ошибочные строки)
#
# Изоляция блоков по умолчанию намеренная: примеры из разных разделов не
# обязаны сочетаться, одинаковые имена в них — норма. Общее объявляется
# явно, а не выводится из порядка блоков.
# Язык документации — АНГЛИЙСКИЙ. Комментарии в самом коде остаются
# русскими: это разные аудитории, и смешивать их в одном файле хуже, чем
# держать границу по типу файла.
set -uo pipefail
cd "$(dirname "$0")/.."

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
    shopt -s nullglob
    FILES=(docs/*.md)
    shopt -u nullglob
fi
[ ${#FILES[@]} -eq 0 ] && { echo "нет файлов документации"; exit 0; }

GEN=Tests/RWKVGenTests/DocExamples.generated.swift

python3 - "$GEN" "${FILES[@]}" <<'PY'
import re, sys
gen, mds = sys.argv[1], sys.argv[2:]
out = [
    "//",
    "//  DocExamples.generated.swift",
    "//  СГЕНЕРИРОВАН Scripts/check_docs.sh из docs/*.md — не править руками.",
    "//",
    "//  Файл существует ради одного: примеры в документации обязаны",
    "//  собираться. Функции никто не вызывает, и это не упущение — проверка",
    "//  здесь именно на тайпчек, потому что протухают у примеров имена и",
    "//  сигнатуры, а не поведение.",
    "//",
    "import Foundation",
    "import MLX",
    "@testable import RWKVGen",
    "@testable import RWKVQuant",
    "@testable import RWKVEmbedding",
    "@testable import RWKVRerank",
    "",
]
total = skipped = 0
for md in mds:
    text = open(md, encoding="utf-8").read()
    name = re.sub(r"\W", "_", md.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    helpers, blocks = [], []
    for m in re.finditer(r"```(swift|swift-skip|swift-helper)\n(.*?)```",
                         text, re.S):
        if m.group(1) == "swift-skip":
            skipped += 1
        elif m.group(1) == "swift-helper":
            helpers.append(m.group(2))
        else:
            blocks.append(m.group(2))
    for i, b in enumerate(blocks):
        # Всё — ВНУТРИ функции примера, включая помощников. Так два разных
        # документа не могут столкнуться именами: `loadBackbone` в
        # reranker.md и в Embedding.md — разные локальные функции, а не
        # повторное объявление одной. Ценой дублирования в файле, который
        # никто не читает.
        out.append("func doc_%s_%d() throws {" % (name, i))
        for h in helpers:
            out.extend("    " + l if l.strip() else "" for l in h.splitlines())
        out.extend("    " + l if l.strip() else "" for l in b.splitlines())
        out.append("}")
        out.append("")
        total += 1
    print("%s: блоков %d, помощников %d" % (md, len(blocks), len(helpers)))
open(gen, "w", encoding="utf-8").write("\n".join(out))
print("всего блоков %d, пропущено %d → %s" % (total, skipped, gen))
PY

echo "── сборка тестовой цели ──"
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
if swift build --build-tests > "$LOG" 2>&1; then
    echo "примеры собираются"
    exit 0
fi
echo "ПРИМЕРЫ НЕ СОБИРАЮТСЯ:"
grep -E "error:" "$LOG" | grep -v "^error: " | head -10
exit 1
