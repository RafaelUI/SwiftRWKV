#!/bin/bash
# Мутационная проверка: внести один намеренный дефект, прогнать тесты,
# убедиться, что он пойман, вернуть как было.
#
#   ./Scripts/mutate.sh <файл> <что> <на что> [--filter <набор>]
#
# Выход 0 — мутация ПОЙМАНА (тесты упали, как и должны).
# Выход 1 — мутация НЕ поймана: тесты зелёные при сломанном коде.
set -uo pipefail
cd "$(dirname "$0")/.."

FILE="$1"; FROM="$2"; TO="$3"; shift 3
FILTER=""
if [ "${1:-}" = "--filter" ]; then FILTER="--filter $2"; fi

BACKUP=$(mktemp)
cp "$FILE" "$BACKUP"
restore() { cp "$BACKUP" "$FILE"; rm -f "$BACKUP"; }
trap restore EXIT

python3 - "$FILE" "$FROM" "$TO" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding='utf-8').read()
n = s.count(frm)
if n != 1:
    print(f"ОШИБКА: подстрока встречается {n} раз(а), нужна ровно одна", file=sys.stderr)
    sys.exit(2)
open(path, 'w', encoding='utf-8').write(s.replace(frm, to))
PY
if [ $? -ne 0 ]; then echo "мутация не применена"; exit 2; fi

OUT=$(swift test $FILTER 2>&1); RC=$?

# Признак «код собрался и тесты пошли». Без него любое "error:" — это
# ошибка компиляции, то есть мутация просто невалидна, а не поймана.
# Через here-string, а не `echo | grep -q`: grep -q выходит на первом совпадении,
# echo на большом выводе получает SIGPIPE, и под pipefail конвейер считается
# упавшим — мутация с сотней падений (то есть ПОЙМАННАЯ) объявлялась несобранной.
if ! grep -q "Test Case '" <<< "$OUT"; then
    echo "НЕ СОБРАЛОСЬ (мутация невалидна)"
    grep -E "error:" <<< "$OUT" | head -3 || true
    exit 2
fi

# Решает КОД ВОЗВРАТА, а не разбор текста. Разбор здесь уже подводил дважды.
# Первый раз падение по сигналу и Fatal error он считал ошибкой компиляции.
# Второй раз (и это стоило четырёх ложных «не поймана») строка итога при
# наличии пропущенных тестов читается
#
#     Executed 15 tests, with 1 test skipped and 5 failures (0 unexpected)
#
# то есть после «with» идёт «1 test skipped», а не число падений, и шаблон
# "with [1-9][0-9]* failure" не совпадает НИКОГДА. Достаточно одного
# пропускаемого теста в наборе, чтобы весь прогон мутаций молча объявил
# сломанный код здоровым — то есть ровно тот отказ, от которого мутации и
# защищают.
#
# `swift test` возвращает ненулевой код при любом падении, включая сигнал и
# fatal error. Текстовые признаки оставлены вторым рубежом.
if [ $RC -ne 0 ] \
   || grep -qE "and [1-9][0-9]* failure|with [1-9][0-9]* failure" <<< "$OUT" \
   || grep -qE "unexpected signal code|Fatal error" <<< "$OUT"; then
    echo "ПОЙМАНА"
    grep -E "error:|Fatal error" <<< "$OUT" | grep -v "^error: Process" | head -3 || true
    exit 0
fi

echo "НЕ ПОЙМАНА — тесты зелёные при сломанном коде"
exit 1
