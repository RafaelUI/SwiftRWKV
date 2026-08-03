#!/bin/bash
# Мутационная проверка сэмплера и цикла генерации.
#
#   ./Scripts/mutations_sampling.sh [фильтр тестов]
#
# Сэмплер отличается от остального пакета тем, что его дефекты НЕ ВИДНЫ в
# выдаче. Модель с проигнорированным top-p, с перепутанными topK и topP, с
# незадействованным сидом — все они выдают связный текст. Единственный
# способ узнать, что параметр действительно работает, — сломать его нарочно
# и проверить, что тесты об этом сказали.
#
# Мутации подобраны так, чтобы каждая соответствовала правдоподобной ошибке
# при написании кода, а не произвольной порче.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
S=Sources/RWKVGen/Sampling.swift
G=Sources/RWKVGen/Generate.swift

FILTER="${1:-SamplingTests|GenerateTests|GenerateSmokeTests}"
run() {
    local desc="$1"; shift
    printf '%-52s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Генератор и сид ──"

# Сид не доходит до генератора: все сэмплеры делят одну последовательность.
run "сид игнорируется" \
    "$S" 'self.rng = SplitMix64(seed: config.seed)
    }

    /// Сбросить' 'self.rng = SplitMix64(seed: 0)
    }

    /// Сбросить'

# Вырожденный генератор: воспроизводимость идеальна, разнообразия нет.
run "генератор вырожден в константу" \
    "$S" 'Float(Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0))' \
         'Float(Double(next() >> 11) * 0.0) + 0.5'

# reset не трогает поток — продолжение вместо перезапуска.
run "reset не перезапускает поток" \
    "$S" 'rng = SplitMix64(seed: config.seed)
        penalty.reset()' \
         'penalty.reset()'

echo
echo "── Температура ──"

# Температура не применяется: распределение всегда исходное.
run "температура не делит логиты" \
    "$S" 'let scaled = adjusted / config.temperature' \
         'let scaled = adjusted'

# Умножение вместо деления — знакомая опечатка, меняющая смысл на обратный.
run "температура умножает вместо деления" \
    "$S" 'let scaled = adjusted / config.temperature' \
         'let scaled = adjusted * config.temperature'

echo
echo "── Отсечения ──"

# topK не применяется вовсе.
run "topK игнорируется" \
    "$S" 'let limit = config.topK > 0 ? min(config.topK, p.count) : p.count' \
         'let limit = p.count'

# Сдвиг границы на единицу — классическая ошибка отсечения.
run "topK оставляет на одного больше" \
    "$S" 'let limit = config.topK > 0 ? min(config.topK, p.count) : p.count' \
         'let limit = config.topK > 0 ? min(config.topK + 1, p.count) : p.count'

# topP не применяется: хвост остаётся.
run "topP игнорируется" \
    "$S" 'if mass >= config.topP { break }' \
         'if mass >= 1.0 { break }'

# Строгое сравнение вместо нестрогого — off-by-one по массе.
run "topP сравнивает строго" \
    "$S" 'if mass >= config.topP { break }' \
         'if mass > config.topP { break }'

# Выход ДО увеличения счётчика: кандидатов на одного меньше.
#
# Здесь раньше стояла мутация «проверять массу до прибавления», то есть
# перенос `if` в начало тела цикла. Она НЕ ЛОВИЛАСЬ — и правильно: при mass=0
# на входе условие ложно, накопление идёт, выход происходит на следующем
# витке, и множество получается ТО ЖЕ САМОЕ. Мутация была не дефектом, а
# переписыванием того же цикла другими словами. Заменена на настоящий
# сдвиг границы.
run "topP теряет последнего кандидата" \
    "$S" 'mass += p[kept]
            kept += 1
            if mass >= config.topP { break }' \
         'mass += p[kept]
            if mass >= config.topP { break }
            kept += 1'

# Нет перенормировки усечённого распределения: частоты остаются исходными,
# а остаток массы уходит последнему кандидату.
run "усечённое распределение не нормируется" \
    "$S" 'let u = rng.uniform() * mass' \
         'let u = rng.uniform()'

echo
echo "── Штрафы ──"

# Знак не учитывается: на отрицательных логитах штраф работает наоборот.
run "штраф не смотрит на знак логита" \
    "$S" 'MLX.where(seen .> 0,
                                    seen / config.repetitionPenalty,
                                    seen * config.repetitionPenalty)' \
         'seen / config.repetitionPenalty'

# Штраф добавляется вместо вычитания.
run "аддитивный штраф прибавляется" \
    "$S" 'out = out.at[idx].subtract(sub)' \
         'out = out.at[idx].add(sub)'

# presence и frequency перепутаны местами: накопление пропадает.
run "частотный штраф не накапливается" \
    "$S" 'let sub = counts * config.frequencyPenalty + config.presencePenalty' \
         'let sub = counts * 0 + config.frequencyPenalty + config.presencePenalty'

# Затухание применяется ПОСЛЕ прибавления: свежий токен ослаблен наравне
# со старыми.
run "затухание задевает свежий токен" \
    "$S" 'occurrence[id, default: 0] += 1
    }

    public mutating func reset()' \
         'occurrence[id, default: 0] += 1
        if decay != 1 { for (k, v) in occurrence { occurrence[k] = v * decay } }
    }

    public mutating func reset()'

echo
echo "── Цикл генерации ──"

# Состояние не догоняет выдачу: продолжение теряет последний токен.
run "состояние отстаёт на токен" \
    "$G" 'while fed < produced.count {
            _ = step(produced[fed], state: &state)
            fed += 1
        }' \
         'if false { _ = step(produced[0], state: &state) }'

# Стоп-строка не вырезается из текста.
run "стоп-строка остаётся в тексте" \
    "$G" 'bytes.removeSubrange(hit.start ..< bytes.count)' \
         '_ = hit.start'

# Стоп-строка ищется только в последнем токене: пропускается всё, что легло
# на границу токенов.
run "стоп-строка ищется только в новых байтах" \
    "$G" 'from: max(0, grewFrom - holdBack))' \
         'from: grewFrom)'

# Хвост не придерживается: начало стоп-строки утекает в onToken.
run "хвост не придерживается" \
    "$G" 'let limit = holdBack == 0 ? bytes.count : bytes.count - holdBack' \
         'let limit = bytes.count'

# Граница UTF-8 не соблюдается: поток рвётся посреди символа.
run "поток рвётся посреди символа" \
    "$G" 'let safeEnd = max(emitted, utf8Boundary(bytes, upTo: max(0, limit)))' \
         'let safeEnd = max(emitted, max(0, limit))'

# Придержанный хвост не отдаётся в конце: последние байты теряются.
run "хвост не досылается в конце" \
    "$G" 'if let onToken, reason != .cancelled, emitted < bytes.count {' \
         'if let onToken, reason == .cancelled, emitted < bytes.count {'

# Стоп-токен попадает в текст.
run "стоп-токен попадает в текст" \
    "$G" 'if config.stopTokens.contains(id) {
                reason = .stopToken(id)
                break loop
            }' \
         'let mustStop = config.stopTokens.contains(id)'

echo
echo "Итог: смотри строки НЕ ПОЙМАНА выше. Каждая — либо пробел в наборе"
echo "тестов, либо утверждение о том, что мутация безвредна; второе"
echo "нужно обосновать разбором, а не оставлять как есть."
