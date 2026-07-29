#!/bin/bash
# Мутационная проверка фундамента реранкера: wkv7Step и вынесенный RWKVBlock.
#
#   ./Scripts/mutations_block.sh
#
# Каждая строка — один правдоподобный дефект. Правдоподобный означает «так
# действительно пишут»: перепутанная ось состояния, знак, забытая ветка.
# Тесты обязаны падать на каждом. Мутация, которая НЕ поймана, — это не
# формальность, а сообщение о дыре в наборе.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
STEP=Sources/RWKVKernel/WKV7Step.swift
BLOCK=Sources/RWKVGen/RWKVBlock.swift

FILTER="${1:-}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    if [ -n "$FILTER" ]; then "$M" "$@" --filter "$FILTER"; else "$M" "$@"; fi
}

echo "── wkv7Step ──"

# Оси состояния: h[dv,dk] против h[dk,dv]. Формы совпадают (D×D), поэтому
# ошибка молчит. Ловиться должна цепочкой шагов против сплошного прохода.
run "sa считается по оси value, а не key" \
    "$STEP" 'let sa = (h * aK).sum(axis: -1)' 'let sa = (h * aK).sum(axis: -2)'

run "outer-произведение v·k перевёрнуто" \
    "$STEP" 'h = h * wK + vV * kK' 'h = h * wK + vV.transposed(0, 1, 3, 2) * kK'

run "чтение out по оси value вместо key" \
    "$STEP" 'var out = (h * rK).sum(axis: -1)' 'var out = (h * rK).sum(axis: -2)'

run "затухание применено к старому h после обновления" \
    "$STEP" 'h = h * wK + vV * kK + sa.expandedDimensions(axis: 3) * bK' \
            'h = h + vV * kK + sa.expandedDimensions(axis: 3) * bK'

run "низкоранговый член sa·b потерян" \
    "$STEP" '+ sa.expandedDimensions(axis: 3) * bK' '+ 0.0 * sa.expandedDimensions(axis: 3) * bK'

run "out снят с состояния ДО обновления" \
    "$STEP" 'var out = (h * rK).sum(axis: -1)' \
            'var out = ((h - vV * kK) * rK).sum(axis: -1)'

echo
echo "── RWKVBlock: арифметика ──"

run "token-shift сдвигает не туда" \
    "$BLOCK" 'let shifted = concatenated([head, x[0..., 0 ..< (T - 1)]], axis: 1)' \
             'let shifted = concatenated([x[0..., 1 ..< T], head], axis: 1)'

run "знак a в ядре WKV" \
    "$BLOCK" 'let (o, h) = wkv7Step(r, ww, kWkv, v, -kk, bWkv, hIn)' \
             'let (o, h) = wkv7Step(r, ww, kWkv, v, kk, bWkv, hIn)'

# НЕ ЛОВИТСЯ, и это ответ, а не пробел в наборе.
#
# kWkv отличается от k только на пад-позициях. Их выход и так объявлен
# мусорным: k=0 убирает вклад позиции в состояние, а модель каузальна, так
# что мусор доходит только до пад-позиций следующих слоёв. Состояние
# снимается на endIdx — последнем РЕАЛЬНОМ токене. То есть мутация меняет
# ровно те числа, которые никто не читает.
#
# Оставлено в списке намеренно: если однажды кто-то начнёт читать выход на
# пад-позициях (или уберёт маску), мутация станет ловиться, и это будет
# сигналом, что инвариант изменился.
run "bonus считается по маскированному k (безвредна)" \
    "$BLOCK" 'let bonus = (r * k * g("r_k")).sum(axis: -1, keepDims: true) * v' \
             'let bonus = (r * kWkv * g("r_k")).sum(axis: -1, keepDims: true) * v'

run "ln_x применён ПОСЛЕ bonus" \
    "$BLOCK" 'out = (out + bonus).reshaped([B, T, D])' \
             'out = (out + bonus * 0.0).reshaped([B, T, D])'

run "gate применён без sigmoid внутри" \
    "$BLOCK" 'let gate = linear(sigmoid(linear(xg, g("g_lora_A.weight"))),' \
             'let gate = linear((linear(xg, g("g_lora_A.weight"))),'

run "iclr a через tanh вместо sigmoid" \
    "$BLOCK" 'let a = sigmoid(linear(linear(xa, g("a_lora_A.weight")),' \
             'let a = tanh(linear(linear(xa, g("a_lora_A.weight")),'

run "маска паддинга не нейтрализует w" \
    "$BLOCK" 'ww = ww * m + (1.0 - m)' 'ww = ww * m'

run "cmix без возведения в квадрат" \
    "$BLOCK" 'return ctx.project(h * h, "cmix.value.weight")' \
             'return ctx.project(h, "cmix.value.weight")'

echo
echo "── RWKVBlock: правила копирования весов ──"

run "первый блок стека всё же берёт v_lora" \
    "$BLOCK" '} else if !needsV {' '} else if false {'

run "нейтрализация v_lora через нулевой bias вместо −10" \
    "$BLOCK" 'MLXArray.full(bias.shape, values: MLXArray(Float(-10)), dtype: bias.dtype)' \
             'MLXArray.zeros(bias.shape, dtype: bias.dtype)'

run "value-residual по слою базы, а не по позиции в стеке" \
    "$BLOCK" 'public var hasValueResidual: Bool { index > 0 }' \
             'public var hasValueResidual: Bool { true }'

run "блок читает веса мимо wOverride (обучение встанет)" \
    "$BLOCK" 'if let o = wOverride?[key] { return o }' 'if false, let o = wOverride?[key] { return o }'

echo
echo "── Контекст бэкбона ──"

# Ловится только СТРУКТУРНО (testBackboneKeepsKernelPathAtT1), и это
# осознанный компромисс, а не лень. Численно .step и ядро расходятся на
# ~1e-7 — ниже любого допуска в наборе, поэтому сравнением такое поймать
# нельзя в принципе. А знать надо: через T == 1 идёт рекуррентный декод, и
# молчаливый переезд сдвинул бы все уже снятые замеры паритета.
run "бэкбон переехал на .step (сдвиг всех замеров паритета)" \
    "$BLOCK" 'backbone.trainLayers.contains(layer) ? .train : .forward' \
             'T == 1 ? .step : (backbone.trainLayers.contains(layer) ? .train : .forward)'

run "value-residual слоя базы по позиции ≥ 0" \
    "$BLOCK" 'var hasValueResidual: Bool { layer > 0 }' \
             'var hasValueResidual: Bool { layer >= 0 }'
