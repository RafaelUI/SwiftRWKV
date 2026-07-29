#!/bin/bash
# Мутационная проверка головы реранкера и её лоссов.
#
#   ./Scripts/mutations_rerank.sh [фильтр тестов]
#
# Правдоподобные дефекты: перепутанный зонд, потерянный ln, не тот слой
# состояния, знак в лоссе, сумма вместо среднего.
set -uo pipefail
cd "$(dirname "$0")/.."

M=Scripts/mutate.sh
HEAD=Sources/RWKVRerank/RerankerHead.swift
LOSS=Sources/RWKVRerank/RerankLoss.swift
MODEL=Sources/RWKVRerank/Reranker.swift

FILTER="${1:-RerankerHeadTests|RerankLossTests|RerankParityTests}"
run() {
    local desc="$1"; shift
    printf '%-58s ' "$desc"
    "$M" "$@" --filter "$FILTER"
}

echo "── Голова ──"

# Zero-init последнего слоя MLP. Без него стартовый лосс перестаёт быть
# ln(C), и самый дешёвый детектор сломанной проводки исчезает.
run "score_fc2 инициализирован не нулём" \
    "$HEAD" 'self.fc2Weight = MLXArray.zeros([1, hidden])' \
            'self.fc2Weight = MLXRandom.normal([1, hidden]) * 0.02'

run "скор снимается с ПЕРВОГО зонда, а не с последнего" \
    "$HEAD" 'let lastProbe = x[0..., cfg.nProbe - 1]' \
            'let lastProbe = x[0..., 0]'

run "ln_out потерян" \
    "$HEAD" 'let h = layerNormLast(lastProbe, lnOutW, lnOutB)' \
            'let h = lastProbe'

run "ln0 потерян" \
    "$HEAD" 'x = layerNormLast(x, ln0W, ln0B)' 'x = x + 0.0'

run "нелинейность MLP выброшена" \
    "$HEAD" 'let mid = tanh(matmul(h, fc1W.transposed()) + fc1B)' \
            'let mid = matmul(h, fc1W.transposed()) + fc1B'

run "bias fc1 не применён" \
    "$HEAD" 'let mid = tanh(matmul(h, fc1W.transposed()) + fc1B)' \
            'let mid = tanh(matmul(h, fc1W.transposed()))'

run "все блоки читают слот 0 вместо своего" \
    "$HEAD" 'let hIn = selected[0..., sourceSlot[bi]]' \
            'let hIn = selected[0..., 0]'

run "v_first между блоками не переносится" \
    "$HEAD" 'let (xo, vf, _) = block(x, vFirst, hIn: hIn)' \
            'let (xo, vf, _) = block(x, nil, hIn: hIn)'

run "sharedState читает первый слой вместо последнего" \
    "$HEAD" 'let last = resolved[resolved.count - 1]' 'let last = resolved[0]'

run "uniqueSources не дедуплицируется по возрастанию" \
    "$HEAD" 'let uniq = Array(Set(srcs)).sorted()' \
            'let uniq = Array(Set(srcs)).sorted().reversed().map { $0 }'

run "порядок параметров: fc1/fc2 переставлены" \
    "$HEAD" 'let fc1W = next(), fc1B = next(), fc2W = next()' \
            'let fc1B = next(), fc1W = next(), fc2W = next()'

echo
echo "── Чекпоинт ──"

run "метаданные слоя не проверяются при загрузке" \
    "$MODEL" 'for key in ["layer_idx", "shared_state", "n_probe",' \
             'for key in ["shared_state", "n_probe",'

run "layer_idx пишется до нормализации отрицательных" \
    "$MODEL" 'head.layerIdx.map(String.init).joined(separator: ",")' \
             'head.cfg.layerIdx.map(String.init).joined(separator: ",")'

echo
echo "── Лоссы ──"

run "listwise: знак перевёрнут" \
    "$LOSS" 'return (logZ - picked).mean()' 'return (picked - logZ).mean()'

run "listwise: сумма вместо среднего" \
    "$LOSS" 'return (logZ - picked).mean()' 'return (logZ - picked).sum()'

run "listwise: температура умножает, а не делит" \
    "$LOSS" 'let s = scores.asType(.float32) / temperature' \
            'let s = scores.asType(.float32) * temperature'

run "listwise: logsumexp без вычета максимума" \
    "$LOSS" 'let logZ = mx.squeezed(axis: -1) + log(exp(s - mx).sum(axis: -1))' \
            'let logZ = log(exp(s).sum(axis: -1))'

run "listwise: редукция по оси батча" \
    "$LOSS" 'let mx = s.max(axis: -1, keepDims: true)' \
            'let mx = s.max(axis: 0, keepDims: true)'

run "BCE: цель на позиции 0 вместо label" \
    "$LOSS" 'let targets = (cols .== labels.reshaped([B, 1])).asType(.float32)' \
            'let targets = (cols .== MLXArray.zeros([B, 1], dtype: .int32)).asType(.float32)'

run "BCE: наивная форма вместо устойчивой" \
    "$LOSS" 'let loss = maximum(s, MLXArray(Float(0))) - s * targets
              + log1p(exp(-MLX.abs(s)))' \
            'let loss = -(targets * log(sigmoid(s)) + (1 - targets) * log(1 - sigmoid(s)))'

run "mixed: доли перепутаны местами" \
    "$LOSS" 'return alpha * listwiseLoss(scores, labels, temperature: temperature)
         + (1.0 - alpha) * bceLoss(scores, labels)' \
            'return (1.0 - alpha) * listwiseLoss(scores, labels, temperature: temperature)
         + alpha * bceLoss(scores, labels)'
