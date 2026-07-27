#!/bin/bash
# Пакет мутаций по коду дообучения эмбеддингов.
# Каждая строка — один намеренный дефект. Интересны НЕПОЙМАННЫЕ.
cd "$(dirname "$0")/.."
M=./Scripts/mutate.sh

run() {
    echo "════════════════════════════════════════════════════════"
    echo "МУТАЦИЯ: $1"
    shift
    $M "$@"
    echo
}

# 2. Композиция: вторая часть получает НАЧАЛО массива вместо своего среза.
run "Composite: срез не сдвигается между частями" \
  Sources/RWKVGen/Training/TrainableSet.swift \
  'body(part, Array(ps[offset ..< offset + n]))
            offset += n' \
  'body(part, Array(ps[0 ..< n]))
            offset += n' \
  --filter EmbeddingFinetuneTests

# 3. Голова не подставляется — обучение головы уходит в никуда.
run "Голова: inject ничего не делает" \
  Sources/RWKVEmbedding/EmbeddingTrainable.swift \
  'public func inject(_ ps: [MLXArray]) {
        head.setParameters(ps)
    }' \
  'public func inject(_ ps: [MLXArray]) {
    }' \
  --filter EmbeddingFinetuneTests

# 5. Ранг: ничья засчитывается как проигрыш.
run "Метрики: ничья считается проигрышем" \
  Sources/RWKVEmbedding/EmbeddingMetrics.swift \
  'for j in 0 ..< width where rowsData[base + j] > target { better += 1 }' \
  'for j in 0 ..< width where rowsData[base + j] >= target { better += 1 }' \
  --filter EmbeddingFinetuneTests

# 6. nDCG: сдвиг в логарифме.
run "Метрики: nDCG без +1 в логарифме" \
  Sources/RWKVEmbedding/EmbeddingMetrics.swift \
  'ndcg += 1.0 / log2(Double(rank) + 1.0)' \
  'ndcg += 1.0 / log2(Double(rank) + 2.0)' \
  --filter EmbeddingFinetuneTests

# 7. nDCG: отсечка на 10 снята — метрика перестаёт быть @10.
run "Метрики: nDCG@10 без отсечки" \
  Sources/RWKVEmbedding/EmbeddingMetrics.swift \
  'if rank <= 10 { ndcg += 1.0 / log2(Double(rank) + 1.0) }' \
  'ndcg += 1.0 / log2(Double(rank) + 1.0)' \
  --filter EmbeddingParityTests

# 11. GradCache: подстановка параметров вынесена наружу фаз.
run "GradCache: нет inject внутри embedChunk" \
  Sources/RWKVEmbedding/EmbeddingObjective.swift \
  '                embedChunk: { ps, start in
                    trainable.inject(ps)' \
  '                embedChunk: { ps, start in' \
  --filter EmbeddingFinetuneTests

# 13. Классификация: argmax по всем меткам, а не по пулу строки.
run "Метрики: классификация игнорирует пул строки" \
  Sources/RWKVEmbedding/EmbeddingMetrics.swift \
  'for (j, label) in pool.enumerated() {
                let s = flat[i * width + labelIndex[label]!]
                if s > best { best = s; bestJ = j }
            }' \
  'for (j, label) in labelTexts.enumerated() {
                let s = flat[i * width + labelIndex[label]!]
                if s > best { best = s; bestJ = j }
            }' \
  --filter EmbeddingFinetuneTests

# 14. Выравнивание: округление вниз вместо вверх.
run "encodeBatch: padMultiple округляет вниз" \
  Sources/RWKVEmbedding/EmbeddingDataset.swift \
  'maxLen += mult - (maxLen % mult)' \
  'maxLen -= maxLen % mult' \
  --filter EmbeddingFinetuneTests

# 15. Полный режим тянет в обучение LM-голову и таблицу эмбеддингов.
run "Режим .full включает emb и LM-голову" \
  Sources/RWKVEmbedding/EmbeddingTrainable.swift \
  'let keys = bb.bodyWeightKeys' \
  'let keys = bb.weightKeys' \
  --filter EmbeddingFinetuneTests

# 16. Токенизатор: \x в str-repr снова как сырой байт.
run "Токенизатор: \\xNN в str-repr как сырой байт" \
  Sources/RWKVGen/WorldTokenizer.swift \
  '                    appendUTF8(codePoint: UInt32(code), to: &bytes)' \
  '                    bytes.append(code)' \
  --filter WorldTokenizerTests

echo "════════════════════════════════════════════════════════"
echo "готово"
