"""
Эталон блока реранкера: снимает с rwkv-metal (Python/MLX) на реальной 0.1B то,
против чего проверяется Swift.

Что здесь проверяется и почему именно это
──────────────────────────────────────────
Паритет body() говорит, что база перенесена верно. Голова реранкера — не база:
это отдельный стек RWKV-блоков, который читает состояние базы одним токеном-
зондом. У него три места, где перенос может разойтись внутренне непротиворечиво:

  1. wkv7_step — один шаг рекуррентности мимо Metal-ядра. Раскладка состояния
     h[dv, dk] не самоочевидна, а формы при перепутанных осях совпадают.
  2. Правила КОПИРОВАНИЯ весов слоя базы в блок головы. Их три, и все три
     молчаливые: первый блок стека не берёт v_lora; блок не-первый,
     инициализированный слоем 0 (где v_lora нет), получает НЕЙТРАЛЬНУЮ
     (B=0, bias=−10), а не случайную; всё остальное копируется как есть.
  3. Сам проход блока поверх заданного h_in.

Веса блока в дамп НЕ кладутся: они целиком выводятся из базы, паритет которой
уже доказан, и класть их значило бы проверять копирование против самого себя.
Вместо этого кладутся контрольные суммы нескольких весов — если правила
копирования разошлись, расходятся и они, и это видно сразу, до разбора выхода.

Запуск:
    cd ~/Develop/rwkv-metal
    .venv/bin/python ~/Develop/SwiftRWKV/Scripts/dump_block_reference.py \
        --model world_0.1b_x070.safetensors \
        --out   ~/Develop/SwiftRWKV/.testdata/block_ref.safetensors
"""
import argparse
import os

import mlx.core as mx

import rwkv_metal as rk
from rwkv_metal.kernel.wkv7 import wkv7_step
from rwkv_metal.model.rwkv7_x070 import RWKV7X070, RWKVBlock, lora_ranks

# Слои базы, из которых собираются блоки. 5 — «середина стека», рекомендуемый
# источник по docs/reranker.md; 0 — единственный слой без v_lora, то есть тот,
# на котором срабатывает правило нейтрализации; -1 — умолчание конфига.
SOURCE_LAYERS = [0, 5, 11]

# Текст пары: то, что реально сворачивается в состояние. Длина неважна —
# важно, чтобы состояние было НЕнулевым и не вырожденным.
PAIR_TEXT = (
    "Instruct: Given a query, retrieve the passage that answers it\n"
    "Document: Пчёлы зимуют, сбиваясь в плотный клуб внутри улья: наружные "
    "особи периодически меняются местами с внутренними, а тепло вырабатывается "
    "работой грудных мышц.\n"
    "Query: как пчёлы переживают зиму?"
)

TERMINATOR = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="world_0.1b_x070.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4321)
    args = ap.parse_args()

    # Геометрия — из форм весов: конфигурации, которую можно задать неверно,
    # тут просто нет (тот же приём, что в dump_reference.py).
    weights = mx.load(os.path.expanduser(args.model))
    n_layer = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
    n_embd = weights["ln_out.weight"].shape[0]
    vocab = weights["head.weight"].shape[0]
    head_size = weights["blocks.0.tmix.k_k"].shape[1]
    print(f"model: L={n_layer} D={n_embd} V={vocab} head={head_size}")

    class Cfg:
        pass

    cfg = Cfg()
    cfg.n_layer, cfg.n_embd, cfg.vocab_size = n_layer, n_embd, vocab
    cfg.head_size, cfg.n_head = head_size, n_embd // head_size
    ranks = lora_ranks(n_embd)

    base = RWKV7X070(cfg, ranks)
    base.load_weights(list(weights.items()))
    mx.eval(base.parameters())

    H, S, D = cfg.n_head, cfg.head_size, cfg.n_embd
    out = {}
    mx.random.seed(args.seed)

    # ── 1. wkv7_step на синтетике ─────────────────────────────────────
    #
    # Входы кладутся в дамп вместе с выходом: Swift обязан считать ТЕ ЖЕ
    # числа, а не «похожие на своих случайных». Распределения — как у
    # настоящих: w из exp(-0.606531·sigmoid), (a, b) — DPLR-пара из
    # нормированного k.
    B = 2
    r_ = mx.random.normal([B, 1, H, S]) * 0.5
    v_ = mx.random.normal([B, 1, H, S]) * 0.5
    k_ = mx.random.normal([B, 1, H, S]) * 0.5
    kk = k_ / mx.sqrt(mx.sum(k_ * k_, axis=-1, keepdims=True) + 1e-12)
    iclr = mx.sigmoid(mx.random.normal([B, 1, H, S]))
    a_ = -kk
    b_ = kk * iclr
    w_ = mx.exp(-0.606531 * mx.sigmoid(mx.random.normal([B, 1, H, S])))
    h0 = mx.random.normal([B, H, S, S]) * 0.1
    mx.eval(r_, w_, k_, v_, a_, b_, h0)

    s_out, s_h = wkv7_step(r_, w_, k_, v_, a_, b_, h0)
    for name, arr in [("r", r_), ("w", w_), ("k", k_), ("v", v_),
                      ("a", a_), ("b", b_), ("h_in", h0),
                      ("out", s_out), ("h_out", s_h)]:
        out["step/" + name] = arr.astype(mx.float32)

    # ── 2. Состояние базы на реальном тексте ──────────────────────────
    tok = rk.WorldTokenizer()
    ids = tok.encode(PAIR_TEXT) + [TERMINATOR]
    idx = mx.array([ids])
    state = base.states(idx)
    out["pair/idx"] = idx.astype(mx.int32)
    for L in SOURCE_LAYERS:
        out[f"pair/wkv_{L}"] = state.wkv[L].astype(mx.float32)
    print(f"пара: {len(ids)} токенов, состояние {state.wkv[0].shape}")

    # ── 3. Блок головы поверх этого состояния ─────────────────────────
    #
    # (source_layer, index_in_stack) покрывает все три правила копирования:
    #   (5, 0)  — обычный случай: слой с v_lora, но блок первый ⇒ не копируем
    #   (5, 1)  — обычный случай с v_lora
    #   (0, 1)  — слой БЕЗ v_lora на позиции, где она нужна ⇒ нейтрализация
    #   (11, 0) — умолчание конфига (-1)
    probe = mx.random.normal([1, 1, D]) * 0.5
    v_first_in = mx.random.normal([1, 1, H, S]) * 0.5
    mx.eval(probe, v_first_in)
    out["block/probe"] = probe.astype(mx.float32)
    out["block/v_first_in"] = v_first_in.astype(mx.float32)

    for src, index in [(5, 0), (5, 1), (0, 1), (11, 0)]:
        tag = f"block/s{src}_i{index}"
        blk = RWKVBlock(cfg, index, ranks)
        _init_from_base(blk, base, src)

        h_in = state.wkv[src].astype(mx.float32)
        v_in = None if index == 0 else v_first_in
        # return_state отдаёт ещё и сдвиги token-shift — блоку головы они не
        # нужны (зонд один, продолжать нечего), но пятёрка есть пятёрка.
        x, v_out, h_out, _, _ = blk(probe, v_in, h_in=h_in, return_state=True)
        mx.eval(x, v_out, h_out)

        out[tag + "/x"] = x.astype(mx.float32)
        out[tag + "/v_first_out"] = v_out.astype(mx.float32)
        out[tag + "/h_out"] = h_out.astype(mx.float32)

        # Контрольные суммы весов: разошлись правила копирования — разойдутся
        # и они, и это будет видно ДО разбора выхода.
        out[tag + "/wsum"] = _weight_checksums(blk)

    # ── 4. ГОЛОВА целиком ─────────────────────────────────────────────
    #
    # Веса головы кладутся в дамп и оттуда же загружаются в Swift. Здесь это
    # необходимо, а не избыточно (в отличие от весов блока, которые целиком
    # выводятся из базы): у головы есть зонды и MLP, которых в базе нет
    # вовсе, и породить их одинаково двумя генераторами случайных чисел
    # нельзя.
    #
    # score_fc2 РАНДОМИЗИРУЕТСЯ. Штатная инициализация задаёт ноль — и тогда
    # скор тождественно ноль при любой, в том числе полностью сломанной,
    # реализации всего, что до него. Паритет на нулях не значит ничего.
    from rwkv_metal.reranker import Reranker, RerankerConfig
    from mlx.utils import tree_flatten

    # ДВЕ конфигурации, и вторая существует не для полноты.
    #
    # Голова из одного блока с одним зондом не исполняет три ветки: выбор
    # слота состояния (слот всегда 0), перенос v_first между блоками
    # (переносить некуда) и снятие скора с ПОСЛЕДНЕГО зонда (он же первый).
    # Мутационная проверка это и показала — все три дефекта проходили мимо
    # тестов, пока эталон был только одноблочный. Вторая конфигурация
    # (два блока над РАЗНЫМИ слоями, два зонда) закрывает ровно их.
    head_cfgs = {
        "h1": RerankerConfig(layer_idx=(5,)),
        "h2": RerankerConfig(layer_idx=(0, 5), n_probe=2),
    }

    for tag, rcfg in head_cfgs.items():
        rr = Reranker(base, rcfg, head_dtype=mx.float32)
        hd = rr.head
        hidden = hd.score_fc1.weight.shape[0]
        hd.score_fc2.weight = mx.random.normal([1, hidden]) * 0.05
        mx.eval(hd.parameters())

        for name, arr in tree_flatten(hd.parameters()):
            out[f"head/{tag}/w/" + name] = arr.astype(mx.float32)

        # Батч из ЧЕТЫРЁХ разных состояний: одинаковые дали бы одинаковые
        # скоры, и перепутанная ось батча прошла бы мимо. Каждый читаемый
        # слой идёт своим слотом — если слоты перепутать, скоры разъедутся.
        def four(layer):
            w = state.wkv[layer][0]
            return mx.stack([w, state.wkv[0][0], state.wkv[11][0],
                             0.5 * (w + state.wkv[11][0])], axis=0)

        sel = mx.stack([four(s) for s in hd.unique_sources], axis=1)
        scores = hd(sel.astype(mx.float32))
        mx.eval(scores)
        out[f"head/{tag}/selected"] = sel.astype(mx.float32)
        out[f"head/{tag}/scores"] = scores.astype(mx.float32)
        out[f"head/{tag}/unique_sources"] = mx.array(hd.unique_sources).astype(mx.int32)
        print(f"{tag}: слои {hd.layer_idx}, uniq {hd.unique_sources}, "
              f"зондов {rcfg.n_probe}, скоры "
              f"{[round(float(x), 5) for x in scores]}")

    # Лоссы считаются на скорах одноблочной головы — какие именно, неважно,
    # важно чтобы они были разные и воспроизводимые.
    scores = out["head/h1/scores"]

    # Лоссы на этих скорах: они проверяются отдельно от головы, потому что
    # ломаются отдельно (устойчивая форма BCE, температура, ось softmax).
    from rwkv_metal.reranker import listwise_loss, bce_loss, mixed_loss

    sc = scores.reshape(1, 4)
    lbl = mx.array([2])
    out["loss/listwise"] = listwise_loss(sc, lbl).astype(mx.float32)
    out["loss/listwise_t05"] = listwise_loss(sc, lbl, 0.5).astype(mx.float32)
    out["loss/bce"] = bce_loss(sc, lbl).astype(mx.float32)
    out["loss/mixed07"] = mixed_loss(sc, lbl, 0.7).astype(mx.float32)
    out["loss/scores"] = sc.astype(mx.float32)
    out["loss/labels"] = lbl.astype(mx.int32)

    mx.eval(list(out.values()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mx.save_safetensors(args.out, out)

    print(f"\nstep out    max|·| = {float(mx.max(mx.abs(s_out))):.6f}")
    for src, index in [(5, 0), (5, 1), (0, 1), (11, 0)]:
        t = f"block/s{src}_i{index}"
        print(f"{t}  max|x| = {float(mx.max(mx.abs(out[t + '/x']))):.6f}")
    print(f"listwise {float(out['loss/listwise']):.6f}  "
          f"bce {float(out['loss/bce']):.6f}  "
          f"mixed(0.7) {float(out['loss/mixed07']):.6f}")
    print(f"\n-> {args.out}")


def _init_from_base(blk, base, src_idx):
    """То же, что RerankerHead.init_from_base, но для одного блока.

    Продублировано здесь, а не импортировано, намеренно: дамп обязан снимать
    ПРАВИЛА, а не вызывать ту же функцию, которую проверяет.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    src = dict(tree_flatten(base.blocks[src_idx].parameters()))
    dst_keys = set(k for k, _ in tree_flatten(blk.parameters()))
    upd = {k: v for k, v in src.items() if k in dst_keys}
    blk.update(tree_unflatten(list(upd.items())))

    missing_v = [k for k in dst_keys if k.startswith("tmix.v_lora") and k not in src]
    if missing_v:
        tm = blk.tmix
        tm.v_lora_B.weight = mx.zeros_like(tm.v_lora_B.weight)
        tm.v_lora_B.bias = mx.full(tm.v_lora_B.bias.shape, -10.0,
                                   dtype=tm.v_lora_B.bias.dtype)

    # fp32: официальные веса bf16, и голова унаследовала бы 8 бит мантиссы.
    from mlx.utils import tree_map
    blk.update(tree_map(lambda x: x.astype(mx.float32) if isinstance(x, mx.array) else x,
                        blk.parameters()))
    mx.eval(blk.parameters())
    return blk


def _weight_checksums(blk):
    """Суммы нескольких весов блока в фиксированном порядке.

    Не хеш: сумма fp32 сравнивается с допуском, а хеш требовал бы побитового
    совпадения представления, которого между реализациями никто не обещал.
    Порядок фиксирован списком — сортировка перемешала бы записи при
    переименовании.
    """
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(blk.parameters()))
    keys = ["ln1.weight", "ln2.weight", "tmix.x_r", "tmix.x_k", "tmix.k_k",
            "tmix.r_k", "tmix.r_proj.weight", "tmix.o_proj.weight",
            "tmix.w_lora_B.bias", "tmix.a_lora_B.bias",
            "tmix.ln_x.weight", "cmix.x_k", "cmix.key.weight",
            "cmix.value.weight"]
    vals = [mx.sum(flat[k].astype(mx.float32)) for k in keys]
    # v_lora: если её нет — записываем NaN, чтобы «нет веса» и «вес нулевой»
    # не слились в одно значение. Нулевая сумма как раз бывает у
    # нейтрализованной v_lora_B, и спутать эти два случая было бы легко.
    for k in ["tmix.v_lora_B.weight", "tmix.v_lora_B.bias"]:
        vals.append(mx.sum(flat[k].astype(mx.float32)) if k in flat
                    else mx.array(float("nan")))
    return mx.stack(vals).astype(mx.float32)


if __name__ == "__main__":
    main()
