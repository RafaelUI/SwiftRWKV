import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Gradient checkpointing на чистом публичном MLX-Swift API (без Cmlx-FFI).
//
//  Forward считает f(inputs) — но за границей CustomFunction внутренние
//  промежуточные тензоры f НЕ удерживаются внешним autodiff-графом (та же
//  механика, что у wkv7Train, чей Forward отбрасывает sa_fwd/h_ckpts).
//  В backward VJP ПЕРЕСЧИТЫВАЕТ f через vjp(...) — recompute checkpointing.
//
//  Стоимость: +1 forward f в backward. Выигрыш: пик памяти не держит
//  активации f между forward и backward.
//
//  ВАЖНО: vjp(f, primals:) считает котангенты ТОЛЬКО к явным аргументам f.
//  Всё, к чему нужен grad (вход x, v_first, LoRA-адаптеры), ОБЯЗАНО быть
//  элементом inputs, а не захваченным в замыкании. Frozen-веса — можно
//  захватывать (grad к ним не нужен).
// ───────────────────────────────────────────────────────────────────────

/// Обернуть `f: [MLXArray] -> [MLXArray]` в gradient-checkpointed функцию.
/// Значения идентичны `f`; отличается только память/скорость backward.
public func checkpointed(
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray] {
    { inputs in
        let fn = CustomFunction {
            Forward { ins in f(ins) }
            VJP { primals, cotangents in
                vjp(f, primals: primals, cotangents: cotangents).1
            }
        }
        return fn(inputs)
    }
}
