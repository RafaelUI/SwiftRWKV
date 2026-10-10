import Foundation
import MLX
import MLXFast

// ───────────────────────────── Константы ─────────────────────────────
// Должны совпадать с model/wkv7.py
public let WKV7_HEAD_SIZE = 64
public let WKV7_CHUNK     = 16  // 32 расходился на 1.5B (см. chunk × lr матрицу), 16 численно эквивалентен

// Кэш скомпилированных ядер по числу голов H (как _fwd_cache в Python)
private var fwdKernelCache: [Int: MLXFastKernel] = [:]

// Тело forward-ядра — ДОСЛОВНО как _get_fwd в model/wkv7.py.
// Сигнатура (имена буферов, *_shape) генерируется автоматически.
private func wkv7ForwardKernel(H: Int) -> MLXFastKernel {
    if let k = fwdKernelCache[H] { return k }

    let header = """
    constant uint HEAD_SIZE_C = \(WKV7_HEAD_SIZE);
    constant uint CHUNK_C     = \(WKV7_CHUNK);
    constant uint H_C         = \(H);
    """

    let source = """
        uint dv  = thread_position_in_grid.y;
        uint bhi = thread_position_in_grid.x;
        uint bi  = bhi / H_C; uint hi = bhi % H_C;

        float h_row[HEAD_SIZE_C];
        uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

        for (uint t=0; t<CHUNK_C; t++) {
            uint base = ((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C;

            float sa = 0.0f;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
            sa_out[base+dv] = sa;

            float v_dv = v[base+dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*b[base+dk];

            float y = 0.0f;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
            out[base+dv] = y;
        }
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
    """

    let kern = MLXFast.metalKernel(
        name: "wkv7_fwd_\(H)",
        inputNames: ["r", "w", "k", "v", "a", "b", "h_in"],
        outputNames: ["out", "h_out", "sa_out"],
        source: source,
        header: header
    )
    fwdKernelCache[H] = kern
    return kern
}

// Один чанк forward (T == CHUNK). Входы [B,T,H,D], h_in [B,H,D,D].
// Возвращает (out [B,T,H,D], h_out [B,H,D,D], sa_out [B,T,H,D]).
public func wkv7ChunkForward(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray
) -> (MLXArray, MLXArray, MLXArray) {
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
    precondition(T == WKV7_CHUNK, "T (\(T)) must equal CHUNK \(WKV7_CHUNK)")
    precondition(D == WKV7_HEAD_SIZE, "D (\(D)) must equal HEAD_SIZE \(WKV7_HEAD_SIZE)")

    let kern = wkv7ForwardKernel(H: H)
    let inputs = [r, w, k, v, a, b, hIn].map { $0.asType(.float32) }
    let outputs = kern(
        inputs,
        grid: (B * H, D, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [[B, T, H, D], [B, H, D, D], [B, T, H, D]],
        outputDTypes: [.float32, .float32, .float32]
    )
    return (outputs[0], outputs[1], outputs[2])
}

// Полный forward по всей длине T с ЯВНЫМ граничным состоянием (frozen backbone):
// чанкуем по CHUNK, переносим h_out → h_in. Backward НЕ нужен.
//
// hIn == nil ⇒ нулевое состояние (начало последовательности).
// Возвращает (out [B,T,H,D], hOut [B,H,D,D]) — hOut годится как hIn следующего
// куска той же последовательности. Это то, на чём стоит префикс-кэш реранкера:
// документ сворачивается один раз, запрос продолжает с готового состояния.
//
// Хвост длиной < CHUNK добивается no-op шагами (w=1, r=k=v=a=b=0):
//   h' = 1*h + 0*kᵀ + 0*bᵀ = h
// поэтому hOut с паддингом равен hOut без него — добитые шаги состояние не
// трогают (ср. _pad_to_chunk в kernel/wkv7.py).
public func wkv7ForwardWithState(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    // Одним запуском на всю длину — выход побитно тот же (см. ниже).
    if wkv7WholeSequenceForward { return wkv7SequenceForward(r, w, k, v, a, b, hIn) }
    return wkv7ForwardChunked(r, w, k, v, a, b, hIn)
}

/// Тот же проход НАРЕЗКОЙ по CHUNK — прежняя реализация. Оставлена эталоном
/// для `wkv7SequenceForward` и вторым плечом замера.
public func wkv7ForwardChunked(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
    var h = hIn?.asType(.float32) ?? MLXArray.zeros([B, H, D, D], dtype: .float32)
    precondition(h.shape == [B, H, D, D],
                 "hIn \(h.shape) должен быть [B,H,D,D] = [\(B),\(H),\(D),\(D)]")
    var outs: [MLXArray] = []

    var start = 0
    while start < T {
        let end = min(start + WKV7_CHUNK, T)
        let cl  = end - start
        func slice(_ x: MLXArray, pad val: Float) -> MLXArray {
            var c = x[0..., start ..< end]
            if cl < WKV7_CHUNK {
                let padW = WKV7_CHUNK - cl
                c = padded(c, widths: [.init((0, 0)), .init((0, padW)),
                                       .init((0, 0)), .init((0, 0))],
                           value: MLXArray(val))
            }
            return c
        }
        let rc = slice(r, pad: 0), wc = slice(w, pad: 1), kc = slice(k, pad: 0)
        let vc = slice(v, pad: 0), ac = slice(a, pad: 0), bc = slice(b, pad: 0)

        let (oc, hOut, _) = wkv7ChunkForward(rc, wc, kc, vc, ac, bc, h)
        h = hOut
        outs.append(oc[0..., 0 ..< cl])
        start += WKV7_CHUNK
    }
    return (concatenated(outs, axis: 1), h)
}

// Совместимая обёртка: нулевое начальное состояние, конечное отбрасывается.
public func wkv7Forward(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray
) -> MLXArray {
    wkv7ForwardWithState(r, w, k, v, a, b, nil).0
}

// ───────────────────────────────────────────────────────────────────────
//  Замороженный WKV ОДНИМ ЗАПУСКОМ на всю длину.
//
//  Порт `wkv7_infer` из rwkv-metal (kernel/wkv7.py), с одним отличием:
//  длина приходит ВХОДОМ ядра, а не константой в его исходнике. В Python
//  ядро собирается заново на каждую новую длину (кеш по (H, T)); для чата,
//  где длина промпта каждый раз другая, это компиляция шейдера на каждый
//  запрос. Здесь ядро одно на число голов.
//
//  Зачем. `wkv7ForwardWithState` резал последовательность по CHUNK = 16:
//  срезы шести входов, дополнение хвоста, запуск, склейка выходов — 32
//  запуска на слой при T = 512. По замеру префилла 1.5B это 124 мс из 839
//  против 42 мс у одного запуска (на 0.4B — 111 из 332: весь разрыв с
//  rwkv-quant). Нарезка нужна ОБУЧАЕМОМУ ядру (обратный проход
//  восстанавливает состояние по чанкам), прямому она не нужна вовсе.
//
//  Арифметика шага та же и в том же порядке, состояние между шагами живёт
//  в регистрах потока — выход ПОБИТНО равен нарезанному пути (тест
//  WKV7SequenceTests), меняется только число запусков.
//
//  a / w / k / b / r шага одинаковы для всех 64 потоков группы и читаются
//  из общей памяти группы (как в ядре rwkv-metal): без этого каждое
//  значение читалось бы из глобальной памяти 64 раза.
// ───────────────────────────────────────────────────────────────────────

/// Считать ли замороженный WKV одним запуском на всю длину
/// (`wkv7SequenceForward`) вместо нарезки по CHUNK. Выключатель — для
/// замера и сверки в одном процессе.
nonisolated(unsafe) public var wkv7WholeSequenceForward = true
/// Сколько раз WKV посчитан одним запуском на всю длину (счётчик для
/// тестов: выход побитно тот же, и потеря ускорения иначе не видна).
nonisolated(unsafe) public private(set) var wkv7SequenceLaunches = 0

nonisolated(unsafe) private var seqKernelCache: [Int: MLXFastKernel] = [:]

private func wkv7SequenceKernel(H: Int) -> MLXFastKernel {
    if let k = seqKernelCache[H] { return k }

    let header = """
    constant uint HEAD_SIZE_C = \(WKV7_HEAD_SIZE);
    constant uint H_C         = \(H);
    """

    let source = """
        // dv — индекс потока ВНУТРИ группы: одна группа на пару (b, h)
        uint dv  = thread_position_in_threadgroup.y;
        uint bhi = threadgroup_position_in_grid.x;
        uint bi  = bhi / H_C; uint hi = bhi % H_C;
        uint T   = tlen[0];

        threadgroup float a_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C], k_sh[HEAD_SIZE_C];
        threadgroup float b_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];

        float h_row[HEAD_SIZE_C];
        uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

        for (uint t=0; t<T; t++) {
            uint base = ((bi*T+t)*H_C+hi)*HEAD_SIZE_C;
            a_sh[dv]=a[base+dv]; w_sh[dv]=w[base+dv]; k_sh[dv]=k[base+dv];
            b_sh[dv]=b[base+dv]; r_sh[dv]=r[base+dv];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sa = 0.0f;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];
            float v_dv = v[base+dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = w_sh[dk]*h_row[dk] + v_dv*k_sh[dk] + sa*b_sh[dk];
            float y = 0.0f;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r_sh[dk];
            out[base+dv] = y;

            // следующий шаг перезапишет *_sh, пока кто-то ещё читает текущие
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
    """

    let kern = MLXFast.metalKernel(
        name: "wkv7_seq_\(H)",
        inputNames: ["r", "w", "k", "v", "a", "b", "h_in", "tlen"],
        outputNames: ["out", "h_out"],
        source: source,
        header: header
    )
    seqKernelCache[H] = kern
    return kern
}

/// Прямой WKV по всей длине одним запуском. Входы [B,T,H,D], hIn [B,H,D,D]
/// (nil — нули). Возвращает (out [B,T,H,D], hOut [B,H,D,D]). Обратного
/// прохода нет: только там, где через слой не идёт градиент.
public func wkv7SequenceForward(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
    precondition(T >= 1, "T (\(T)) must be >= 1")
    precondition(D == WKV7_HEAD_SIZE, "D (\(D)) must equal HEAD_SIZE \(WKV7_HEAD_SIZE)")
    let h = hIn?.asType(.float32) ?? MLXArray.zeros([B, H, D, D], dtype: .float32)
    precondition(h.shape == [B, H, D, D],
                 "hIn \(h.shape) должен быть [B,H,D,D] = [\(B),\(H),\(D),\(D)]")
    // Типы входов фиксированы: MLX строит исходник ядра из типов входов, а
    // кеширует по имени (см. WKV7Train.swift).
    let inputs = [r, w, k, v, a, b, h].map { $0.asType(.float32) } + [MLXArray([UInt32(T)])]
    let outputs = wkv7SequenceKernel(H: H)(
        inputs,
        grid: (B * H, D, 1),
        threadGroup: (1, D, 1),
        outputShapes: [[B, T, H, D], [B, H, D, D]],
        outputDTypes: [.float32, .float32]
    )
    wkv7SequenceLaunches += 1
    return (outputs[0], outputs[1])
}
