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

// Полный forward по всей длине T (frozen backbone): чанкуем по CHUNK,
// переносим h_out → h_in. Backward НЕ нужен. Эквивалент wkv7_train без autodiff.
public func wkv7Forward(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray
) -> MLXArray {
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
    var h = MLXArray.zeros([B, H, D, D], dtype: .float32)
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
    return concatenated(outs, axis: 1)
}
