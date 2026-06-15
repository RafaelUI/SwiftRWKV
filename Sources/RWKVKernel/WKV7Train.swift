import Foundation
import MLX
import MLXFast

// ───────────────────────────────────────────────────────────────────────
//  Дифференцируемое WKV-7 для ОБУЧЕНИЯ верхних слоёв.
//  Порт wkv7_checkpoint.py: forward + backward в один GPU-вызов на весь T.
//
//  Для frozen-бэкбона остаётся wkv7Forward (WKV7.swift, без autodiff).
//  Здесь же — версия, завёрнутая в CustomFunction, чтобы MLX-autodiff
//  подхватил наш ручной backward.
//
//  Ключевые отличия от Python-версии:
//   • Swift-овский VJP { primals, cotangents } НЕ получает outputs forward'а,
//     поэтому sa_out и h_checkpoints ПЕРЕСЧИТЫВАЕМ внутри VJP (один лишний
//     forward — это и есть gradient checkpointing, экономит память).
//   • Функция возвращает ТОЛЬКО out (h_out/sa/ckpts — внутренние),
//     значит cotangents = [d_out], а grad по финальному состоянию = 0.
//
//  Требования: T % WKV7_CHUNK == 0, D == WKV7_HEAD_SIZE. Всё в fp32.
// ───────────────────────────────────────────────────────────────────────

private var ckptFwdCache: [String: MLXFastKernel] = [:]
private var ckptBwdCache: [String: MLXFastKernel] = [:]

private func ckptFwdKernel(H: Int, T: Int) -> MLXFastKernel {
    let key = "\(H)_\(T)"
    if let k = ckptFwdCache[key] { return k }
    let N = T / WKV7_CHUNK
    let header = """
    constant uint HEAD_SIZE_C = \(WKV7_HEAD_SIZE);
    constant uint T_C         = \(T);
    constant uint CHUNK_C     = \(WKV7_CHUNK);
    constant uint N_CHUNKS_C  = \(N);
    constant uint H_C         = \(H);
    """
    // дословно из _get_ckpt_fwd
    let source = """
        uint dv = thread_position_in_grid.y;
        uint bhi = thread_position_in_grid.x;
        uint bi = bhi / H_C, hi = bhi % H_C;

        float h_row[HEAD_SIZE_C];
        uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[hb+dk];

        for (uint c=0; c<N_CHUNKS_C; c++) {
            for (uint t=0; t<CHUNK_C; t++) {
                uint base = ((bi*T_C + c*CHUNK_C + t)*H_C + hi)*HEAD_SIZE_C;
                float sa = 0;
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
                sa_out[base+dv] = sa;
                float vv = v[base+dv];
                for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                    h_row[dk] = w[base+dk]*h_row[dk] + vv*k[base+dk] + sa*b[base+dk];
                float y = 0;
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
                out[base+dv] = y;
            }
            uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
        }
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
    """
    let kern = MLXFast.metalKernel(
        name: "wkv7_ckpt_fwd_H\(H)_T\(T)",
        inputNames: ["r","w","k","v","a","b","h_in"],
        outputNames: ["out","h_out","sa_out","h_checkpoints"],
        source: source, header: header
    )
    ckptFwdCache[key] = kern
    return kern
}

private func ckptBwdKernel(H: Int, T: Int) -> MLXFastKernel {
    let key = "\(H)_\(T)"
    if let k = ckptBwdCache[key] { return k }
    let N = T / WKV7_CHUNK
    let header = """
    constant uint HEAD_SIZE_C = \(WKV7_HEAD_SIZE);
    constant uint T_C         = \(T);
    constant uint CHUNK_C     = \(WKV7_CHUNK);
    constant uint N_CHUNKS_C  = \(N);
    constant uint H_C         = \(H);
    """
    // дословно из _get_ckpt_bwd
    let source = """
        uint dv = thread_position_in_threadgroup.x;
        uint bhi = threadgroup_position_in_grid.x;
        uint bi = bhi / H_C, hi = bhi % H_C;

        threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C];
        threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];
        threadgroup float w_sh[HEAD_SIZE_C], a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
        threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C], dsa_sh[HEAD_SIZE_C];

        float C_row[HEAD_SIZE_C], h_row[HEAD_SIZE_C];
        uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[hb+dk];

        for (int c=(int)N_CHUNKS_C-1; c>=0; c--) {
            uint ckb = ((bi*H_C+hi)*N_CHUNKS_C+(uint)c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_ckpts[ckb+dk];

            for (int t=(int)CHUNK_C-1; t>=0; t--) {
                uint base = ((bi*T_C+(uint)c*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;
                k_sh[dv]=k[base+dv]; v_sh[dv]=v[base+dv]; r_sh[dv]=r[base+dv];
                w_sh[dv]=w[base+dv]; a_sh[dv]=a[base+dv]; b_sh[dv]=b[base+dv];
                dy_sh[dv]=d_out[base+dv]; sa_sh[dv]=sa_fwd[base+dv];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float dy_dv = dy_sh[dv];
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] += dy_dv*r_sh[dk];

                float dsa_dv=0, dv_val=0;
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                    dsa_dv += C_row[dk]*b_sh[dk];
                    dv_val += C_row[dk]*k_sh[dk];
                }
                dv_out[base+dv] = dv_val;
                dsa_sh[dv] = dsa_dv;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
                dr_out[base+dv] = dr_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float sa_dv=sa_sh[dv], v_dv=v_sh[dv];
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                    float hp=(h_row[dk]-v_dv*k_sh[dk]-sa_dv*b_sh[dk])/w_sh[dk];
                    accum[dv][dk]=C_row[dk]*hp; h_row[dk]=hp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float dw_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dw_val+=accum[s][dv];
                dw_out[base+dv] = dw_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=C_row[dk]*v_dv;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float dk_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dk_val+=accum[s][dv];
                dk_out[base+dv] = dk_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=dsa_sh[dv]*h_row[dk];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
                da_out[base+dv] = da_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=sa_sh[dv]*C_row[dk];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
                db_out[base+dv] = db_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                    C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];
            }
        }
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[hb+dk] = C_row[dk];
    """
    let kern = MLXFast.metalKernel(
        name: "wkv7_ckpt_bwd_H\(H)_T\(T)",
        inputNames: ["r","w","k","v","a","b","h_ckpts","sa_fwd","d_out","d_h_out"],
        outputNames: ["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        source: source, header: header
    )
    ckptBwdCache[key] = kern
    return kern
}

// Дифференцируемый WKV-7 на весь T. Входы [B,T,H,D]. Возвращает out [B,T,H,D].
// Использовать ТОЛЬКО в обучаемых слоях (frozen — через wkv7Forward).
public func wkv7Train(_ r: MLXArray, _ w: MLXArray, _ k: MLXArray,
               _ v: MLXArray, _ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let fn = CustomFunction {
        Forward { inp in
            let r = inp[0], w = inp[1], k = inp[2], v = inp[3], a = inp[4], b = inp[5]
            let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
            let N = T / WKV7_CHUNK
            let hIn = MLXArray.zeros([B, H, D, D], dtype: .float32)
            let ins = [r,w,k,v,a,b,hIn].map { $0.asType(.float32) }
            let o = ckptFwdKernel(H: H, T: T)(
                ins, grid: (B*H, D, 1), threadGroup: (1,1,1),
                outputShapes: [[B,T,H,D],[B,H,D,D],[B,T,H,D],[B,H,N,D,D]],
                outputDTypes: [.float32,.float32,.float32,.float32]
            )
            return [o[0]]                 // только out; остальное — внутреннее
        }
        VJP { primals, cotangents in
            let r = primals[0], w = primals[1], k = primals[2]
            let v = primals[3], a = primals[4], b = primals[5]
            let dOut = cotangents[0]
            let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
            let N = T / WKV7_CHUNK
            let hIn = MLXArray.zeros([B, H, D, D], dtype: .float32)
            // 1) пересчёт forward → sa_fwd, h_checkpoints (out отбрасываем)
            let f = ckptFwdKernel(H: H, T: T)(
                [r,w,k,v,a,b,hIn].map { $0.asType(.float32) },
                grid: (B*H, D, 1), threadGroup: (1,1,1),
                outputShapes: [[B,T,H,D],[B,H,D,D],[B,T,H,D],[B,H,N,D,D]],
                outputDTypes: [.float32,.float32,.float32,.float32]
            )
            let saFwd = f[2], hCkpts = f[3]
            let dHOut = MLXArray.zeros([B, H, D, D], dtype: .float32)
            // 2) backward
            let g = ckptBwdKernel(H: H, T: T)(
                [r,w,k,v,a,b,hCkpts,saFwd,dOut.asType(.float32),dHOut],
                grid: (B*H*D, 1, 1), threadGroup: (D, 1, 1),
                outputShapes: Array(repeating: [B,T,H,D], count: 6) + [[B,H,D,D]],
                outputDTypes: Array(repeating: DType.float32, count: 7)
            )
            // grad по r,w,k,v,a,b к dtype примала; dh_in (g[6]) отбрасываем
            return zip([g[0],g[1],g[2],g[3],g[4],g[5]], primals).map { $0.asType($1.dtype) }
        }
    }
    return fn([r, w, k, v, a, b])[0]
}
