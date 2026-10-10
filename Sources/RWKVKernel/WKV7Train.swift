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
//   • Возвращаются два выхода — out и h_out (sa/ckpts остаются внутренними),
//     значит cotangents = [d_out, d_h_out].
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
    // Из _get_ckpt_bwd (wkv7_checkpoint.py), с одним отличием: общих на
    // threadgroup копий входов шага (k_sh, v_sh, r_sh, w_sh, a_sh, b_sh,
    // dy_sh, sa_sh, dsa_sh) здесь НЕТ.
    //
    // Почему. Память threadgroup у Metal ограничена 32768 байтами, и считает
    // он её вдвое против объявленного (4096 float — уже предел, проверено
    // валидацией). Одна матрица обмена accum[64][64] занимает предел
    // целиком; девять векторов поверх давали 37376. Без валидации ядро
    // при этом ЗАПУСКАЕТСЯ и обычно считает верно, а портит память рядом —
    // отсюда срывы GPU на обучении настоящих моделей (CommandBuffer Hang /
    // PageFault, NaN в градиентах нижних слоёв; найдено 10.10 прогоном с
    // MTL_SHADER_VALIDATION=1).
    //
    // Чем заменены. Копии были не нужны по существу: k, v, r, w, a, b,
    // d_out, sa_fwd — входные буферы, их читает любой поток напрямую, а
    // dsa каждый поток читал только свой — это локальная переменная.
    // Значения и порядок операций те же, поэтому выход ПОБИТНО прежний
    // (сверено снимком до и после на четырёх формах). Заодно ушли два
    // барьера на шаг.
    let source = """
        uint dv = thread_position_in_threadgroup.x;
        uint bhi = threadgroup_position_in_grid.x;
        uint bi = bhi / H_C, hi = bhi % H_C;

        threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C];

        float C_row[HEAD_SIZE_C], h_row[HEAD_SIZE_C];
        uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[hb+dk];

        for (int c=(int)N_CHUNKS_C-1; c>=0; c--) {
            uint ckb = ((bi*H_C+hi)*N_CHUNKS_C+(uint)c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_ckpts[ckb+dk];

            for (int t=(int)CHUNK_C-1; t>=0; t--) {
                uint base = ((bi*T_C+(uint)c*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

                float dy_dv = d_out[base+dv];
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] += dy_dv*r[base+dk];

                float dsa_dv=0, dv_val=0;
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                    dsa_dv += C_row[dk]*b[base+dk];
                    dv_val += C_row[dk]*k[base+dk];
                }
                dv_out[base+dv] = dv_val;

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
                dr_out[base+dv] = dr_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float sa_dv=sa_fwd[base+dv], v_dv=v[base+dv];
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                    float hp=(h_row[dk]-v_dv*k[base+dk]-sa_dv*b[base+dk])/w[base+dk];
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

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=dsa_dv*h_row[dk];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
                da_out[base+dv] = da_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=sa_dv*C_row[dk];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
                db_out[base+dv] = db_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                    C_row[dk] = C_row[dk]*w[base+dk] + dsa_dv*a[base+dk];
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

// Дифференцируемый WKV-7 на весь T с ЯВНЫМ граничным состоянием.
// Входы [B,T,H,D], hIn [B,H,D,D] (nil ⇒ нули). Возвращает (out, hOut).
//
// Дифференцируемо и по hIn: backward-ядро уже считает dh_in_out, здесь он
// доходит до вызывающего, а не отбрасывается. Это то, что позволяет учить
// голову ПОВЕРХ кэшированного состояния (реранкер) и тюнить само состояние.
//
// hOut — второй выход, поэтому котангентов два: [d_out, d_h_out]. Если
// вызывающий не использует hOut, MLX подаёт нулевой d_h_out, и результат
// совпадает с прежним поведением.
//
// ВАЖНО: T обязана делиться на WKV7_CHUNK. Ядро считает N = T/CHUNK целочисленно,
// поэтому некратный T молча обработал бы меньше токенов — precondition делает
// это громким. (Python добивает T паддингом в _pad_to_chunk; здесь паддинг
// остаётся на стороне вызывающего, как и было.)
public func wkv7TrainWithState(
    _ r: MLXArray, _ w: MLXArray, _ k: MLXArray, _ v: MLXArray,
    _ a: MLXArray, _ b: MLXArray, _ hIn: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
    precondition(T % WKV7_CHUNK == 0,
                 "T (\(T)) должна делиться на CHUNK \(WKV7_CHUNK)")
    precondition(D == WKV7_HEAD_SIZE,
                 "D (\(D)) должен равняться HEAD_SIZE \(WKV7_HEAD_SIZE)")
    let h0 = hIn ?? MLXArray.zeros([B, H, D, D], dtype: .float32)
    precondition(h0.shape == [B, H, D, D],
                 "hIn \(h0.shape) должен быть [B,H,D,D] = [\(B),\(H),\(D),\(D)]")

    let fn = CustomFunction {
        Forward { inp in
            let r = inp[0], w = inp[1], k = inp[2], v = inp[3], a = inp[4], b = inp[5]
            let hIn = inp[6]
            let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
            let N = T / WKV7_CHUNK
            let ins = [r,w,k,v,a,b,hIn].map { $0.asType(.float32) }
            let o = ckptFwdKernel(H: H, T: T)(
                ins, grid: (B*H, D, 1), threadGroup: (1,1,1),
                outputShapes: [[B,T,H,D],[B,H,D,D],[B,T,H,D],[B,H,N,D,D]],
                outputDTypes: [.float32,.float32,.float32,.float32]
            )
            return [o[0], o[1]]           // out, h_out; sa/ckpts — внутренние
        }
        VJP { primals, cotangents in
            let r = primals[0], w = primals[1], k = primals[2]
            let v = primals[3], a = primals[4], b = primals[5]
            let hIn = primals[6]
            let B = r.shape[0], T = r.shape[1], H = r.shape[2], D = r.shape[3]
            let N = T / WKV7_CHUNK
            let dOut  = cotangents[0]
            // Swift-овский VJP не получает outputs forward'а (третий аргумент
            // C-замыкания — argnums — отбрасывается), поэтому sa_fwd и
            // h_checkpoints пересчитываем: один лишний forward и есть
            // gradient checkpointing.
            let f = ckptFwdKernel(H: H, T: T)(
                [r,w,k,v,a,b,hIn].map { $0.asType(.float32) },
                grid: (B*H, D, 1), threadGroup: (1,1,1),
                outputShapes: [[B,T,H,D],[B,H,D,D],[B,T,H,D],[B,H,N,D,D]],
                outputDTypes: [.float32,.float32,.float32,.float32]
            )
            let saFwd = f[2], hCkpts = f[3]
            let dHOut = cotangents.count > 1
                ? cotangents[1].asType(.float32)
                : MLXArray.zeros([B, H, D, D], dtype: .float32)
            // ВХОДЫ ОБРАТНОГО ЯДРА — СТРОГО fp32, как у прямого.
            //
            // MLX строит исходник пользовательского ядра из ТИПОВ входов, а
            // кеширует библиотеку по ИМЕНИ. Одно имя с другими типами — для
            // него «исходник сменился»: старая библиотека и её конвейер
            // освобождаются НЕМЕДЛЕННО (Device::clear_library), хотя уже
            // закодированные, но ещё не исполненные запуски в командном
            // буфере на них ссылаются. Здесь именно это и происходило:
            // примелы слоя 0 приходят в типе параметров (bf16 / fp16), а слоёв
            // выше — в fp32 (выход WKV), и на обратном проходе ядро слоя 0
            // выбивало конвейер из-под ядра слоя 1. Отсюда недетерминированные
            // NaN в градиентах r/k/v слоя 1 и всего слоя 0 и срывы GPU
            // (Hang / PageFault / InnocentVictim) — и отсюда же их исчезновение
            // при малых командных буферах, под валидацией Metal и при
            // однотипных параметрах (найдено 10.10).
            // Значения те же: ядро и раньше читало bf16 с неявным
            // приведением к float.
            let g = ckptBwdKernel(H: H, T: T)(
                [r,w,k,v,a,b].map { $0.asType(.float32) }
                    + [hCkpts,saFwd,dOut.asType(.float32),dHOut],
                grid: (B*H*D, 1, 1), threadGroup: (D, 1, 1),
                outputShapes: Array(repeating: [B,T,H,D], count: 6) + [[B,H,D,D]],
                outputDTypes: Array(repeating: DType.float32, count: 7)
            )
            // argnums Swift игнорирует ⇒ возвращаем градиенты по ВСЕМ семи
            // примелам по порядку, включая dh_in (g[6]).
            return zip(g, primals).map { $0.asType($1.dtype) }
        }
    }
    let out = fn([r, w, k, v, a, b, h0])
    return (out[0], out[1])
}

// Совместимая обёртка: нулевое начальное состояние, конечное отбрасывается.
// Использовать ТОЛЬКО в обучаемых слоях (frozen — через wkv7Forward).
public func wkv7Train(_ r: MLXArray, _ w: MLXArray, _ k: MLXArray,
               _ v: MLXArray, _ a: MLXArray, _ b: MLXArray) -> MLXArray {
    wkv7TrainWithState(r, w, k, v, a, b, nil).0
}
