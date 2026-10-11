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
    // Из _get_ckpt_fwd (wkv7_checkpoint.py), нынешняя редакция:
    //  • a / w / k / b / r одинаковы для всех 64 потоков группы — читаются из
    //    глобальной памяти один раз и раздаются через threadgroup (без этого
    //    каждое значение читалось 64 раза; в rwkv-metal это стоило половины
    //    времени ядра, здесь замерено 58 → 28 мс на 24 слоя 1.5B при T = 512);
    // Порядок сложения НЕ тронут (в rwkv-metal суммы ещё и разорваны на восемь
    // аккумуляторов, ~4e-8 сдвига): выход обязан остаться ПОБИТНО равным
    // замороженному ядру — на этом стоит согласие кэша состояний с обучением
    // (WKV7StateTests.testForwardAndTrainAgreeOnState). Разрыв сумм мерился:
    // см. NEXT_SESSION.
    // Память threadgroup: 5 × 64 float = 1280 байт (Metal считает вдвое —
    // 2560 при пределе 32768).
    let source = """
        uint dv  = thread_position_in_threadgroup.y;
        uint bhi = threadgroup_position_in_grid.x;
        uint bi  = bhi / H_C, hi = bhi % H_C;

        threadgroup float a_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C], k_sh[HEAD_SIZE_C];
        threadgroup float b_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];

        float h_row[HEAD_SIZE_C];
        uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[hb+dk];

        for (uint c=0; c<N_CHUNKS_C; c++) {
            for (uint t=0; t<CHUNK_C; t++) {
                uint base = ((bi*T_C + c*CHUNK_C + t)*H_C + hi)*HEAD_SIZE_C;

                a_sh[dv]=a[base+dv]; w_sh[dv]=w[base+dv]; k_sh[dv]=k[base+dv];
                b_sh[dv]=b[base+dv]; r_sh[dv]=r[base+dv];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float sa = 0;
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];
                sa_out[base+dv] = sa;

                float vv = v[base+dv];
                for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                    h_row[dk] = w_sh[dk]*h_row[dk] + vv*k_sh[dk] + sa*b_sh[dk];

                float y = 0;
                for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r_sh[dk];
                out[base+dv] = y;

                // следующий t перезапишет *_sh, пока их ещё читают
                threadgroup_barrier(mem_flags::mem_threadgroup);
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

private var ckptBwd2Cache: [String: MLXFastKernel] = [:]

/// Скалярное произведение четырьмя независимыми аккумуляторами (цепочка
/// зависимых FMA рвётся — ядро упирается в её латентность).
private func dot4(_ dst: String, _ a: (String) -> String, _ b: (String) -> String) -> String {
    var lines = ["float \(dst)0=0.0f,\(dst)1=0.0f,\(dst)2=0.0f,\(dst)3=0.0f;",
                 "for (uint j=0; j<HEAD_SIZE_C; j+=4) {"]
    for u in 0 ..< 4 { lines.append("    \(dst)\(u) += \(a("j+\(u)")) * \(b("j+\(u)"));") }
    lines.append("}")
    lines.append("float \(dst) = (\(dst)0+\(dst)1)+(\(dst)2+\(dst)3);")
    return lines.joined(separator: "\n                ")
}

/// Обратное ядро по схеме ядра Бо Пэна (RWKV-CUDA/rwkv7_fast_fused); порт
/// `_get_ckpt_bwd_v2` из rwkv-metal.
///
/// Прежнее ядро держит СТРОКУ состояния, как прямое, и потому считает
/// dr / dw / dk / da / db МЕЖПОТОКОВЫМИ редукциями: пять обменов через
/// threadgroup на шаг времени, каждый — запись матрицы 64×64 и два барьера.
/// Здесь поток держит КОЛОНКУ: stT[j] = h[j][i]. Тогда все пять величин —
/// скалярные произведения внутри потока с общими векторами шага, а
/// межпотоковый обмен остаётся один (dSb, нужен для da). Пять произведений
/// идут одним проходом по j: порознь они читали три регистровых массива по
/// пять раз.
///
/// Математика та же, порядок сложения другой: с прежним ядром сходится до
/// ~1e-5 (fp32). Память threadgroup: 9 × 64 float = 2304 байта.
private func ckptBwdKernelV2(H: Int, T: Int) -> MLXFastKernel {
    let key = "\(H)_\(T)"
    if let k = ckptBwd2Cache[key] { return k }
    let N = T / WKV7_CHUNK
    let header = """
    constant uint HEAD_SIZE_C = \(WKV7_HEAD_SIZE);
    constant uint T_C         = \(T);
    constant uint CHUNK_C     = \(WKV7_CHUNK);
    constant uint N_CHUNKS_C  = \(N);
    constant uint H_C         = \(H);
    """
    let dr = dot4("dr", { "stT[\($0)]" }, { "dy_sh[\($0)]" })
    let da = dot4("da", { "stT[\($0)]" }, { "dsb_sh[\($0)]" })
    let source = """
        uint i    = thread_position_in_threadgroup.x;
        uint bhi  = threadgroup_position_in_grid.x;
        uint bidx = bhi / H_C, hi = bhi % H_C;

        threadgroup float r_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C], k_sh[HEAD_SIZE_C];
        threadgroup float v_sh[HEAD_SIZE_C], a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
        threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C];
        threadgroup float dsb_sh[HEAD_SIZE_C];

        // stT — колонка i состояния; dst — строка i градиента по состоянию,
        // dstT — его же колонка: dv / dSb берутся по строке, dw / dk / db — по колонке.
        float stT[HEAD_SIZE_C], dst[HEAD_SIZE_C], dstT[HEAD_SIZE_C];

        uint hbase = (bidx*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C;
        uint hrow  = hbase + i*HEAD_SIZE_C;
        for (uint j=0; j<HEAD_SIZE_C; j++) {
            dst[j]  = d_h_out[hrow + j];
            dstT[j] = d_h_out[hbase + j*HEAD_SIZE_C + i];
        }

        for (int c=(int)N_CHUNKS_C-1; c>=0; c--) {
            uint ckb = ((bidx*H_C+hi)*N_CHUNKS_C+(uint)c)*HEAD_SIZE_C*HEAD_SIZE_C;
            for (uint j=0; j<HEAD_SIZE_C; j++)
                stT[j] = h_ckpts[ckb + j*HEAD_SIZE_C + i];

            for (int t=(int)CHUNK_C-1; t>=0; t--) {
                uint base = ((bidx*T_C+(uint)c*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

                r_sh[i]=r[base+i]; w_sh[i]=w[base+i]; k_sh[i]=k[base+i];
                v_sh[i]=v[base+i]; a_sh[i]=a[base+i]; b_sh[i]=b[base+i];
                dy_sh[i]=d_out[base+i]; sa_sh[i]=sa_fwd[base+i];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float ri=r_sh[i], wi=w_sh[i], ki=k_sh[i];
                float ai=a_sh[i], bb=b_sh[i], dyi=dy_sh[i];

                // dr[i] = sum_j h[j][i]*dy[j] — состояние ДО реконструкции
                \(dr)
                dr_out[base+i] = dr;

                // h_prev[j][i] = (h[j][i] - v[j]*k[i] - sa[j]*b[i]) / w[i]
                float iwi = 1.0f/wi;
                for (uint j=0; j<HEAD_SIZE_C; j++) {
                    stT[j]   = (stT[j] - v_sh[j]*ki - sa_sh[j]*bb) * iwi;
                    dst[j]  += dyi*r_sh[j];
                    dstT[j] += ri*dy_sh[j];
                }

                float dw0=0.0f,dw1=0.0f, dkv0=0.0f,dkv1=0.0f, dvv0=0.0f,dvv1=0.0f;
                float db0=0.0f,db1=0.0f, dsb0=0.0f,dsb1=0.0f;
                for (uint j=0; j<HEAD_SIZE_C; j+=2) {
                    float tA=dstT[j],   tB=dstT[j+1];
                    float sA=stT[j],    sB=stT[j+1];
                    float uA=dst[j],    uB=dst[j+1];
                    dw0  += tA*sA;         dw1  += tB*sB;
                    dkv0 += tA*v_sh[j];    dkv1 += tB*v_sh[j+1];
                    db0  += tA*sa_sh[j];   db1  += tB*sa_sh[j+1];
                    dvv0 += uA*k_sh[j];    dvv1 += uB*k_sh[j+1];
                    dsb0 += uA*b_sh[j];    dsb1 += uB*b_sh[j+1];
                }
                float dw=dw0+dw1, dkv=dkv0+dkv1, dvv=dvv0+dvv1;
                float db=db0+db1, dsb=dsb0+dsb1;
                dw_out[base+i] = dw;
                dk_out[base+i] = dkv;
                dv_out[base+i] = dvv;
                db_out[base+i] = db;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dsb_sh[i] = dsb;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // единственная межпотоковая величина: da[i] = sum_j h_prev[j][i]*dSb[j]
                \(da)
                da_out[base+i] = da;

                for (uint j=0; j<HEAD_SIZE_C; j++) {
                    dst[j]  = dst[j]*w_sh[j] + dsb*a_sh[j];
                    dstT[j] = dstT[j]*wi + ai*dsb_sh[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint j=0; j<HEAD_SIZE_C; j++) dh_in_out[hrow + j] = dst[j];
    """
    let kern = MLXFast.metalKernel(
        name: "wkv7_ckpt_bwd2_H\(H)_T\(T)",
        inputNames: ["r","w","k","v","a","b","h_ckpts","sa_fwd","d_out","d_h_out"],
        outputNames: ["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        source: source, header: header
    )
    ckptBwd2Cache[key] = kern
    return kern
}

/// Обратное ядро обучаемого WKV: true — по схеме Бо Пэна (`ckptBwdKernelV2`),
/// false — прежнее. Читается на каждый вызов VJP: A/B в одном процессе.
public nonisolated(unsafe) var wkv7BackwardV2 = true

/// Держать выходы прямого ядра (sa и чекпоинты состояния) до обратного
/// прохода — true, либо пересчитывать прямое ядро внутри VJP — false.
///
/// Пересчёт — лишний запуск прямого ядра на слой и шаг. Держать — дешевле по
/// времени, но sa [B,T,H,S] и чекпоинты [B,H,T/16,S,S] живут до обратного
/// прохода: ~40 КБ на токен и слой при H = 32. Под блочным чекпоинтом
/// (`useBlockCheckpoint`) слой в обратном проходе живёт один, и это ничего не
/// стоит; без него на длинной последовательности набегает (T = 2048, 24 слоя
/// — ~2 ГБ), тогда выгоднее false. В rwkv-metal выходы держатся всегда.
public nonisolated(unsafe) var wkv7KeepForwardOutputs = true

/// Сколько раз поставлено в граф прямое ядро обучаемого WKV (счётчик для
/// тестов: при удержании выходов — один запуск на вызов, при пересчёте — два).
public nonisolated(unsafe) var wkv7TrainForwardLaunches = 0

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

    let N = T / WKV7_CHUNK

    func forwardKernel(_ ins: [MLXArray]) -> [MLXArray] {
        wkv7TrainForwardLaunches += 1
        return ckptFwdKernel(H: H, T: T)(
            ins, grid: (B*H, D, 1), threadGroup: (1, D, 1),
            outputShapes: [[B,T,H,D],[B,H,D,D],[B,T,H,D],[B,H,N,D,D]],
            outputDTypes: [.float32,.float32,.float32,.float32]
        )
    }

    // ВХОДЫ ОБОИХ ЯДЕР — СТРОГО fp32.
    //
    // MLX строит исходник пользовательского ядра из ТИПОВ входов, а
    // кеширует библиотеку по ИМЕНИ. Одно имя с другими типами — для него
    // «исходник сменился»: старая библиотека и её конвейер освобождаются
    // НЕМЕДЛЕННО (Device::clear_library), хотя уже закодированные, но ещё
    // не исполненные запуски в командном буфере на них ссылаются. Примелы
    // слоя 0 приходят в типе параметров (bf16 / fp16), а слоёв выше — в
    // fp32 (выход WKV), и на обратном проходе ядро слоя 0 выбивало конвейер
    // из-под ядра слоя 1: недетерминированные NaN в градиентах r/k/v слоя 1
    // и всего слоя 0 и срывы GPU (найдено 10.10).
    func backward(_ primals: [MLXArray], saFwd: MLXArray, hCkpts: MLXArray,
                  dOut: MLXArray, dHOut: MLXArray?) -> [MLXArray] {
        let dH = dHOut?.asType(.float32) ?? MLXArray.zeros([B, H, D, D], dtype: .float32)
        let kern = wkv7BackwardV2 ? ckptBwdKernelV2(H: H, T: T) : ckptBwdKernel(H: H, T: T)
        return kern(
            primals[0 ..< 6].map { $0.asType(.float32) } + [hCkpts, saFwd, dOut.asType(.float32), dH],
            grid: (B*H*D, 1, 1), threadGroup: (D, 1, 1),
            outputShapes: Array(repeating: [B,T,H,D], count: 6) + [[B,H,D,D]],
            outputDTypes: Array(repeating: DType.float32, count: 7)
        )
    }

    if RWKVAblation.skipWKVBackward || !wkv7KeepForwardOutputs {
        // Пересчёт прямого ядра внутри VJP: Swift-овский VJP { primals,
        // cotangents } не получает выходов прямого прохода, а sa и чекпоинты
        // состояния обратному ядру нужны.
        let fn = CustomFunction {
            Forward { inp in
                let o = forwardKernel(inp.map { $0.asType(.float32) })
                return [o[0], o[1]]           // out, h_out; sa/ckpts — внутренние
            }
            VJP { primals, cotangents in
                if RWKVAblation.skipWKVBackward {                 // только замер, см. RWKVAblation
                    return primals.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                }
                let f = forwardKernel(primals.map { $0.asType(.float32) })
                let g = backward(primals, saFwd: f[2], hCkpts: f[3], dOut: cotangents[0],
                                 dHOut: cotangents.count > 1 ? cotangents[1] : nil)
                // argnums Swift игнорирует ⇒ градиенты по ВСЕМ семи примелам
                // по порядку, включая dh_in (g[6]).
                return zip(g, primals).map { $0.asType($1.dtype) }
            }
        }
        let out = fn([r, w, k, v, a, b, h0])
        return (out[0], out[1])
    }

    // Выходы прямого ядра держатся до обратного прохода — без пересчёта.
    //
    // Обход того, что VJP в Swift не видит выходов Forward: прямое ядро
    // считается СНАРУЖИ дифференцируемой функции, на входах под stopGradient
    // (для автограда его выходы — константы, VJP у ядра не спрашивают), а
    // сама функция принимает их дополнительными входами и пропускает out и
    // h_out как есть. VJP тогда находит sa и чекпоинты среди своих примелов.
    // Градиент идёт только по первым семи входам: остальные четыре от
    // примелов дифференцирования не зависят, и MLX их котангенты не берёт.
    let f = forwardKernel([r, w, k, v, a, b, h0].map { stopGradient($0.asType(.float32)) })
    let fn = CustomFunction {
        Forward { inp in [stopGradient(inp[7]), stopGradient(inp[8])] }
        VJP { primals, cotangents in
            let g = backward(primals, saFwd: primals[9], hCkpts: primals[10], dOut: cotangents[0],
                             dHOut: cotangents.count > 1 ? cotangents[1] : nil)
            return zip(g, primals).map { $0.asType($1.dtype) }
                + primals[7...].map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
        }
    }
    let out = fn([r, w, k, v, a, b, h0, f[0], f[1], f[2], f[3]])
    return (out[0], out[1])
}

// Совместимая обёртка: нулевое начальное состояние, конечное отбрасывается.
// Использовать ТОЛЬКО в обучаемых слоях (frozen — через wkv7Forward).
public func wkv7Train(_ r: MLXArray, _ w: MLXArray, _ k: MLXArray,
               _ v: MLXArray, _ a: MLXArray, _ b: MLXArray) -> MLXArray {
    wkv7TrainWithState(r, w, k, v, a, b, nil).0
}
