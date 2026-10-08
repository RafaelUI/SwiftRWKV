import Foundation
import MLX
import MLXFast

// ───────────────────────────────────────────────────────────────────────
//  Fused Metal-деквантизация sb6 → dense, один launch.
//
//  Порт rwkv_metal/lora/rwkvq_kernel.py. Исходник ядра перенесён ДОСЛОВНО:
//  расхождение здесь не «чуть хуже качество», а тихий сдвиг относительно
//  калибровки. Пресет REDUCTION откалиброван ровно под эту арифметику —
//  в частности под ФИНАЛЬНЫЙ КОМБАЙН В FP32 (не в half). Половинная точность
//  на этом шаге даёт ~18% расхождений на одном бите мантиссы bf16, и
//  калибровка, которая этого не предполагает, начинает мерить не то.
//
//  ТИП АРИФМЕТИКИ И ТИП ХРАНЕНИЯ — РАЗНЫЕ ВЕЩИ, и путать их дорого.
//  Комбайн `code·s + m` считается в float ВСЕГДА, независимо от `dtype`;
//  меняется только то, чем результат записывается в память. Округление при
//  этом ровно одно и ровно там же, где оно происходило раньше: вызывающий
//  всё равно немедленно приводил fp32-транзиент к bf16 (`baseProj`,
//  `embed`), потому что вся модель считает в bf16. То есть `dtype: .bfloat16`
//  даёт БИТ-В-БИТ тот же результат, что старый путь, — и это утверждение
//  проверяется тестом, а не предполагается.
//
//  Зачем: транзиент fp32 вдвое больше нужного, и на векторной проекции всё
//  упирается в пропускную способность памяти. Замерено на 2.9B — деквантизация
//  занимает 57–62% времени проекции, а один только матмул по bf16-весам
//  против fp32 идёт 0.608 против 0.965 мс (2560×2560) и 3.613 против 6.918 мс
//  (65536×2560).
//
//  Раскладка (источник истины — rwkv_quant/formats/schema.py):
//    qblk[row, blk] = 16Б кодов (block-local split-ниббл, gs=32)
//                     [+ 4Б qh (бит 4) [+ 4Б qh2 (бит 5)]]
//    qsqm[row, blk] = uchar2 (qs, qm+31 как int8) на блок
//    ddm[row, sblk] = half2 (d, dm) на суперблок, sblk = blk / gw_sb
//    w = code · half(qs·d) + half(qm·dm), финальная сборка в float32
//
//  Почему не деквантовать композицией MLX-операций: замер в Python
//  (tests/dev_check_rwkvq_fused_kernel.py) дал 3.8–7.1× в пользу одного
//  launch против ~8 отдельных операций, при бит-в-бит совпадении.
// ───────────────────────────────────────────────────────────────────────

/// Типы, которыми ядро умеет записывать результат.
///
/// Список закрытый намеренно: имя типа уходит в ИСХОДНИК Metal, и опечатка
/// проявится ошибкой компиляции ядра в рантайме, а не при сборке.
private func metalTypeName(_ d: DType) -> String {
    switch d {
    case .float32: return "float"
    case .bfloat16: return "bfloat16_t"
    case .float16: return "float16_t"
    default: preconditionFailure("деквантизация в \(d) не поддержана")
    }
}

/// Ключ кэша скомпилированных ядер: геометрия и тип выхода входят в исходник
/// константами, значит на каждое сочетание нужно своё ядро.
private struct DequantKey: Hashable {
    let inFeatures: Int
    let outFeatures: Int
    let xbits: Int
    let superBlock: Int
    let dtype: String
}

private var dequantKernelCache: [DequantKey: MLXFastKernel] = [:]

private func dequantKernel(inFeatures IN: Int, outFeatures OUT: Int,
                           xbits: Int, superBlock gwSb: Int,
                           dtype: DType) -> MLXFastKernel {
    let outType = metalTypeName(dtype)
    let key = DequantKey(inFeatures: IN, outFeatures: OUT,
                         xbits: xbits, superBlock: gwSb, dtype: outType)
    if let k = dequantKernelCache[key] { return k }

    precondition((0 ... 2).contains(xbits), "xbits \(xbits) вне {0,1,2}")
    precondition(IN % 32 == 0, "IN (\(IN)) должен делиться на размер блока 32")

    let NB = IN / 32                 // блоков на строку
    let NSB = NB / gwSb              // суперблоков на строку
    let SU = 4 + xbits               // слов uint32 на блок в qblk

    let header = """
    constant uint IN_C  = \(IN);
    constant uint OUT_C = \(OUT);
    constant uint NB_C  = \(NB);
    constant uint NSB_C = \(NSB);
    constant uint SB_C  = \(gwSb);
    constant uint SU_C  = \(SU);
    constant uint XBITS = \(xbits);
    constant uint TOTAL = \(OUT * NB);
    """

    // Дополнительные битплоскости подставляются в ИСХОДНИК по xbits (а не
    // только гасятся рантайм-условием): при xbits=0 буферов qh/qh2 в блоке
    // просто нет, и читать их нельзя.
    let qhBody = xbits >= 1 ? """
        if (XBITS >= 1) {
            uint hb = qb[4];
            for (uint c = 0; c < 32; c++) nib[c] |= uchar(((hb >> c) & 1u) << 4);
        }
    """ : ""

    let qh2Body = xbits >= 2 ? """
        if (XBITS >= 2) {
            uint hb2 = qb[5];
            for (uint c = 0; c < 32; c++) nib[c] |= uchar(((hb2 >> c) & 1u) << 5);
        }
    """ : ""

    let source = """
        uint idx = thread_position_in_grid.x;
        if (idx >= TOTAL) return;
        uint row = idx / NB_C;
        uint blk = idx % NB_C;

        device const uint* qb = (device const uint*)qblk + (row * NB_C + blk) * SU_C;
        thread uchar nib[32];
        for (uint w = 0; w < 4; w++) {
            uint word = qb[w];
            for (uint b = 0; b < 4; b++) {
                uchar byte = uchar((word >> (b * 8)) & 0xFFu);
                uint j = w * 4 + b;
                nib[j]      = byte & 0xFu;
                nib[j + 16] = (byte >> 4) & 0xFu;
            }
        }
    \(qhBody)\(qh2Body)
        uchar2 sm = ((device const uchar2*)qsqm)[row * NB_C + blk];
        half2  dd = ((device const half2*)ddm)[row * NSB_C + blk / SB_C];
        half s  = (half)((float)sm.x * (float)dd.x);
        half mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);
        float sf = float(s);
        float mf = float(mn);

        device \(outType)* orow = out + row * IN_C + blk * 32;
        for (uint c = 0; c < 32; c++) {
            // Комбайн В FLOAT при любом типе хранения: приведение стоит
            // одним округлением на записи, а не на каждом слагаемом.
            orow[c] = (\(outType))(float(nib[c]) * sf + mf);
        }
    """

    let kern = MLXFast.metalKernel(
        name: "rwkvq_dequant\(4 + xbits)_\(IN)_\(OUT)_\(outType)",
        inputNames: ["qblk", "qsqm", "ddm"],
        outputNames: ["out"],
        source: source,
        header: header
    )
    dequantKernelCache[key] = kern
    return kern
}

/// packed sb6 → dense [OUT, IN] за один launch.
///
/// `dtype` — тип ХРАНЕНИЯ результата; арифметика комбайна всегда float.
/// Умолчание fp32 оставлено, чтобы эталоны бит-в-бит и все существующие
/// вызовы не сдвинулись: менять умолчание там, где на нём стоят проверки, —
/// самый дешёвый способ выродить их молча.
public func rwkvqDequantDense(qblk: MLXArray, qsqm: MLXArray, ddm: MLXArray,
                              outFeatures OUT: Int, inFeatures IN: Int,
                              superBlock gwSb: Int, xbits: Int,
                              dtype: DType = .float32) -> MLXArray {
    let NB = IN / 32
    let kern = dequantKernel(inFeatures: IN, outFeatures: OUT,
                             xbits: xbits, superBlock: gwSb, dtype: dtype)
    let threadgroup = 256
    let total = OUT * NB
    let groups = (total + threadgroup - 1) / threadgroup
    return kern([qblk, qsqm, ddm],
                grid: (groups * threadgroup, 1, 1),
                threadGroup: (threadgroup, 1, 1),
                outputShapes: [[OUT, IN]],
                outputDTypes: [dtype])[0]
}

// ───────────────────────────────────────────────────────────────────────
//  Деквантизация asym/rtn — обычными MLX-операциями, не fused-ядром.
//
//  Почему не Metal, как у sb6: это низкоранговые LoRA-матрицы модели
//  (w1/w2/a1/a2/v1/v2/g1/g2, 2560×96..320) — на 2-3 порядка меньше
//  sb6-проекций (2560×2560..10240). Разворачиваются ОДИН раз при сборке
//  бэкбона из сайдкара (RwkvqSidecar.buildDenseWeights), не на каждом
//  forward — цена одноразовая, писать под неё отдельный кернель незачем.
// ───────────────────────────────────────────────────────────────────────

/// gw-asym (LoRA @6, gw64): codes[OUT,IN] unsigned-контейнер, scale/min
/// [OUT, NB] fp32 на блок ширины `groupSize`. Порт reader.py::_dequantize_gw_asym.
public func rwkvqDequantAsym(codes: MLXArray, scale: MLXArray, min: MLXArray,
                             groupSize gs: Int, dtype: DType = .float32) -> MLXArray {
    let inFeatures = codes.shape[1]
    // ceil(IN/gs) блоков — IN не обязан делиться на gs нацело (напр. w1/a1:
    // IN=96, gs=64 -> 2 блока, второй неполный). Индекс строится ПОКОЛОНОЧНО
    // (c / gs), а не через nb*gs: целочисленное nb=IN/gs при неполном
    // последнем блоке даёт КОРОТКИЙ индекс и молча берёт scale/min не из
    // того блока — ровно так проявлялось на w1/a1 (96 % 64 != 0).
    var idx = [Int32](); idx.reserveCapacity(inFeatures)
    for c in 0 ..< inFeatures { idx.append(Int32(c / gs)) }
    let idxArr = MLXArray(idx)
    let scaleC = scale.asType(.float32).take(idxArr, axis: 1)   // [OUT, IN]
    let minC = min.asType(.float32).take(idxArr, axis: 1)
    let w = codes.asType(.float32) * scaleC + minC
    return w.asType(dtype)
}

/// per-row RTN: codes[OUT,IN] int8 (уже распакован, если источник был
/// нибблами), scale[OUT,1] fp16 -- один множитель на строку.
/// Порт reader.py::_dequantize_one (без SpQR-ветки -- см. вызывающий код:
/// текущие пресеты outlier_fracs не используют).
public func rwkvqDequantRTN(codes: MLXArray, scale: MLXArray,
                            dtype: DType = .float32) -> MLXArray {
    let w = codes.asType(.float32) * scale.asType(.float32)
    return w.asType(dtype)
}

/// uint8 [rows, ceil(cols/2)] -> int8 [rows, cols], BIASED SPLIT-раскладка
/// (см. schema.py: низкий ниббл байта i = колонка i, высокий = колонка
/// i + ceil(cols/2), код хранится как code+8 без знака).
/// Порт schema.py::unpack_int4.
public func rwkvqUnpackInt4(_ packed: MLXArray, columns nCols: Int) -> MLXArray {
    let lo = (packed & MLXArray(UInt8(0x0F))).asType(.int32) - 8
    let hi = (packed >> MLXArray(UInt8(4))).asType(.int32) - 8
    let full = concatenated([lo, hi], axis: 1)     // split-раскладка
    return full[0..., 0 ..< nCols].asType(.int8)
}

// ───────────────────────────────────────────────────────────────────────
//  Раскладка sym (Q6_K): блок 16 БЕЗ min, масштаб блока — int8-код против
//  одной fp16 d на суперблок из 16 блоков (256 весов). 6 и 8 бит.
//
//  Порт rwkv_quant/backends/metal/quant_linear_sym.py::_get_kernel_dequant
//  (скалярные ветки "half", "float" и "bf16"). Арифметика перенесена
//  ДОСЛОВНО, включая то, чем ветки отличаются друг от друга:
//
//    fp16:  w = (half)q * s          — произведение в half. Под этой веткой
//                                      измерены числа пресета; её текст в
//                                      rwkv-quant заморожен гейтом равенства.
//    fp32:  w = (float)q * (float)s  — бит-в-бит codec.dequant_sym.
//    bf16:  то же, что fp32, с одним округлением при записи.
//
//  Масштаб `s = half(qs · d)` округляется через half ВО ВСЕХ ветках, как в
//  файле: без half-роундтрипа ядро расходится с writer на последнем бите.
//
//  Интерлив (qblk) — раскладка загрузчика, не диска:
//    bits=8: qblk[row] = IN знаковых байт-кодов, как есть;
//    bits=6: на ПАРУ блоков (32 колонки) 16Б нибблов + 4Б qh + 4Б qh2.
//            Коды лежат СО СДВИГОМ +32 в 0...63; сдвиг снимается в ядре.
//            Split нибблов блок-локальный, а блок 16 — поэтому колонки
//            пары идут l0,l1,h0,h1 (первый блок), l2,l3,h2,h3 (второй).
//            Перепутать порядок — получить правдоподобный, но неверный
//            результат: величины те же, места не те.
//
//  Почему не родной quantizedMM, как у sb6: MLX принимает группы 32, 64 и
//  128, блока 16 там нет. Точной перекладки не существует, а неточная была
//  бы ДРУГОЙ квантованной базой под тем же именем.
// ───────────────────────────────────────────────────────────────────────

/// Порядок регистров → порядок колонок внутри пары блоков по 16.
private let symPairRegs = ["l0", "l1", "h0", "h1", "l2", "l3", "h2", "h3"]

/// Мульт-трюк битплоскости: ниббл → 4 байта одним умножением, порядок
/// регистров — парный.
private func symPlanePair(_ src: String, shift: Int) -> String {
    var out = ""
    for (i, reg) in symPairRegs.enumerated() {
        let sh = i * 4
        let nib = sh == 0 ? "(\(src) & 0xFu)"
            : sh == 28 ? "(\(src) >> 28)" : "((\(src) >> \(sh)) & 0xFu)"
        out += "    \(reg) |= as_type<uchar4>((\(nib) * 0x00204081u & 0x01010101u) << \(shift));\n"
    }
    return out
}

/// Восемь записей на регистр × четыре регистра = 16 колонок одного блока.
private func symWrites(_ regs: ArraySlice<String>, scale: String, base: Int,
                       outType: String) -> String {
    var out = ""
    for (i, reg) in regs.enumerated() {
        for (c, comp) in ["x", "y", "z", "w"].enumerated() {
            let idx = base + i * 4 + c
            switch outType {
            case "half":
                out += "    o[\(idx)] = ((half)(float)\(reg).\(comp) - (half)32.0h) * \(scale);\n"
            case "bfloat16_t":
                out += "    o[\(idx)] = (bfloat16_t)(((float)\(reg).\(comp) - 32.0f) * (float)\(scale));\n"
            default:
                out += "    o[\(idx)] = ((float)\(reg).\(comp) - 32.0f) * (float)\(scale);\n"
            }
        }
    }
    return out
}

private struct SymDequantKey: Hashable {
    let inFeatures: Int
    let outFeatures: Int
    let bits: Int
    let dtype: String
}

private var symDequantKernelCache: [SymDequantKey: MLXFastKernel] = [:]

private func symMetalTypeName(_ d: DType) -> String {
    switch d {
    case .float32: return "float"
    case .bfloat16: return "bfloat16_t"
    case .float16: return "half"
    default: preconditionFailure("деквантизация sym в \(d) не поддержана")
    }
}

private func symDequantKernel(inFeatures IN: Int, outFeatures OUT: Int,
                              bits: Int, dtype: DType) -> MLXFastKernel {
    let T = symMetalTypeName(dtype)
    let key = SymDequantKey(inFeatures: IN, outFeatures: OUT, bits: bits, dtype: T)
    if let k = symDequantKernelCache[key] { return k }

    precondition(bits == 6 || bits == 8, "sym: битность \(bits) вне {6, 8}")
    precondition(IN % 256 == 0, "sym-ядро: IN=\(IN) не кратен суперблоку 256")

    let total = bits == 8 ? OUT * (IN / 16) : OUT * (IN / 32)
    let header = """
    constant uint IN_C  = \(IN);
    constant uint OUT_C = \(OUT);
    constant uint NB    = \(IN / 16);
    constant uint NSB   = \(IN / 256);
    constant uint NPAIR = \(IN / 32);
    constant uint TOTAL = \(total);
    """

    let source: String
    if bits == 8 {
        let write: String
        switch T {
        case "half":
            write = "o[i] = (half)((float)c[row*IN_C + b*16 + i]) * s;"
        case "bfloat16_t":
            write = "o[i] = (bfloat16_t)((float)c[row*IN_C + b*16 + i] * (float)s);"
        default:
            write = "o[i] = (float)c[row*IN_C + b*16 + i] * (float)s;"
        }
        source = """
            uint gid = thread_position_in_grid.x;
            if (gid >= TOTAL) return;
            uint row = gid / NB;
            uint b   = gid % NB;
            device const char* c  = (device const char*)qblk;
            device const char* sc = (device const char*)qs;
            half s = (half)((float)sc[row*NB + b] * (float)d[row*NSB + b/16]);
            device \(T)* o = out + row*IN_C + b*16;
            for (uint i = 0; i < 16; i++)
                \(write)
        """
    } else {
        source = """
            uint gid = thread_position_in_grid.x;
            if (gid >= TOTAL) return;
            uint row = gid / NPAIR;
            uint p   = gid % NPAIR;
            device const uint* qb = ((device const uint*)qblk) + gid*6;
            uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
            uint hb = qb[4];
        \(symPlanePair("hb", shift: 4))    uint hb2 = qb[5];
        \(symPlanePair("hb2", shift: 5))
            device const char* sc = (device const char*)qs;
            float dsb = (float)d[row*NSB + p/8];
            half s0 = (half)((float)sc[row*NB + 2*p]     * dsb);
            half s1 = (half)((float)sc[row*NB + 2*p + 1] * dsb);
            device \(T)* o = out + row*IN_C + p*32;
        \(symWrites(symPairRegs[0 ..< 4], scale: "s0", base: 0, outType: T))
        \(symWrites(symPairRegs[4 ..< 8], scale: "s1", base: 16, outType: T))
        """
    }

    let kern = MLXFast.metalKernel(
        name: "rwkvq_symdq\(bits)_\(IN)_\(OUT)_\(T)",
        inputNames: ["qblk", "qs", "d"],
        outputNames: ["out"],
        source: source,
        header: header
    )
    symDequantKernelCache[key] = kern
    return kern
}

/// sym-интерлив → dense [OUT, IN].
///
/// `dtype` здесь — НЕ только тип хранения: ветка fp16 считает произведение
/// в half (норма инференса rwkv-quant), ветки fp32 и bf16 — во float.
/// При IN, не кратном суперблоку 256, ядро не применимо; тогда считает
/// цепочка MLX-операций (`rwkvqDequantSymReference`) — fp32-арифметика
/// codec с приведением результата.
public func rwkvqDequantSym(qblk: MLXArray, qs: MLXArray, d: MLXArray,
                            outFeatures OUT: Int, inFeatures IN: Int, bits: Int,
                            dtype: DType = .float32) -> MLXArray {
    guard IN % 256 == 0 else {
        return rwkvqDequantSymReference(qblk: qblk, qs: qs, d: d, outFeatures: OUT,
                                        inFeatures: IN, bits: bits).asType(dtype)
    }
    let kern = symDequantKernel(inFeatures: IN, outFeatures: OUT, bits: bits, dtype: dtype)
    let threadgroup = 256
    let total = OUT * (IN / (bits == 8 ? 16 : 32))
    let groups = (total + threadgroup - 1) / threadgroup
    return kern([qblk, qs, d],
                grid: (groups * threadgroup, 1, 1),
                threadGroup: (threadgroup, 1, 1),
                outputShapes: [[OUT, IN]],
                outputDTypes: [dtype])[0]
}

/// Байт int8, лежащий в uint8-контейнере, → значение (int32). Арифметикой,
/// а не сменой вида: не зависит от того, как версия MLX трактует каст.
private func signedByte(_ u: MLXArray) -> MLXArray {
    ((u.asType(.int32) + 128) % 256) - 128
}

/// ЭТАЛОН: тот же деквант цепочкой MLX-операций, fp32, формула
/// codec.dequant_sym. Медленный, но от ядра не зависит — ради этого и есть:
/// им сверяется ядро и считаются формы, которых ядро не берёт.
public func rwkvqDequantSymReference(qblk: MLXArray, qs: MLXArray, d: MLXArray,
                                     outFeatures OUT: Int, inFeatures IN: Int,
                                     bits: Int) -> MLXArray {
    let NB = IN / 16
    var q: MLXArray
    if bits == 8 {
        q = signedByte(qblk.reshaped([OUT, IN])).asType(.float32)
    } else {
        let NP = IN / 32
        let blk = qblk.reshaped([OUT, NP, 24])
        // блок-локальный split: байт j блока несёт колонку j (low) и j+8 (high)
        let cb = blk[0..., 0..., 0 ..< 16].reshaped([OUT, NP * 2, 8])
        var qi = concatenated([cb & 0xF, cb >> 4], axis: 2).asType(.int32)
        let shifts = MLXArray(Array(0 ..< 8).map { UInt8($0) })
        for (off, weight) in [(16, Int32(16)), (20, Int32(32))] {
            let plane = blk[0..., 0..., off ..< (off + 4)].reshaped([OUT, IN / 8])
            let b = (plane.expandedDimensions(axis: -1) >> shifts) & 1
            qi = qi + b.reshaped([OUT, NP * 2, 16]).asType(.int32) * weight
        }
        q = (qi - 32).asType(.float32).reshaped([OUT, IN])    // снятие сдвига упаковки
    }
    let dd = repeated(d.asType(.float32), count: 16, axis: 1)[0..., 0 ..< NB]
    let scale = (signedByte(qs).asType(.float32) * dd).asType(.float16).asType(.float32)
    return (q.reshaped([OUT, NB, 16]) * scale.reshaped([OUT, NB, 1])).reshaped([OUT, IN])
}
