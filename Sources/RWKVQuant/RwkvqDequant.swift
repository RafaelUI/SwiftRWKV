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
    let nb = inFeatures / gs
    var idx = [Int32](); idx.reserveCapacity(inFeatures)
    for b in 0 ..< nb { idx.append(contentsOf: repeatElement(Int32(b), count: gs)) }
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
