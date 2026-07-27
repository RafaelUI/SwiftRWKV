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

/// Ключ кэша скомпилированных ядер: геометрия входит в исходник константами.
private struct DequantKey: Hashable {
    let inFeatures: Int
    let outFeatures: Int
    let xbits: Int
    let superBlock: Int
}

private var dequantKernelCache: [DequantKey: MLXFastKernel] = [:]

private func dequantKernel(inFeatures IN: Int, outFeatures OUT: Int,
                           xbits: Int, superBlock gwSb: Int) -> MLXFastKernel {
    let key = DequantKey(inFeatures: IN, outFeatures: OUT,
                         xbits: xbits, superBlock: gwSb)
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

        device float* orow = out + row * IN_C + blk * 32;
        for (uint c = 0; c < 32; c++) {
            orow[c] = float(nib[c]) * sf + mf;
        }
    """

    let kern = MLXFast.metalKernel(
        name: "rwkvq_dequant\(4 + xbits)_\(IN)_\(OUT)",
        inputNames: ["qblk", "qsqm", "ddm"],
        outputNames: ["out"],
        source: source,
        header: header
    )
    dequantKernelCache[key] = kern
    return kern
}

/// packed sb6 → dense [OUT, IN] float32 за один launch.
public func rwkvqDequantDense(qblk: MLXArray, qsqm: MLXArray, ddm: MLXArray,
                              outFeatures OUT: Int, inFeatures IN: Int,
                              superBlock gwSb: Int, xbits: Int) -> MLXArray {
    let NB = IN / 32
    let kern = dequantKernel(inFeatures: IN, outFeatures: OUT,
                             xbits: xbits, superBlock: gwSb)
    let threadgroup = 256
    let total = OUT * NB
    let groups = (total + threadgroup - 1) / threadgroup
    return kern([qblk, qsqm, ddm],
                grid: (groups * threadgroup, 1, 1),
                threadGroup: (threadgroup, 1, 1),
                outputShapes: [[OUT, IN]],
                outputDTypes: [.float32])[0]
}
