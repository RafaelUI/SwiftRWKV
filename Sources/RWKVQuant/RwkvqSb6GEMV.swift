import Foundation
import MLX
import MLXFast

// ───────────────────────────────────────────────────────────────────────
//  GEMV для раскладки sb6: y = x · Wᵀ БЕЗ плотной матрицы. Путь декода
//  пресета COMPRESSION.
//
//  Порт rwkv_quant/backends/metal/quant_linear_gw.py::_get_kernel_k3 и
//  GwQuantLinear._gemv1 (умолчания бэкенда: K3_HALF и K3_XSUM выключены).
//  Исходник ядра перенесён ДОСЛОВНО, включая порядок суммирования
//  (32 дорожки на подгруппу, simd_sum в конце): выход обязан быть БИТ-В-БИТ
//  равен ядру rwkv-quant, и это проверяется эталоном, а не допуском.
//
//  Ядро читает тот же K3-интерлив, что и ядро декванта (RwkvqDequant.swift):
//    qblk[row, blk] = 16Б нибблов [+4Б qh] [+4Б qh2]
//    qsqm[row, blk] = (qs, qm) двумя байтами, qm — int8 без сдвига
//    ddm[row, sblk] = (d, dm) двумя fp16
//  Вес блока — q · s + m, где s = half(qs · d), m = half(qm · dm), поэтому
//    Σ (q·s + m)·x = s · Σ q·x + m · Σ x,
//  и вторая сумма (xbsum, по блокам 32) считается один раз на вход.
//
//  Зачем, если у sb6 есть родной контейнер MLX (`quantizedMM`). Замерено на
//  декоде 1.5B: quantizedMM — 18.5 мс/ток при 13.2 у Metal-бэкенда
//  rwkv-quant с этим ядром. Родной контейнер и K3 — одни и те же байты в
//  двух раскладках; держать обе значило бы удвоить память модели, поэтому
//  база собирается под что-то одно (см. `RwkvqAttachOptions.useNativeKernel`).
//
//  ЧИСЛА. Вес здесь — точное q·s + m при s и m, округлённых через half, как
//  в файле; сумма копится в fp32. Вырожденный блок (qs·d = 0 в half) даёт
//  s = 0, как и ядро декванта: зажим scale снизу 1e-8, который держит
//  codec, сюда не переносится (так же в rwkv-quant; на вес это не больше
//  63·1e-8).
// ───────────────────────────────────────────────────────────────────────

/// (NSG, RS) для одной строки входа — таблица `_k3_cfg` из rwkv-quant
/// (свип bench_kernel3_protoD.py): подгруппа из 32 потоков ведёт RS строк,
/// в threadgroup NSG подгрупп.
func rwkvqSb6GEMVConfig(outFeatures OUT: Int, inFeatures IN: Int, xbits: Int) -> (nsg: Int, rs: Int) {
    if OUT >= 32768 { return (4, 4) }                       // head
    if OUT >= 8192 { return xbits == 0 ? (2, 2) : (4, 4) }  // cmix key
    return (4, 4)                                           // tmix / cmix value
}

private struct Sb6GemvKey: Hashable {
    let inFeatures: Int
    let outFeatures: Int
    let xbits: Int
    let nsg: Int
    let rs: Int
}

private var sb6GemvKernelCache: [Sb6GemvKey: MLXFastKernel] = [:]

/// Битплоскость одним умножением: ниббл → 4 байта. Порядок регистров
/// ПРЯМОЙ (l0…l3, h0…h3) — split нибблов у sb6 локален блоку 32, в отличие
/// от sym, где блок 16 и порядок парный.
private func sb6Plane(_ src: String, shift: Int) -> String {
    var out = ""
    for (i, reg) in ["l0", "l1", "l2", "l3", "h0", "h1", "h2", "h3"].enumerated() {
        let sh = i * 4
        let nib = sh == 0 ? "(\(src) & 0xFu)"
            : sh == 28 ? "(\(src) >> 28)" : "((\(src) >> \(sh)) & 0xFu)"
        out += "            \(reg) |= as_type<uchar4>((\(nib) * 0x00204081u & 0x01010101u) << \(shift));\n"
    }
    return out
}

private func sb6GemvKernel(inFeatures IN: Int, outFeatures OUT: Int, xbits: Int,
                           nsg NSG: Int, rs RS: Int) -> MLXFastKernel {
    let key = Sb6GemvKey(inFeatures: IN, outFeatures: OUT, xbits: xbits, nsg: NSG, rs: RS)
    if let k = sb6GemvKernelCache[key] { return k }
    precondition((0 ... 2).contains(xbits), "sb6: битплоскостей \(xbits) вне 0...2")
    precondition(IN % 256 == 0 && OUT % (NSG * RS) == 0,
                 "sb6-GEMV: форма \(OUT)×\(IN) не берётся ядром при (\(NSG), \(RS))")

    let header = """
    constant uint IN_C  = \(IN);
    constant uint OUT_C = \(OUT);
    constant uint NB    = \(IN / 32);
    constant uint NSB   = \(IN / 256);
    constant uint NSG   = \(NSG);
    constant uint RS    = \(RS);
    constant uint SU    = \(4 + xbits);
    constant uint OUT_PER = \(OUT);
    """

    var dec = """
                uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
                uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
                uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
                uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
                uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
                uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
                uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
                uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
                uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);

    """
    if xbits >= 1 { dec += "            uint hb = qb[4];\n" + sb6Plane("hb", shift: 4) }
    if xbits >= 2 { dec += "            uint hb2 = qb[5];\n" + sb6Plane("hb2", shift: 5) }

    let source = """
        uint tgid = threadgroup_position_in_grid.x;
        uint tix  = thread_position_in_threadgroup.x;
        uint sg   = tix / 32;
        uint lane = tix % 32;
        uint row0 = tgid * (NSG * RS) + sg * RS;
        uint xi   = row0 / OUT_PER;

        device const float4* x4  = (device const float4*)(x + xi*IN_C);
        device const uint*   qu  = (device const uint*)qblk;
        device const uchar2* sm2 = (device const uchar2*)qsqm;
        device const half2*  dd2 = (device const half2*)ddm;
        float acc[RS];
        for (uint j = 0; j < RS; j++) acc[j] = 0.0f;

        for (uint p = lane; p < NB; p += 32) {
            float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
            float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
            float xbs = xbsum[xi*NB + p];
            for (uint j = 0; j < RS; j++) {
                device const uint* qb = qu + ((row0+j)*NB + p) * SU;
    \(dec)
                float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                         + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                         + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                         + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                         + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                         + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                         + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                         + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w));
                uchar2 sm = sm2[(row0+j)*NB + p];
                half2  dd = dd2[(row0+j)*NSB + p/8];
                half  s  = (half)((float)sm.x * (float)dd.x);
                half  mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);
                acc[j] += (float)s * dv + (float)mn * xbs;
            }
        }
        for (uint j = 0; j < RS; j++) {
            float a = simd_sum(acc[j]);
            if (lane == 0)
                out[row0 + j] = a;
        }
    """

    let kern = MLXFast.metalKernel(
        name: "rwkvq_sb6_gemv\(4 + xbits)_s\(NSG)r\(RS)_\(IN)_\(OUT)",
        inputNames: ["x", "qblk", "qsqm", "ddm", "xbsum"],
        outputNames: ["out"],
        source: source,
        header: header
    )
    sb6GemvKernelCache[key] = kern
    return kern
}

/// Берёт ли ядро такую форму (одна строка входа; суперблок 8).
public func rwkvqSb6GEMVApplies(outFeatures OUT: Int, inFeatures IN: Int, xbits: Int,
                                superBlock: Int, rows: Int) -> Bool {
    let (nsg, rs) = rwkvqSb6GEMVConfig(outFeatures: OUT, inFeatures: IN, xbits: xbits)
    return (0 ... 2).contains(xbits) && superBlock == 8 && rows == 1
        && IN % 256 == 0 && OUT % (nsg * rs) == 0
}

/// y = x · Wᵀ по K3-интерливу sb6. x — [1, IN]; выход [1, OUT], fp32.
///
/// `config` — только для тестов (другая раскладка потоков обязана давать
/// тот же ответ с точностью до порядка суммирования).
public func rwkvqSb6GEMV(_ x: MLXArray, qblk: MLXArray, qsqm: MLXArray, ddm: MLXArray,
                         outFeatures OUT: Int, inFeatures IN: Int, xbits: Int,
                         config: (nsg: Int, rs: Int)? = nil) -> MLXArray {
    precondition(x.ndim == 2 && x.shape == [1, IN], "sb6-GEMV: вход \(x.shape), ожидался [1, \(IN)]")
    // Типы входов ядра фиксированы НАМЕРЕННО: MLX строит исходник ядра из
    // типов входов, а кеширует по имени, и вызов того же ядра с другими
    // типами освобождает конвейер из-под ещё не исполненных запусков
    // (см. WKV7Train.swift). Чужой тип — ошибка вызывающего, и громкая.
    precondition(qblk.dtype == .uint8 && qsqm.dtype == .uint8 && ddm.dtype == .float16,
                 "sb6-GEMV: типы буферов \(qblk.dtype)/\(qsqm.dtype)/\(ddm.dtype), нужны uint8/uint8/float16")
    let (NSG, RS) = config ?? rwkvqSb6GEMVConfig(outFeatures: OUT, inFeatures: IN, xbits: xbits)
    let x32 = x.asType(.float32)
    let xbsum = x32.reshaped([1, IN / 32, 32]).sum(axis: 2)
    let kern = sb6GemvKernel(inFeatures: IN, outFeatures: OUT, xbits: xbits, nsg: NSG, rs: RS)
    let nTG = OUT / (NSG * RS)
    return kern([x32, qblk, qsqm, ddm, xbsum],
                grid: (nTG * NSG * 32, 1, 1),
                threadGroup: (NSG * 32, 1, 1),
                outputShapes: [[1, OUT]],
                outputDTypes: [.float32])[0]
}

extension RwkvqSidecar {

    /// Проекция по sb6-тензору ядром GEMV; nil — если тензор не sb6 либо
    /// ядро не берёт его форму или число строк входа (тогда вызывающий идёт
    /// путём декванта). x — [1, IN].
    public func sb6GEMV(_ key: String, _ x: MLXArray) -> MLXArray? {
        guard let info = tensors[key], info.kind == .sb6, x.ndim == 2,
              x.shape[1] == info.inFeatures,
              rwkvqSb6GEMVApplies(outFeatures: info.outFeatures, inFeatures: info.inFeatures,
                                  xbits: info.xbits, superBlock: info.superBlock,
                                  rows: x.shape[0]),
              let qblk = arrays["\(key)::qblk"], let qsqm = arrays["\(key)::qsqm"],
              let ddm = arrays["\(key)::ddm"] else { return nil }
        return rwkvqSb6GEMV(x, qblk: qblk, qsqm: qsqm, ddm: ddm,
                            outFeatures: info.outFeatures, inFeatures: info.inFeatures,
                            xbits: info.xbits)
    }
}
