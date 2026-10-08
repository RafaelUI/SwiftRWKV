import Foundation
import MLX
import MLXFast

// ───────────────────────────────────────────────────────────────────────
//  GEMV для раскладки sym: y = x · Wᵀ БЕЗ плотной матрицы. Путь декода.
//
//  Порт rwkv_quant/backends/metal/quant_linear_sym.py::_get_kernel_sym8 /
//  _get_kernel_sym6 и SymQuantLinear._gemv. Исходник ядер перенесён
//  ДОСЛОВНО, включая порядок суммирования (32 дорожки на подгруппу,
//  simd_sum в конце): выход обязан быть БИТ-В-БИТ равен ядру rwkv-quant,
//  и это проверяется эталоном, а не допуском.
//
//  Зачем. Путь «деквант + матмул» на каждую проекцию читает сжатое, ПИШЕТ
//  плотный транзиент и читает его обратно: на декоде 1.5B REDUCTION это
//  84.8 мс/ток. Здесь ядро декодирует коды прямо в скалярном произведении;
//  плотная матрица не материализуется вовсе.
//
//  ЧИСЛА ДРУГИЕ, ЧЕМ У ПУТИ «ДЕКВАНТ + МАТМУЛ», и это свойство формата, а
//  не дефект порта. Здесь вес — точное произведение q · half(qs · d),
//  сумма копится в fp32; там вес округлён до fp16 и матмул идёт в fp16.
//  Так же расходятся декод и GEMM-префилл в самом rwkv-quant (там это
//  «пол нарезки», KL порядка 1e-6). Декод точнее.
//
//  Отличия от sb6-ядра (у которого здесь родной контейнер MLX):
//    • min нет — нет второго слагаемого и второй пары скаляров;
//    • масштаб блока — целый байт int8, распаковка не нужна;
//    • при bits=6 сдвиг +32 снимается в ядре:
//        Σ (q − 32)·x = Σ q·x − 32·Σ x,
//      и вторая сумма (xbsum, по блокам 16) считается один раз на вход.
//
//  Конфигурация (NSG, RS) = (4, 4) — та, что стоит в rwkv-quant для всех
//  форм: подгруппа из 32 потоков ведёт RS строк, в threadgroup NSG подгрупп.
//  Отсюда требование OUT % 16 == 0; вместе с IN % 256 == 0 оно выполнено на
//  всех sym-тензорах выпущенных моделей. Прочие формы ядро не берёт —
//  вызывающий идёт путём декванта.
// ───────────────────────────────────────────────────────────────────────

/// Сколько строк входа ядро берёт за один запуск (NB_MAX в rwkv-quant).
public let rwkvqSymGEMVMaxRows = 4

private let symGemvNSG = 4
private let symGemvRS = 4

private struct SymGemvKey: Hashable {
    let inFeatures: Int
    let outFeatures: Int
    let bits: Int
    let rows: Int
}

private var symGemvKernelCache: [SymGemvKey: MLXFastKernel] = [:]

private func symGemvHeader(IN: Int, OUT: Int, NN: Int) -> String {
    """
    constant uint IN_C   = \(IN);
    constant uint OUT_C  = \(OUT);
    constant uint OUT_PER= \(OUT);
    constant uint NK     = 1;
    constant uint NB    = \(IN / 16);
    constant uint NSB   = \(IN / 256);
    constant uint NPAIR = \(IN / 32);
    constant uint NSG   = \(symGemvNSG);
    constant uint RS    = \(symGemvRS);
    constant uint NN    = \(NN);
    """
}

/// Порядок регистров → порядок колонок внутри пары блоков по 16 (тот же,
/// что в ядре декванта: split нибблов блок-локальный, блок 16).
private let gemvPairRegs = ["l0", "l1", "h0", "h1", "l2", "l3", "h2", "h3"]

private func gemvPlanePair(_ src: String, shift: Int) -> String {
    var out = ""
    for (i, reg) in gemvPairRegs.enumerated() {
        let sh = i * 4
        let nib = sh == 0 ? "(\(src) & 0xFu)"
            : sh == 28 ? "(\(src) >> 28)" : "((\(src) >> \(sh)) & 0xFu)"
        out += "            \(reg) |= as_type<uchar4>((\(nib) * 0x00204081u & 0x01010101u) << \(shift));\n"
    }
    return out
}

private func symGemvKernel(inFeatures IN: Int, outFeatures OUT: Int,
                           bits: Int, rows NN: Int) -> MLXFastKernel {
    let key = SymGemvKey(inFeatures: IN, outFeatures: OUT, bits: bits, rows: NN)
    if let k = symGemvKernelCache[key] { return k }
    precondition(bits == 6 || bits == 8, "sym: битность \(bits) вне {6, 8}")
    precondition(IN % 256 == 0 && OUT % (symGemvNSG * symGemvRS) == 0,
                 "sym-GEMV: форма \(OUT)×\(IN) не берётся ядром")

    let head = """
        uint tgid = threadgroup_position_in_grid.x;
        uint tix  = thread_position_in_threadgroup.x;
        uint sg   = tix / 32;
        uint lane = tix % 32;
        uint row0 = tgid * (NSG * RS) + sg * RS;
        uint xi   = row0 / OUT_PER;          // при OUT_PER=OUT_C всегда 0

        device const uint* qu = (device const uint*)qblk;
        device const char* sc = (device const char*)qs;
        float acc[RS * NN];
        for (uint j = 0; j < RS * NN; j++) acc[j] = 0.0f;

    """
    let tail = """

        for (uint n = 0; n < NN; n++)
            for (uint j = 0; j < RS; j++) {
                float a = simd_sum(acc[n*RS + j]);
                if (lane == 0)
                    out[n*OUT_C + row0 + j] = a;
            }
    """

    let body: String
    let names: [String]
    if bits == 8 {
        body = """
            for (uint p = lane; p < NB; p += 32) {          // p -- блок из 16 колонок
                for (uint j = 0; j < RS; j++) {
                    device const uint* qb = qu + ((row0+j)*NB + p) * 4;
                    char4 c0 = as_type<char4>(qb[0]);
                    char4 c1 = as_type<char4>(qb[1]);
                    char4 c2 = as_type<char4>(qb[2]);
                    char4 c3 = as_type<char4>(qb[3]);
                    float4 q0 = float4(c0.x, c0.y, c0.z, c0.w);
                    float4 q1 = float4(c1.x, c1.y, c1.z, c1.w);
                    float4 q2 = float4(c2.x, c2.y, c2.z, c2.w);
                    float4 q3 = float4(c3.x, c3.y, c3.z, c3.w);
                    half s = (half)((float)sc[(row0+j)*NB + p]
                                    * (float)d[(row0+j)*NSB + p/16]);
                    for (uint n = 0; n < NN; n++) {
                        device const float4* x4 =
                            (device const float4*)(x + (n*NK + xi)*IN_C);
                        float dv = dot(x4[p*4+0], q0) + dot(x4[p*4+1], q1)
                                 + dot(x4[p*4+2], q2) + dot(x4[p*4+3], q3);
                        acc[n*RS + j] += (float)s * dv;
                    }
                }
            }
        """
        names = ["x", "qblk", "qs", "d"]
    } else {
        body = """
            for (uint p = lane; p < NPAIR; p += 32) {   // p -- ПАРА блоков = 32 кол.
                for (uint j = 0; j < RS; j++) {
                    device const uint* qb = qu + ((row0+j)*NPAIR + p) * 6;
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
        \(gemvPlanePair("hb", shift: 4))            uint hb2 = qb[5];
        \(gemvPlanePair("hb2", shift: 5))
                    float4 f0 = float4(l0.x, l0.y, l0.z, l0.w);
                    float4 f1 = float4(l1.x, l1.y, l1.z, l1.w);
                    float4 f2 = float4(h0.x, h0.y, h0.z, h0.w);
                    float4 f3 = float4(h1.x, h1.y, h1.z, h1.w);
                    float4 f4 = float4(l2.x, l2.y, l2.z, l2.w);
                    float4 f5 = float4(l3.x, l3.y, l3.z, l3.w);
                    float4 f6 = float4(h2.x, h2.y, h2.z, h2.w);
                    float4 f7 = float4(h3.x, h3.y, h3.z, h3.w);
                    // суперблок из 16 блоков = 8 пар: оба масштаба пары
                    // всегда живут под ОДНОЙ d
                    float dsb = (float)d[(row0+j)*NSB + p/8];
                    half s0 = (half)((float)sc[(row0+j)*NB + 2*p]     * dsb);
                    half s1 = (half)((float)sc[(row0+j)*NB + 2*p + 1] * dsb);
                    for (uint n = 0; n < NN; n++) {
                        uint xn = n*NK + xi;
                        device const float4* x4 =
                            (device const float4*)(x + xn*IN_C);
                        float dv0 = dot(x4[p*8+0], f0) + dot(x4[p*8+1], f1)
                                  + dot(x4[p*8+2], f2) + dot(x4[p*8+3], f3);
                        float dv1 = dot(x4[p*8+4], f4) + dot(x4[p*8+5], f5)
                                  + dot(x4[p*8+6], f6) + dot(x4[p*8+7], f7);
                        // снятие сдвига +32: sum((q-32)*x) = sum(q*x) - 32*sum(x)
                        float b0 = xbsum[xn*NB + 2*p], b1 = xbsum[xn*NB + 2*p + 1];
                        acc[n*RS + j] += (float)s0 * (dv0 - 32.0f * b0)
                                       + (float)s1 * (dv1 - 32.0f * b1);
                    }
                }
            }
        """
        names = ["x", "qblk", "qs", "d", "xbsum"]
    }

    let kern = MLXFast.metalKernel(
        name: "rwkvq_sym\(bits)_gemv_n\(NN)_\(IN)_\(OUT)",
        inputNames: names,
        outputNames: ["out"],
        source: head + body + tail,
        header: symGemvHeader(IN: IN, OUT: OUT, NN: NN)
    )
    symGemvKernelCache[key] = kern
    return kern
}

/// Берёт ли ядро такую форму и столько строк входа.
public func rwkvqSymGEMVApplies(outFeatures OUT: Int, inFeatures IN: Int,
                                bits: Int, rows: Int) -> Bool {
    (bits == 6 || bits == 8) && IN % 256 == 0
        && OUT % (symGemvNSG * symGemvRS) == 0
        && rows >= 1 && rows <= rwkvqSymGEMVMaxRows
}

/// y = x · Wᵀ по sym-интерливу. x — [N, IN], N ≤ 4; выход [N, OUT], fp32.
///
/// Вход приводится к fp32: ядро читает его как float4.
public func rwkvqSymGEMV(_ x: MLXArray, qblk: MLXArray, qs: MLXArray, d: MLXArray,
                         outFeatures OUT: Int, inFeatures IN: Int, bits: Int) -> MLXArray {
    precondition(x.ndim == 2 && x.shape[1] == IN, "sym-GEMV: вход \(x.shape), ожидался [N, \(IN)]")
    let N = x.shape[0]
    precondition(rwkvqSymGEMVApplies(outFeatures: OUT, inFeatures: IN, bits: bits, rows: N),
                 "sym-GEMV: форма \(OUT)×\(IN), bits=\(bits), строк \(N) ядром не берётся")
    let x32 = x.asType(.float32)
    let kern = symGemvKernel(inFeatures: IN, outFeatures: OUT, bits: bits, rows: N)
    var inputs: [MLXArray] = [x32, qblk, qs, d]
    if bits == 6 {
        inputs.append(x32.reshaped([N, IN / 16, 16]).sum(axis: 2))
    }
    let nTG = OUT / (symGemvNSG * symGemvRS)
    return kern(inputs,
                grid: (nTG * symGemvNSG * 32, 1, 1),
                threadGroup: (symGemvNSG * 32, 1, 1),
                outputShapes: [[N, OUT]],
                outputDTypes: [.float32])[0]
}

extension RwkvqSidecar {

    /// Проекция по sym-тензору ядром GEMV; nil — если тензор не sym либо
    /// ядро не берёт его форму или число строк входа (тогда вызывающий
    /// идёт путём декванта). x — [N, IN].
    public func symGEMV(_ key: String, _ x: MLXArray) -> MLXArray? {
        guard let info = tensors[key], info.kind == .sym, x.ndim == 2,
              x.shape[1] == info.inFeatures,
              rwkvqSymGEMVApplies(outFeatures: info.outFeatures, inFeatures: info.inFeatures,
                                  bits: info.bits, rows: x.shape[0]),
              let qblk = arrays["\(key)::qblk"], let qs = arrays["\(key)::qs"],
              let d = arrays["\(key)::d"] else { return nil }
        return rwkvqSymGEMV(x, qblk: qblk, qs: qs, d: d, outFeatures: info.outFeatures,
                            inFeatures: info.inFeatures, bits: info.bits)
    }
}
