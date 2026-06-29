import Foundation
import MLX
import RWKVKernel

// Резидентное потребление памяти процессом, МБ.
func residentMemoryMB() -> Double {
    // phys_footprint — то, что показывает Xcode и по чему считает jetsam (вкл. wired/Metal).
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / (1024 * 1024) : 0
}

// ───────────────────────────────────────────────────────────────────────
//  Диск-кэш граничных состояний (mmap, один файл на split, перезаписывается).
//  На диск идёт ТОЛЬКО x [N,T,D] bf16 — плотно, по примерам.
//  НЕ храним: vFirst (пересчёт из id, слой 0) и xPrev (= x[:, -1]).
//  id храним в RAM (предтокенизированы, padded до T) — нужны для vFirst.
// ───────────────────────────────────────────────────────────────────────

struct BoundaryCache {
    let mapped: Data          // mmap файла: x [N,T,D] bf16
    let T: Int
    let D: Int
    var ids: [[Int32]]        // padded до T, для пересчёта vFirst
    var length: [Int]
    var label: [Int32]
    var count: Int { label.count }

    private var rowBytes: Int { T * D * DType.bfloat16.size }

    // x для набора индексов → [B,T,D] bf16 (читается из mmap)
    func readX(_ idx: [Int]) -> MLXArray {
        var arrs: [MLXArray] = []; arrs.reserveCapacity(idx.count)
        for i in idx {
            let off = i * rowBytes
            let chunk = mapped.subdata(in: off ..< off + rowBytes)
            arrs.append(MLXArray(chunk, [1, T, D], dtype: .bfloat16))
        }
        return concatenated(arrs, axis: 0)
    }

    // id батча → [B,T] int32
    func idsBatch(_ idx: [Int]) -> MLXArray {
        var flat = [Int32](); flat.reserveCapacity(idx.count * T)
        for i in idx { flat.append(contentsOf: ids[i]) }
        return MLXArray(flat, [idx.count, T])
    }
}

enum PartialFinetune {

    // Удаляет диск-кэши (после успеха/отмены/ошибки).
    static func clearCache() {
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        for n in ["train", "val"] {
            try? FileManager.default.removeItem(at: dir.appendingPathComponent("pf_\(n).bin"))
        }
    }

    static func buildBoundaryCache(
        backbone: RWKVBackbone,
        examples: [Example],
        freeze: Int,
        name: String,                  // имя файла кэша (pf_<name>.bin), перезапись
        ctxLen: Int = 128,
        batch: Int = 5,
        isCancelled: @escaping () -> Bool = { false },
        progress: @escaping (_ pct: Double, _ peakMB: Double) -> Void = { _, _ in }
    ) -> BoundaryCache {
        precondition(ctxLen % WKV7_CHUNK == 0,
                     "ctxLen (\(ctxLen)) должен делиться на CHUNK \(WKV7_CHUNK)")
        backbone.wOverride = nil
        backbone.trainLayers = []

        let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pf_\(name).bin")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try! FileHandle(forWritingTo: url)

        var ids: [[Int32]] = []; ids.reserveCapacity(examples.count)
        var length: [Int] = []
        var label: [Int32] = []
        var D = -1

        var start = 0
        while start < examples.count {
            if isCancelled() { break }
            let end = min(start + batch, examples.count)
            let bs = Array(examples[start ..< end]); let B = bs.count

            var idsFlat = [Int32](repeating: 0, count: B * ctxLen)   // 0 = pad справа
            for (bi, ex) in bs.enumerated() {
                let L = min(ex.ids.count, ctxLen)
                for t in 0 ..< L { idsFlat[bi * ctxLen + t] = Int32(ex.ids[t]) }
                length.append(L); label.append(Int32(ex.label))
                ids.append(Array(idsFlat[bi * ctxLen ..< bi * ctxLen + ctxLen]))
            }
            let idsArr = MLXArray(idsFlat, [B, ctxLen])
            let (x, _, _) = backbone.boundaryState(idsArr, upTo: freeze)   // нужен только x
            let xb = x.asType(.bfloat16)
            eval(xb)
            if D < 0 { D = xb.shape[2] }
            try! fh.write(contentsOf: xb.asData(access: .copy).data)       // [B,T,D] bf16 → диск

            progress(Double(end) / Double(examples.count), residentMemoryMB())
            start = end
        }
        try! fh.close()
        let mapped = try! Data(contentsOf: url, options: .alwaysMapped)     // mmap
        return BoundaryCache(mapped: mapped, T: ctxLen, D: max(D, 1),
                             ids: ids, length: length, label: label)
    }
}
