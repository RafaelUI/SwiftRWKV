import Foundation
import MLX
import RWKVGen

/// Токенизатор-заглушка: байтов нет, считаются только токены.
final class NullTokenizer: RWKVTokenizer {
    var count: Int { 65536 }
    func encode(_ text: String) -> [Int] { [] }
    func decode(_ tokens: [Int]) -> String { "" }
    func rawBytes(_ id: Int) -> [UInt8]? { [] }
}

/// Батчевый декод против поочерёдного (25.10):
/// `decode-bench --batch-bench <модель> --batch B --arm batch|single [--steps N]`.
/// Одна рука — один процесс; чередование — снаружи. Жадно, без остановок,
/// промпт 32 токена (у каждой строки свой). Печатает время, токены в секунду
/// на ВСЕ строки и пик физического следа процесса.
func batchBench(_ path: String, batch B: Int, arm: String, steps: Int) throws {
    let bb: X070Backbone
    if path.hasSuffix(".rwkvq") {
        bb = try X070Backbone.load(fromRwkvq: expand(path))
    } else {
        let w = try loadArrays(url: URL(fileURLWithPath: expand(path)))
        let nL = w.keys.compactMap { k -> Int? in k.hasPrefix("blocks.") ? Int(k.split(separator: ".")[1]) : nil }.max()! + 1
        let cfg = X070Config(nLayer: nL, nEmbd: w["ln_out.weight"]!.shape[0],
                             headSize: w["blocks.0.tmix.k_k"]!.shape[1], vocab: w["head.weight"]!.shape[0])
        bb = X070Backbone(weights: w, cfg: cfg)
    }
    var rng = SplitMix64(seed: 7)
    let prompts = (0 ..< B).map { _ in (0 ..< 32).map { _ in Int(rng.next() % 60000) + 1 } }
    let tok = NullTokenizer()
    let cfg = GenerationConfig(maxTokens: steps)
    let warm = GenerationConfig(maxTokens: 4)
    // прогрев тем же путём
    if arm == "batch" { _ = bb.generateBatch(prompts: prompts, config: warm, tokenizer: tok) }
    else { var st = RWKVState(cfg: bb.cfg); _ = bb.generate(prompt: prompts[0], config: warm, tokenizer: tok, state: &st) }
    let t0 = Date()
    var produced = 0
    if arm == "batch" {
        produced = bb.generateBatch(prompts: prompts, config: cfg, tokenizer: tok).reduce(0) { $0 + $1.tokens.count }
    } else {
        for p in prompts {
            var st = RWKVState(cfg: bb.cfg)
            produced += bb.generate(prompt: p, config: cfg, tokenizer: tok, state: &st).tokens.count
        }
    }
    let dt = Date().timeIntervalSince(t0)
    print(String(format: "{\"arm\": \"%@\", \"batch\": %d, \"steps\": %d, \"tokens\": %d, \"seconds\": %.3f, \"tok_per_s\": %.1f, \"peak_footprint_gb\": %.2f}",
                 arm, B, steps, produced, dt, Double(produced) / dt, processPeakFootprintGB()))
}

/// Пик физического следа процесса, ГБ.
func processPeakFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let ok = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return ok == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1e9 : 0
}
