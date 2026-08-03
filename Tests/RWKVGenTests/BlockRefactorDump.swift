//
//  BlockRefactorDump.swift
//  Временный характеризационный дамп: снимок ВСЕХ путей бэкбона в файл.
//
//  Как пользоваться (см. NEXT_SESSION.md, раздел про выделение RWKVBlock):
//      RWKV_DUMP=/tmp/after.safetensors  swift test --filter BlockRefactorDump
//      git stash …                       # вернуться к дорефакторной версии
//      RWKV_DUMP=/tmp/before.safetensors swift test --filter BlockRefactorDump
//      # сравнить файлы: разница обязана быть РОВНО нулевой
//
//  Использует ТОЛЬКО публичный API, существовавший до рефакторинга, поэтому
//  компилируется по обе стороны правки. Без RWKV_DUMP тест ничего не делает —
//  в обычном прогоне он пропускается.
//
import XCTest
import MLX
import MLXRandom
@testable import RWKVGen

final class BlockRefactorDump: XCTestCase {

    func testDumpAllPaths() throws {
        guard let path = ProcessInfo.processInfo.environment["RWKV_DUMP"] else {
            throw XCTSkip("RWKV_DUMP не задан — дамп снимается вручную")
        }

        var out: [String: MLXArray] = [:]

        // ── 1. Чистый forward: body и логиты, 3 слоя ──
        let (bb, cfg) = TinyBackbone.make(nLayer: 3)
        let ids = TinyBackbone.ids(2, 24, vocab: cfg.vocab, seed: 5)
        out["body"] = bb.body(ids)
        out["logits"] = bb(ids)

        // ── 2. Путь с состоянием: маска, endIdx, продолжение ──
        let mask = buildMask(lengths: [24, 17], total: 24)
        let endIdx = lastRealIndex(lengths: [24, 17])
        let (hs, st) = bb.bodyWithState(ids, state: nil, mask: mask, endIdx: endIdx)
        out["masked_body"] = hs
        out["masked_wkv"] = st.wkv
        out["masked_tmix_shift"] = st.tmixShift
        out["masked_cmix_shift"] = st.cmixShift

        let ids2 = TinyBackbone.ids(2, 16, vocab: cfg.vocab, seed: 6)
        let (hs2, st2) = bb.bodyWithState(ids2, state: st)
        out["cont_body"] = hs2
        out["cont_wkv"] = st2.wkv

        // ── 3. Разрез сети (partial-finetune) ──
        let (bx, bv) = bb.boundaryState(ids, upTo: 1)
        out["boundary_x"] = bx
        out["boundary_v"] = bv!
        out["tail"] = bb.forwardFrom(bx, bv, from: 1)

        // ── 4. Обучаемое ядро на части слоёв + gradient checkpoint ──
        let (bt, cfgT) = TinyBackbone.make(nLayer: 3)
        bt.trainLayers = [1, 2]
        let idsT = TinyBackbone.ids(2, 32, vocab: cfgT.vocab, seed: 7)
        out["train_kernel_body"] = bt.body(idsT)
        bt.useBlockCheckpoint = true
        out["train_kernel_ckpt_body"] = bt.body(idsT)

        // ── 5. LoRA: forward и ГРАДИЕНТ по адаптерам ──
        //
        // Градиент здесь важнее выхода: рефакторинг переставляет вызовы, и
        // ошибка в порядке аргументов вполне может сохранить forward и
        // сломать backward (или наоборот).
        let (bl, cfgL) = TinyBackbone.make(nLayer: 2)
        _ = LoRA.add(to: bl, spec: LoRASpec(rank: 8, alpha: 16))
        bl.trainLayers = [0, 1]
        let idsL = TinyBackbone.ids(2, 32, vocab: cfgL.vocab, seed: 8)
        out["lora_body"] = bl.body(idsL)

        let keys = LoRA.adapterState(bl).keys.sorted()
        let params = keys.map { LoRA.adapterState(bl)[$0]! }
        func loss(_ ps: [MLXArray]) -> [MLXArray] {
            for (i, k) in keys.enumerated() {
                if k.hasSuffix(".A") { bl.loraA[String(k.dropLast(2))] = ps[i] }
                else { bl.loraB[String(k.dropLast(2))] = ps[i] }
            }
            return [bl.body(idsL).square().mean()]
        }
        let grads = grad(loss, argumentNumbers: Array(0 ..< params.count))(params)
        for (i, k) in keys.enumerated() { out["lora_grad_\(k)"] = grads[i] }

        eval(Array(out.values))
        try save(arrays: out.mapValues { $0.asType(.float32) },
                 url: URL(fileURLWithPath: path))
        print("дамп записан: \(path), тензоров \(out.count)")
    }
}
