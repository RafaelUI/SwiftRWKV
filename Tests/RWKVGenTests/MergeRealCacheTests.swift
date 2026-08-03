//
//  MergeRealCacheTests.swift
//  Слияние кэшей на НАСТОЯЩЕМ кэше от 0.1B, а не на синтетике.
//
//  Синтетика проверяет укладку байтов; здесь проверяется, что механизм
//  работает на кэше, который реально собран кодированием: другой тип (fp16),
//  другие формы, контракт от настоящего прогона. Пропускается, если кэша нет.
//
import XCTest
import MLX
@testable import RWKVGen
@testable import RWKVRerank

final class MergeRealCacheTests: XCTestCase {

    /// Разрезать реальный двухслойный кэш на два односло́йных и склеить
    /// обратно — результат обязан совпасть с исходным ПОБИТОВО.
    ///
    /// Разрез делается тем же `gather(slots:)`, которым пользуется обучение,
    /// поэтому тест заодно смыкает две вещи: чтение по срезу и слияние.
    func testSplitAndMergeRealCacheRoundTrips() throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let base = home.appendingPathComponent(
            "Develop/SwiftRWKV/runs/sweep_l5_l11/cache_eval")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: base.appendingPathExtension("idx.json").path),
            "нет реального кэша — тест пропущен")

        let full = try StateCache.load(base)
        try XCTSkipUnless(full.sources == [5, 11],
                          "кэш не тех слоёв: \(full.sources ?? [])")

        func single(_ layer: Int) throws -> StateCache {
            let slot = full.sources!.firstIndex(of: layer)!
            let w = try StateCacheWriter(
                shape: [full.nPairs, 1] + Array(full.shape.dropFirst(2)),
                dtype: full.dtype)
            var start = 0
            while start < full.nPairs {
                let end = Swift.min(start + 128, full.nPairs)
                let rows = Array(start ..< end)
                try w.write(rows: rows, full.gather(rows, slots: [slot]))
                start = end
            }
            return try w.finish(pairIndex: full.pairIndex, labels: full.labels,
                                hardNegs: full.hardNegs,
                                contract: full.contract, sources: [layer])
        }

        let merged = try single(5).merged(with: single(11))
        XCTAssertEqual(merged.sources, [5, 11])
        XCTAssertEqual(merged.nPairs, full.nPairs)
        XCTAssertEqual(merged.contract, full.contract)

        let rows = Array(stride(from: 0, to: full.nPairs, by: 37))
        let a = merged.gather(rows), b = full.gather(rows)
        eval(a, b)
        XCTAssertEqual(MLX.abs(a - b).max().item(Float.self), 0,
                       "слияние реального кэша разошлось с исходным")
    }
}
