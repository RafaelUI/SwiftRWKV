import CryptoKit
import Foundation
import RWKVGen

/// Скорость токенизации (25.10): `decode-bench --tok-bench <corpus.json> --vocab <словарь>`.
/// Корпус — JSON-массив строк; печатает число токенов, отпечаток id (sha256 тех же
/// байтов, что bench_py.py: id через запятую, «;» после текста) и лучшее из трёх
/// время. Отпечаток сверяет реализации между собой: быстрее, но иначе — не быстрее.
func tokBench(_ corpusPath: String, _ vocabPath: String, rounds: Int = 3) throws {
    let data = try Data(contentsOf: URL(fileURLWithPath: (corpusPath as NSString).expandingTildeInPath))
    let bridged = try JSONSerialization.jsonObject(with: data) as! [String]
    // Строки из JSONSerialization — мосты к NSString (UTF-16): перевод в UTF-8
    // стоил бы на каждом вызове. Замер — на родных строках, перевод — отдельно.
    let texts = bridged.map { (t: String) -> String in var s = t; s.makeContiguousUTF8(); return s }
    var bridgeBest = Double.infinity
    for _ in 0 ..< rounds {
        let s = Date()
        var total = 0
        for t in bridged { var x = t; x.withUTF8 { total &+= $0.count } }
        bridgeBest = min(bridgeBest, Date().timeIntervalSince(s))
        if total < 0 { print(total) }
    }
    let t0 = Date()
    let tok = try loadTokenizer(URL(fileURLWithPath: (vocabPath as NSString).expandingTildeInPath))
    let load = Date().timeIntervalSince(t0)
    let nbytes = texts.reduce(0) { $0 + $1.utf8.count }
    for t in texts.prefix(50) { _ = tok.encode(t) }
    var best = Double.infinity
    var n = 0
    var sha = ""
    for _ in 0 ..< rounds {
        var all: [[Int]] = []
        all.reserveCapacity(texts.count)
        let s = Date()
        for t in texts { all.append(tok.encode(t)) }
        best = min(best, Date().timeIntervalSince(s))
        var h = SHA256()
        n = 0
        for ids in all {
            n += ids.count
            h.update(data: Data(ids.map(String.init).joined(separator: ",").utf8))
            h.update(data: Data(";".utf8))
        }
        sha = h.finalize().map { String(format: "%02x", $0) }.joined()
    }
    print(String(format: "{\"impl\": \"swift\", \"tokens\": %d, \"sha\": \"%@\", \"best_s\": %.3f, \"MB_per_s\": %.2f, \"tok_per_s\": %.0f, \"load_s\": %.2f, \"bridge_s\": %.3f}",
                 n, String(sha.prefix(16)), best, Double(nbytes) / 1e6 / best, Double(n) / best, load, bridgeBest))
}
