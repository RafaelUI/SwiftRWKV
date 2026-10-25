import Foundation

/// Токенизатор модели: то, что нужно генерации, эмбеддеру, реранкеру и
/// претрейну. Реализации — `WorldTokenizer` (словарь RWKV World, жадный
/// самый длинный префикс) и `BPETokenizer` (HuggingFace `tokenizer.json`,
/// байтовый уровень GPT-2: ru60m и другие модели со своим словарём).
public protocol RWKVTokenizer: AnyObject {
    /// Размер словаря: id лежат в 0 ..< count. Путь, где токенизатор и
    /// модель приходят из разных мест, обязан сверить его с размером
    /// эмбеддинга (выход за таблицу в MLX не падает, а возвращает мусор).
    var count: Int { get }
    func encode(_ text: String) -> [Int]
    func decode(_ tokens: [Int]) -> String
    /// Сырые байты токена — для потокового вывода без разрыва UTF-8.
    func rawBytes(_ id: Int) -> [UInt8]?
}

extension WorldTokenizer: RWKVTokenizer {}

/// Загрузить токенизатор по файлу: `tokenizer.json` (или любой .json) —
/// `BPETokenizer`, иначе — словарь World (`rwkv_vocab_*.txt`).
public func loadTokenizer(_ url: URL) throws -> any RWKVTokenizer {
    if url.pathExtension.lowercased() == "json" { return try BPETokenizer(tokenizerURL: url) }
    guard let w = WorldTokenizer(vocabURL: url) else {
        throw BPETokenizerError.unreadable("\(url.path): не словарь World")
    }
    return w
}
