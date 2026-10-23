//
//  PEFTStateFile.swift
//  Обученное состояние в формате RWKV-PEFT (`.pth`): запись и чтение без torch.
//
import Foundation
import MLX

// ───────────────────────────────────────────────────────────────────────
//  Состояние RWKV-PEFT
// ───────────────────────────────────────────────────────────────────────
//
// RWKV-PEFT (state tuning) хранит обученное состояние файлом torch `.pth`:
// словарь `blocks.{i}.att.time_state` → тензор (голов, S, S). Его подключают
// RWKV Runner, Ai00 и другие раннеры. Для x070 тензор лежит в той же
// раскладке, что матрица WKV здесь, и раннеры x070 берут его без
// транспонирования. Сверено свёрткой префикса: здесь против mlx-lm (через неё
// Preen экспортирует .pth, проверенные в RWKV Runner) — `PEFTStateCrossCheck`.
//
// Чего в формате нет. Сдвигов токена (tmix/cmix shift): RWKV-PEFT их не
// обучает и не хранит. Отпечатка базы и контракта: файл не знает, на какой
// модели получен. Поэтому запись отказывает при ненулевых сдвигах (с
// `dropShifts: true` они теряются осознанно), а чтение проверяет только
// архитектуру — слои, головы, размер головы, но не базу.
//
// Контейнер — то, что пишет `torch.save`: zip без сжатия, внутри
// `<архив>/data.pkl` (pickle со ссылками на хранилища), `<архив>/data/<ключ>`
// (сырые байты хранилищ), `byteorder`, `version`. Читатель понимает только
// подмножество pickle, которое даёт `torch.save` словаря тензоров, и ничего
// не исполняет: незнакомый глобал или опкод — отказ, как у
// `torch.load(weights_only: True)`.

extension StateFile {

    public enum PEFTFailure: Error, CustomStringConvertible, Equatable {
        /// Не zip, не pickle torch, битая запись или контрольная сумма.
        case notATorchFile(String)
        /// Файл torch, но с тем, чего читатель не знает (сжатие, zip64,
        /// чужой глобал, порядок байт, тип хранилища).
        case unsupported(String)
        /// В файле нет ни одного `blocks.{i}.att.time_state`.
        case noTimeState
        /// Запись состояния с ненулевыми сдвигами без `dropShifts`.
        case nonZeroShifts
        /// Форма состояния или набор слоёв не те.
        case shape(String)

        public var description: String {
            switch self {
            case .notATorchFile(let why): return "не файл torch (.pth): \(why)"
            case .unsupported(let why): return "файл torch не поддержан: \(why)"
            case .noTimeState: return "в файле нет ни одного blocks.{i}.att.time_state — не состояние RWKV-PEFT"
            case .nonZeroShifts:
                return "у состояния ненулевые сдвиги токена, а RWKV-PEFT хранит только матрицу WKV; "
                    + "dropShifts: true — записать без них"
            case .shape(let why): return "форма состояния: \(why)"
            }
        }
    }

    static func peftKey(_ layer: Int) -> String { "blocks.\(layer).att.time_state" }

    /// Записать состояние в формате RWKV-PEFT: `blocks.{i}.att.time_state`,
    /// (голов, S, S), fp32. Сдвиги токена формат не хранит: ненулевые —
    /// отказ, пока не `dropShifts`. Обученное по умолчанию (`train: [.wkv]`)
    /// состояние сдвигов не имеет; свёрнутый промпт — имеет.
    public static func exportPEFT(_ state: RWKVBatchState, to url: URL, dropShifts: Bool = false) throws {
        let w = state.wkv
        guard w.ndim == 5, w.dim(1) == 1, w.dim(3) == w.dim(4) else {
            throw PEFTFailure.shape("wkv \(w.shape), нужно [слоёв, 1, голов, S, S]")
        }
        let L = w.dim(0), H = w.dim(2), S = w.dim(3)
        if !dropShifts {
            for s in [state.tmixShift, state.cmixShift] {
                let m = abs(s.asType(.float32)).max()
                eval(m)
                if !(m.item(Float.self) == 0) { throw PEFTFailure.nonZeroShifts }
            }
        }
        let wf = w.asType(.float32)
        eval(wf)
        let all = wf.asArray(Float.self)
        let per = H * S * S
        let tensors: [(String, [Int], Data)] = (0 ..< L).map { l in
            (peftKey(l), [H, S, S], all[l * per ..< (l + 1) * per].withUnsafeBytes { Data($0) })
        }
        try TorchPth.write(tensors, to: url)
    }

    /// Прочитать состояние RWKV-PEFT. Сдвиги — нули (формат их не хранит).
    /// С `base` проверяется архитектура (отказ — `Failure.otherArchitecture`);
    /// базу проверить нечем — отпечатка в файле нет. `ignored` — ключи
    /// файла, которые не `blocks.{i}.att.time_state` (например LoRA-веса из
    /// того же чекпойнта), по алфавиту.
    public static func importPEFT(_ url: URL, base bb: X070Backbone? = nil) throws
        -> (state: RWKVBatchState, ignored: [String])
    {
        let read = try TorchPth.read(url)
        var layers: [Int: MLXArray] = [:]
        var ignored = read.other
        for (name, a) in read.tensors {
            let p = name.split(separator: ".", omittingEmptySubsequences: false)
            if p.count == 4, p[0] == "blocks", let l = Int(p[1]), String(l) == p[1], l >= 0,
               p[2] == "att", p[3] == "time_state" {
                layers[l] = a
            } else {
                ignored.append(name)
            }
        }
        guard !layers.isEmpty else { throw PEFTFailure.noTimeState }
        let L = layers.count
        let have = layers.keys.sorted()
        guard have == Array(0 ..< L) else {
            throw PEFTFailure.shape("слои \(have) — нужны 0…\(L - 1) подряд")
        }
        let s0 = layers[0]!.shape
        guard s0.count == 3, s0[1] == s0[2] else {
            throw PEFTFailure.shape("time_state \(s0), нужно (голов, S, S)")
        }
        for l in 1 ..< max(L, 1) where layers[l]!.shape != s0 {
            throw PEFTFailure.shape("слой \(l): \(layers[l]!.shape), слой 0: \(s0)")
        }
        let H = s0[0], S = s0[1]
        if let bb {
            let c = bb.cfg
            if L != c.nLayer || H != c.nHead || S != c.headSize {
                throw Failure.otherArchitecture("в файле слоёв \(L), голов \(H), размер головы \(S); "
                    + "у модели \(c.nLayer), \(c.nHead), \(c.headSize)")
            }
        }
        let wkv = stacked((0 ..< L).map { layers[$0]!.asType(.float32) }).expandedDimensions(axis: 1)
        let zeros = MLXArray.zeros([L, 1, 1, H * S])
        let state = RWKVBatchState(wkv: wkv, tmixShift: zeros, cmixShift: zeros)
        eval(state.wkv, state.tmixShift, state.cmixShift)
        return (state, ignored.sorted())
    }
}

// ───────────────────────────────────────────────────────────────────────
//  Файл torch: zip без сжатия + pickle
// ───────────────────────────────────────────────────────────────────────

enum TorchPth {

    typealias Failure = StateFile.PEFTFailure

    // ── CRC-32 (zip) ────────────────────────────────────────────────────

    static let crcTable: [UInt32] = (0 ..< 256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0 ..< 8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func crc32<C: Collection>(_ bytes: C) -> UInt32 where C.Element == UInt8 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in bytes { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }

    static func cStride(_ size: [Int]) -> [Int] {
        var s = [Int](repeating: 1, count: size.count)
        if size.count >= 2 { for i in stride(from: size.count - 2, through: 0, by: -1) { s[i] = s[i + 1] * size[i + 1] } }
        return s
    }

    // ── Запись ──────────────────────────────────────────────────────────

    /// zip без сжатия. Данные записей выровнены на 64 байта (дополнением
    /// в extra-поле, как у torch), дата — 1980-01-01.
    struct StoredZip {
        var out = Data()
        var central = Data()
        var count = 0

        mutating func add(_ name: String, _ payload: Data, align: Int = 64) {
            let nameBytes = Data(name.utf8)
            let offset = out.count
            let bare = offset + 30 + nameBytes.count + 4
            let pad = (align - bare % align) % align
            let crc = TorchPth.crc32(payload)
            var h = Data()
            h.le32(0x0403_4B50); h.le16(20); h.le16(0); h.le16(0); h.le16(0); h.le16(0x21)
            h.le32(crc); h.le32(UInt32(payload.count)); h.le32(UInt32(payload.count))
            h.le16(UInt16(nameBytes.count)); h.le16(UInt16(4 + pad))
            h.append(nameBytes)
            h.le16(0x4246); h.le16(UInt16(pad)); h.append(Data(repeating: 0x5A, count: pad))
            out.append(h)
            out.append(payload)
            var c = Data()
            c.le32(0x0201_4B50); c.le16(20); c.le16(20); c.le16(0); c.le16(0); c.le16(0); c.le16(0x21)
            c.le32(crc); c.le32(UInt32(payload.count)); c.le32(UInt32(payload.count))
            c.le16(UInt16(nameBytes.count)); c.le16(0); c.le16(0); c.le16(0); c.le16(0); c.le32(0)
            c.le32(UInt32(offset))
            c.append(nameBytes)
            central.append(c)
            count += 1
        }

        mutating func finish() -> Data {
            var d = out
            let cdOffset = d.count
            d.append(central)
            d.le32(0x0605_4B50); d.le16(0); d.le16(0); d.le16(UInt16(count)); d.le16(UInt16(count))
            d.le32(UInt32(central.count)); d.le32(UInt32(cdOffset)); d.le16(0)
            return d
        }
    }

    /// Писатель pickle протокола 2 — ровно те опкоды, что нужны словарю
    /// тензоров (как у `torch.save`, без мемо).
    struct Pickle {
        var out = Data([0x80, 2])                       // PROTO 2
        mutating func op(_ b: UInt8) { out.append(b) }
        mutating func str(_ s: String) {
            let u = Data(s.utf8)
            out.append(0x58); out.le32(UInt32(u.count)); out.append(u)       // BINUNICODE
        }
        mutating func global(_ module: String, _ name: String) {
            out.append(0x63); out.append(Data("\(module)\n\(name)\n".utf8))    // GLOBAL
        }
        mutating func int(_ v: Int) {
            if (0 ... 0xFF).contains(v) { out.append(0x4B); out.append(UInt8(v)) }                   // BININT1
            else if (0 ... 0xFFFF).contains(v) { out.append(0x4D); out.le16(UInt16(v)) }             // BININT2
            else if (Int(Int32.min) ... Int(Int32.max)).contains(v) { out.append(0x4A); out.le32(UInt32(bitPattern: Int32(v))) }
            else { out.append(0x8A); out.append(8); withUnsafeBytes(of: Int64(v).littleEndian) { out.append(contentsOf: $0) } }
        }
        mutating func ints(_ v: [Int]) { op(0x28); v.forEach { int($0) }; op(0x74) }      // MARK … TUPLE
        mutating func orderedDict() { global("collections", "OrderedDict"); op(0x29); op(0x52) }
    }

    /// Словарь fp32-тензоров (C-раскладка) → `.pth`, который читает `torch.load`.
    static func write(_ tensors: [(name: String, shape: [Int], bytes: Data)], to url: URL) throws {
        var p = Pickle()
        p.orderedDict()
        p.op(0x28)                                                   // MARK
        for (i, t) in tensors.enumerated() {
            p.str(t.name)
            p.global("torch._utils", "_rebuild_tensor_v2")
            p.op(0x28)
            p.op(0x28); p.str("storage"); p.global("torch", "FloatStorage"); p.str(String(i)); p.str("cpu")
            p.int(t.bytes.count / 4); p.op(0x74); p.op(0x51)          // … TUPLE, BINPERSID
            p.int(0)
            p.ints(t.shape)
            p.ints(cStride(t.shape))
            p.op(0x89)                                               // NEWFALSE: requires_grad
            p.orderedDict()                                          // backward_hooks
            p.op(0x74); p.op(0x52)                                   // TUPLE, REDUCE
        }
        p.op(0x75); p.op(0x2E)                                       // SETITEMS, STOP
        var z = StoredZip()
        z.add("archive/data.pkl", p.out)
        z.add("archive/byteorder", Data("little".utf8))
        for (i, t) in tensors.enumerated() { z.add("archive/data/\(i)", t.bytes) }
        z.add("archive/version", Data("3\n".utf8))
        try z.finish().write(to: url, options: .atomic)
    }

    // ── Чтение ──────────────────────────────────────────────────────────

    struct StoredZipReader {
        struct Entry { let data: Int, size: Int, method: Int, crc: UInt32 }
        let b: [UInt8]
        var entries: [String: Entry] = [:]
        var names: [String] = []

        func u16(_ i: Int) -> Int { Int(b[i]) | Int(b[i + 1]) << 8 }
        func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }

        init(_ bytes: [UInt8]) throws {
            b = bytes
            guard b.count >= 22 else { throw Failure.notATorchFile("короче заголовка zip") }
            var eocd = -1
            var i = b.count - 22
            while i >= max(0, b.count - 22 - 0xFFFF) {
                if u32(i) == 0x0605_4B50 { eocd = i; break }
                i -= 1
            }
            guard eocd >= 0 else { throw Failure.notATorchFile("не zip (нет конца каталога)") }
            let n = u16(eocd + 10), cdOffset = u32(eocd + 16)
            if n == 0xFFFF || cdOffset == 0xFFFF_FFFF { throw Failure.unsupported("zip64") }
            var p = cdOffset
            for _ in 0 ..< n {
                guard p + 46 <= b.count, u32(p) == 0x0201_4B50 else {
                    throw Failure.notATorchFile("битый центральный каталог zip")
                }
                let method = u16(p + 10), crc = UInt32(u32(p + 16)), csize = u32(p + 20), usize = u32(p + 24)
                let nl = u16(p + 28), el = u16(p + 30), cl = u16(p + 32), local = u32(p + 42)
                guard p + 46 + nl <= b.count else { throw Failure.notATorchFile("битый центральный каталог zip") }
                let name = String(decoding: b[(p + 46) ..< (p + 46 + nl)], as: UTF8.self)
                p += 46 + nl + el + cl
                if csize == 0xFFFF_FFFF || usize == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
                    throw Failure.unsupported("zip64")
                }
                guard local + 30 <= b.count, u32(local) == 0x0403_4B50 else {
                    throw Failure.notATorchFile("битый заголовок записи \(name)")
                }
                let data = local + 30 + u16(local + 26) + u16(local + 28)
                guard data + csize <= b.count else { throw Failure.notATorchFile("запись \(name) за концом файла") }
                if method == 0 && csize != usize { throw Failure.notATorchFile("размеры записи \(name) не сходятся") }
                entries[name] = Entry(data: data, size: csize, method: method, crc: crc)
                names.append(name)
            }
        }

        func read(_ name: String) throws -> [UInt8] {
            guard let e = entries[name] else { throw Failure.notATorchFile("нет записи \(name)") }
            guard e.method == 0 else {
                throw Failure.unsupported("запись \(name) сжата (метод \(e.method)); torch.save пишет без сжатия")
            }
            let d = Array(b[e.data ..< (e.data + e.size)])
            guard TorchPth.crc32(d) == e.crc else { throw Failure.notATorchFile("контрольная сумма \(name) не сходится") }
            return d
        }
    }

    struct StorageRef: Equatable { let type: String, key: String, numel: Int }

    indirect enum Value {
        case none, bool(Bool), int(Int), float(Double), str(String)
        case tuple([Value]), list([Value]), dict([(Value, Value)])
        case global(String, String)
        case storage(StorageRef)
        case tensor(StorageRef, offset: Int, size: [Int], stride: [Int])
        case mark
    }

    /// Разбор pickle без исполнения: стековая машина на опкодах, которые
    /// даёт `torch.save` (протоколы 2–4), и трёх разрешённых конструкторах.
    struct Unpickler {
        let b: [UInt8]
        var p = 0
        var stack: [Value] = []
        var memo: [Int: Value] = [:]

        init(_ bytes: [UInt8]) { b = bytes }

        mutating func take(_ n: Int) throws -> ArraySlice<UInt8> {
            guard n >= 0, p + n <= b.count else { throw Failure.notATorchFile("pickle оборван") }
            defer { p += n }
            return b[p ..< (p + n)]
        }
        mutating func uint(_ n: Int) throws -> Int {
            try take(n).reversed().reduce(0) { $0 << 8 | Int($1) }
        }
        mutating func line() throws -> String {
            guard let e = b[p...].firstIndex(of: 0x0A) else { throw Failure.notATorchFile("pickle оборван") }
            defer { p = e + 1 }
            return String(decoding: b[p ..< e], as: UTF8.self)
        }
        mutating func pop() throws -> Value {
            guard let v = stack.popLast() else { throw Failure.notATorchFile("pickle: пустой стек") }
            return v
        }
        mutating func popMark() throws -> [Value] {
            guard let m = stack.lastIndex(where: { if case .mark = $0 { return true }; return false }) else {
                throw Failure.notATorchFile("pickle: нет метки")
            }
            let items = Array(stack[(m + 1)...])
            stack.removeSubrange(m...)
            return items
        }
        mutating func str(_ n: Int) throws -> Value { .str(String(decoding: try take(n), as: UTF8.self)) }

        mutating func load() throws -> Value {
            while p < b.count {
                let op = b[p]
                p += 1
                switch op {
                case 0x80: _ = try take(1)                                        // PROTO
                case 0x95: _ = try take(8)                                        // FRAME
                case 0x2E: return try pop()                                       // STOP
                case 0x28: stack.append(.mark)                                    // MARK
                case 0x29: stack.append(.tuple([]))                               // EMPTY_TUPLE
                case 0x74: stack.append(.tuple(try popMark()))                    // TUPLE
                case 0x85: let a = try pop(); stack.append(.tuple([a]))           // TUPLE1
                case 0x86: let y = try pop(), x = try pop(); stack.append(.tuple([x, y]))
                case 0x87: let z = try pop(), y = try pop(), x = try pop(); stack.append(.tuple([x, y, z]))
                case 0x5D: stack.append(.list([]))                                // EMPTY_LIST
                case 0x61, 0x65:                                                  // APPEND, APPENDS
                    let items = op == 0x61 ? [try pop()] : try popMark()
                    guard case .list(let l) = try pop() else { throw Failure.notATorchFile("pickle: APPEND не к списку") }
                    stack.append(.list(l + items))
                case 0x7D: stack.append(.dict([]))                                // EMPTY_DICT
                case 0x73, 0x75:                                                  // SETITEM, SETITEMS
                    var items: [Value]
                    if op == 0x73 { let v = try pop(), k = try pop(); items = [k, v] } else { items = try popMark() }
                    guard items.count % 2 == 0, case .dict(let d) = try pop() else {
                        throw Failure.notATorchFile("pickle: SETITEM не к словарю")
                    }
                    stack.append(.dict(d + stride(from: 0, to: items.count, by: 2).map { (items[$0], items[$0 + 1]) }))
                case 0x4E: stack.append(.none)                                    // NONE
                case 0x88: stack.append(.bool(true))                              // NEWTRUE
                case 0x89: stack.append(.bool(false))                             // NEWFALSE
                case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4)))))   // BININT
                case 0x4B: stack.append(.int(try uint(1)))                        // BININT1
                case 0x4D: stack.append(.int(try uint(2)))                        // BININT2
                case 0x8A:                                                        // LONG1
                    let n = try uint(1)
                    guard n <= 8 else { throw Failure.unsupported("целое длиннее 8 байт") }
                    let raw = try uint(n)
                    let shift = 64 - 8 * n
                    stack.append(.int(n == 0 ? 0 : (raw << shift) >> shift))
                case 0x47:                                                        // BINFLOAT
                    let bits = try take(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
                    stack.append(.float(Double(bitPattern: bits)))
                case 0x58: stack.append(try str(try uint(4)))                     // BINUNICODE
                case 0x8C: stack.append(try str(try uint(1)))                     // SHORT_BINUNICODE
                case 0x8D: stack.append(try str(try uint(8)))                     // BINUNICODE8
                case 0x63: let m = try line(), n = try line(); stack.append(.global(m, n))   // GLOBAL
                case 0x93:                                                        // STACK_GLOBAL
                    guard case .str(let n) = try pop(), case .str(let m) = try pop() else {
                        throw Failure.notATorchFile("pickle: STACK_GLOBAL без строк")
                    }
                    stack.append(.global(m, n))
                case 0x71: memo[try uint(1)] = stack.last                         // BINPUT
                case 0x72: memo[try uint(4)] = stack.last                         // LONG_BINPUT
                case 0x94: memo[memo.count] = stack.last                          // MEMOIZE
                case 0x68, 0x6A:                                                  // BINGET, LONG_BINGET
                    guard let v = memo[try uint(op == 0x68 ? 1 : 4)] else { throw Failure.notATorchFile("pickle: нет в мемо") }
                    stack.append(v)
                case 0x51: stack.append(try persistent(try pop()))                // BINPERSID
                case 0x52: let args = try pop(), fn = try pop(); stack.append(try reduce(fn, args))   // REDUCE
                case 0x62:                                                        // BUILD
                    _ = try pop()
                    guard case .dict = stack.last else { throw Failure.unsupported("pickle: BUILD не у словаря") }
                default:
                    throw Failure.unsupported(String(format: "опкод pickle 0x%02x", op))
                }
            }
            throw Failure.notATorchFile("pickle без STOP")
        }

        func persistent(_ pid: Value) throws -> Value {
            guard case .tuple(let t) = pid, t.count == 5, case .str("storage") = t[0],
                  case .global("torch", let type) = t[1], case .str(let key) = t[2], case .int(let numel) = t[4],
                  numel >= 0
            else { throw Failure.unsupported("ссылка на хранилище не torch") }
            return .storage(StorageRef(type: type, key: key, numel: numel))
        }

        func reduce(_ fn: Value, _ args: Value) throws -> Value {
            guard case .global(let m, let n) = fn, case .tuple(let a) = args else {
                throw Failure.unsupported("pickle: REDUCE не глобала")
            }
            func ints(_ v: Value) -> [Int]? {
                guard case .tuple(let t) = v else { return nil }
                let r = t.compactMap { x -> Int? in if case .int(let i) = x { return i }; return nil }
                return r.count == t.count ? r : nil
            }
            switch (m, n) {
            case ("collections", "OrderedDict") where a.isEmpty:
                return .dict([])
            case ("torch._utils", "_rebuild_tensor_v2"):
                guard a.count >= 4, case .storage(let s) = a[0], case .int(let off) = a[1],
                      let size = ints(a[2]), let st = ints(a[3])
                else { throw Failure.unsupported("_rebuild_tensor_v2 с незнакомыми аргументами") }
                return .tensor(s, offset: off, size: size, stride: st)
            case ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_parameter_with_state"):
                guard let first = a.first, case .tensor = first else {
                    throw Failure.unsupported("\(n) не над тензором")
                }
                return first
            default:
                throw Failure.unsupported("глобал \(m).\(n) не разрешён")
            }
        }
    }

    static let storageTypes: [String: DType] = ["FloatStorage": .float32, "BFloat16Storage": .bfloat16,
                                                 "HalfStorage": .float16]

    /// `.pth` → тензоры в порядке словаря и имена записей, которые не тензоры.
    static func read(_ url: URL) throws -> (tensors: [(String, MLXArray)], other: [String]) {
        let zip = try StoredZipReader([UInt8](try Data(contentsOf: url)))
        guard let pkl = zip.names.first(where: { $0 == "data.pkl" || $0.hasSuffix("/data.pkl") }) else {
            throw Failure.notATorchFile("в zip нет data.pkl")
        }
        let prefix = String(pkl.dropLast("data.pkl".count))
        if zip.entries[prefix + "byteorder"] != nil {
            let order = String(decoding: try zip.read(prefix + "byteorder"), as: UTF8.self)
            guard order == "little" else { throw Failure.unsupported("порядок байт \(order)") }
        }
        var u = Unpickler(try zip.read(pkl))
        guard case .dict(let items) = try u.load() else { throw Failure.notATorchFile("наверху pickle не словарь") }
        var storages: [String: MLXArray] = [:]
        var tensors: [(String, MLXArray)] = []
        var other: [String] = []
        for (k, v) in items {
            guard case .str(let name) = k else { throw Failure.unsupported("ключ словаря не строка") }
            guard case .tensor(let s, let offset, let size, let st) = v else { other.append(name); continue }
            guard let dt = storageTypes[s.type] else { throw Failure.unsupported("хранилище \(s.type)") }
            if storages[s.key] == nil {
                let raw = try zip.read(prefix + "data/" + s.key)
                guard raw.count == s.numel * dt.size else {
                    throw Failure.notATorchFile("хранилище \(s.key): \(raw.count) байт, нужно \(s.numel * dt.size)")
                }
                storages[s.key] = MLXArray(Data(raw), [s.numel], dtype: dt)
            }
            let flat = storages[s.key]!
            guard offset >= 0, size.count == st.count, size.allSatisfy({ $0 >= 0 }), st.allSatisfy({ $0 >= 0 }) else {
                throw Failure.notATorchFile("раскладка \(name) некорректна")
            }
            let n = size.reduce(1, *)
            if n > 0 {
                let last = offset + zip2(size, st).reduce(0) { $0 + ($1.0 - 1) * $1.1 }
                guard last < s.numel else { throw Failure.notATorchFile("\(name) выходит за хранилище") }
            }
            if n == 0 {
                tensors.append((name, MLXArray.zeros(size, dtype: dt)))
            } else if st == cStride(size) {
                tensors.append((name, flat[offset ..< (offset + n)].reshaped(size)))
            } else {
                var idx = [Int32](repeating: Int32(offset), count: 1)
                for (d, s) in zip2(size, st) {
                    idx = idx.flatMap { base in (0 ..< d).map { base + Int32($0 * s) } }
                }
                tensors.append((name, flat[MLXArray(idx)].reshaped(size)))
            }
        }
        return (tensors, other)
    }

    private static func zip2<A, B>(_ a: [A], _ b: [B]) -> Zip2Sequence<[A], [B]> { Swift.zip(a, b) }
}

private extension Data {
    mutating func le16(_ v: UInt16) { Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) } }
    mutating func le32(_ v: UInt32) { Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) } }
}
