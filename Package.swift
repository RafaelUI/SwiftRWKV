// swift-tools-version: 5.10
//
// SwiftRWKV — on-device RWKV-7 training & inference for the Apple ecosystem.
//
// Author:  ImpulseLeap / Alexei Goncharov
// Website: https://www.impulseleap.com
// Repo:    https://github.com/RafaelUI/SwiftRWKV
// License: Apache 2.0
//
import PackageDescription

let package = Package(
    name: "SwiftRWKV",
    platforms: [
        // Минимумы продиктованы mlx-swift (macOS 14 / iOS 17 / visionOS 1).
        // iPadOS покрывается продуктом .iOS.
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    // Каждый таргет, которым можно пользоваться снаружи, обязан быть
    // ПРОДУКТОМ. Таргет без продукта виден только внутри пакета: собирается,
    // тестируется, документируется — и не импортируется потребителем.
    // Реранкер, эмбеддинги и квантованная база прожили так до 0.1.2.
    products: [
        // Инференс крупных RWKV-7 World моделей, генерация с сэмплером,
        // LoRA/QLoRA-файнтюн, партиал-файнтюн, претрейн.
        //   import RWKVGen
        .library(name: "RWKVGen", targets: ["RWKVGen"]),
        // Квантованная база .rwkvq: формат, сайдкар, fused-деквантизация.
        // Отдельным продуктом, а не частью RWKVGen: сборке, которая не
        // трогает квантованные веса, всё это не нужно. RWKVGen подключает
        // его сам — импортировать напрямую надо только чтобы читать сайдкар
        // без модели (например ради замеров).
        //   import RWKVQuant
        .library(name: "RWKVQuant", targets: ["RWKVQuant"]),
        // Текстовые векторы: пулинг, обучаемая голова, контрастные лоссы,
        // GradCache, curriculum, метрики.
        //   import RWKVEmbedding
        .library(name: "RWKVEmbedding", targets: ["RWKVEmbedding"]),
        // Cross-encoder реранкер: голова над состоянием, кэш состояний,
        // listwise-обучение, выдача с индексом.
        //   import RWKVRerank
        .library(name: "RWKVRerank", targets: ["RWKVRerank"]),
        // Прогон реранкера: данные → кэш состояний → обучение головы → отчёт.
        .executable(name: "rerank-run", targets: ["RerankRun"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.31.4"
        ),
    ],
    targets: [
        // Кастомное WKV-7 Metal-ядро (forward + дифференцируемый checkpoint backward).
        // Фундамент для RWKVGen.
        .target(
            name: "RWKVKernel",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        // Бэкенд квантованной базы .rwkvq (gw_mode="sb6"): формат + fused
        // Metal-деквантизация. Намеренно НЕ знает про X070Backbone — это
        // нижний слой, который RWKVGen подключает сверху. Отдельный таргет
        // потому, что владеет форматом файла и своим ядром: сборке, которая
        // не трогает квантованные веса, всё это не нужно.
        .target(
            name: "RWKVQuant",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "RWKVGen",
            dependencies: [
                "RWKVKernel",
                "RWKVQuant",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        // Текстовые векторы: пулинг, обучаемая голова, контрастные лоссы,
        // GradCache, дообучение и метрики. Стоит поверх RWKVGen, потому что
        // нужен body() бэкбона. RWKVKernel — ради одной константы WKV7_CHUNK:
        // батчи для обучения обязаны быть кратны ей по длине, и знать это
        // число здесь честнее, чем продублировать его литералом.
        .target(
            name: "RWKVEmbedding",
            dependencies: [
                "RWKVGen",
                "RWKVKernel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        // Cross-encoder реранкер: голова из RWKV-блоков поверх состояния
        // базы, кэш состояний, listwise-обучение. Отдельный таргет от
        // RWKVEmbedding, потому что это ВТОРАЯ стадия поиска и живёт она без
        // первой: реранкер не считает и не сравнивает векторы вовсе, он
        // читает состояние. Общее у них — только бэкбон.
        .target(
            name: "RWKVRerank",
            dependencies: [
                "RWKVGen",
                "RWKVKernel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        // Прогон реранкера от данных до обученной головы. Исполняемая цель,
        // а не тест: тест обязан быть быстрым и воспроизводимым, а здесь
        // минуты кодирования на реальной модели и числа, которые сравниваются
        // с питоновскими замерами вручную.
        .executableTarget(
            name: "RerankRun",
            dependencies: ["RWKVRerank", "RWKVGen",
                           .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "RWKVGenTests",
            dependencies: ["RWKVGen", "RWKVKernel", "RWKVQuant",
                           "RWKVEmbedding", "RWKVRerank"]
        ),
    ]
)
