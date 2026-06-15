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
    products: [
        // Обучение верхних слоёв + головы, классификация, извлечение фич,
        // инференс малых моделей. Кастомное WKV-7 Metal-ядро.
        //   import RWKVTrain
        .library(name: "RWKVTrain", targets: ["RWKVTrain"]),

        // Инференс крупных RWKV-7 World моделей (генерация) + LoRA/QLoRA-файнтюн.
        //   import RWKVGen
        .library(name: "RWKVGen", targets: ["RWKVGen"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.31.4"
        ),
    ],
    targets: [
        // Кастомное WKV-7 Metal-ядро (forward + дифференцируемый checkpoint backward).
        // Общий фундамент для RWKVTrain и RWKVGen.
        .target(
            name: "RWKVKernel",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "RWKVTrain",
            dependencies: [
                "RWKVKernel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "RWKVGen",
            dependencies: [
                "RWKVKernel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "RWKVTrainTests",
            dependencies: ["RWKVTrain", "RWKVKernel"],
            resources: [
                // Эталон паритета WKV-7-ядра: входы + эталонные выходы одного чанка.
                .copy("Resources/wkv7_kernel_parity.safetensors"),
            ]
        ),
        .testTarget(
            name: "RWKVGenTests",
            dependencies: ["RWKVGen", "RWKVKernel"],
            resources: [
                // Эталон паритета x070: веса World-0.1B + ожидаемые ln_out/logits.
                .copy("Resources/world_0.1b_x070.safetensors"),
                .copy("Resources/x070_parity.safetensors"),
                .copy("Resources/x070_stages.safetensors"),
                .copy("Resources/x070_tmix.safetensors"),
                .copy("Resources/x070_perlayer.safetensors"),
            ]
        ),
    ]
)
