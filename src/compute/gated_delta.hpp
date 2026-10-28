// GDN 前向接口：原生 float32 路径与参考混合精度 Metal 路径；BF16 路径尚未定义反向。
#pragma once
#include <mlx/mlx.h>
#include <optional>
#include <vector>
namespace lab::compute {
// q/k 已归一化并将读取尺度折入 q；g 是乘法衰减而非 log gate。
std::vector<mlx::core::array> gated_delta(
    const mlx::core::array& q, const mlx::core::array& k, const mlx::core::array& v,
    const mlx::core::array& g, const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state = std::nullopt);
}
