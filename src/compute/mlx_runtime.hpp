// 计算运行时的初始化接口与设备身份信息，供实际计算入口和集成测试共用。
// 这里只约定运行时准备能力，不暴露固定数值样例或测试断言。

#pragma once

#include <expected>
#include <string>

namespace lab::compute {
struct RuntimeInfo {
  std::string device_name;
  std::string mlx_version;
};

// Initialize at worker startup, before other MLX work or worker threads begin.
// Selects the GPU and reports its identity; does not execute a numerical test.
std::expected<RuntimeInfo, std::string> initialize_gpu();
}  // namespace lab::compute
