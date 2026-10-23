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
