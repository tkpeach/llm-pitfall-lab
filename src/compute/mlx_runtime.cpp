#include "compute/mlx_runtime.hpp"
#include "lab_build_info.hpp"

#include <mlx/mlx.h>

#include <filesystem>
#include <stdexcept>
#include <variant>

namespace lab::compute {
namespace mx = mlx::core;

std::expected<RuntimeInfo, std::string> initialize_gpu() {
  try {
    if (std::string_view(mx::version()) != build::mlx_version) {
      return std::unexpected("Loaded MLX version differs from the configured SDK");
    }
    if (!std::filesystem::is_regular_file(build::metallib)) {
      return std::unexpected("Configured mlx.metallib is missing; restore the selected SDK");
    }
    mx::metal::set_metallib_path(std::string(build::metallib));
    if (!mx::metal::is_available()) {
      return std::unexpected("Metal GPU unavailable; this runtime has no CPU fallback");
    }
    mx::set_default_device(mx::Device(mx::Device::gpu));
    const auto& info = mx::device_info();
    const auto device_name = std::get<std::string>(info.at("device_name"));
    return RuntimeInfo{device_name, mx::version()};
  } catch (const std::exception& error) {
    return std::unexpected(error.what());
  }
}
}  // namespace lab::compute
