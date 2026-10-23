#include "compute/mlx_runtime.hpp"
#include "lab_build_info.hpp"

#include <mlx/mlx.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <print>
#include <stdexcept>

// D0 numerical fixture. Test inputs, reference results, tolerances and memory
// counters belong here; production initialization must not execute this case.
int main() {
  try {
    const auto runtime = lab::compute::initialize_gpu();
    if (!runtime) {
      std::println(stderr, "GPU initialization failed: {}", runtime.error());
      return 1;
    }
    namespace mx = mlx::core;
    if (!(mx::default_device() == mx::Device::gpu)) {
      throw std::runtime_error("Runtime did not select the GPU");
    }
    mx::reset_peak_memory();
    const auto x = mx::array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3});
    const auto w = mx::array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {3, 2});
    const auto expected_y = mx::array({22.f, 28.f, 49.f, 64.f}, {2, 2});
    const auto expected_g = mx::array({109.f, 142.f, 144.5f, 188.f, 180.f, 234.f}, {3, 2});
    const std::function<mx::array(const mx::array&)> loss = [&](const auto& weights) {
      return mx::mean(mx::square(mx::matmul(x, weights)));
    };
    const auto y = mx::matmul(x, w);
    const auto gradient = mx::grad(loss)(w);
    const auto y_error = mx::max(mx::abs(y - expected_y));
    const auto g_error = mx::max(mx::abs(gradient - expected_g));
    mx::eval(y_error, g_error);
    const auto forward_error = y_error.item<float>();
    const auto gradient_error = g_error.item<float>();
    constexpr auto tolerance = 1e-5f;
    if (!std::isfinite(forward_error) || !std::isfinite(gradient_error) ||
        forward_error > tolerance || gradient_error > tolerance) {
      throw std::runtime_error("GPU forward/gradient differs from the analytic reference");
    }
    std::println("status: ok\ndevice: GPU\ndevice_name: {}\nmlx_version: {}\ncpp_standard: {}\n"
                 "forward_max_error: {}\ngradient_max_error: {}\npeak_bytes: {}",
                 runtime->device_name, runtime->mlx_version, __cplusplus,
                 forward_error, gradient_error, mx::get_peak_memory());
    return 0;
  } catch (const std::exception& error) {
    std::println(stderr, "GPU smoke test failed: {}", error.what());
    return 1;
  }
}
