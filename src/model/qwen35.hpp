// 固定 Qwen3.5-4B 语言模型：量化权重加载、前向与独立请求的混合缓存。
#pragma once
#include <mlx/mlx.h>
#include <filesystem>
#include <functional>
#include <string_view>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lab::model {
struct Qwen35Config {
  int hidden, intermediate, layers, heads, kv_heads, head_dim, vocab;
  int key_heads, value_heads, key_dim, value_dim, conv_kernel, attention_interval, max_positions;
  float eps, rope_base, rotary_factor;
  static Qwen35Config parse(const std::string& json);
};
struct LayerState {
  std::optional<mlx::core::array> first, second;
};
struct Qwen35State {
  int offset = 0;
  std::vector<LayerState> layers;
};
using TensorMap = std::unordered_map<std::string, mlx::core::array>;
using Observer = std::function<void(std::string_view, const mlx::core::array&)>;

class Qwen35 {
 public:
  explicit Qwen35(const std::filesystem::path& artifact);
  const Qwen35Config& config() const noexcept { return config_; }
  Qwen35State make_state() const;
  // Nonempty token IDs for one unpadded text request. State belongs to that request.
  mlx::core::array forward(const std::vector<int>& tokens, Qwen35State& state,
                          const Observer& observer = {}) const;
  TensorMap state_tensors(const Qwen35State& state) const;
 private:
  Qwen35Config config_;
  TensorMap weights_;
  mlx::core::array linear(const mlx::core::array& x, const std::string& name) const;
  mlx::core::array norm(const mlx::core::array& x, const std::string& name) const;
  mlx::core::array attention(const mlx::core::array& x, int layer, int offset, LayerState& state) const;
  mlx::core::array delta(const mlx::core::array& x, int layer, LayerState& state, const Observer& observer) const;
};
}  // namespace lab::model
