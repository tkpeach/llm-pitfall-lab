// Bounded mechanism probe, not a complete Qwen implementation or training engine.
#include <mlx/mlx.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <csignal>
#include <fstream>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>

namespace mx = mlx::core;
using mx::array;
using Tensors = std::unordered_map<std::string, array>;
volatile std::sig_atomic_t pause_requested = 0;
void pause_handler(int) { pause_requested = 1; }

// Same scalar-gate recurrence as mlx_lm.models.gated_delta.gated_delta_ops.
// Inputs already have q/k normalization and gate activation applied.
std::vector<array> recurrence(const std::vector<array>& p) {
  auto q = mx::repeat(p[0], 2, 2), k = mx::repeat(p[1], 2, 2);
  auto state = p[5];
  std::vector<array> ys;
  for (int t = 0; t < q.shape(1); ++t) {
    auto qt = mx::expand_dims(mx::take(q, t, 1), -2);
    auto kt = mx::expand_dims(mx::take(k, t, 1), -2);
    auto vt = mx::take(p[2], t, 1);
    auto gt = mx::reshape(mx::take(p[3], t, 1), {1, 32, 1, 1});
    auto bt = mx::expand_dims(mx::take(p[4], t, 1), -1);
    state = state * gt;
    auto delta = (vt - mx::sum(state * kt, -1)) * bt;
    state = state + kt * mx::expand_dims(delta, -1);
    ys.push_back(mx::sum(state * qt, -1));
  }
  return {mx::stack(ys, 1), state};
}

void operations(const std::string& input, const std::string& output) {
  auto f = mx::load_safetensors(input).first;
  Tensors out;
  auto w = mx::quantize(f.at("weight"), 64, 4);
  auto project = [&](const array& x) {
    return mx::quantized_matmul(x, w[0], w[1], w[2], true, 64, 4);
  };
  std::function<array(const std::vector<array>&)> objective = [&](const auto& p) {
    auto y = project(p[0]) + mx::matmul(mx::matmul(p[0], p[1]), p[2]) * array(0.25f);
    return mx::mean(mx::square(y - f.at("target")));
  };
  std::vector<array> params{f.at("x"), f.at("a"), f.at("b")};
  auto result = mx::value_and_grad(objective, {0, 1, 2})(params);
  out.emplace("quant_output", project(params[0]));
  out.emplace("lora_loss", result.first);
  for (int i = 0; i < 3; ++i) out.emplace("lora_grad_" + std::to_string(i), result.second[i]);
  auto checkpointed = mx::checkpoint([&](const auto& p) {
    return std::vector<array>{objective(p)};
  });
  auto cp = mx::value_and_grad(checkpointed, std::vector<int>{0, 1, 2})(params);
  for (int i = 0; i < 3; ++i) out.emplace("checkpoint_grad_" + std::to_string(i), cp.second[i]);
  // The memory-constrained target also needs a low-precision differentiable path.
  auto bw = mx::quantize(mx::astype(f.at("weight"), mx::bfloat16), 64, 4);
  std::function<array(const std::vector<array>&)> bf_loss = [&](const auto& p) {
    auto y = mx::quantized_matmul(p[0], bw[0], bw[1], bw[2], true, 64, 4)
        + mx::matmul(mx::matmul(p[0], p[1]), p[2]) * array(0.25f);
    return mx::mean(mx::square(mx::astype(y, mx::float32) - f.at("target")));
  };
  std::vector<array> bp;
  for (auto p : params) bp.push_back(mx::astype(p, mx::bfloat16));
  auto br = mx::value_and_grad(bf_loss, {0, 1, 2})(bp);
  out.emplace("bf16_lora_loss", br.first);
  for (int i = 0; i < 3; ++i) out.emplace("bf16_lora_grad_" + std::to_string(i), br.second[i]);

  out.emplace("attention", mx::fast::scaled_dot_product_attention(
      f.at("attn_q"), f.at("attn_k"), f.at("attn_v"), 1.f / 16.f, "causal"));
  std::vector<array> decode;
  auto ak = f.at("attn_k"), av = f.at("attn_v");
  for (int t = 0; t < 8; ++t) {
    auto q = mx::take(f.at("attn_q"), array({t}), 2);
    auto k = mx::take(ak, mx::arange(t + 1), 2);
    auto v = mx::take(av, mx::arange(t + 1), 2);
    decode.push_back(mx::fast::scaled_dot_product_attention(q, k, v, 1.f / 16.f));
  }
  out.emplace("attention_decode", mx::concatenate(decode, 2));
  out.emplace("norm", mx::fast::rms_norm(f.at("x"), std::nullopt, 1e-6f));
  out.emplace("rope", mx::fast::rope(f.at("attn_q"), 64, false, 10000000.f, 1.f, 0));
  out.emplace("conv", mx::conv1d(f.at("conv_x"), f.at("conv_w"), 1, 0, 1, 4));
  auto kernel = mx::fast::metal_kernel("probe_square", {"x"}, {"y"},
      "uint i = thread_position_in_grid.x; y[i] = x[i] * x[i];");
  out.emplace("custom_metal", kernel({params[0]}, {params[0].shape()}, {mx::float32},
      {int(params[0].size()), 1, 1}, {256, 1, 1}, {}, std::nullopt, false, {})[0]);

  std::vector<array> rp{f.at("q"), f.at("k"), f.at("v"), f.at("g"), f.at("beta"), f.at("state")};
  auto r = recurrence(rp);
  out.emplace("recurrence_output", r[0]);
  out.emplace("recurrence_state", r[1]);
  std::function<array(const std::vector<array>&)> loss = [](const auto& p) {
    auto r = recurrence(p);
    return mx::mean(mx::square(r[0])) + array(0.01f) * mx::mean(mx::square(r[1]));
  };
  auto rg = mx::value_and_grad(loss, {0, 1, 2, 3, 4, 5})(rp);
  out.emplace("recurrence_loss", rg.first);
  for (int i = 0; i < 6; ++i) out.emplace("recurrence_grad_" + std::to_string(i), rg.second[i]);
  // Prefill vs two chunks with persistent recurrent state.
  std::vector<array> left, right;
  for (int i = 0; i < 5; ++i) {
    left.push_back(mx::take(rp[i], mx::arange(0, 4), 1));
    right.push_back(mx::take(rp[i], mx::arange(4, 8), 1));
  }
  left.push_back(rp[5]);
  auto l = recurrence(left);
  right.push_back(l[1]);
  auto rr = recurrence(right);
  out.emplace("chunk_output", mx::concatenate({l[0], rr[0]}, 1));
  out.emplace("chunk_state", rr[1]);
  mx::save_safetensors(output, out, {{"scope", "synthetic mechanisms; not full Qwen"}});
  std::cout << "operations peak_bytes=" << mx::get_peak_memory() << "\n";
}

void fused_recurrence(const std::string& input, const std::string& output, bool backward) {
  auto f = mx::load_safetensors(input).first;
  std::vector<array> p{f.at("q"), f.at("k"), f.at("v"), f.at("g"), f.at("beta"), f.at("state")};
  auto fused = [](const std::vector<array>& p) {
    return mx::fast::gated_delta_update(p[0], p[1], p[2], p[3], p[4], p[5]);
  };
  Tensors out;
  if (!backward) {
    auto r = fused(p);
    out.emplace("output", r[0]); out.emplace("state", r[1]);
  } else {
    std::function<array(const std::vector<array>&)> loss = [&](const auto& p) {
      auto r = fused(p);
      return mx::mean(mx::square(r[0])) + array(0.01f) * mx::mean(mx::square(r[1]));
    };
    auto r = mx::value_and_grad(loss, {0, 1, 2, 3, 4, 5})(p);
    for (int i = 0; i < 6; ++i) out.emplace("grad_" + std::to_string(i), r.second[i]);
  }
  mx::save_safetensors(output, out);
  std::cout << "fused_recurrence peak_bytes=" << mx::get_peak_memory() << "\n";
}

void durable_save(const std::string& path, const Tensors& tensors, int step) {
  // MLX appends .safetensors to paths without that suffix; keep staging suffix valid.
  auto temp = path + ".tmp.safetensors";
  mx::save_safetensors(temp, tensors, {{"identity", "cpp-mlx-probe-v1"}, {"step", std::to_string(step)}});
  int fd = open(temp.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open checkpoint staging failed");
  auto rc = fsync(fd); close(fd);
  if (rc != 0) throw std::runtime_error("checkpoint fsync failed");
  std::filesystem::rename(temp, path);
  fd = open(std::filesystem::path(path).parent_path().c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open checkpoint directory failed");
  rc = fsync(fd); close(fd);
  if (rc != 0) throw std::runtime_error("checkpoint directory fsync failed");
}

void train(const std::string& path, bool resume, int stop, bool signal_test = false) {
  Tensors s;
  int step = 0;
  if (resume) {
    auto loaded = mx::load_safetensors(path);
    if (loaded.second.at("identity") != "cpp-mlx-probe-v1") throw std::runtime_error("checkpoint identity mismatch");
    step = std::stoi(loaded.second.at("step"));
    s = std::move(loaded.first);
  } else {
    if (std::filesystem::exists(path)) throw std::runtime_error("refuse to overwrite existing run");
    s.emplace("a", mx::random::normal({256, 4}, mx::float32, 0.f, 0.02f, mx::random::key(12)));
    s.emplace("b", mx::zeros({4, 128}));
    for (auto name : {"ma", "va"}) s.emplace(name, mx::zeros({256, 4}));
    for (auto name : {"mb", "vb"}) s.emplace(name, mx::zeros({4, 128}));
    s.emplace("key", mx::random::key(42));
    s.emplace("losses", mx::zeros({0}));
    s.emplace("sample_ids", mx::zeros({0}, mx::int32));
  }
  if (step < 0 || step > stop || stop > 8) throw std::runtime_error("invalid progress");
  auto base = mx::quantize(mx::random::normal({128, 256}, mx::float32, 0.f, 0.05f, mx::random::key(7)), 64, 4);
  std::signal(SIGINT, pause_handler);
  std::signal(SIGTERM, pause_handler);
  for (; step < stop;) {
    auto keys = mx::random::split(s.at("key"), 3);
    s.at("key") = mx::take(keys, 0, 0);
    auto sample = mx::random::randint(0, 17, {1}, mx::int32, mx::take(keys, 1, 0));
    auto x = mx::random::normal({4, 256}, mx::float32, mx::take(keys, 2, 0)) + mx::astype(sample, mx::float32) * array(0.01f);
    std::function<array(const std::vector<array>&)> loss = [&](const auto& p) {
      auto y = mx::quantized_matmul(x, base[0], base[1], base[2], true, 64, 4)
          + mx::matmul(mx::matmul(x, p[0]), p[1]) * array(0.25f);
      return mx::mean(mx::square(y - array(0.1f)));
    };
    auto r = mx::value_and_grad(loss, {0, 1})({s.at("a"), s.at("b")});
    for (int j = 0; j < 2; ++j) {
      std::string p = j == 0 ? "a" : "b", m = "m" + p, v = "v" + p;
      auto g = r.second[j];
      s.at(m) = array(0.9f) * s.at(m) + array(0.1f) * g;
      s.at(v) = array(0.999f) * s.at(v) + array(0.001f) * mx::square(g);
      auto mh = s.at(m) / array(float(1. - std::pow(0.9, step + 1)));
      auto vh = s.at(v) / array(float(1. - std::pow(0.999, step + 1)));
      s.at(p) = s.at(p) - array(0.01f) * mh / (mx::sqrt(vh) + array(1e-8f));
    }
    s.at("losses") = mx::concatenate({s.at("losses"), mx::reshape(r.first, {1})});
    s.at("sample_ids") = mx::concatenate({s.at("sample_ids"), sample});
    std::vector<array> materialize;
    for (const auto& kv : s) materialize.push_back(kv.second);
    mx::eval(materialize);
    durable_save(path, s, step + 1);
    ++step;
    // Test-only rendezvous gives the controller a deterministic chance to send SIGINT.
    // This does not measure real-model kernel cancellation or long-step pause latency.
    if (signal_test && step == 3) {
      { std::ofstream ready(path + ".ready"); ready << getpid() << "\n"; }
      for (int i = 0; i < 500 && !pause_requested; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (!pause_requested) throw std::runtime_error("signal test timed out");
    }
    if (pause_requested) break;
  }
  std::cout << "training step=" << step << " peak_bytes=" << mx::get_peak_memory() << "\n";
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: probe METALLIB ops INPUT OUTPUT | probe METALLIB train CHECKPOINT STOP | probe METALLIB resume CHECKPOINT STOP");
    mx::metal::set_metallib_path(argv[1]);
    if (!mx::metal::is_available()) throw std::runtime_error("Metal GPU unavailable; no CPU fallback");
    mx::set_default_device(mx::Device(mx::Device::gpu));
    std::cout << "MLX " << mx::version() << " device=GPU\n";
    std::string mode = argv[2];
    if (mode == "ops") operations(argv[3], argv[4]);
    else if (mode == "fused" || mode == "fused-grad") fused_recurrence(argv[3], argv[4], mode == "fused-grad");
    else if (mode == "train" || mode == "resume" || mode == "signal-test")
      train(argv[3], mode == "resume", std::stoi(argv[4]), mode == "signal-test");
    else throw std::runtime_error("unknown mode");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
