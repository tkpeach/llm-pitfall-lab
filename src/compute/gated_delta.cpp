// 混合精度 GDN 前向接入：直接保留 BF16 输入与 float32 gate/state。
// Metal recurrence adapted from MLX-LM 0.32.2, Copyright © 2025 Apple Inc.
// MIT permission notice and source provenance: third_party/mlx-lm-gdn/{LICENSE,README.md}.
#include "compute/gated_delta.hpp"
#include <stdexcept>

namespace lab::compute {
namespace mx=mlx::core;
std::vector<mx::array> gated_delta(
    const mx::array& q, const mx::array& k, const mx::array& v,
    const mx::array& g, const mx::array& beta, const std::optional<mx::array>& state) {
  // Preserve the established native float32 route. The BF16 mixed-precision
  // contract needs both float32 decay and the reference reduction order.
  if(q.dtype()==mx::float32)
    return mx::fast::gated_delta_update(q,k,v,g,beta,state);
  if(q.dtype()!=mx::bfloat16)
    throw std::invalid_argument("GDN supports float32 or mixed BF16 inputs only");
  if(mx::default_device()!=mx::Device(mx::Device::gpu))
    throw std::runtime_error("mixed-precision GDN requires the GPU");
  if(q.ndim()!=4 || k.shape()!=q.shape() || v.ndim()!=4 || g.ndim()!=3 || beta.shape()!=g.shape())
    throw std::invalid_argument("invalid mixed-precision GDN ranks/shapes");
  const int B=q.shape(0),T=q.shape(1),Hk=q.shape(2),Dk=q.shape(3),Hv=v.shape(2),Dv=v.shape(3);
  if(B<=0 || T<=0 || Hk<=0 || Dk<=0 || Dk%32 || Hv<=0 || Hv%Hk || Dv<=0 ||
      v.shape(0)!=B || v.shape(1)!=T || g.shape()!=mx::Shape{B,T,Hv} ||
      k.dtype()!=mx::bfloat16 || v.dtype()!=mx::bfloat16 || g.dtype()!=mx::float32 ||
      beta.dtype()!=mx::bfloat16)
    throw std::invalid_argument("unsupported mixed-precision GDN geometry or dtype");
  auto initial=state?*state:mx::zeros({B,Hv,Dv,Dk},mx::float32);
  if(initial.shape()!=mx::Shape{B,Hv,Dv,Dk} || initial.dtype()!=mx::float32)
    throw std::invalid_argument("invalid float32 GDN recurrent state");
  static const auto operation=mx::fast::metal_kernel(
      "lab_gated_delta_bf16", {"q","k","v","g","beta","state_in","T"}, {"y","state_out"}, R"MSL(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float local_state[n_per_t];
    for (int i = 0; i < n_per_t; i++) {
      local_state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; t++) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; i++) {
        auto s_idx = n_per_t * dk_idx + i;
        local_state[i] = local_state[i] * g_[hv_idx];
        kv_mem += local_state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; i++) {
        auto s_idx = n_per_t * dk_idx + i;
        local_state[i] = local_state[i] + k_[s_idx] * delta;
        out += local_state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<InT>(out);
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
      y += Hv * Dv;
    }
    for (int i = 0; i < n_per_t; i++) {
      o_state[n_per_t * dk_idx + i] = static_cast<StT>(local_state[i]);
    }
)MSL");
  return operation({q,k,v,g,beta,initial,mx::array(T)}, {{B,T,Hv,Dv},{B,Hv,Dv,Dk}},
      {q.dtype(),mx::float32},{32,Dv,B*Hv},{32,4,1},{{"Dk",Dk},{"Dv",Dv},{"Hk",Hk},{"Hv",Hv},{"InT",q.dtype()},{"StT",mx::float32}},
      std::nullopt,false,{});
}
}  // namespace lab::compute
