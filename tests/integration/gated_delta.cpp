// 用解析递推和分块对照验证正式计算接口，包括 BF16 输入下的 float32 衰减与状态。
#include "compute/gated_delta.hpp"
#include "compute/mlx_runtime.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
  try {
    const auto runtime=lab::compute::initialize_gpu();
    if(!runtime) throw std::runtime_error(runtime.error());
    namespace mx=mlx::core;
    // Unit key/query direction: state first component follows s'=g*s+beta*(v-g*s).
    auto one=mx::concatenate({mx::ones({1,2,1,1}),mx::zeros({1,2,1,127})},-1);
    auto q=mx::astype(one,mx::bfloat16);
    auto v=mx::full({1,2,2,128},2.f,mx::bfloat16);
    auto g=mx::full({1,2,2},0.999f,mx::float32);
    auto beta=mx::full({1,2,2},0.25f,mx::bfloat16);
    auto result=lab::compute::gated_delta(q,q,v,g,beta);
    const float expected=0.5f*0.999f+0.25f*(2.f-0.5f*0.999f);
    auto error=mx::max(mx::abs(mx::take(result[1],0,-1)-mx::array(expected)));
    auto first=lab::compute::gated_delta(mx::slice(q,{0,0,0,0},{1,1,1,128}),
        mx::slice(q,{0,0,0,0},{1,1,1,128}),mx::slice(v,{0,0,0,0},{1,1,2,128}),
        mx::slice(g,{0,0,0},{1,1,2}),mx::slice(beta,{0,0,0},{1,1,2}));
    auto second=lab::compute::gated_delta(mx::slice(q,{0,1,0,0},{1,2,1,128}),
        mx::slice(q,{0,1,0,0},{1,2,1,128}),mx::slice(v,{0,1,0,0},{1,2,2,128}),
        mx::slice(g,{0,1,0},{1,2,2}),mx::slice(beta,{0,1,0},{1,2,2}),first[1]);
    const auto difference=mx::max(mx::abs(result[1]-second[1])).item<float>();
    const auto analytic_error=error.item<float>();
    const auto fp32=lab::compute::gated_delta(mx::astype(q,mx::float32),mx::astype(q,mx::float32),
        mx::astype(v,mx::float32),g,mx::astype(beta,mx::float32));
    const auto fp32_error=mx::max(mx::abs(mx::take(fp32[1],0,-1)-mx::array(expected))).item<float>();
    if(!std::isfinite(fp32_error) || fp32_error>1e-6f) throw std::runtime_error("native float32 GDN changed");
    // Observe the unadapted SDK path for learning/diagnosis; its error is not
    // the acceptance oracle for the formal interface.
    const auto unadapted=mx::fast::gated_delta_update(q,q,v,g,beta);
    const auto sdk_error=mx::max(mx::abs(mx::take(unadapted[1],0,-1)-mx::array(expected))).item<float>();
    if(!std::isfinite(analytic_error) || analytic_error>1e-6f || difference>1e-6f || result[1].dtype()!=mx::float32)
      throw std::runtime_error("GDN differs from analytic decay or chunked state");
    std::cout << "expected_state_first_component: " << expected
              << "\nunadapted_sdk_state_max_error: " << sdk_error
              << "\nunadapted_output_matches: " << mx::array_equal(result[0],unadapted[0]).item<bool>()
              << "\nanalytic_state_max_error: " << analytic_error << "\nchunk_state_max_error: " << difference << '\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
