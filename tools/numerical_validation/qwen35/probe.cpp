// 独立进程数值观测工具；输入、生成长度和输出由外部调用者提供。
#include "model/qwen35.hpp"
#include "compute/mlx_runtime.hpp"
#include "resources/preflight.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
namespace mx=mlx::core;
namespace fs=std::filesystem;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv){
  try{
    if(argc!=7)throw std::runtime_error("usage: qwen35_probe ROOT ARTIFACT INPUT OUTPUT CHUNK GENERATION_STEPS");
    const fs::path root=argv[1],artifact=argv[2],input_file=argv[3],output=argv[4];
    const int chunk=std::stoi(argv[5]),generation_steps=std::stoi(argv[6]);
    if(chunk<0 || generation_steps<0 || generation_steps>16)throw std::runtime_error("invalid probe work bound");
    std::ifstream input(input_file);Json request;input>>request;
    const auto tokens=request.at("tokens").get<std::vector<int>>();
    if(tokens.empty() || tokens.size()>128)throw std::runtime_error("probe requires 1..128 prepared token IDs");
    auto policy=lab::resources::load_policy(root/"configs/resources.toml");
    if(!policy)throw std::runtime_error(policy.error());
    auto snapshot=lab::resources::inspect(root,*policy);
    if(!snapshot)throw std::runtime_error(snapshot.error());
    auto decision=lab::resources::assess(*policy,*snapshot,1024ULL*1024*1024);
    if(!decision || !decision->allowed)throw std::runtime_error("probe disk peak refused");
    auto runtime=lab::compute::initialize_gpu();
    if(!runtime)throw std::runtime_error(runtime.error());
    mx::set_memory_limit(10ULL*1024*1024*1024);
    mx::reset_peak_memory();
    auto started=Clock::now();
    lab::model::Qwen35 model(artifact);
    auto state=model.make_state();
    lab::model::TensorMap observations;
    const bool diagnostic=std::getenv("LAB_D4_DIAGNOSTIC")!=nullptr;
    const lab::model::Observer observe=[&](std::string_view name,const mx::array& value) {
      const bool layer_output=name.starts_with("layer.") && name.find('.',6)==std::string_view::npos;
      if(name=="embedding" || name=="final_hidden" || layer_output || (diagnostic && name.starts_with("layer.0.")))
        observations.insert_or_assign(std::string(name),value);
    };
    std::vector<mx::array> parts;
    auto forward_start=Clock::now();
    const int size=static_cast<int>(tokens.size());
    for(int begin=0;begin<size;){
      const int end=chunk==0?size:std::min(size,begin+chunk);
      parts.push_back(model.forward(std::vector<int>(tokens.begin()+begin,tokens.begin()+end),state,chunk==0?observe:lab::model::Observer{}));
      // Establish the same materialized boundary for every chunk.
      auto cache=model.state_tensors(state);
      std::vector<mx::array> pending{parts.back()};
      for(auto& [name,tensor]:cache){(void)name;pending.push_back(tensor);}
      mx::eval(pending);
      begin=end;
    }
    auto logits=mx::concatenate(parts,1);
    observations.insert_or_assign("logits",logits);
    for(auto& [name,tensor]:model.state_tensors(state)) observations.insert_or_assign("cache."+name,tensor);
    const double prefill_seconds=std::chrono::duration<double>(Clock::now()-forward_start).count();
    std::vector<int> generated;
    auto next=mx::take(logits,size-1,1);
    for(int n=0;n<generation_steps;++n){
      const int id=mx::argmax(next,-1).item<int>();
      generated.push_back(id);
      if(id==248044 || id==248046)break;
      next=model.forward({id},state);
      mx::eval(next);
    }
    if(!generated.empty()) observations.insert_or_assign("generated",mx::array(generated.data(),{static_cast<int>(generated.size())},mx::int32));
    fs::create_directories(output.parent_path());
    mx::save_safetensors(output.string(),observations);
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    Json report{{"device",runtime->device_name},{"input_tokens",tokens.size()},{"chunk",chunk},
        {"prefill_seconds",prefill_seconds},{"total_seconds",std::chrono::duration<double>(Clock::now()-started).count()},
        {"mlx_peak_bytes",mx::get_peak_memory()},{"max_rss_bytes",usage.ru_maxrss},
        {"generated",generated},{"final_offset",state.offset}};
    std::cout<<report.dump()<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<"Qwen3.5 probe failed: "<<e.what()<<'\n';return 1;}
}
