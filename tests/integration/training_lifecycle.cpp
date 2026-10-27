// 用确定性小型 LoRA/Adam 更新验证安全边界暂停、检查点恢复和完整状态续接。
#include "checkpoint/store.hpp"
#include "compute/mlx_runtime.hpp"
#include "lab_build_info.hpp"
#include "runner/control.hpp"

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
namespace mx=mlx::core;
namespace fs=std::filesystem;
using Tensors=std::unordered_map<std::string,mx::array>;
using Json=nlohmann::json;

const lab::checkpoint::Identity identity{
    "d3-synthetic-lora-v1", "fixed-17-sample-sequence-v1", "none",
    "rank4-adam-seed42-step8-v1"};
const lab::resources::Policy policy{1024ULL*1024*1024, 800ULL*1024*1024, 1, {}};

Tensors initial_state() {
  Tensors state;
  state.emplace("a",mx::random::normal({256,4},mx::float32,0.f,0.02f,mx::random::key(12)));
  state.emplace("b",mx::zeros({4,128}));
  for (const auto* name:{"ma","va","mb","vb"}) state.emplace(name,mx::zeros(
      std::string_view(name)[1]=='a' ? mx::Shape{256,4} : mx::Shape{4,128}));
  state.emplace("rng_key",mx::random::key(42));
  state.emplace("losses",mx::zeros({0}));
  state.emplace("sample_ids",mx::zeros({0},mx::int32));
  return state;
}

std::vector<mx::array> frozen_weights() {
  return mx::quantize(mx::random::normal({128,256},mx::float32,0.f,0.05f,mx::random::key(7)),64,4);
}

void save_state(const fs::path& file,const Tensors& state) {
  fs::create_directories(file.parent_path());
  mx::save_safetensors(file.string(),state);
}

void publish_checkpoint(const fs::path& root,std::string name,const Tensors& state,int step) {
  const auto state_file=root/"runs/d3-state"/(name+".safetensors");
  save_state(state_file,state);
  Json control{{"step",step},{"data_position",step},{"optimizer_step",step}};
  lab::checkpoint::Request request{name,identity,control.dump(),
      {{"training_state.safetensors",state_file}},16ULL*1024*1024};
  const auto saved=lab::checkpoint::save(root,policy,request);
  std::error_code error;
  fs::remove(state_file,error);
  if (!saved) throw std::runtime_error("checkpoint save failed: "+saved.error());
}

std::pair<Tensors,int> load_checkpoint(const fs::path& root,std::string_view name) {
  const auto loaded=lab::checkpoint::verify(root,name,identity);
  if (!loaded) throw std::runtime_error("checkpoint verification failed: "+loaded.error());
  const auto control=Json::parse(loaded->control_json);
  const int step=control.at("step").get<int>();
  if (control.at("data_position").get<int>()!=step || control.at("optimizer_step").get<int>()!=step)
    throw std::runtime_error("checkpoint progress fields disagree");
  auto tensors=mx::load_safetensors((loaded->directory/"training_state.safetensors").string()).first;
  for (const auto* required:{"a","b","ma","va","mb","vb","rng_key","losses","sample_ids"})
    if (!tensors.contains(required)) throw std::runtime_error(std::string("checkpoint tensor missing: ")+required);
  if (tensors.at("sample_ids").size()!=static_cast<std::size_t>(step) ||
      tensors.at("losses").size()!=static_cast<std::size_t>(step))
    throw std::runtime_error("checkpoint tensor progress differs from control state");
  return {std::move(tensors),step};
}

lab::runner::RunResult train(Tensors& state,int& step,int total_steps,
           int pause_after,const fs::path& root,std::string_view checkpoint_name,
           const fs::path& ready_file,lab::runner::PauseControl& pause_control) {
  const auto weights=frozen_weights();
  auto result=lab::runner::run(pause_control,static_cast<std::uint64_t>(step),
      static_cast<std::uint64_t>(total_steps),[&]() -> std::expected<void,std::string> {
    const auto keys=mx::random::split(state.at("rng_key"),3);
    state.at("rng_key")=mx::take(keys,0,0);
    const auto sample=mx::random::randint(0,17,{1},mx::int32,mx::take(keys,1,0));
    const auto x=mx::random::normal({4,256},mx::float32,mx::take(keys,2,0))+
        mx::astype(sample,mx::float32)*mx::array(0.01f);
    const std::function<mx::array(const std::vector<mx::array>&)> loss=[&](const auto& p) {
      const auto y=mx::quantized_matmul(x,weights[0],weights[1],weights[2],true,64,4)+
          mx::matmul(mx::matmul(x,p[0]),p[1])*mx::array(0.25f);
      return mx::mean(mx::square(y-mx::array(0.1f)));
    };
    const auto result=mx::value_and_grad(loss,{0,1})({state.at("a"),state.at("b")});
    for (int i=0;i<2;++i) {
      const std::string parameter=i==0?"a":"b";
      const std::string moment="m"+parameter,variance="v"+parameter;
      const auto& gradient=result.second[i];
      state.at(moment)=mx::array(0.9f)*state.at(moment)+mx::array(0.1f)*gradient;
      state.at(variance)=mx::array(0.999f)*state.at(variance)+mx::array(0.001f)*mx::square(gradient);
      const auto bias1=1.0f-std::pow(0.9f,static_cast<float>(step+1));
      const auto bias2=1.0f-std::pow(0.999f,static_cast<float>(step+1));
      const auto corrected_m=state.at(moment)/mx::array(bias1);
      const auto corrected_v=state.at(variance)/mx::array(bias2);
      state.at(parameter)=state.at(parameter)-mx::array(0.01f)*corrected_m/
          (mx::sqrt(corrected_v)+mx::array(1e-8f));
    }
    state.at("losses")=mx::concatenate({state.at("losses"),mx::reshape(result.first,{1})});
    state.at("sample_ids")=mx::concatenate({state.at("sample_ids"),sample});
    std::vector<mx::array> materialized;
    for (const auto& [name,value]:state) { (void)name;materialized.push_back(value); }
    mx::eval(materialized);
    ++step;

    if (pause_after>0 && step==pause_after) {
      std::ofstream ready(ready_file);
      ready << "step=" << step << '\n';
      ready.close();
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
      while (!pause_control.requested() && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (!pause_control.requested())
        return std::unexpected("pause request did not arrive at test boundary");
    }
    return {};
  },[&](std::uint64_t completed_steps) -> std::expected<void,std::string> {
    if (checkpoint_name.empty()) return std::unexpected("pause requested without checkpoint target");
    try {
      publish_checkpoint(root,std::string(checkpoint_name),state,static_cast<int>(completed_steps));
      return {};
    } catch (const std::exception& error) {
      return std::unexpected(error.what());
    }
  });
  if (!result) throw std::runtime_error(result.error());
  step=static_cast<int>(result->completed_steps);
  return *result;
}

void run_worker(char** argv) {
  const std::string mode=argv[1];
  const fs::path root=fs::absolute(argv[2]);
  lab::runner::PauseControl pause_control;
  const auto control_installed=pause_control.install();
  if (!control_installed) throw std::runtime_error("pause control initialization failed: "+control_installed.error());
  const auto runtime=lab::compute::initialize_gpu();
  if (!runtime) throw std::runtime_error("GPU initialization failed: "+runtime.error());
  mx::reset_peak_memory();
  if (mode=="continuous") {
    const fs::path output=argv[3];
    const int total=std::stoi(argv[4]);
    auto state=initial_state();
    int step=0;
    train(state,step,total,0,root,{},{},pause_control);
    save_state(output,state);
    std::cout << "status: completed\nstep: " << step << "\npeak_bytes: " << mx::get_peak_memory() << '\n';
  } else if (mode=="pause") {
    const std::string checkpoint_name=argv[3];
    const int pause_after=std::stoi(argv[4]),total=std::stoi(argv[5]);
    const fs::path ready=argv[6];
    auto state=initial_state();
    int step=0;
    const auto result=train(state,step,total,pause_after,root,checkpoint_name,ready,pause_control);
    if (!result.paused) throw std::runtime_error("pause request did not stop the training loop");
    if (step!=pause_after) throw std::runtime_error("paused at an unexpected optimizer step");
    std::cout << "worker_exiting_after_pause=1\npeak_bytes: " << mx::get_peak_memory() << '\n';
  } else if (mode=="resume") {
    const std::string checkpoint_name=argv[3];
    const fs::path output=argv[4];
    const int total=std::stoi(argv[5]);
    auto [state,step]=load_checkpoint(root,checkpoint_name);
    train(state,step,total,0,root,{},{},pause_control);
    save_state(output,state);
    std::cout << "status: resumed\nstep: " << step << "\npeak_bytes: " << mx::get_peak_memory() << '\n';
  } else {
    throw std::runtime_error("unknown training lifecycle mode");
  }
}
}  // namespace

int main(int argc,char** argv) {
  try {
    if (argc<2 || argc>7) throw std::runtime_error(
        "usage: training_lifecycle continuous ROOT OUTPUT STEPS | pause ROOT CHECKPOINT STEP TOTAL READY | resume ROOT CHECKPOINT OUTPUT TOTAL");
    run_worker(argv);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "training lifecycle failed: " << error.what() << '\n';
    return 1;
  }
}
