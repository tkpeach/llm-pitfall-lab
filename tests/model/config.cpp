// 固定模型配置与制品拒绝契约；使用小型配置，不下载或分配目标权重。
#include "model/qwen35.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
namespace fs=std::filesystem;
using Json=nlohmann::json;
void check(bool v,const char* message){if(!v)throw std::runtime_error(message);}
template<class F> void reject(F f,const char* message){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected,message);}
int main(int argc,char** argv){
  fs::path temp;
  try{
    check(argc==2,"expected configuration fixture");
    std::ifstream file(argv[1]);Json config;file>>config;
    const auto valid=lab::model::Qwen35Config::parse(config.dump());
    check(valid.layers==32 && valid.vocab==248320 && valid.key_dim==128,"fixed geometry differs");
    for(const auto* field:{"hidden_size","num_hidden_layers","head_dim","linear_conv_kernel_dim","full_attention_interval"}){
      auto changed=config;changed["text_config"][field]=0;
      reject([&]{lab::model::Qwen35Config::parse(changed.dump());},"invalid geometry accepted");
    }
    auto changed=config;changed["text_config"]["num_hidden_layers"]=32.5;
    reject([&]{lab::model::Qwen35Config::parse(changed.dump());},"fractional layer count accepted");
    changed=config;changed["text_config"]["layer_types"][3]="linear_attention";
    reject([&]{lab::model::Qwen35Config::parse(changed.dump());},"wrong layer sequence accepted");
    changed=config;changed["quantization"]["bits"]=8;
    reject([&]{lab::model::Qwen35Config::parse(changed.dump());},"wrong quantization accepted");
    changed=config;changed["text_config"]["rope_parameters"]["rope_type"]="yarn";
    reject([&]{lab::model::Qwen35Config::parse(changed.dump());},"unsupported rope accepted");
    changed=config;changed["text_config"]["tie_word_embeddings"]=false;
    reject([&]{lab::model::Qwen35Config::parse(changed.dump());},"untied embedding accepted");
    const auto candidate=fs::temp_directory_path()/("qwen35-contract-"+std::to_string(::getpid()));
    check(fs::create_directory(candidate),"temporary fixture directory collision");
    temp=candidate;
    std::ofstream(temp/"config.json")<<config.dump();
    Json manifest{{"schema_version",1},{"repository","wrong-model"},{"revision","wrong-revision"},{"component","language_only"}};
    std::ofstream(temp/"artifact.json")<<manifest.dump();
    reject([&]{lab::model::Qwen35 model(temp);},"untrusted artifact identity accepted");
    manifest["repository"]="Qwen/Qwen3.5-4B";manifest["revision"]="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
    manifest["conversion"]={{"mlx","0.32.4"},{"mlx_lm","0.32.2"},{"bits",4},{"group_size",64},{"mode","affine"},{"dtype","bfloat16"}};
    manifest["files"]={{"config.json",{{"bytes",1},{"sha256","wrong-digest"}}}};
    std::ofstream(temp/"artifact.json")<<manifest.dump();
    reject([&]{lab::model::Qwen35 model(temp);},"artifact corruption accepted");
    manifest["files"]={{"../outside",{{"bytes",1},{"sha256","wrong-digest"}}}};
    std::ofstream(temp/"artifact.json")<<manifest.dump();
    reject([&]{lab::model::Qwen35 model(temp);},"artifact path traversal accepted");
    fs::remove_all(temp);temp.clear();
    std::cout<<"PASS: fixed Qwen3.5 config, geometry, dtype and artifact rejection contracts\n";
    return 0;
  }catch(const std::exception& e){if(!temp.empty())fs::remove_all(temp);std::cerr<<e.what()<<'\n';return 1;}
}
