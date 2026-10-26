// 验证检查点容器的正常往返及拒绝边界；训练状态是否完整留待真实训练/恢复实验判定。
#include "checkpoint/store.hpp"
#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <unistd.h>
namespace {
std::size_t checks=0;
void require(bool value,std::string_view message) { ++checks;if (!value) throw std::runtime_error(std::string(message)); }
}
int main() {
  auto pattern=(std::filesystem::temp_directory_path()/"pitfall-checkpoint-XXXXXX").string();
  auto* temp=::mkdtemp(pattern.data());
  if (!temp) return 1;
  const std::filesystem::path root=temp;
  try {
    std::ofstream(root/"tensor.bin",std::ios::binary)<<"abc";
    lab::resources::Policy policy{16*1024*1024,12*1024*1024,1,{}};
    lab::checkpoint::Identity identity{"model-v1","data-v1","template-v1","config-v1"};
    lab::checkpoint::Request request{"step-0001",identity,R"({"step":1})",{{"adapter.bin",root/"tensor.bin"}},2*1024*1024};
    auto saved=lab::checkpoint::save(root,policy,request);
    require(saved.has_value(),saved?"saved":saved.error());
    auto loaded=lab::checkpoint::verify(root,"step-0001",identity);
    require(loaded && loaded->control_json==R"({"step":1})","control state did not round trip");
    require(loaded->artifacts.size()==2 && loaded->artifacts[1].sha256==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","known SHA-256 vector differs");
    require(!lab::checkpoint::save(root,policy,request),"same generation overwritten");
    auto wrong=identity;wrong.configuration="other";
    require(!lab::checkpoint::verify(root,"step-0001",wrong),"identity mismatch accepted");
    for (auto value:{"../step-0001",".pending-anything","step/child",""})
      require(!lab::checkpoint::verify(root,value,identity),"unsafe generation accepted");
    for (auto value:{"[]","true","not-json","{\"step\":1,\"step\":2}"}) {
      auto invalid=request;invalid.name="invalid-state";invalid.control_json=value;
      require(!lab::checkpoint::save(root,policy,invalid),"invalid state accepted");
    }
    auto missing=request;missing.name="missing-source";missing.files[0].source=root/"absent";
    require(!lab::checkpoint::save(root,policy,missing),"missing source accepted");
    auto low=request;low.name="low-peak";low.peak_bytes=1024;
    require(lab::checkpoint::save(root,policy,low).has_value(),"project budget must not reject a checkpoint when volume space is available");
    require(lab::checkpoint::verify(root,"low-peak",identity).has_value(),"low-peak checkpoint failed verification");
    std::filesystem::remove_all(root/"runs/checkpoints/low-peak");
    for (const auto& entry:std::filesystem::directory_iterator(root/"runs/checkpoints"))
      require(entry.path().filename()=="step-0001","ordinary failure left an abandoned stage");
    require(lab::checkpoint::verify(root,"step-0001",identity).has_value(),"failed save damaged old checkpoint");
    std::ofstream(root/"runs/checkpoints/step-0001/adapter.bin",std::ios::binary)<<"bad";
    require(!lab::checkpoint::verify(root,"step-0001",identity),"same-size corruption accepted");
    std::println("PASS: {} checkpoint round-trip, hash, identity and refusal assertions",checks);
  } catch (const std::exception& error) {
    std::println(stderr,"FAIL: {}",error.what());std::filesystem::remove_all(root);return 1;
  }
  std::filesystem::remove_all(root);return 0;
}
