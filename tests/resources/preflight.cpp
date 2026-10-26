// 验证轻量资源预检的配置、卷空间判断和明确的项目占用未知状态。
#include "resources/preflight.hpp"
#include <fstream>
#include <limits>
#include <print>
#include <stdexcept>
#include <unistd.h>

namespace {
namespace fs=std::filesystem;
std::size_t checks=0;
void require(bool condition,const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
struct Temp {
  fs::path root;
  Temp() {
    auto pattern=(fs::temp_directory_path()/"pitfall-resource-test-XXXXXX").string();
    auto* name=::mkdtemp(pattern.data());
    if (!name) throw std::runtime_error("cannot create test directory");
    root=name;
  }
  ~Temp() { std::error_code error;fs::remove_all(root,error); }
};
std::string config(std::string limit="1000",std::string warning="800",std::string floor="200",
                   std::string extra="[]") {
  return "project_limit_bytes="+limit+"\nwarning_bytes="+warning+"\nmin_free_bytes="+floor+
      "\nextra_owned_paths="+extra+"\n";
}
}

int main() {
  try {
    using namespace lab::resources;
    const Policy policy{1000,800,200,{}};
    auto volume_ok=assess(policy,Snapshot{std::nullopt,600},300);
    require(volume_ok && volume_ok->allowed && !volume_ok->warning && volume_ok->max_growth_bytes==400,
            "unknown project usage does not invent a project-budget warning");
    auto volume_excess=assess(policy,Snapshot{std::nullopt,600},401);
    require(volume_excess && !volume_excess->allowed && volume_excess->reason=="volume_peak_exceeded",
            "volume safety floor rejects excessive peak");
    require(!assess(policy,Snapshot{std::nullopt,199},0)->allowed,"already below volume floor refuses");
    require(assess(Policy{100,101,1,{}},Snapshot{std::nullopt,1000},0).has_value()==false,
            "invalid policy refuses");
    auto measured=assess(policy,Snapshot{950,1000},51);
    require(measured && measured->allowed && measured->warning && measured->reason=="project_budget_exceeded",
            "richer callers can supply measured usage for a soft-budget warning");
    require(!assess(policy,Snapshot{std::nullopt,600},std::numeric_limits<std::uint64_t>::max())->allowed,
            "peak overflow cannot bypass volume floor");

    require(parse_policy(config()).has_value(),"valid policy");
    require(parse_policy("# comment\n"+config("1_000")).has_value(),"standard TOML syntax");
    for (const auto& invalid : {std::string(""),config("true"),config("1000.0"),config("\"1000\""),
        config("0"),config("-1"),config("1000","1001"),config("1000","0"),
        config("1000","800","0"),config("9223372036854775808"),config()+"unknown=1\n",
        config()+"warning_bytes=1\n",config("1000","800","200","[1]"),
        config("1000","800","200","[\"relative\"]"),config("1000","800","200","true")})
      require(!parse_policy(invalid),"invalid TOML or policy must refuse");

    Temp temp;
    const auto root=temp.root/"project",extra=temp.root/"external";
    fs::create_directories(root/"many-files");
    fs::create_directory(extra);
    for (int i=0;i<100;++i) std::ofstream(root/"many-files"/("item-"+std::to_string(i)))<<"x";
    auto snapshot=inspect(root,policy);
    require(snapshot && !snapshot->allocated_bytes && snapshot->available_bytes>0,
            "lightweight inspect reports volume space and leaves project usage unknown");
    Policy with_extra=policy;with_extra.extra_owned_paths={extra};
    auto declared=inspect(root,with_extra);
    require(declared && !declared->allocated_bytes,"declared external directories are not recursively scanned");
    fs::create_directory_symlink(extra,root/"extra-link");
    with_extra.extra_owned_paths={root/"extra-link"};
    require(!inspect(root,with_extra),"symlink cannot masquerade as an owned directory");
    with_extra.extra_owned_paths={temp.root/"absent"};
    require(!inspect(root,with_extra),"missing owned root refuses");
    with_extra.extra_owned_paths={"relative"};
    require(!inspect(root,with_extra),"relative owned root refuses");
    with_extra.extra_owned_paths={"/dev"};
    require(!inspect(root,with_extra),"external volume refuses");
    require(!inspect(temp.root/"absent",policy),"missing project refuses");
    require(!inspect(root/"many-files"/"item-0",policy),"file is not a project directory");
    require(!load_policy(temp.root/"missing.toml"),"missing policy refuses");
    const auto policy_file=temp.root/"policy.toml";
    std::ofstream(policy_file)<<config();
    require(load_policy(policy_file).has_value(),"load policy from file");
    std::ofstream(policy_file)<<std::string(65537,' ');
    require(!load_policy(policy_file),"oversized policy refuses before parse");
    std::println("PASS: {} lightweight resource policy and volume assertions",checks);
    return 0;
  } catch (const std::exception& error) {
    std::println(stderr,"FAIL: {}",error.what());
    return 1;
  }
}
