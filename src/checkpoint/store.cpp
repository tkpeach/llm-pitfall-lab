// 将检查点写入项目内独立暂存目录，回读校验清单后以不覆盖旧版本的原子重命名发布。
// 读取显式指定的已发布版本，拒绝损坏、额外文件和身份不一致，不尝试猜测或从头恢复。
#include "checkpoint/store.hpp"
#include "resources/writer.hpp"
#include "storage/files.hpp"
#include "lab_build_info.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <set>
#include <stdexcept>
#include <unistd.h>
namespace lab::checkpoint {
namespace {
namespace fs=std::filesystem;
namespace io=lab::storage;
using Json=nlohmann::json;
constexpr std::size_t metadata_limit=1024*1024;
Json json(std::string_view text) {
  if (text.size()>metadata_limit) throw std::runtime_error("checkpoint metadata exceeds 1 MiB");
  std::vector<std::set<std::string>> keys;
  auto result=Json::parse(text.begin(),text.end(),[&](int depth,Json::parse_event_t event,Json& value) {
    if (depth>128) throw std::runtime_error("checkpoint metadata nesting exceeds 128 levels");
    if (event==Json::parse_event_t::object_start) keys.emplace_back();
    if (event==Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
      throw std::runtime_error("duplicate checkpoint metadata key");
    if (event==Json::parse_event_t::object_end) keys.pop_back();
    return true;
  });
  return result;
}
void require(bool condition,std::string_view message) { if (!condition) throw std::runtime_error(std::string(message)); }
void exact_keys(const Json& value,const std::set<std::string>& names) {
  require(value.is_object() && value.size()==names.size(),"checkpoint metadata keys differ");
  for (const auto& name:names) require(value.contains(name),"checkpoint metadata key missing");
}
std::uint64_t integer(const Json& value) {
  require(value.is_number_unsigned() || (value.is_number_integer() && value.get<std::int64_t>()>=0),"expected non-negative integer metadata");
  return value.get<std::uint64_t>();
}
void version(const Json& value) { require(integer(value)==1,"unsupported checkpoint format version"); }
void name(std::string_view value,bool generation=false) {
  require(!value.empty() && value.size()<=(generation?64:128) &&
          ((value[0]>='A' && value[0]<='Z') || (value[0]>='a' && value[0]<='z') || (value[0]>='0' && value[0]<='9')),
          "invalid checkpoint or artifact name");
  for (char c:value)
    require((c>='A' && c<='Z') || (c>='a' && c<='z') || (c>='0' && c<='9') || c=='_' || c=='-' || (!generation && c=='.'),
            "unsafe checkpoint or artifact name");
}
Json identity_json(const Identity& identity) {
  require(!identity.model.empty() && !identity.data.empty() && !identity.template_id.empty() && !identity.configuration.empty(),
          "checkpoint identity fields must be non-empty");
  return {{"model",identity.model},{"data",identity.data},{"template",identity.template_id},{"configuration",identity.configuration}};
}
Identity identity_value(const Json& value) {
  exact_keys(value,{"model","data","template","configuration"});
  for (const auto& item:value) require(item.is_string(),"checkpoint identity must use strings");
  Identity identity{value.at("model"),value.at("data"),value.at("template"),value.at("configuration")};
  (void)identity_json(identity);return identity;
}
void must(std::expected<void,std::string> result) { if (!result) throw std::runtime_error(result.error()); }
io::Fd open_relative_directory(int root,const fs::path& relative) {
  auto parent=io::parent_at(root,relative);
  const int fd=::openat(parent.directory.get(),parent.name.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
  if (fd<0) io::fail("open checkpoint directory");
  return io::Fd(fd);
}
Verified verify_directory(int directory,const Identity& expected) {
  auto manifest_file=io::file_at(directory,"manifest.json");
  const auto manifest=json(io::read_all(manifest_file.get(),metadata_limit));
  exact_keys(manifest,{"format_version","identity","files"});version(manifest.at("format_version"));
  const auto identity=identity_value(manifest.at("identity"));
  require(identity==expected,"checkpoint identity mismatch");
  const auto& files=manifest.at("files");
  require(files.is_array() && files.size()>=2 && files.size()<=129,"checkpoint needs control and 1..128 artifact files");
  std::set<std::string> names{"manifest.json"};
  Verified result{{},identity,{}, {}};
  for (const auto& entry:files) {
    exact_keys(entry,{"name","bytes","sha256"});
    const auto filename=entry.at("name").get<std::string>();name(filename);
    require(names.insert(filename).second,"duplicate or reserved checkpoint file");
    const auto size=integer(entry.at("bytes"));
    const auto hash=entry.at("sha256").get<std::string>();
    require(hash.size()==64 && std::ranges::all_of(hash,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}),"invalid SHA-256 metadata");
    auto file=io::file_at(directory,filename);
    const auto info=io::stat_fd(file.get());
    require(info.st_size>=0 && static_cast<std::uint64_t>(info.st_size)==size,
            "checkpoint file size mismatch before hashing");
    const auto actual=io::fingerprint(file.get());
    require(actual.bytes==size && actual.sha256==hash,"checkpoint file size or SHA-256 mismatch");
    if (filename=="control.json") {
      const auto control=json(io::read_all(file.get(),metadata_limit));
      exact_keys(control,{"format_version","identity","state"});version(control.at("format_version"));
      require(identity_value(control.at("identity"))==identity,"control/manifest identity mismatch");
      require(control.at("state").is_object(),"checkpoint control state must be an object");
      result.control_json=control.at("state").dump();
    }
    result.artifacts.push_back({filename,size,hash});
  }
  require(names.contains("control.json"),"checkpoint control file missing");
  const auto entries=io::entries(directory);
  require(std::set<std::string>(entries.begin(),entries.end())==names,"checkpoint has missing or unexpected files");
  return result;
}
}  // namespace

std::expected<Verified,std::string> save(const fs::path& root,const resources::Policy& policy,const Request& request) {
  try {
    name(request.name,true);
    const auto identity=identity_json(request.identity);
    const auto state=json(request.control_json);
    require(state.is_object(),"control state must be a JSON object");
    require(!request.files.empty() && request.files.size()<=128,"checkpoint needs 1..128 artifact files");
    std::set<std::string> filenames{"control.json","manifest.json"};
    for (const auto& file:request.files) {
      name(file.name);require(filenames.insert(file.name).second,"duplicate/reserved artifact name");
      require(file.source.is_absolute() && file.source.native().find('\0')==std::string::npos,
              "artifact source must be an absolute path");
    }
    const auto snapshot=resources::inspect(root,policy);
    if (!snapshot) throw std::runtime_error(snapshot.error());
    const auto decision=resources::assess(policy,*snapshot,request.peak_bytes);
    if (!decision) throw std::runtime_error(decision.error());
    if (!decision->allowed) throw std::runtime_error("checkpoint peak does not fit volume safety floor");
    auto started=resources::WriteSession::begin(root);
    if (!started) throw std::runtime_error(started.error());
    auto writer=std::move(*started);
    must(writer.ensure_directory("runs"));must(writer.ensure_directory("runs/checkpoints"));
    const auto stage=fs::path("runs/checkpoints")/(".pending-"+std::to_string(::getpid())+"-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto target=fs::path("runs/checkpoints")/request.name;
    try {
      must(writer.create_directory(stage));
      must(writer.write_file(stage/"control.json",Json{{"format_version",1},{"identity",identity},{"state",state}}.dump()));
      for (const auto& file:request.files) {
        const auto source=fs::canonical(file.source);
        auto under=[](const fs::path& file,const fs::path& base) {
          const auto rel=file.lexically_relative(base);
          return !rel.empty() && *rel.begin()!="..";
        };
        bool owned=under(source,writer.root());
        for (const auto& extra:policy.extra_owned_paths) owned=owned || under(source,fs::canonical(extra));
        require(owned,"checkpoint source must belong to the project or declared owned roots");
        const auto forbidden=writer.root()/"research";
        auto relative=source.lexically_relative(forbidden);
        require(relative.empty() || *relative.begin()=="..","formal checkpoint cannot read research artifacts");
        must(writer.copy_file(file.source,stage/file.name));
      }
      auto root_directory=io::directory(writer.root());
      auto stage_directory=open_relative_directory(root_directory.get(),stage);
      Json entries=Json::array();
      auto record=[&](const std::string& filename) {
        auto file=io::file_at(stage_directory.get(),filename);const auto fp=io::fingerprint(file.get());
        entries.push_back({{"name",filename},{"bytes",fp.bytes},{"sha256",fp.sha256}});
      };
      record("control.json");for (const auto& file:request.files) record(file.name);
      must(writer.write_file(stage/"manifest.json",Json{{"format_version",1},{"identity",identity},{"files",entries}}.dump()));
      // 写后回读检查与恢复时使用同一校验逻辑，再发布已完整验证的暂存目录。
      auto result=verify_directory(stage_directory.get(),request.identity);
      must(writer.publish(stage,target));
      result.directory=writer.root()/target;
      return result;
    } catch (...) { writer.discard_directory(stage);throw; }
  } catch (const std::exception& error) { return std::unexpected(error.what()); }
}
std::expected<Verified,std::string> verify(const fs::path& root,std::string_view generation,const Identity& expected) {
  try {
    name(generation,true);(void)identity_json(expected);
    auto project=io::directory(fs::canonical(root));
    const auto relative=fs::path("runs/checkpoints")/std::string(generation);
    auto directory=open_relative_directory(project.get(),relative);
    auto result=verify_directory(directory.get(),expected);
    result.directory=fs::canonical(root)/relative;
    return result;
  } catch (const std::exception& error) { return std::unexpected(error.what()); }
}
std::expected<Identity,std::string> parse_identity(std::string_view text) {
  try { return identity_value(json(text)); }
  catch (const std::exception& error) { return std::unexpected(error.what()); }
}
std::expected<Request,std::string> parse_request(std::string_view text) {
  try {
    const auto request=json(text);
    exact_keys(request,{"name","identity","control","files","peak_bytes"});
    require(request.at("control").is_object(),"request control must be an object");
    require(request.at("files").is_array(),"request files must be an array");
    Request result{request.at("name").get<std::string>(),identity_value(request.at("identity")),
        request.at("control").dump(),{},integer(request.at("peak_bytes"))};
    for (const auto& file:request.at("files")) {
      exact_keys(file,{"name","source"});
      const auto path=file.at("source").get<std::string>();
      require(!path.empty() && fs::path(path).is_absolute() && path.find('\0')==std::string::npos,"artifact source must be an absolute path");
      result.files.push_back({file.at("name").get<std::string>(),path});
    }
    return result;
  } catch (const std::exception& error) { return std::unexpected(error.what()); }
}
std::string report_json(const Verified& checkpoint) {
  Json artifacts=Json::array();
  for (const auto& item:checkpoint.artifacts) artifacts.push_back({{"name",item.name},{"bytes",item.bytes},{"sha256",item.sha256}});
  return Json{{"directory",checkpoint.directory.string()},{"identity",identity_json(checkpoint.identity)},
      {"control",json(checkpoint.control_json)},{"artifacts",artifacts},{"format_version",1}}.dump(2);
}
}  // namespace lab::checkpoint
