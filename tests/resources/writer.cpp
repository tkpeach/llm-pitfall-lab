// 验证轻量文件会话的相对路径、防覆盖发布和失败清理。
#include "resources/writer.hpp"
#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <unistd.h>
namespace {
namespace fs=std::filesystem;
std::size_t checks=0;
void require(bool condition,std::string_view message) {
  ++checks;
  if (!condition) throw std::runtime_error(std::string(message));
}
}
int main() {
  auto pattern=(fs::temp_directory_path()/"pitfall-writer-XXXXXX").string();
  const auto* temp=::mkdtemp(pattern.data());
  if (!temp) return 1;
  const fs::path root=temp;
  try {
    fs::create_directory(root/"existing");
    std::ofstream(root/"existing/value")<<"user";
    auto started=lab::resources::WriteSession::begin(root);
    require(started.has_value(),"cannot begin write session");
    auto writer=std::move(*started);
    require(lab::resources::WriteSession::begin(root).has_value(),"concurrent session unexpectedly blocked");
    require(writer.ensure_directory("existing").has_value(),"cannot reuse existing directory");
    require(writer.create_directory("stage").has_value(),"cannot create stage");
    require(writer.write_file("stage/value","new").has_value(),"cannot write file");
    require(!writer.write_file("stage/value","overwrite"),"overwrote existing file");
    require(fs::exists(root/"existing/value"),"pre-existing user file changed");
    writer.discard_directory("stage");
    require(!fs::exists(root/"stage"),"owned staging directory was not cleaned");

    fs::create_directory(root/"outside");
    fs::create_directory_symlink(root/"outside",root/"jump");
    auto paths=lab::resources::WriteSession::begin(root);
    require(paths.has_value(),"cannot begin path test session");
    for (const auto& unsafe:{fs::path("../escape"),fs::path("/absolute"),fs::path("jump/escape")}) {
      auto attempt=lab::resources::WriteSession::begin(root);
      require(attempt && !attempt->create_directory(unsafe),"unsafe controlled path accepted");
    }
    require(!fs::exists(root/"outside/escape"),"followed directory symlink");

    auto publication=lab::resources::WriteSession::begin(root);
    require(publication.has_value(),"cannot begin publication session");
    auto publisher=std::move(*publication);
    require(publisher.create_directory("publish-stage") &&
            publisher.write_file("publish-stage/content","abc"),"prepare publication");
    require(publisher.publish("publish-stage","published").has_value(),"atomic publication failed");
    require(fs::exists(root/"published/content"),"published content missing");
    require(!publisher.publish("publish-stage","published"),"existing publication was replaced");

    std::println("PASS: {} lightweight file-session assertions",checks);
  } catch (const std::exception& error) {
    std::println(stderr,"FAIL: {}",error.what());
    fs::remove_all(root);
    return 1;
  }
  fs::remove_all(root);
  return 0;
}
