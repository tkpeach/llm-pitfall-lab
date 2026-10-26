// 原生 CLI 入口：连接任务评估、资源预检和检查点保存/核验，统一处理输入输出与退出状态。
// 业务校验留在对应模块，不在命令层修复答案或启动无关的 GPU 计算。

#include "lab_build_info.hpp"
#include "task/evaluator.hpp"
#include "resources/preflight.hpp"
#include "checkpoint/store.hpp"

#include <cstdio>
#include <charconv>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::string read_input(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error(std::string("cannot open input: ") + path);
  const auto size = input.tellg();
  // 在分配读取缓冲前限制小批量输入大小，避免误读大文件；这不替代 D2 的统一资源保护。
  constexpr std::streamoff limit = 16 * 1024 * 1024;
  if (size < 0 || size > limit)
    throw std::runtime_error("input is unreadable or exceeds 16 MiB");
  std::string contents(static_cast<std::size_t>(size), '\0');
  input.seekg(0);
  if (!input.read(contents.data(), static_cast<std::streamsize>(size)))
    throw std::runtime_error(std::string("cannot read input: ") + path);
  return contents;
}
}  // namespace

int main(int argc, char** argv) {
  const auto help = "Usage: pitfall-lab --version | --help\n"
                    "       pitfall-lab evaluate --gold GOLD.json --predictions PREDICTIONS.json\n"
                    "       pitfall-lab resources --root PROJECT --config POLICY.toml --peak-bytes BYTES\n"
                    "       pitfall-lab checkpoint-save --root PROJECT --config POLICY.toml --request REQUEST.json\n"
                    "       pitfall-lab checkpoint-verify --root PROJECT --name NAME --identity IDENTITY.json";
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::println("{}", help);
    return 0;
  }
  if (argc == 6 && std::string_view(argv[1]) == "evaluate" &&
      std::string_view(argv[2]) == "--gold" && std::string_view(argv[4]) == "--predictions") {
    // CLI 只连接文件输入、任务模块和报告输出；不在这里解析/修复答案，也不启动 GPU。
    try {
      auto report = lab::task::evaluate_batch(read_input(argv[3]), read_input(argv[5]));
      if (!report) {
        // 评估材料不合法与“答案得零分”不同：前者退出报错，后者仍正常生成评分报告。
        std::println(stderr, "Evaluation input error: {}", report.error());
        return 1;
      }
      // stdout 保持为可供工具读取的 JSON；故障诊断走 stderr，避免混入报告。
      std::println("{}", lab::task::report_json(*report));
      return 0;
    } catch (const std::exception& error) {
      std::println(stderr, "Evaluation failed: {}", error.what());
      return 1;
    }
  }
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::println("pitfall-lab {} (C++23, MLX SDK {})", lab::build::version, lab::build::mlx_version);
    return 0;
  }
  if (argc == 8 && std::string_view(argv[1]) == "resources" &&
      std::string_view(argv[2]) == "--root" && std::string_view(argv[4]) == "--config" &&
      std::string_view(argv[6]) == "--peak-bytes") {
    std::uint64_t peak{};
    const std::string_view value(argv[7]);
    const auto parsed = std::from_chars(value.data(), value.data()+value.size(), peak);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data()+value.size()) {
      std::println(stderr,"{}",help);
      return 2;
    }
    // 只读取配置、核算占用与报告决定；不创建额度预留、不删除文件，也不启动 GPU。
    const auto policy = lab::resources::load_policy(argv[5]);
    if (!policy) { std::println(stderr,"Resource policy error: {}",policy.error()); return 1; }
    const auto snapshot = lab::resources::inspect(argv[3],*policy);
    if (!snapshot) { std::println(stderr,"Resource inspection error: {}",snapshot.error()); return 1; }
    const auto decision = lab::resources::assess(*policy,*snapshot,peak);
    if (!decision) { std::println(stderr,"Resource decision error: {}",decision.error()); return 1; }
    std::println("{}",lab::resources::report_json(*policy,*snapshot,*decision));
    return decision->allowed ? 0 : 1;
  }
  if (argc==8 && std::string_view(argv[1])=="checkpoint-save" && std::string_view(argv[2])=="--root" &&
      std::string_view(argv[4])=="--config" && std::string_view(argv[6])=="--request") {
    try {
      const auto policy=lab::resources::load_policy(argv[5]);
      if (!policy) throw std::runtime_error(policy.error());
      const auto request=lab::checkpoint::parse_request(read_input(argv[7]));
      if (!request) throw std::runtime_error(request.error());
      const auto saved=lab::checkpoint::save(argv[3],*policy,*request);
      if (!saved) throw std::runtime_error(saved.error());
      std::println("{}",lab::checkpoint::report_json(*saved));return 0;
    } catch (const std::exception& error) {
      std::println(stderr,"Checkpoint save failed: {}",error.what());return 1;
    }
  }
  if (argc==8 && std::string_view(argv[1])=="checkpoint-verify" && std::string_view(argv[2])=="--root" &&
      std::string_view(argv[4])=="--name" && std::string_view(argv[6])=="--identity") {
    try {
      const auto identity=lab::checkpoint::parse_identity(read_input(argv[7]));
      if (!identity) throw std::runtime_error(identity.error());
      const auto checked=lab::checkpoint::verify(argv[3],argv[5],*identity);
      if (!checked) throw std::runtime_error(checked.error());
      std::println("{}",lab::checkpoint::report_json(*checked));return 0;
    } catch (const std::exception& error) {
      std::println(stderr,"Checkpoint verify failed: {}",error.what());return 1;
    }
  }
  std::println(stderr, "{}", help);
  return 2;
}
