#pragma once
#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace lab::task {
// 统一字段顺序贯穿记录、逐字段判分和报告；JSON 输入的键顺序不影响这一对应关系。
inline constexpr std::array<std::string_view, 6> field_names{
    "order_id", "product", "quantity", "unit_price_minor", "currency", "delivery_date"};
using Value = std::variant<std::monostate, std::string, std::int64_t>;
// 只表达任务允许的值，monostate 对应 JSON null；公开接口无需依赖 JSON 库的 DOM 类型。
struct Record {
  std::array<Value, 6> fields;
};
struct Validation {
  // 阶段标志用于定位失败层；仅全部通过时才提供可用于判分的 record。
  bool json_valid = false;
  bool schema_valid = false;
  bool type_valid = false;
  bool date_valid = false;
  std::optional<Record> record;
  std::string error;
};
struct Count {
  std::size_t correct = 0;
  std::size_t total = 0;
};
struct SampleResult {
  std::string id;
  Validation validation;
  std::array<bool, 6> fields_correct{};
  bool exact = false;
  double reward = 0;
};
struct Report {
  // 汇总计数保留真实分母；逐样例结果则用于追查错误，避免只有一个总分而无法诊断。
  Count json, schema, type, date, exact, fields, non_null, missing;
  std::array<Count, 6> per_field{}, per_field_non_null{}, per_field_missing{};
  std::vector<SampleResult> samples;
};

// 校验逐级累积：JSON → 六键结构 → 类型/范围 → 日期。不提取原文，也不修复模型答案。
Validation validate_output(std::string_view output);
// gold 必须来自成功的契约校验。单条评分与批次汇总分开，便于后续训练/评估复用。
SampleResult evaluate_record(const Record& gold, std::string_view output);
// 文件或 gold 损坏时返回输入错误；答案非法或缺失时仍生成报告，并把该样例计错。
std::expected<Report, std::string> evaluate_batch(std::string_view gold_json,
                                               std::string_view predictions_json);
std::string report_json(const Report& report);
}  // namespace lab::task
