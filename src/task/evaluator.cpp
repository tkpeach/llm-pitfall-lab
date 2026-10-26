// 将模型原始输出转为任务分数：严格解析 → 契约校验 → 单条判分 → 批次汇总。
// 同一解析器服务于评估文件和答案文本，调用层分别处理中止评估与保留错误样例。

#include "task/evaluator.hpp"
#include "lab_build_info.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace lab::task {
namespace {
using Json = nlohmann::json;
std::expected<Json, std::string> parse_json(std::string_view text) {
  // DOM 默认会覆盖同名键，解析完成后再检查已经来不及；用每层对象的键集合保留证据。
  // 回调看到的是解码后的键，因此转义写法不同但名称相同的键也会被拒绝。
  std::vector<std::set<std::string>> keys;
  bool duplicate = false;
  try {
    auto value = Json::parse(text.begin(), text.end(), [&](int depth, Json::parse_event_t event, Json& item) {
      // 限制异常嵌套的解析成本；只做校验，不通过丢弃元素来“修复”输入。
      if (depth > 128) throw std::runtime_error("JSON nesting exceeds 128 levels");
      if (event == Json::parse_event_t::object_start) keys.emplace_back();
      if (event == Json::parse_event_t::key && !keys.back().insert(item.get<std::string>()).second)
        duplicate = true;
      if (event == Json::parse_event_t::object_end) keys.pop_back();
      return true;
    });
    if (duplicate) return std::unexpected("duplicate JSON key");
    return value;
  } catch (const Json::exception& error) {
    return std::unexpected(error.what());
  } catch (const std::runtime_error& error) {
    return std::unexpected(error.what());
  }
}

bool exact_keys(const Json& value, const std::vector<std::string_view>& names) {
  return value.is_object() && value.size() == names.size() &&
      std::ranges::all_of(names, [&](auto name) { return value.contains(name); });
}

bool valid_date(std::string_view value) {
  // 先约束输出写法，再检查日历合法性；只有 YYYY-MM-DD 外形不能排除不存在的日期。
  if (value.size() != 10 || value[4] != '-' || value[7] != '-') return false;
  for (std::size_t i = 0; i < value.size(); ++i)
    if (i != 4 && i != 7 && (value[i] < '0' || value[i] > '9')) return false;
  const int year = (value[0]-'0')*1000 + (value[1]-'0')*100 + (value[2]-'0')*10 + value[3]-'0';
  const unsigned month = (value[5]-'0')*10 + value[6]-'0';
  const unsigned day = (value[8]-'0')*10 + value[9]-'0';
  return year >= 1 && std::chrono::year_month_day{
      std::chrono::year{year}, std::chrono::month{month}, std::chrono::day{day}}.ok();
}

Validation validate_value(const Json& value) {
  // 每通过一层才设置相应标志。失败即返回，后续汇总便能区分格式、类型和日期问题，
  // 同时保证不完整或非法记录不会进入字段比较。
  Validation result;
  result.json_valid = true;
  if (!exact_keys(value, {field_names.begin(), field_names.end()})) {
    result.error = "expected one object containing exactly the six task fields";
    return result;
  }
  result.schema_valid = true;
  Record record;
  for (std::size_t i = 0; i < field_names.size(); ++i) {
    const auto& v = value.at(field_names[i]);
    // 缺失信息是合法的 null，保留为 monostate，不推断默认值；遗漏键则已在结构层拒绝。
    if (v.is_null()) continue;
    if (i == 2 || i == 3) {
      // 只接收契约中的整数类型，并在转成 int64 前排除过大的无符号值。
      // 不把 1.0、数字字符串或 bool 自动转成整数，否则会掩盖模型的格式错误。
      if (!v.is_number_integer() ||
          (v.is_number_unsigned() && v.get<std::uint64_t>() > std::numeric_limits<std::int64_t>::max())) {
        result.error = std::string(field_names[i]) + " must be an int64 or null";
        return result;
      }
      const auto number = v.get<std::int64_t>();
      if (number < (i == 2 ? 1 : 0)) {
        result.error = std::string(field_names[i]) + " is outside its allowed range";
        return result;
      }
      record.fields[i] = number;
    } else {
      if (!v.is_string()) {
        result.error = std::string(field_names[i]) + " must be a string or null";
        return result;
      }
      const auto text = v.get<std::string>();
      if (i == 4 && text != "CNY" && text != "USD") {
        result.error = "currency must be CNY, USD or null";
        return result;
      }
      record.fields[i] = text;
    }
  }
  result.type_valid = true;
  if (const auto* date = std::get_if<std::string>(&record.fields[5]); date && !valid_date(*date)) {
    result.error = "delivery_date must be a real YYYY-MM-DD date in years 0001..9999";
    return result;
  }
  result.date_valid = true;
  result.record = std::move(record);
  return result;
}

void add(Count& count, bool correct) {
  ++count.total;
  count.correct += correct;
}
Json metric(const Count& count) {
  // 某批次可能没有 gold 非空字段；此时准确率未定义，用 null 而非 0 或 100% 表示。
  return {{"correct", count.correct}, {"total", count.total},
          {"rate", count.total ? Json(static_cast<double>(count.correct) / count.total) : Json(nullptr)}};
}
}  // namespace

Validation validate_output(std::string_view output) {
  auto parsed = parse_json(output);
  if (!parsed) {
    Validation result;
    result.error = parsed.error();
    return result;
  }
  return validate_value(*parsed);
}

SampleResult evaluate_record(const Record& gold, std::string_view output) {
  SampleResult result;
  result.validation = validate_output(output);
  // 非法答案整条及全部字段都计错：保留默认的 false/0，不能从其中抢救匹配字段。
  if (!result.validation.record) return result;
  std::size_t correct = 0;
  // 只比较规范值，不裁剪字符串或做二次归一化；归一化属于标注阶段的职责。
  for (std::size_t i = 0; i < field_names.size(); ++i) {
    result.fields_correct[i] = gold.fields[i] == result.validation.record->fields[i];
    correct += result.fields_correct[i];
  }
  result.exact = correct == field_names.size();
  // 字段准确率提供部分正确的信号，供后续受控奖励实验参考；它本身不是 RL 训练闭环。
  result.reward = static_cast<double>(correct) / field_names.size();
  return result;
}

std::expected<Report, std::string> evaluate_batch(std::string_view gold_json,
                                               std::string_view predictions_json) {
  // 先确认评估材料可信，再给答案打分。gold 或外层文件损坏会让分数失去意义，
  // 应中止评估；原始 output 内部的格式错误则属于模型错误，留到逐条判分处理。
  auto gold = parse_json(gold_json);
  auto predictions = parse_json(predictions_json);
  if (!gold) return std::unexpected("invalid gold JSON: " + gold.error());
  if (!predictions) return std::unexpected("invalid predictions JSON: " + predictions.error());
  if (!gold->is_array() || gold->empty()) return std::unexpected("gold must be a non-empty array");
  if (!predictions->is_array()) return std::unexpected("predictions must be an array");

  std::set<std::string> ids;
  std::vector<Record> records;
  // gold 确定评估全集和样例顺序；这里只验证标注契约，不判断答案是否忠实于原文。
  for (const auto& row : *gold) {
    if (!exact_keys(row, {"id", "input", "gold"}) || !row["id"].is_string() ||
        row["id"].get<std::string>().empty() || !row["input"].is_string())
      return std::unexpected("gold rows require non-empty string id, string input and gold object");
    const auto id = row["id"].get<std::string>();
    if (!ids.insert(id).second) return std::unexpected("duplicate gold id: " + id);
    auto validation = validate_value(row["gold"]);
    if (!validation.record) return std::unexpected("invalid gold record " + id + ": " + validation.error);
    records.push_back(*validation.record);
  }
  std::map<std::string, std::string> outputs;
  // 通过 ID 对齐而不依赖文件顺序。未知或重复 ID 不能被静默忽略，否则可能错配样例。
  // output 保持原始字符串，让围栏、解释文字和重复键等错误进入正式校验路径。
  for (const auto& row : *predictions) {
    if (!exact_keys(row, {"id", "output"}) || !row["id"].is_string() || !row["output"].is_string())
      return std::unexpected("prediction rows require string id and raw string output");
    const auto id = row["id"].get<std::string>();
    if (!ids.contains(id)) return std::unexpected("unknown prediction id: " + id);
    if (!outputs.emplace(id, row["output"].get<std::string>()).second)
      return std::unexpected("duplicate prediction id: " + id);
  }

  Report report;
  // 遍历 gold 而非预测集合，使漏答也进入分母，避免只统计已回答样例而虚高分数。
  for (std::size_t row = 0; row < records.size(); ++row) {
    const auto id = gold->at(row)["id"].get<std::string>();
    const auto prediction = outputs.find(id);
    SampleResult result;
    if (prediction == outputs.end()) result.validation.error = "missing prediction";
    else result = evaluate_record(records[row], prediction->second);
    result.id = id;
    add(report.json, result.validation.json_valid);
    add(report.schema, result.validation.schema_valid);
    add(report.type, result.validation.type_valid);
    add(report.date, result.validation.date_valid);
    add(report.exact, result.exact);
    // 同时观察字段总体和 gold 的空/非空分组，暴露“全部输出 null”获得表面高分的捷径。
    // 分组依据是 gold；若按预测是否为空分组，模型就能通过选择不回答改变统计对象。
    for (std::size_t i = 0; i < field_names.size(); ++i) {
      const bool correct = result.fields_correct[i];
      add(report.fields, correct);
      add(report.per_field[i], correct);
      if (std::holds_alternative<std::monostate>(records[row].fields[i])) {
        add(report.missing, correct);
        add(report.per_field_missing[i], correct);
      } else {
        add(report.non_null, correct);
        add(report.per_field_non_null[i], correct);
      }
    }
    report.samples.push_back(std::move(result));
  }
  return report;
}

std::string report_json(const Report& report) {
  // 这里只组织报告，不重新判分；整体指标、分组指标和逐条诊断共享同一份计算结果。
  Json fields = Json::object(), samples = Json::array();
  for (std::size_t i = 0; i < field_names.size(); ++i) {
    fields[field_names[i]] = {{"all", metric(report.per_field[i])},
        {"gold_non_null", metric(report.per_field_non_null[i])},
        {"gold_null", metric(report.per_field_missing[i])}};
  }
  for (const auto& sample : report.samples) {
    samples.push_back({{"id", sample.id}, {"valid", sample.validation.record.has_value()},
        {"json_valid", sample.validation.json_valid}, {"schema_valid", sample.validation.schema_valid},
        {"type_valid", sample.validation.type_valid}, {"date_valid", sample.validation.date_valid},
        {"error", sample.validation.error}, {"exact", sample.exact},
        {"fields_correct", sample.fields_correct}, {"candidate_reward", sample.reward}});
  }
  return Json{{"task_version", "chinese-order-v1"}, {"samples", samples},
      {"json_valid", metric(report.json)}, {"schema_valid", metric(report.schema)},
      {"type_valid", metric(report.type)}, {"date_valid", metric(report.date)},
      {"exact", metric(report.exact)}, {"fields", metric(report.fields)},
      {"gold_non_null", metric(report.non_null)}, {"gold_null", metric(report.missing)},
      {"per_field", fields}}.dump(2);
}
}  // namespace lab::task
