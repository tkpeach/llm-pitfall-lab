// 隔离验证订单输出契约和单条计分：合法基准、非法变体、类型/日期边界与部分得分。
// 直接调用正式任务模块，检查评分规则本身，不生成模型答案或衡量模型效果。

#include "task/evaluator.hpp"
#include <algorithm>
#include <cmath>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// 从合法基准只改一个字段，隔离触发条件；测试关注契约行为，而非解析器内部实现。
const std::string all_null = R"({"order_id":null,"product":null,"quantity":null,"unit_price_minor":null,"currency":null,"delivery_date":null})";
std::size_t checks = 0;
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
std::string changed(std::string_view field, std::string_view value) {
  auto text = all_null;
  const auto begin = text.find('"' + std::string(field) + "\":") + field.size() + 3;
  text.replace(begin, 4, value);
  return text;
}
void invalid(std::string_view text, bool json, bool schema, bool type) {
  // 不仅检查“被拒绝”，还检查在哪层失败，防止诊断指标与真实校验路径脱节。
  const auto result = lab::task::validate_output(text);
  require(!result.record && !result.error.empty() && !result.date_valid &&
          result.json_valid == json && result.schema_valid == schema && result.type_valid == type,
          "wrong refusal or stage for: " + std::string(text));
}
}  // namespace
int main() {
  try {
    using namespace lab::task;
    // 先建立计分基线：合法等价写法不扣分；部分正确有部分分；非法答案不能保留任何字段分。
    const auto gold = validate_output(all_null);
    require(gold.record.has_value(), "all-null record must be valid");
    auto perfect = evaluate_record(*gold.record, " \n" + all_null + "\t");
    require(perfect.exact && perfect.reward == 1, "whitespace must not change scoring");
    const auto different = evaluate_record(*gold.record, changed("product", R"("纸箱")"));
    require(!different.exact && std::abs(different.reward - 5.0/6) < 1e-12 &&
            !different.fields_correct[1], "partial reward must be field accuracy");
    const auto broken = evaluate_record(*gold.record, changed("quantity", "true"));
    require(broken.reward == 0 && !broken.exact &&
            std::ranges::none_of(broken.fields_correct, [](bool b) { return b; }),
            "invalid output must make even matching null fields wrong");

    // 区分 JSON 文本不合法与任务结构不合法：两者都不给内容分，但阶段统计不同。
    for (const auto& text : std::vector<std::string>{
        "", "not json", "```json\n"+all_null+"\n```", all_null+" explanation",
        all_null+all_null, "{\"a\":NaN}", "{\"a\":Infinity}", "{\"a\":1e999}",
        "{\"a\":null,}", "{/*comment*/\"a\":null}",
        "{\"order_id\":null,\"order_\\u0069d\":null}",
        "{\"a\":{\"b\":1,\"b\":2}}", "{\"a\":\"\\uD800\"}"})
      invalid(text, false, false, false);
    auto duplicate = all_null;
    duplicate.insert(1, "\"order_id\":\"second\",");
    invalid(duplicate, false, false, false);
    auto bad_utf8 = changed("product", "\"x\"");
    bad_utf8[bad_utf8.find('x')] = static_cast<char>(0xff);
    invalid(bad_utf8, false, false, false);
    for (auto text : {std::string("[]"), std::string("null"), std::string("{}"),
                      std::string("{\"extra\":null,")+all_null.substr(1)})
      invalid(text, true, false, false);

    // 用边界与相近类型检查严格性，避免隐式转换或“长得像日期”绕过契约。
    for (auto value : {"true", "false", "0", "-1", "1.0", "1e0", "\"三\"", "[]",
                       "9223372036854775808", "-9223372036854775809"})
      invalid(changed("quantity", value), true, true, false);
    for (auto value : {"-1", "true", "0.0", "\"3990\""})
      invalid(changed("unit_price_minor", value), true, true, false);
    for (auto field : {"order_id", "product", "currency", "delivery_date"})
      invalid(changed(field, "42"), true, true, false);
    for (auto value : {"\"EUR\"", "\"cny\"", "\"$\""})
      invalid(changed("currency", value), true, true, false);
    for (auto value : {"\"2026-02-29\"", "\"1900-02-29\"", "\"2026-04-31\"",
                       "\"0000-01-01\"", "\"2026-13-01\"", "\"2026-01-00\"",
                       "\"2026-1-01\"", "\"明天\"", "\"2026-01-01T00:00:00\""})
      invalid(changed("delivery_date", value), true, true, true);
    for (auto value : {"\"2000-02-29\"", "\"2024-02-29\"", "\"0001-01-01\"", "\"9999-12-31\""})
      require(validate_output(changed("delivery_date", value)).record.has_value(), "valid Gregorian date");
    require(validate_output(changed("quantity", "9223372036854775807")).record.has_value(), "int64 boundary");
    require(validate_output(changed("unit_price_minor", "0")).record.has_value(), "zero unit price");
    auto unicode = validate_output(changed("product", "\"\\u7eb8\\u7bb1\""));
    require(unicode.record && std::get<std::string>(unicode.record->fields[1]) == "纸箱", "JSON Unicode decoding");
    const auto exact_string = evaluate_record(*unicode.record, changed("product", "\"纸箱 \""));
    require(!exact_string.exact && !exact_string.fields_correct[1], "do not trim answers");
    invalid(std::string(130, '[')+"0"+std::string(130, ']'), false, false, false);
    std::println("PASS: {} strict parsing, type, Gregorian date and scoring assertions", checks);
    return 0;
  } catch (const std::exception& error) {
    std::println(stderr, "FAIL: {}", error.what());
    return 1;
  }
}
