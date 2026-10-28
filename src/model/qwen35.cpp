// 参考固定 MLX-LM 0.32.2 的 Qwen3.5 语言结构；专用计算使用 MLX 原生接口。
// 制品来源和摘要在张量加载前核验，不从研究目录或 Python 运行时加载模型。
#include "model/qwen35.hpp"
#include "compute/gated_delta.hpp"
#include "storage/files.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace lab::model {
namespace mx = mlx::core;
namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
constexpr auto repository = "Qwen/Qwen3.5-4B";
constexpr auto revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
const std::string prefix = "language_model.model.";
std::string layer_name(int i) { return prefix + "layers." + std::to_string(i) + "."; }
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::string read_file(const fs::path& directory, const std::string& file, std::size_t limit) {
  auto dir = storage::directory(directory);
  auto fd = storage::file_at(dir.get(), file);
  return storage::read_all(fd.get(), limit);
}
mx::array silu(const mx::array& x) {
  static auto operation=mx::compile([](const std::vector<mx::array>& a) {
    return std::vector<mx::array>{a[0]*mx::sigmoid(a[0])};
  },true);
  return operation({x})[0];
}
mx::array swiglu(const mx::array& gate,const mx::array& value) {
  static auto operation=mx::compile([](const std::vector<mx::array>& a) {
    return std::vector<mx::array>{a[0]*mx::sigmoid(a[0])*a[1]};
  },true);
  return operation({gate,value})[0];
}
mx::array compute_decay(const mx::array& a,const mx::array& bias,const mx::array& log_rate) {
  static auto operation=mx::compile([](const std::vector<mx::array>& x) {
    auto input=x[0]+x[1];
    return std::vector<mx::array>{mx::exp(-mx::exp(mx::astype(x[2],mx::float32))*
        mx::logaddexp(input,mx::array(0.f,input.dtype())))};
  },true);
  return operation({a,bias,log_rate})[0];
}
mx::array cut(const mx::array& x, int axis, int start, int end) {
  mx::Shape starts(x.ndim(), 0), stops = x.shape();
  starts[axis] = start;
  stops[axis] = end;
  return mx::slice(x, starts, stops);
}
}  // namespace

Qwen35Config Qwen35Config::parse(const std::string& text) {
  const auto root = Json::parse(text);
  require(root.at("model_type") == "qwen3_5", "expected Qwen3.5 conditional-generation configuration");
  const auto& t = root.at("text_config");
  const auto& rope = t.at("rope_parameters");
  for (const auto* name : {"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
      "num_key_value_heads", "head_dim", "vocab_size", "linear_num_key_heads", "linear_num_value_heads",
      "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "full_attention_interval",
      "max_position_embeddings"})
    require(t.at(name).is_number_integer(), std::string("expected integer text configuration: ")+name);
  Qwen35Config c{t.at("hidden_size"), t.at("intermediate_size"), t.at("num_hidden_layers"),
      t.at("num_attention_heads"), t.at("num_key_value_heads"), t.at("head_dim"), t.at("vocab_size"),
      t.at("linear_num_key_heads"), t.at("linear_num_value_heads"), t.at("linear_key_head_dim"),
      t.at("linear_value_head_dim"), t.at("linear_conv_kernel_dim"), t.at("full_attention_interval"),
      t.at("max_position_embeddings"), t.at("rms_norm_eps"), rope.at("rope_theta"), rope.at("partial_rotary_factor")};
  require(c.hidden == 2560 && c.intermediate == 9216 && c.layers == 32 && c.heads == 16 &&
      c.kv_heads == 4 && c.head_dim == 256 && c.vocab == 248320 && c.key_heads == 16 &&
      c.value_heads == 32 && c.key_dim == 128 && c.value_dim == 128 && c.conv_kernel == 4 &&
      c.attention_interval == 4 && c.max_positions == 262144 && c.eps == 1e-6f &&
      c.rope_base == 10000000.f && c.rotary_factor == 0.25f,
      "unsupported configuration: only the fixed Qwen3.5-4B text geometry is supported");
  require(t.at("tie_word_embeddings") == true && root.at("tie_word_embeddings") == true &&
      t.at("attention_bias") == false && t.at("attn_output_gate") == true &&
      t.at("hidden_act") == "silu" && t.at("dtype") == "bfloat16" &&
      t.at("model_type") == "qwen3_5_text" && rope.value("rope_type", rope.value("type", "")) == "default",
      "unsupported text attention, embedding or dtype configuration");
  const auto& types = t.at("layer_types");
  require(types.is_array() && types.size() == static_cast<std::size_t>(c.layers), "invalid layer_types");
  for (int i = 0; i < c.layers; ++i)
    require(types[i] == ((i+1)%c.attention_interval ? "linear_attention" : "full_attention"),
            "layer_types disagree with full_attention_interval");
  const auto& q = root.at("quantization");
  require(q.size() == 3 && q.at("bits") == 4 && q.at("group_size") == 64 && q.at("mode") == "affine",
          "expected uniform affine 4-bit group-64 weights");
  return c;
}

Qwen35::Qwen35(const fs::path& directory)
    : config_(Qwen35Config::parse(read_file(directory, "config.json", 64*1024))) {
  const auto manifest = Json::parse(read_file(directory, "artifact.json", 1024*1024));
  require(manifest.at("schema_version") == 1 && manifest.at("repository") == repository &&
      manifest.at("revision") == revision && manifest.at("component") == "language_only",
      "artifact source identity mismatch");
  const auto& conversion = manifest.at("conversion");
  require(conversion.at("mlx") == "0.32.4" && conversion.at("mlx_lm") == "0.32.2" &&
      conversion.at("bits") == 4 && conversion.at("group_size") == 64 &&
      conversion.at("mode") == "affine" && conversion.at("dtype") == "bfloat16",
      "unsupported artifact conversion identity");
  auto dir = storage::directory(directory);
  std::set<std::string> verified_files;
  std::uint64_t bytes = 0;
  for (const auto& [name, entry] : manifest.at("files").items()) {
    require(fs::path(name).filename() == name && name != "." && name != "..", "invalid artifact filename");
    auto fd = storage::file_at(dir.get(), name);
    const auto actual = storage::fingerprint(fd.get());
    require(actual.bytes == entry.at("bytes").get<std::uint64_t>() && actual.sha256 == entry.at("sha256").get<std::string>(),
            "artifact digest mismatch: " + name);
    bytes += actual.bytes;
    require(bytes <= 3ULL*1024*1024*1024, "artifact exceeds 3 GiB bound");
    verified_files.insert(name);
  }
  for (const auto* name : {"config.json", "model.safetensors.index.json", "tokenizer.json",
                           "tokenizer_config.json", "chat_template.jinja"})
    require(verified_files.contains(name), std::string("unverified required artifact file: ")+name);
  const auto index = Json::parse(read_file(directory, "model.safetensors.index.json", 1024*1024));
  std::set<std::string> shards;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    require(shard.is_string() && verified_files.contains(shard.get<std::string>()), "unverified tensor shard");
    shards.insert(shard.get<std::string>());
  }
  for (const auto& shard : shards) {
    auto tensors = mx::load_safetensors((directory/shard).string()).first;
    for (auto& [name, tensor] : tensors) {
      require(index.at("weight_map").contains(name) && index.at("weight_map").at(name) == shard,
              "tensor does not match shard index: " + name);
      require(weights_.emplace(name, std::move(tensor)).second, "duplicate tensor: " + name);
    }
  }
  require(weights_.size() == index.at("weight_map").size(), "missing indexed tensors");
  std::set<std::string> expected;
  auto check = [&](const std::string& name, mx::Shape shape, bool packed=false) {
    require(weights_.contains(name), "missing weight: " + name);
    const auto& w = weights_.at(name);
    require(w.shape() == shape, "weight shape mismatch: " + name);
    require(packed ? w.dtype() == mx::uint32 : (w.dtype() == mx::bfloat16 || w.dtype() == mx::float32),
            "weight dtype mismatch: " + name);
    expected.insert(name);
  };
  auto matrix = [&](const std::string& name, int out, int in) {
    check(name+".weight", {out, in/8}, true);
    check(name+".scales", {out, in/64});
    check(name+".biases", {out, in/64});
  };
  const auto& c = config_;
  matrix(prefix+"embed_tokens", c.vocab, c.hidden);
  check(prefix+"norm.weight", {c.hidden});
  for (int i=0; i<c.layers; ++i) {
    const auto p = layer_name(i);
    check(p+"input_layernorm.weight", {c.hidden});
    check(p+"post_attention_layernorm.weight", {c.hidden});
    matrix(p+"mlp.gate_proj", c.intermediate, c.hidden);
    matrix(p+"mlp.up_proj", c.intermediate, c.hidden);
    matrix(p+"mlp.down_proj", c.hidden, c.intermediate);
    if ((i+1)%c.attention_interval == 0) {
      matrix(p+"self_attn.q_proj", c.heads*c.head_dim*2, c.hidden);
      matrix(p+"self_attn.k_proj", c.kv_heads*c.head_dim, c.hidden);
      matrix(p+"self_attn.v_proj", c.kv_heads*c.head_dim, c.hidden);
      matrix(p+"self_attn.o_proj", c.hidden, c.heads*c.head_dim);
      check(p+"self_attn.q_norm.weight", {c.head_dim});
      check(p+"self_attn.k_norm.weight", {c.head_dim});
    } else {
      const int kd=c.key_heads*c.key_dim, vd=c.value_heads*c.value_dim, cd=2*kd+vd;
      matrix(p+"linear_attn.in_proj_qkv", cd, c.hidden);
      matrix(p+"linear_attn.in_proj_z", vd, c.hidden);
      matrix(p+"linear_attn.in_proj_a", c.value_heads, c.hidden);
      matrix(p+"linear_attn.in_proj_b", c.value_heads, c.hidden);
      matrix(p+"linear_attn.out_proj", c.hidden, vd);
      check(p+"linear_attn.conv1d.weight", {cd, c.conv_kernel, 1});
      check(p+"linear_attn.norm.weight", {c.value_dim});
      check(p+"linear_attn.A_log", {c.value_heads});
      check(p+"linear_attn.dt_bias", {c.value_heads});
    }
  }
  require(expected.size() == weights_.size(), "unexpected tensor outside fixed text model");
}

Qwen35State Qwen35::make_state() const {
  return {0, std::vector<LayerState>(config_.layers)};
}
mx::array Qwen35::linear(const mx::array& x, const std::string& name) const {
  return mx::quantized_matmul(x, weights_.at(name+".weight"), weights_.at(name+".scales"),
                              weights_.at(name+".biases"), true, 64, 4);
}
mx::array Qwen35::norm(const mx::array& x, const std::string& name) const {
  return mx::fast::rms_norm(x, weights_.at(name+".weight"), config_.eps);
}
mx::array Qwen35::attention(const mx::array& x, int i, int offset, LayerState& state) const {
  const auto& c=config_;
  const auto p=layer_name(i)+"self_attn.";
  const int n=x.shape(1);
  auto qg=mx::split(mx::reshape(linear(x,p+"q_proj"), {1,n,c.heads,c.head_dim*2}),2,-1);
  auto q=mx::transpose(norm(qg[0],p+"q_norm"), {0,2,1,3});
  auto k=mx::transpose(norm(mx::reshape(linear(x,p+"k_proj"),{1,n,c.kv_heads,c.head_dim}),p+"k_norm"),{0,2,1,3});
  auto v=mx::transpose(mx::reshape(linear(x,p+"v_proj"),{1,n,c.kv_heads,c.head_dim}),{0,2,1,3});
  q=mx::fast::rope(q, static_cast<int>(c.head_dim*c.rotary_factor), false, c.rope_base, 1.f, offset);
  k=mx::fast::rope(k, static_cast<int>(c.head_dim*c.rotary_factor), false, c.rope_base, 1.f, offset);
  if (state.first) {
    k=mx::concatenate({*state.first,k},2);
    v=mx::concatenate({*state.second,v},2);
  }
  state.first=k;state.second=v;
  // Explicit absolute positions also cover chunked prefill with a nonzero offset.
  auto mask=mx::expand_dims(mx::arange(offset,offset+n),1)>=mx::expand_dims(mx::arange(offset+n),0);
  auto out=mx::fast::scaled_dot_product_attention(q,k,v,1.f/std::sqrt(static_cast<float>(c.head_dim)),"",mask);
  out=mx::reshape(mx::transpose(out,{0,2,1,3}),{1,n,c.heads*c.head_dim});
  return linear(out*mx::sigmoid(mx::reshape(qg[1],{1,n,c.heads*c.head_dim})),p+"o_proj");
}
mx::array Qwen35::delta(const mx::array& x, int i, LayerState& state, const Observer& observer) const {
  const auto& c=config_;
  const auto p=layer_name(i)+"linear_attn.";
  const int n=x.shape(1),kd=c.key_heads*c.key_dim,vd=c.value_heads*c.value_dim,cd=2*kd+vd;
  auto trace=[&](const std::string& name,const mx::array& value) {
    if(observer) observer("layer."+std::to_string(i)+".delta."+name,value);
  };
  auto qkv=linear(x,p+"in_proj_qkv");
  trace("qkv",qkv);
  auto previous=state.first ? *state.first : mx::zeros({1,c.conv_kernel-1,cd},x.dtype());
  auto joined=mx::concatenate({previous,qkv},1);
  state.first=mx::contiguous(cut(joined,1,joined.shape(1)-(c.conv_kernel-1),joined.shape(1)));
  auto conv_pre=mx::conv1d(joined,weights_.at(p+"conv1d.weight"),1,0,1,cd);
  trace("conv_pre",conv_pre);
  auto conv=silu(conv_pre);
  trace("conv",conv);
  auto q=mx::reshape(cut(conv,2,0,kd),{1,n,c.key_heads,c.key_dim});
  auto k=mx::reshape(cut(conv,2,kd,2*kd),{1,n,c.key_heads,c.key_dim});
  auto v=mx::reshape(cut(conv,2,2*kd,cd),{1,n,c.value_heads,c.value_dim});
  const float inv=1.f/std::sqrt(static_cast<float>(c.key_dim));
  q=mx::array(inv*inv,q.dtype())*mx::fast::rms_norm(q,std::nullopt,1e-6f*inv*inv);
  k=mx::array(inv,k.dtype())*mx::fast::rms_norm(k,std::nullopt,1e-6f*inv*inv);
  auto decay=compute_decay(linear(x,p+"in_proj_a"),weights_.at(p+"dt_bias"),weights_.at(p+"A_log"));
  auto beta=mx::sigmoid(linear(x,p+"in_proj_b"));
  trace("q",q);trace("k",k);trace("v",v);trace("decay",decay);trace("beta",beta);
  // C++ SDK's native GDN accepts normalized q/k and multiplicative decay.
  auto updated=compute::gated_delta(q,k,v,decay,beta,state.second);
  state.second=updated[1];
  trace("raw_out",updated[0]);
  auto z=mx::reshape(linear(x,p+"in_proj_z"),{1,n,c.value_heads,c.value_dim});
  trace("z",z);
  auto normalized=norm(updated[0],p+"norm");
  trace("normalized",normalized);
  auto gated=mx::astype(mx::astype(normalized,mx::float32)*silu(mx::astype(z,mx::float32)),x.dtype());
  trace("gated",gated);
  auto output=linear(mx::reshape(gated,{1,n,vd}),p+"out_proj");
  trace("out",output);
  return output;
}
mx::array Qwen35::forward(const std::vector<int>& tokens, Qwen35State& state, const Observer& observer) const {
  const auto& c=config_;
  require(!tokens.empty() && tokens.size() <= static_cast<std::size_t>(c.max_positions), "invalid token count");
  require(state.offset>=0 && state.offset<=c.max_positions-static_cast<int>(tokens.size()) &&
      state.layers.size()==static_cast<std::size_t>(c.layers), "invalid request cache progress");
  for (int id:tokens) require(id>=0 && id<c.vocab, "token id outside vocabulary");
  for(int i=0;i<c.layers;++i) {
    const auto& s=state.layers[i];
    require(s.first.has_value()==s.second.has_value() && s.first.has_value()==(state.offset>0), "incomplete request cache");
    if (s.first) {
      const bool full=(i+1)%c.attention_interval==0;
      const mx::Shape first=full?mx::Shape{1,c.kv_heads,state.offset,c.head_dim}:mx::Shape{1,c.conv_kernel-1,2*c.key_heads*c.key_dim+c.value_heads*c.value_dim};
      const mx::Shape second=full?first:mx::Shape{1,c.value_heads,c.value_dim,c.key_dim};
      require(s.first->shape()==first && s.second->shape()==second,"request cache shape/progress mismatch");
      require(s.first->dtype()==mx::bfloat16 && s.second->dtype()==(full?mx::bfloat16:mx::float32),
              "request cache dtype mismatch");
    }
  }
  auto ids=mx::array(tokens.data(),{1,static_cast<int>(tokens.size())},mx::int32);
  const auto ep=prefix+"embed_tokens";
  // Gather packed embedding rows first; never expand the complete vocabulary matrix.
  auto x=mx::dequantize(mx::take(weights_.at(ep+".weight"),ids,0),
      mx::take(weights_.at(ep+".scales"),ids,0),mx::take(weights_.at(ep+".biases"),ids,0),64,4);
  if(observer) observer("embedding",x);
  for(int i=0;i<c.layers;++i) {
    const auto p=layer_name(i);
    auto input=norm(x,p+"input_layernorm");
    const auto trace_prefix="layer."+std::to_string(i)+".";
    if(observer) observer(trace_prefix+"inputnorm",input);
    auto mixed=(i+1)%c.attention_interval==0?attention(input,i,state.offset,state.layers[i]):delta(input,i,state.layers[i],observer);
    x=x+mixed;
    auto h=norm(x,p+"post_attention_layernorm");
    auto gate=linear(h,p+"mlp.gate_proj"), up=linear(h,p+"mlp.up_proj");
    auto glu=swiglu(gate,up);
    auto mlp_out=linear(glu,p+"mlp.down_proj");
    if(observer) {
      observer(trace_prefix+"postnorm",h);
      observer(trace_prefix+"mlp.gate",gate);observer(trace_prefix+"mlp.up",up);
      observer(trace_prefix+"mlp.glu",glu);observer(trace_prefix+"mlp.out",mlp_out);
    }
    x=x+mlp_out;
    if(observer) observer("layer."+std::to_string(i),x);
  }
  auto hidden=norm(x,prefix+"norm");
  state.offset+=static_cast<int>(tokens.size());
  if(observer) observer("final_hidden",hidden);
  return linear(hidden,ep);
}
TensorMap Qwen35::state_tensors(const Qwen35State& state) const {
  require(state.layers.size()==static_cast<std::size_t>(config_.layers),"invalid cache layer count");
  TensorMap result;
  for(int i=0;i<config_.layers;++i) {
    const auto& s=state.layers[i];
    const auto p="layer."+std::to_string(i)+".";
    const bool full=(i+1)%config_.attention_interval==0;
    if(s.first) result.emplace(p+(full?"keys":"conv"),*s.first);
    if(s.second) result.emplace(p+(full?"values":"recurrent"),*s.second);
  }
  return result;
}
}  // namespace lab::model
