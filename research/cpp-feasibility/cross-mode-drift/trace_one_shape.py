"""Run one shape in isolation for xctrace Metal System Trace."""
import json,pathlib,sys
import mlx.core as mx
from mlx_lm.utils import load_model
from transformers import AutoTokenizer
root=pathlib.Path.cwd();artifact=root/'runs/models/qwen35-4b-text-4bit';n=int(sys.argv[1])
if n not in (1,3,22):raise ValueError('supported shapes: 1, 3, 22')
mx.set_default_device(mx.gpu);mx.set_memory_limit(10*1024**3)
t=AutoTokenizer.from_pretrained(artifact,local_files_only=True,trust_remote_code=False)
messages=json.loads((root/'fixtures/qwen35-4b/numerical-input.json').read_text())['messages']
ids=t.apply_chat_template(messages,tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
model,_=load_model(artifact);model.eval();x=model.layers[0].input_layernorm(model.model.embed_tokens(mx.array([ids],mx.int32)));mx.eval(x)
start=2 if n==1 else 0;inp=mx.contiguous(x[:,start:start+n,:]);mx.eval(inp)
y=model.layers[0].linear_attn.in_proj_qkv(inp);mx.eval(y);mx.synchronize()
print(json.dumps({'tokens':n,'output':y[0,0 if n==1 else 2,2523].item()}),flush=True)
