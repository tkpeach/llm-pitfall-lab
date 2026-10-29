import json, pathlib, hashlib
import mlx.core as mx
from mlx_lm.utils import load_model
from transformers import AutoTokenizer
root=pathlib.Path.cwd(); artifact=root/'runs/models/qwen35-4b-text-4bit';dest=root/'runs/numerical-validation/diagnostic/qkv-isolation-report.json'
mx.set_default_device(mx.gpu);mx.set_memory_limit(10*1024**3);mx.reset_peak_memory()
tok=AutoTokenizer.from_pretrained(artifact,local_files_only=True,trust_remote_code=False);messages=json.loads((root/'fixtures/qwen35-4b/numerical-input.json').read_text())['messages'];ids=tok.apply_chat_template(messages,tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
model,_=load_model(artifact);model.eval(); x=model.layers[0].input_layernorm(model.model.embed_tokens(mx.array([ids],mx.int32)));mx.eval(x)
qkv=model.layers[0].linear_attn.in_proj_qkv
full=qkv(x);one=mx.concatenate([qkv(x[:,i:i+1,:]) for i in range(len(ids))],axis=1);three=mx.concatenate([qkv(x[:,i:i+3,:]) for i in range(0,len(ids),3)],axis=1);mx.eval(full,one,three)
def stats(a,b):
 af=a.astype(mx.float32);bf=b.astype(mx.float32);d=af-bf;absd=mx.abs(d);mx.eval(absd);flat=mx.argmax(absd.reshape(-1)).item();T,D=a.shape[1],a.shape[2];token=flat//D;feature=flat%D; rms=mx.sqrt(mx.mean(d*d)).item()/max(mx.sqrt(mx.mean(bf*bf)).item(),1e-6);return {'equal':bool(mx.array_equal(a,b).item()),'scaled_rmse':rms,'max_abs':mx.max(absd).item(),'max_token_index_zero_based':token,'max_feature_index_zero_based':feature,'token_id':ids[token],'chunk_value':a[0,token,feature].item(),'full_value':b[0,token,feature].item(),'first_token_max_abs':mx.max(absd[:,0,:]).item()}
report={'input':'same materialized layer-0 inputnorm tensor and fixed in_proj_qkv weights, shared in one process','artifact_sha256':hashlib.sha256((artifact/'artifact.json').read_bytes()).hexdigest(),'token_ids':ids,'input_dtype':str(x.dtype),'input_shape':list(x.shape),'qkv_dtype':str(full.dtype),'token_vs_full':stats(one,full),'three_vs_full':stats(three,full),'peak_mlx_bytes':mx.get_peak_memory()}
dest.write_text(json.dumps(report,indent=2));print(json.dumps(report))
