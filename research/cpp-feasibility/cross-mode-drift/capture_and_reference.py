"""Bounded Metal capture and float64 audit for a fixed QKV output coordinate."""
import hashlib,json,os,pathlib,re,subprocess,tempfile,zlib,shutil
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load_model
from transformers import AutoTokenizer
root=pathlib.Path.cwd();artifact=root/'runs/models/qwen35-4b-text-4bit';dest=root/'runs/numerical-validation/diagnostic';binary=root/'runs/d2/build/pitfall-lab'
if os.environ.get('MTL_CAPTURE_ENABLED')!='1':raise RuntimeError('Metal capture must be enabled before process startup')
mx.set_default_device(mx.gpu);mx.set_memory_limit(10*1024**3);mx.reset_peak_memory()
t=AutoTokenizer.from_pretrained(artifact,local_files_only=True,trust_remote_code=False);messages=json.loads((root/'fixtures/qwen35-4b/numerical-input.json').read_text())['messages'];ids=t.apply_chat_template(messages,tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids'];model,_=load_model(artifact);model.eval()
x=model.layers[0].input_layernorm(model.model.embed_tokens(mx.array([ids],mx.int32)));q=model.layers[0].linear_attn.in_proj_qkv;mx.eval(x,q.weight,q.scales,q.biases)
report={'artifact_sha256':hashlib.sha256((artifact/'artifact.json').read_bytes()).hexdigest(),'input_shape':list(x.shape),'selected_token_index':2,'selected_feature_index':2523,'captures':{}}
with tempfile.TemporaryDirectory(prefix='metal-qkv-',dir=dest) as td:
 for n in (1,3,22):
  check=subprocess.run([str(binary),'resources','--root',str(root),'--config',str(root/'configs/resources.toml'),'--peak-bytes',str(4*1024**3)],capture_output=True,text=True,timeout=15,check=True)
  if not json.loads(check.stdout)['allowed']:raise RuntimeError('insufficient disk space')
  used_kib=int(subprocess.run(['du','-sk',str(root)],capture_output=True,text=True,check=True,timeout=60).stdout.split()[0])
  if used_kib*1024+4*1024**3>20*1024**3:raise RuntimeError('project budget insufficient for capture peak')
  start=2 if n==1 else 0;inp=mx.contiguous(x[:,start:start+n,:]);mx.eval(inp)
  path=pathlib.Path(td)/f'm{n}.gputrace';mx.metal.start_capture(str(path))
  try:
   y=q(inp);mx.eval(y);mx.synchronize()
  finally:mx.metal.stop_capture()
  idx=0 if n==1 else 2
  value=y[0,idx,2523].item(); found=set(); size=0; files=[]
  for f in path.rglob('*'):
   if not f.is_file():continue
   size+=f.stat().st_size; files.append(str(f.relative_to(path)))
   if f.stat().st_size < 1024**2:
    data=f.read_bytes()
    try:data+=zlib.decompressobj().decompress(data,2*1024**2)
    except zlib.error:pass
    for m in re.findall(rb'[A-Za-z0-9_]{3,180}',data):
     s=m.decode('ascii')
     if re.search(r'(?i)(qmv|qmm|quantized)',s):found.add(s)
  report['captures'][str(n)]={'value':value,'trace_bytes':size,'file_count':len(files),'candidate_strings':sorted(found)[:100]}
  print('captured',n,'bytes',size,'labels',sorted(found)[:20],flush=True);shutil.rmtree(path)
 # Reconstruct effective affine quantized row in float32, then sum products in float64 on CPU.
 row=2523; w=mx.dequantize(q.weight[row:row+1],q.scales[row:row+1].astype(mx.float32),q.biases[row:row+1].astype(mx.float32),64,4);v=x[0,2,:].astype(mx.float32);mx.eval(w,v)
 w_np=np.asarray(w).astype(np.float64).reshape(-1);v_np=np.asarray(v).astype(np.float64).reshape(-1);reference=float(np.dot(v_np,w_np));report['float64_reference']={'value':reference,'interpretation':'float64 dot of BF16 activation values and affine weights reconstructed via float32 dequantization; not original BF16 weights or a reference GPU kernel','weight_shape':list(w.shape),'weight_dtype':str(w.dtype),'activation_dtype':str(x.dtype),'abs_errors':{k:abs(v['value']-reference) for k,v in report['captures'].items()}}
 report['peak_mlx_bytes']=mx.get_peak_memory();(dest/'capture-and-reference-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
