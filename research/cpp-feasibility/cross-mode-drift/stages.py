import json, os, subprocess, tempfile, pathlib, sys, time
root=pathlib.Path.cwd(); artifact=root/'runs/models/qwen35-4b-text-4bit'; preflight=root/'runs/d2/build/pitfall-lab'; ref=root/'tools/numerical_validation/qwen35/reference.py'; dest=root/'runs/numerical-validation/diagnostic'; report=dest/'cross-mode-detail-report.json'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(artifact,local_files_only=True,trust_remote_code=False); msg=json.loads((root/'fixtures/qwen35-4b/numerical-input.json').read_text())['messages']; ids=t.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True)['input_ids']
orig=ref.read_text()
hooks = """    d=model.layers[0].linear_attn
    l=model.layers[0]
    selected={'inputnorm':l.input_layernorm,'qkv':d.in_proj_qkv,'conv':d.conv1d,'a':d.in_proj_a,'b':d.in_proj_b,'z':d.in_proj_z,'gatednorm':d.norm,'out':d.out_proj,'postnorm':l.post_attention_layernorm,'mlp_gate':l.mlp.gate_proj,'mlp_up':l.mlp.up_proj,'mlp_down':l.mlp.down_proj}
    names={id(v):k for k,v in selected.items()}
    stage_parts={}
    for cls in {type(v) for v in selected.values()}:
        original=cls.__call__
        def wrapper(obj,*a,_fn=original,**kw):
            result=_fn(obj,*a,**kw)
            if id(obj) in names:
                stage_parts.setdefault(names[id(obj)],[]).append(result)
            return result
        cls.__call__=wrapper
"""
obs=orig.replace('    layer_type = type(model.layers[0])',hooks+'    layer_type = type(model.layers[0])',1).replace('    observations["logits"] = logits','    for key, chunks in stage_parts.items():\n        observations["stage."+key]=mx.concatenate(chunks,axis=1)\n    observations["logits"] = logits',1)

assert obs!=orig and 'stage_parts.setdefault' in obs
with tempfile.TemporaryDirectory(prefix='layer-drift-',dir=dest) as temp:
 d=pathlib.Path(temp); (d/'instrumented.py').write_text(obs);(d/'input.json').write_text(json.dumps({'tokens':ids}))
 results={}; files={}
 for mode,script,chunk in [('base_full',ref,0),('observed_full',d/'instrumented.py',0),('base_token',ref,1),('observed_token',d/'instrumented.py',1)]:
  pre=subprocess.run([str(preflight),'resources','--root',str(root),'--config',str(root/'configs/resources.toml'),'--peak-bytes',str(1024**3)],capture_output=True,text=True,timeout=15,check=True); assert json.loads(pre.stdout)['allowed']
  out=d/f'{mode}.safetensors'; cmd=[sys.executable,str(script),'--artifact',str(artifact),'--input',str(d/'input.json'),'--output',str(out),'--chunk',str(chunk)]; p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
  if p.returncode: raise RuntimeError(f'{mode}: {p.stdout[-2000:]} {p.stderr[-2000:]}')
  results[mode]=json.loads(p.stdout.strip().splitlines()[-1]);files[mode]=out;print(mode,results[mode]['prefill_seconds'],flush=True)
 import mlx.core as mx
 a={k:mx.load(str(v)) for k,v in files.items()}
 def metrics(x,y):
  xx=x.astype(mx.float32);yy=y.astype(mx.float32);diff=mx.abs(xx-yy); mse=mx.sqrt(mx.mean((xx-yy)**2)).item();den=max(mx.sqrt(mx.mean(yy**2)).item(),1e-6);return {'max_abs':mx.max(diff).item(),'scaled_rmse':mse/den,'equal':bool(mx.array_equal(x,y).item())}
 verification={}
 for mode in ('full','token'):
  x=a['base_'+mode];y=a['observed_'+mode];names=sorted(set(x)&set(y)); diffs=[n for n in names if not mx.array_equal(x[n],y[n]).item()];verification[mode]={'matched':len(names),'different':diffs,'generated':results['base_'+mode]['generated'],'observed_generated':results['observed_'+mode]['generated']}
  if diffs or results['base_'+mode]['generated']!=results['observed_'+mode]['generated']:raise RuntimeError(f'observation changed {mode}: {diffs}')
 full=a['observed_full'];token=a['observed_token'];rows=[]
 names=['inputnorm','qkv','a','b','z','conv','gatednorm','out','postnorm','mlp_gate','mlp_up','mlp_down']
 for name in names:
  x=token['stage.'+name];y=full['stage.'+name];value=metrics(x,y);value.update(stage=name,shape=list(x.shape),first_token=next((j for j in range(len(ids)) if not mx.array_equal(x[:,j],y[:,j]).item()),None));rows.append(value)
 summary={'scope':'MLX-LM reference full vs token, layer 0 component hooks; baseline compared','token_ids':ids,'artifact_sha256':__import__('hashlib').sha256((artifact/'artifact.json').read_bytes()).hexdigest(),'workers':results,'verification':verification,'stages':rows,'logits':metrics(token['logits'],full['logits'])}
 report.write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('REPORT',report,flush=True)
