import json, os, subprocess, tempfile, pathlib, sys, time
root=pathlib.Path.cwd(); artifact=root/'runs/models/qwen35-4b-text-4bit'; preflight=root/'runs/d2/build/pitfall-lab'; ref=root/'tools/numerical_validation/qwen35/reference.py'; dest=root/'runs/numerical-validation/diagnostic'; report=dest/'cross-mode-layer-report.json'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(artifact,local_files_only=True,trust_remote_code=False); msg=json.loads((root/'fixtures/qwen35-4b/numerical-input.json').read_text())['messages']; ids=t.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True)['input_ids']
orig=ref.read_text(); obs=orig.replace('    manual_first_layer = None','    layer_parts = {}\n    manual_first_layer = None',1).replace('        observations[f"layer.{index}"] = output','        observations[f"layer.{index}"] = output\n        layer_parts.setdefault(index, []).append(output)',1).replace('            observations = {}','            observations = {f"layer.{i}": mx.concatenate(parts, axis=1) for i, parts in layer_parts.items()}',1)
assert obs!=orig and 'layer_parts.setdefault' in obs
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
 for i in range(32):
  name=f'layer.{i}'; x=token[name];y=full[name];value=metrics(x,y); value.update(layer=i,first_token=next((j for j in range(len(ids)) if not mx.array_equal(x[:,j,:],y[:,j,:]).item()),None));rows.append(value)
 summary={'scope':'MLX-LM reference full vs token, 22 tokens; observer hook output and state checked against unmodified workers','artifact_sha256':__import__('hashlib').sha256((artifact/'artifact.json').read_bytes()).hexdigest(),'token_ids':ids,'workers':results,'verification':verification,'embedding':metrics(t.backend_tokenizer and mx.array([0]),mx.array([0])) if False else None,'layers':rows,'logits':metrics(token['logits'],full['logits']),'first_nonzero_layer':next((r['layer'] for r in rows if not r['equal']),None)}
 report.write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('REPORT',report,flush=True)
