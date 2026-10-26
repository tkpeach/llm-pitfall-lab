"""验证检查点原子发布、损坏拒绝和失败后的旧版本保留。

二进制 payload 只是容器测试材料，不表示已具备模型张量序列化或训练恢复能力。
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
args=parser.parse_args()
binary=args.binary.resolve()
checks=0

def check(value,message):
    global checks
    checks+=1
    if not value:
        raise AssertionError(message)

with tempfile.TemporaryDirectory() as temporary:
    base=Path(temporary);root=base/'project';root.mkdir()
    policy=base/'policy.toml'
    policy.write_text('project_limit_bytes=16777216\nwarning_bytes=12582912\nmin_free_bytes=1\nextra_owned_paths=[]\n')
    payload=root/'source.bin';payload.write_bytes(bytes(range(256))*256)
    identity={'model':'toy-model-v1','data':'toy-data-v1','template':'toy-template-v1','configuration':'toy-config-v1'}
    identity_file=base/'identity.json';identity_file.write_text(json.dumps(identity))
    request={'name':'step-0001','identity':identity,'control':{'step':1,'data_position':4},
             'files':[{'name':'adapter.bin','source':str(payload)}],'peak_bytes':2097152}
    request_file=base/'request.json'
    def save(value,preexec_fn=None):
        request_file.write_text(json.dumps(value))
        return subprocess.run([str(binary),'checkpoint-save','--root',str(root),'--config',str(policy),
                               '--request',str(request_file)],cwd=base,capture_output=True,text=True,
                              timeout=30,preexec_fn=preexec_fn)
    def verify(name='step-0001',expected=identity):
        identity_file.write_text(json.dumps(expected))
        return subprocess.run([str(binary),'checkpoint-verify','--root',str(root),'--name',name,
                               '--identity',str(identity_file)],cwd=base,capture_output=True,text=True,timeout=30)
    saved=save(request);check(saved.returncode==0,saved)
    result=json.loads(saved.stdout)
    check(result['control']==request['control'],'control state mismatch')
    target=root/'runs/checkpoints/step-0001'
    check((target/'adapter.bin').read_bytes()==payload.read_bytes(),'binary copy mismatch')
    artifact=next(a for a in result['artifacts'] if a['name']=='adapter.bin')
    check(artifact['sha256']==hashlib.sha256(payload.read_bytes()).hexdigest(),'independent SHA-256 mismatch')
    check(verify().returncode==0,'new-process verification failed')
    for field in identity:
        changed={**identity,field:'different'}
        bad=verify(expected=changed)
        check(bad.returncode==1 and not bad.stdout,f'accepted mismatched {field}')
    check(save(request).returncode==1,'existing checkpoint overwritten')

    # 在已有有效版本之外建立损坏副本，逐一验证读取拒绝，不拿旧版本当故障实验材料。
    original_manifest=json.loads((target/'manifest.json').read_text())
    def corrupt(name,change):
        folder=root/'runs/checkpoints'/name;shutil.copytree(target,folder)
        change(folder)
        output=verify(name)
        check(output.returncode==1 and not output.stdout,f'corrupt checkpoint accepted: {name}')
    corrupt('bad-content',lambda p:(p/'adapter.bin').write_bytes(b'wrong'))
    def sparse_wrong_size(folder):
        with (folder/'adapter.bin').open('wb') as stream:
            stream.truncate(1 << 40)
    corrupt('sparse-wrong-size',sparse_wrong_size)
    corrupt('missing-control',lambda p:(p/'control.json').unlink())
    corrupt('extra-file',lambda p:(p/'unlisted.bin').write_bytes(b'extra'))
    def manifest_edit(folder,edit):
        data=copy.deepcopy(original_manifest);edit(data)
        (folder/'manifest.json').write_text(json.dumps(data))
    corrupt('bad-version',lambda p:manifest_edit(p,lambda m:m.update(format_version=2)))
    corrupt('bad-size',lambda p:manifest_edit(p,lambda m:m['files'][0].update(bytes='1')))
    corrupt('bad-path',lambda p:manifest_edit(p,lambda m:m['files'][0].update(name='../escape')))
    corrupt('duplicate-entry',lambda p:manifest_edit(p,lambda m:m['files'].append(m['files'][0])))
    corrupt('duplicate-key',lambda p:(p/'manifest.json').write_text('{"format_version":1,"format_version":1}'))
    def symlink_artifact(folder):
        (folder/'adapter.bin').unlink();(folder/'adapter.bin').symlink_to(payload)
    corrupt('linked-file',symlink_artifact)
    def fifo_artifact(folder):
        (folder/'adapter.bin').unlink();os.mkfifo(folder/'adapter.bin')
    corrupt('fifo-file',fifo_artifact)

    for name,alter in (
        ('bad-control',lambda r:r.update(control=[])),
        ('reserved-file',lambda r:r['files'][0].update(name='control.json')),
        ('escape-file',lambda r:r['files'][0].update(name='../escape')),
        ('missing-source',lambda r:r['files'][0].update(source=str(root/'absent'))),
    ):
        invalid=copy.deepcopy(request);invalid['name']=name;alter(invalid)
        check(save(invalid).returncode==1,f'invalid request accepted: {name}')
        check(not (root/'runs/checkpoints'/name).exists(),'failed save became visible')
    check(not list((root/'runs/checkpoints').glob('.pending-*')),'ordinary failures left incomplete stages')
    # 文件大小限制只作用于测试子进程，使真实 write 返回 EFBIG；生产代码不包含故障注入接口。
    def file_limit():
        signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096))
    failed=copy.deepcopy(request);failed['name']='write-failure'
    output=save(failed,preexec_fn=file_limit)
    check(output.returncode==1 and not output.stdout,'real partial-write failure was not handled')
    check(not (root/'runs/checkpoints/write-failure').exists(),'partial checkpoint published')
    check(not list((root/'runs/checkpoints').glob('.pending-*')),'write failure stage was not cleaned')
    check(verify().returncode==0,'I/O failure damaged old checkpoint')

    # 新进程可发布新的明确版本；写入路径不依赖跨进程独占租约。
    fresh=copy.deepcopy(request);fresh['name']='step-0002';fresh['control']['step']=2
    check(save(fresh).returncode==0,'new checkpoint could not be published')
    check(verify('step-0002').returncode==0 and verify().returncode==0,'new/old checkpoints not both valid')
    check(not (base/'escape').exists(),'path escaped project')
print(f'PASS: {checks} checkpoint CLI, corruption and partial-write assertions')
