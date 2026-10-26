"""Check that native resource preflight is read-only and distinguishes denial from input errors."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
binary=parser.parse_args().binary.resolve()
with tempfile.TemporaryDirectory() as temporary:
    base=Path(temporary); root=base/'project'; root.mkdir()
    (root/'artifact').write_bytes(b'x'*8192)
    policy=base/'resources.toml'
    valid='project_limit_bytes=1048576\nwarning_bytes=524288\nmin_free_bytes=1\nextra_owned_paths=[]\n'
    policy.write_text(valid)
    def invoke(peak='0',config=policy,directory=root):
        return subprocess.run([str(binary),'resources','--root',str(directory),'--config',str(config),
                               '--peak-bytes',peak],cwd=base,capture_output=True,text=True,timeout=30)
    before={str(p.relative_to(root)):p.read_bytes() for p in root.iterdir() if p.is_file()}
    allowed=invoke()
    assert allowed.returncode==0,allowed
    report=json.loads(allowed.stdout)
    assert report['allowed'] and not report['reservation_created'],report
    assert report['allocated_bytes'] is None and not report['project_usage_measured'],report
    assert report['operation']=='read_only_resource_preflight',report
    policy.write_text('project_limit_bytes=1\nwarning_bytes=1\nmin_free_bytes=1\nextra_owned_paths=[]\n')
    over_budget=invoke('0')
    assert over_budget.returncode==0,over_budget
    report=json.loads(over_budget.stdout)
    assert report['allowed'] and not report['warning'] and report['allocated_bytes'] is None,report
    policy.write_text(valid)
    denied=invoke(str(2**64-1))
    assert denied.returncode==1,denied
    report=json.loads(denied.stdout)
    assert not report['allowed'] and report['reason']=='volume_peak_exceeded',report
    for value in ('-1','+1','1.0','1e3','18446744073709551616',''):
        bad=invoke(value)
        assert bad.returncode==2 and not bad.stdout and 'Usage:' in bad.stderr,bad
    for kwargs in ({'config':base/'absent.toml'},{'directory':base/'absent'}):
        bad=invoke(**kwargs)
        assert bad.returncode==1 and not bad.stdout and bad.stderr,bad
    policy.write_text(valid.replace('1048576','true'))
    bad=invoke()
    assert bad.returncode==1 and not bad.stdout and bad.stderr,bad
    after={str(p.relative_to(root)):p.read_bytes() for p in root.iterdir() if p.is_file()}
    assert before==after and len(list(root.iterdir()))==1,'preflight must not write or delete artifacts'
print('PASS: native read-only preflight, overflow-safe denial and config/argument refusal')
