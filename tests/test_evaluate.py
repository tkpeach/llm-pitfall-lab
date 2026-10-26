"""Exercise the native evaluator against hand-auditable task fixtures."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--fixtures', type=Path, required=True)
args = parser.parse_args()
binary = args.binary.resolve()
fixtures = json.loads(args.fixtures.read_text())
# Python 只准备输入并检查报告，评分仍由原生 CLI 完成；临时目录隔离对仓库工作目录的依赖。
with tempfile.TemporaryDirectory() as directory:
    cwd = Path(directory)
    gold_path, pred_path = cwd / 'gold.json', cwd / 'predictions.json'
    def invoke(gold, predictions):
        gold_path.write_text(json.dumps(gold, ensure_ascii=False))
        pred_path.write_text(json.dumps(predictions, ensure_ascii=False))
        return subprocess.run([str(binary), 'evaluate', '--gold', str(gold_path),
                               '--predictions', str(pred_path)], cwd=cwd,
                              capture_output=True, text=True, timeout=30)
    # 同时改变键顺序、转义和样例顺序，确认比较的是规范值，并按 ID 对齐。
    perfect = [{'id': row['id'], 'output': json.dumps(dict(reversed(list(row['gold'].items()))), ensure_ascii=True, indent=2)}
               for row in fixtures]
    result = invoke(fixtures, list(reversed(perfect)))
    assert result.returncode == 0, result
    report = json.loads(result.stdout)
    assert report['exact'] == {'correct': 6, 'total': 6, 'rate': 1.0}, report
    assert report['fields'] == {'correct': 36, 'total': 36, 'rate': 1.0}, report
    for key in ('json_valid', 'schema_valid', 'type_valid', 'date_valid'):
        assert report[key]['rate'] == 1, report
    assert report['gold_non_null']['total'] == 23, report
    assert report['gold_null']['total'] == 13, report
    assert all(s['candidate_reward'] == 1 for s in report['samples'])

    # 全 null 可以匹配缺失字段，但不能掩盖非空字段全错；漏答则连匹配 null 的分也没有。
    all_null = {key: None for key in fixtures[0]['gold']}
    result = invoke(fixtures, [{'id': r['id'], 'output': json.dumps(all_null)} for r in fixtures])
    report = json.loads(result.stdout)
    assert report['exact']['correct'] == 1, report
    assert report['gold_non_null']['correct'] == 0, report
    assert report['gold_null']['correct'] == 13, report
    assert report['fields']['correct'] == 13, report

    result = invoke(fixtures, [])
    report = json.loads(result.stdout)
    assert report['exact']['total'] == 6 and report['fields']['total'] == 36, report
    assert report['fields']['correct'] == 0 and report['json_valid']['correct'] == 0, report
    assert all(s['error'] == 'missing prediction' for s in report['samples'])

    bad = copy.deepcopy(perfect)
    bad[0]['output'] = '```json\n' + bad[0]['output'] + '\n```'
    result = invoke(fixtures, bad)
    report = json.loads(result.stdout)
    assert report['exact']['correct'] == 5 and report['fields']['correct'] == 30, report
    assert report['samples'][0]['candidate_reward'] == 0, report

    # 阶段指标累积、分母固定；无论类型还是日期失败，全部六个字段都必须计错。
    for field, value, expected in (
        ('quantity', True, (1, 1, 0, 0)),
        ('delivery_date', '2026-02-29', (1, 1, 1, 0)),
        ('extra', None, (1, 0, 0, 0)),
    ):
        answer = copy.deepcopy(fixtures[0]['gold'])
        answer[field] = value
        result = invoke(fixtures[:1], [{'id': fixtures[0]['id'], 'output': json.dumps(answer)}])
        report = json.loads(result.stdout)
        assert tuple(report[k]['correct'] for k in ('json_valid','schema_valid','type_valid','date_valid')) == expected, report
        assert report['fields']['correct'] == 0 and report['fields']['total'] == 6, report
    # 分组没有样本时，比率应为 null，不能伪装成失败或满分。
    result = invoke(fixtures[-1:], perfect[-1:])
    report = json.loads(result.stdout)
    assert report['gold_non_null'] == {'correct': 0, 'total': 0, 'rate': None}, report

    # 模型错误要进入得分，评估材料错误则中止；从退出码和 stdout 验证这一职责边界。
    for predictions in (perfect + [perfect[0]], [{'id': 'unknown', 'output': '{}'}],
                        [{'id': fixtures[0]['id'], 'output': {}}]):
        result = invoke(fixtures, predictions)
        assert result.returncode == 1 and not result.stdout and result.stderr, result
    for gold in ([], fixtures + [fixtures[0]], [{**fixtures[0], 'gold': {'quantity': True}}]):
        result = invoke(gold, [])
        assert result.returncode == 1 and not result.stdout and result.stderr, result
    result = subprocess.run([str(binary), 'evaluate', '--gold', str(cwd / 'absent'),
                             '--predictions', str(pred_path)], capture_output=True, text=True)
    assert result.returncode == 1 and not result.stdout, result
    for arguments in (('evaluate',), ('evaluate', '--gold', str(gold_path)),
                      ('evaluate', '--other', str(gold_path), '--predictions', str(pred_path))):
        result = subprocess.run([str(binary), *arguments], capture_output=True, text=True)
        assert result.returncode == 2 and 'Usage:' in result.stderr, result
print('PASS: six fixture golds, ID matching, null baseline, invalid/missing denominators and input refusal')
