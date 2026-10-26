"""Audit small synthetic evaluation seeds; cited evidence checks are not human review."""
import argparse
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

FIELDS = ('order_id', 'product', 'quantity', 'unit_price_minor', 'currency', 'delivery_date')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def chinese_integer(text):
    # Only normalize the cited numeral, not the whole order: this is a label audit,
    # not a second extraction engine or a reference answer generator.
    if text.isascii() and text.isdecimal():
        return int(text)
    digits = dict(zip('零一二三四五六七八九两', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2)))
    total, number = 0, 0
    for char in text:
        if char in digits:
            number = digits[char]
        elif char in '十百':
            total += (number or 1) * {'十': 10, '百': 100}[char]
            number = 0
        else:
            raise ValueError(f'unsupported cited numeral: {text}')
    return total + number


def evidence_value(field, quote):
    if field == 'quantity':
        return chinese_integer(quote)
    if field == 'unit_price_minor':
        require(re.fullmatch(r'[0-9]+(?:\.[0-9]{1,2})?', quote), 'invalid amount evidence')
        return int(Decimal(quote) * 100)
    if field == 'currency':
        return {'人民币': 'CNY', 'CNY': 'CNY', '美元': 'USD', 'USD': 'USD'}[quote]
    if field == 'delivery_date':
        match = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', quote) or re.fullmatch(
            r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', quote)
        require(match, 'invalid date evidence')
        return date(*map(int, match.groups())).isoformat()
    return quote


def validate_dataset(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['task_version'] == 'chinese-order-v1', 'task identity differs')
    require(set(manifest['files']) == {'dev.json', 'test.json', 'annotations.json'}, 'unexpected dataset files')
    for name, expected in manifest['files'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'checksum differs: {name}')
    annotations = json.loads((root / 'annotations.json').read_text())
    meta = {row['id']: row for row in annotations}
    require(len(meta) == len(annotations), 'duplicate annotation id')
    seen, texts = set(), set()
    groups = defaultdict(list)
    order_groups = {}
    summary = {}
    for split in ('dev', 'test'):
        rows = json.loads((root / f'{split}.json').read_text())
        require(rows, f'{split} is empty')
        tags = Counter()
        for row in rows:
            require(set(row) == {'id', 'input', 'gold'}, 'unexpected evaluation envelope')
            identity, text, gold = row['id'], row['input'], row['gold']
            require(identity not in seen and text not in texts, 'duplicate id or exact input')
            require(identity in meta, 'missing annotation')
            seen.add(identity); texts.add(text)
            info = meta[identity]
            require(info['split'] == split, 'split metadata differs')
            require(set(gold) == set(FIELDS) and set(info['evidence']) == set(FIELDS), 'field coverage differs')
            require(set(info['null_reasons']) == {k for k in FIELDS if gold[k] is None}, 'null reasons differ')
            require(info['group_id'] and info['rule_tags'], 'missing group or rule tags')
            require(info['review_status'], 'missing review status')
            for field in FIELDS:
                quote, value = info['evidence'][field], gold[field]
                if value is None:
                    require(quote is None and info['null_reasons'][field], 'missing null rationale')
                else:
                    require(isinstance(quote, str) and quote and quote in text, f'evidence absent: {identity}/{field}')
                    if field in ('quantity', 'unit_price_minor'):
                        require(type(value) is int and value >= (1 if field == 'quantity' else 0), 'bad integer label')
                    require(evidence_value(field, quote) == value, f'evidence contradicts gold: {identity}/{field}')
            groups[info['group_id']].append((split, gold))
            if gold['order_id'] is not None:
                previous = order_groups.setdefault(gold['order_id'], info['group_id'])
                require(previous == info['group_id'], 'same order crosses groups')
            tags.update(info['rule_tags'])
        actual = {'samples': len(rows), 'groups': len({meta[r['id']]['group_id'] for r in rows}),
                  'nonnull_fields': sum(v is not None for r in rows for v in r['gold'].values()),
                  'null_fields': sum(v is None for r in rows for v in r['gold'].values())}
        require(actual == manifest['counts'][split], f'{split} counts differ')
        summary[split] = {**actual, 'rule_tag_counts': dict(sorted(tags.items()))}
    require(seen == set(meta), 'orphan annotation')
    for group, members in groups.items():
        require(len({split for split, _ in members}) == 1, f'group crosses splits: {group}')
        require(len(members) >= 2, f'group lacks paraphrase: {group}')
        require(all(gold == members[0][1] for _, gold in members), f'group gold differs: {group}')
    return summary


def exercise_native(root, binary):
    # Gold echo is a format/aggregation self-check, not evidence of model accuracy.
    result = {}
    with tempfile.TemporaryDirectory() as temp:
        for split in ('dev', 'test'):
            source = (root / f'{split}.json').resolve()
            rows = json.loads(source.read_text())
            outputs = [{'id': r['id'], 'output': json.dumps(r['gold'], ensure_ascii=False)} for r in rows]
            predictions = Path(temp) / f'{split}-predictions.json'
            predictions.write_text(json.dumps(outputs, ensure_ascii=False))
            run = subprocess.run([str(binary.resolve()), 'evaluate', '--gold', str(source),
                                  '--predictions', str(predictions)], cwd=temp, check=True,
                                 capture_output=True, text=True, timeout=30)
            report = json.loads(run.stdout)
            require(report['exact']['correct'] == len(rows) and report['fields']['correct'] == len(rows)*6,
                    'native evaluator rejects or miscounts seed gold')
            result[split] = {'self_check_samples': len(rows), 'all_gold_valid': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = validate_dataset(args.root)
        native = exercise_native(args.root, args.binary)
        print(json.dumps({'checks': summary, 'native_gold_self_check': native,
                          'independent_human_review': 'pending', 'model_quality_measured': False}, ensure_ascii=False))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'dataset check failed: {error}\n')
