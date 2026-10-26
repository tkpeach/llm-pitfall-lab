"""Ensure dataset checks reject corruption rather than merely accepting the current seed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from check_dataset import validate_dataset


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dataset'
        shutil.copytree(DATASET, self.root)

    def mutate(self, name, change, refresh=True):
        path = self.root / name
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value, ensure_ascii=False))
        if refresh:
            manifest = self.root / 'manifest.json'
            data = json.loads(manifest.read_text())
            data['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))

    def test_checksum_detects_unrecorded_edit(self):
        self.mutate('dev.json', lambda rows: rows[0].update(input='changed'), refresh=False)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_dataset(self.root)

    def test_gold_must_agree_with_cited_value(self):
        self.mutate('dev.json', lambda rows: rows[0]['gold'].update(quantity=999))
        with self.assertRaisesRegex(ValueError, 'contradicts gold'):
            validate_dataset(self.root)

    def test_evidence_must_exist_in_source(self):
        self.mutate('annotations.json', lambda rows: rows[0]['evidence'].update(product='不存在的商品'))
        with self.assertRaisesRegex(ValueError, 'evidence absent'):
            validate_dataset(self.root)

    def test_group_cannot_cross_splits(self):
        def cross(rows):
            source = next(r['group_id'] for r in rows if r['split'] == 'dev')
            target = next(r['group_id'] for r in rows if r['split'] == 'test')
            for row in rows:
                if row['group_id'] == target:
                    row['group_id'] = source
        self.mutate('annotations.json', cross)
        with self.assertRaisesRegex(ValueError, 'group crosses splits'):
            validate_dataset(self.root)

    def test_null_requires_reason(self):
        def remove(rows):
            row = next(r for r in rows if r['null_reasons'])
            row['null_reasons'].clear()
        self.mutate('annotations.json', remove)
        with self.assertRaisesRegex(ValueError, 'null reasons differ'):
            validate_dataset(self.root)

    def test_duplicate_input_is_rejected(self):
        self.mutate('dev.json', lambda rows: rows[1].update(input=rows[0]['input']))
        with self.assertRaisesRegex(ValueError, 'duplicate id or exact input'):
            validate_dataset(self.root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    DATASET = parser.parse_args().root
    unittest.main(argv=['dataset_checks'])
