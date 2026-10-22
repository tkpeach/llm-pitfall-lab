import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_pitfall_lab.checkpoint import load_checkpoint, save_checkpoint
from llm_pitfall_lab.resources import ResourceError, ResourceGuard, physical_usage
from llm_pitfall_lab.task import FIELDS, parse_answer, score_answer

ROOT = Path(__file__).resolve().parents[3]


class TaskContract(unittest.TestCase):
    def test_hand_reviewed_contract_examples(self):
        cases = json.loads((ROOT / "fixtures/task_contract.json").read_text())
        for case in cases:
            with self.subTest(case=case["id"]):
                result = score_answer(json.dumps(case["gold"], ensure_ascii=False), case["gold"])
                self.assertTrue(result["exact"])

    def test_strict_invalid_outputs_are_counted_as_failures(self):
        gold = dict.fromkeys(FIELDS)
        malformed = ["```json\n{}\n```", "{} trailing", '{"quantity":1,"quantity":2}',
                     json.dumps({**gold, "extra": 1}), json.dumps({**gold, "quantity": True}),
                     json.dumps({**gold, "unit_price_minor": 1.5}),
                     json.dumps({**gold, "delivery_date": "2026-02-30"}),
                     json.dumps({**gold, "currency": "EUR"}),
                     json.dumps({**gold, "quantity": float("nan")})]
        for text in malformed:
            with self.subTest(text=text):
                result = score_answer(text, gold)
                self.assertFalse(result["valid"])
                self.assertEqual(result["gold_null_count"], 6)
                self.assertEqual(result["reward"], 0)

    def test_all_null_does_not_hide_missing_content(self):
        gold = dict.fromkeys(FIELDS)
        gold.update(quantity=3, product="纸箱")
        result = score_answer(json.dumps(dict.fromkeys(FIELDS)), gold)
        self.assertFalse(result["exact"])
        self.assertEqual(result["gold_nonnull_correct"], 0)
        self.assertEqual(result["gold_nonnull_count"], 2)


class DiskAndCheckpoint(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/resources.toml").write_text(
            'project_limit_bytes = 100000000\nwarning_bytes = 90000000\n'
            'min_free_bytes = 0\nextra_owned_paths = []\n')
        self.guard = ResourceGuard(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_physical_count_deduplicates_hardlinks_and_overlapping_roots(self):
        first, second = self.root / "first", self.root / "second"
        first.write_bytes(b"x" * 8192)
        os.link(first, second)
        self.assertEqual(physical_usage([first, second]), first.stat().st_blocks * 512)
        self.assertEqual(physical_usage([self.root, first]), physical_usage([self.root]))

    def test_refuses_project_and_volume_overrun_before_writing(self):
        with self.assertRaises(ResourceError):
            self.guard.check(100000001)
        self.guard.policy["min_free_bytes"] = 100
        with patch("llm_pitfall_lab.resources.shutil.disk_usage") as usage:
            usage.return_value.free = 105
            with self.assertRaises(ResourceError):
                self.guard.check(6)
        with self.assertRaises(ValueError):
            self.guard.check(-1)

    def test_atomic_failure_preserves_last_valid_checkpoint(self):
        path = self.root / "run/checkpoint.json"
        identity = {"experiment": "a"}
        save_checkpoint(path, identity, {"step": 1}, self.guard)
        with patch("llm_pitfall_lab.checkpoint.os.replace", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                save_checkpoint(path, identity, {"step": 2}, self.guard)
        self.assertEqual(load_checkpoint(path, identity), {"step": 1})
        self.assertEqual(list(path.parent.glob(".checkpoint-*")), [])

    def test_rejects_identity_changes_and_corruption(self):
        path = self.root / "checkpoint.json"
        save_checkpoint(path, {"model": "4B"}, {"step": 1}, self.guard)
        with self.assertRaises(ValueError):
            load_checkpoint(path, {"model": "9B"})
        envelope = json.loads(path.read_text())
        envelope["payload"]["state"]["step"] = 999
        path.write_text(json.dumps(envelope))
        with self.assertRaises(ValueError):
            load_checkpoint(path, {"model": "4B"})


if __name__ == "__main__":
    unittest.main()
