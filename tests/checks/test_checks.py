"""Meaningful refusal cases for navigation and native dependency boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmake" / "mlx"))
from check_project import check_commands, check_docs
from locate_sdk import sdk_info


class RepositoryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "README.md").write_text("[入口](docs/README.md)\n")
        (self.root / "AGENTS.md").write_text("# Instructions\n")
        (self.root / "docs/README.md").write_text("# 文档入口\n[设计](design.md#边界)\n")
        (self.root / "docs/design.md").write_text("# 设计\n## 边界\n")

    def test_valid_unicode_anchor_and_navigation(self):
        check_docs(self.root)

    def test_missing_document_is_rejected(self):
        (self.root / "docs/design.md").unlink()
        with self.assertRaisesRegex(ValueError, "broken"):
            check_docs(self.root)

    def test_missing_anchor_is_rejected(self):
        (self.root / "docs/design.md").write_text("# 设计\n")
        with self.assertRaisesRegex(ValueError, "heading"):
            check_docs(self.root)

    def test_unindexed_document_is_rejected(self):
        (self.root / "docs/unindexed.md").write_text("# Missing\n")
        with self.assertRaisesRegex(ValueError, "missing from"):
            check_docs(self.root)

    def test_research_source_and_include_paths_are_rejected(self):
        entry = dict(directory=str(self.root), file="src/main.cpp",
                     arguments=["c++", "-std=c++23", "-Iresearch/headers"])
        with self.assertRaisesRegex(ValueError, "research include"):
            check_commands(self.root, [entry])
        entry.update(file="research/old.cpp", arguments=["c++", "-std=c++23"])
        with self.assertRaisesRegex(ValueError, "research source"):
            check_commands(self.root, [entry])

    def test_old_or_extension_language_mode_is_rejected(self):
        for mode in ("c++17", "c++20", "gnu++23"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "C\\+\\+23"):
                check_commands(self.root, [dict(directory=str(self.root), file="src/main.cpp",
                                               arguments=["c++", f"-std={mode}"])])

    def test_clang_cxx23_alias_is_accepted(self):
        check_commands(self.root, [dict(directory=str(self.root), file="src/main.cpp",
                                       arguments=["c++", "-std=c++2b"])])

    def test_mismatched_sdk_is_rejected_without_gpu_import(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            sdk_info(self.root / "not-the-selected-sdk")


if __name__ == "__main__":
    unittest.main()
