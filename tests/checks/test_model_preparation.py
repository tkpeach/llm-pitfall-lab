#!/usr/bin/env python3
"""Exercise pinned-model transfer integrity and resumable ranges without network access."""
import importlib.util
import io
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("prepare_qwen35", Path(__file__).parents[2] / "tools/model_artifacts/qwen35/prepare_qwen35.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Response(io.BytesIO):
    def __init__(self, data, status=206, content_range=""):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


class TransferTests(unittest.TestCase):
    def test_truncated_ranges_resume_at_saved_offsets_and_reassemble(self):
        size = 64 * 1024**2
        block = bytes(range(256))
        starts = []
        lock = threading.Lock()

        def serve(request, timeout):
            begin, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            with lock:
                starts.append(begin)
            length = 1337 if begin % (32 * 1024**2) == 0 else end-begin+1
            data = (block * ((length+256)//256))[begin%256:begin%256+length]
            return Response(data, content_range=f"bytes {begin}-{end}/{size}")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transfer.partial"
            with patch.object(prepare.urllib.request, "urlopen", side_effect=serve):
                prepare.fetch("https://example.invalid/fixed", output, size)
            self.assertEqual(output.stat().st_size, size)
            with output.open("rb") as source:
                for _ in range(64):
                    self.assertEqual(source.read(1024**2), block * 4096)
            self.assertEqual(sorted(starts), [0, 1337, 32*1024**2, 32*1024**2+1337])
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_range_ignored_is_rejected_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transfer.partial"
            with patch.object(prepare.urllib.request, "urlopen", return_value=Response(b"wrong", status=200)):
                with self.assertRaisesRegex(ValueError, "did not honor"):
                    prepare.fetch("https://example.invalid/fixed", output, 64*1024**2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
