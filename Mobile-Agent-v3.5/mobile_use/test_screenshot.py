"""Tests for screenshot capture without a connected Android device."""

import importlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

try:
    importlib.import_module("qwen_vl_utils")
except ImportError:
    module = types.ModuleType("qwen_vl_utils")
    module.smart_resize = mock.MagicMock()
    sys.modules["qwen_vl_utils"] = module

from utils import AdbTools  # noqa: E402


class ScreenshotTest(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (12, 8), "red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.png = buffer.getvalue()

    @mock.patch("utils.time.sleep")
    @mock.patch("utils.subprocess.run")
    def test_retries_failed_and_invalid_captures(self, run, _sleep):
        run.side_effect = [
            subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"device offline"),
            subprocess.CompletedProcess([], 0, stdout=b"not an image", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=self.png, stderr=b""),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen shot.png"
            adb = AdbTools("path with spaces/adb", device="device-1")
            self.assertTrue(adb.get_screenshot(path))
            self.assertEqual(path.read_bytes(), self.png)
            self.assertEqual(adb.image_info, (12, 8))
            self.assertEqual(run.call_count, 3)
            self.assertEqual(
                run.call_args.args[0],
                ["path with spaces/adb", "-s", "device-1", "exec-out", "screencap", "-p"],
            )

    @mock.patch("utils.time.sleep")
    @mock.patch("utils.subprocess.run")
    def test_failed_capture_does_not_reuse_existing_screenshot(self, run, _sleep):
        run.return_value = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"offline")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screenshot.png"
            path.write_bytes(self.png)
            adb = AdbTools("adb")
            self.assertFalse(adb.get_screenshot(path))
            self.assertEqual(run.call_count, 3)
            self.assertEqual(path.read_bytes(), self.png)
            self.assertIsNone(adb.image_info)

    @unittest.skipIf(os.name == "nt", "executable test script requires POSIX")
    def test_capture_with_a_real_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            adb_path = Path(directory) / "fake adb"
            adb_path.write_text(
                f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write({self.png!r})\n"
            )
            adb_path.chmod(0o755)
            path = Path(directory) / "screen shot.png"
            adb = AdbTools(str(adb_path))
            self.assertTrue(adb.get_screenshot(path))
            self.assertEqual(path.read_bytes(), self.png)
            self.assertEqual(adb.image_info, (12, 8))


if __name__ == "__main__":
    unittest.main()
