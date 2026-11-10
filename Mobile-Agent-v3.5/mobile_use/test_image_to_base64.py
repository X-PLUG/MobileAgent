"""
image_to_base64() must open the paths build_messages actually stores, and
resize with the same factor coordinate scaling uses.

Run from this directory:
    python -m unittest test_image_to_base64 -v

build_messages prefixes every screenshot with "file://". Passing that string
to Pillow raises OSError, so the mobile run retries "Error calling LLM" and
never reaches the model. computer_use strips the prefix; this path did not.

Coordinate scaling in run_gui_owl_1_5_for_mobile.py calls smart_resize with
factor=16. Encoding with factor=28 (the old default here) feeds the model a
different grid than the one clicks are mapped back onto — the same mismatch
fixed for computer_use in issue #265.
"""

import base64
import importlib
import os
import sys
import tempfile
import types
import unittest
from io import BytesIO


def _stub_missing(name, **attrs):
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_missing("openai", OpenAI=object)
_stub_missing("numpy", ndarray=object)
_stub_missing("qwen_vl_utils", smart_resize=lambda *a, **k: (1, 1))

from PIL import Image  # noqa: E402

from utils import image_to_base64, smart_resize  # noqa: E402


# A 1080x2400 phone screenshot. factor 16 and factor 28 round to different
# sizes, so the test fails if encoding is still on the old factor.
_WIDTH, _HEIGHT = 1080, 2400
_MIN_PIXELS = 3136
_MAX_PIXELS = 10035200


def _png(path):
    Image.new("RGB", (_WIDTH, _HEIGHT), (10, 20, 30)).save(path)


def _decoded_size(uri):
    raw = base64.b64decode(uri.split(",", 1)[1])
    return Image.open(BytesIO(raw)).size


class ImageToBase64Test(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        _png(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_file_url_prefix_opens(self):
        uri = image_to_base64("file://" + self.path)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        h16, w16 = smart_resize(
            _HEIGHT, _WIDTH, factor=16,
            min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS,
        )
        h28, w28 = smart_resize(
            _HEIGHT, _WIDTH, factor=28,
            min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS,
        )
        self.assertNotEqual((h16, w16), (h28, w28))
        self.assertEqual(_decoded_size(uri), (w16, h16))

    def test_plain_path_still_opens(self):
        uri = image_to_base64(self.path)
        self.assertTrue(uri.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
