# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GUI-Owl image payload encoding."""

import base64
from io import BytesIO

from absl.testing import absltest
from PIL import Image

from android_world.agents import gui_owl


class ImageEncodingTest(absltest.TestCase):

  def test_pil_to_base64_returns_png_data_uri(self):
    screenshot = Image.new('RGB', (2, 3), color='white')
    payload = gui_owl.pil_to_base64(screenshot)
    prefix, encoded = payload.split(',', 1)

    self.assertEqual(prefix, 'data:image/png;base64')
    with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
      self.assertEqual(decoded.format, 'PNG')
      self.assertEqual(decoded.size, (2, 3))


if __name__ == '__main__':
  absltest.main()
