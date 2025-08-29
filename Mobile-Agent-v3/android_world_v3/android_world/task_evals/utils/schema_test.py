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

from absl.testing import absltest
from android_world.task_evals.utils import schema


class SchemaTest(absltest.TestCase):

  def test_create(self):
    test_schema = schema.create([
        schema.string("file_name", is_required=True),
        schema.string("header"),
        schema.string("footer"),
        schema.string("replace_text"),
        schema.enum(
            "edit_type", ["header", "footer", "replace"], is_required=True
        ),
    ])

    self.assertEqual(
        test_schema,
        {
            "type": "object",
            "properties": {
                "file_name": {"type": "string"},
                "header": {"type": "string"},
                "footer": {"type": "string"},
                "replace_text": {"type": "string"},
                "edit_type": {
                    "type": "string",
                    "enum": ["header", "footer", "replace"],
                },
            },
            "required": ["file_name", "edit_type"],
        },
    )


if __name__ == "__main__":
  absltest.main()
