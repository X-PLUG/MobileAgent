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

from unittest import mock

from absl.testing import absltest
from android_world.task_evals.utils import sqlite_schema_utils


class SchemaUtilsTest(absltest.TestCase):

  def generate_mock_item(self, title):
    """Utility function to generate a mock item with a specified title."""
    return sqlite_schema_utils.Recipe(title=title)

  def test_get_random_items_no_replacement(self):
    """Test generating items without replacement, ensuring no duplicates."""
    titles = ['Item 1', 'Item 1', 'Item 2', 'Item 3', 'Item 3']
    generate_item_fn = mock.Mock(
        side_effect=[self.generate_mock_item(title) for title in titles]
    )

    items = sqlite_schema_utils.get_random_items(
        n=3, generate_item_fn=generate_item_fn, replacement=False
    )

    self.assertLen(items, 3)
    self.assertEqual(
        {item.title for item in items}, set(['Item 1', 'Item 2', 'Item 3'])
    )
    generate_item_fn.assert_called()

  def test_get_random_items_with_replacement(self):
    """Test generating items with replacement, allowing duplicates."""
    generate_item_fn = mock.Mock(
        side_effect=[self.generate_mock_item('Item 1') for _ in range(3)]
    )

    items = sqlite_schema_utils.get_random_items(
        n=3, generate_item_fn=generate_item_fn, replacement=True
    )

    self.assertLen(items, 3)
    self.assertTrue(all(item.title == 'Item 1' for item in items))
    generate_item_fn.assert_called()

  def test_get_random_items_with_filter(self):
    """Test generating items with a filter function applied."""
    titles = ['Item 1', 'Item 2', 'Reject', 'Item 3']
    generate_item_fn = mock.Mock(
        side_effect=[self.generate_mock_item(title) for title in titles]
    )
    filter_fn = lambda item: item.title != 'Reject'

    items = sqlite_schema_utils.get_random_items(
        n=3,
        generate_item_fn=generate_item_fn,
        filter_fn=filter_fn,
        replacement=False,
    )

    self.assertLen(items, 3)
    self.assertNotIn('Reject', {item.title for item in items})
    generate_item_fn.assert_called()


if __name__ == '__main__':
  absltest.main()
