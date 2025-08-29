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

import os
import tempfile
from absl.testing import absltest
from android_world import checkpointer


class CheckpointerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.checkpointer = checkpointer.IncrementalCheckpointer(
        directory=self.temp_dir.name
    )

  def tearDown(self) -> None:
    super().tearDown()
    self.temp_dir.cleanup()

  def test_save_and_load_valid_data(self) -> None:
    """Tests if save and load work as expected with valid data."""
    task_group1 = [{'key': 'value1'}]
    task_group2 = [{'key': 'value2'}]
    self.checkpointer.save_episodes(task_group1, 'task_group1')
    self.checkpointer.save_episodes(task_group2, 'task_group2')
    loaded_data = self.checkpointer.load()
    self.assertCountEqual(loaded_data, task_group1 + task_group2)

  def test_load_empty_directory(self) -> None:
    """Tests if loading an empty directory returns empty data."""
    loaded_data = self.checkpointer.load()
    self.assertEqual([], loaded_data)

  def test_overwrite_existing_task_group(self) -> None:
    """Tests if save overwrites an existing task group."""
    initial_data = [{'initial_key': 'initial_value'}]
    self.checkpointer.save_episodes(initial_data, 'task_group')
    new_data = [{'new_key': 'new_value'}]
    self.checkpointer.save_episodes(new_data, 'task_group')
    loaded_data = self.checkpointer.load()
    self.assertEqual(new_data, loaded_data)

  def test_save_and_load_multiple_task_groups(self) -> None:
    """Tests saving and loading multiple task groups."""
    task_groups = [
        ([{'key1': 'value1'}], 'task_group1'),
        ([{'key2': 'value2'}], 'task_group2'),
        ([{'key3': 'value3'}], 'task_group3'),
    ]
    for task_group, task_group_id in task_groups:
      self.checkpointer.save_episodes(task_group, task_group_id)
    loaded_data = self.checkpointer.load()
    expected_data = [data for data, _ in task_groups]
    self.assertCountEqual(
        loaded_data, [item for sublist in expected_data for item in sublist]  # pylint: disable=g-complex-comprehension
    )

  def test_load_invalid_file(self) -> None:
    """Tests if loading an invalid file is gracefully handled."""
    invalid_file = os.path.join(self.temp_dir.name, 'invalid.txt')
    with open(invalid_file, 'w') as f:
      f.write('invalid data')
    loaded_data = self.checkpointer.load()
    self.assertEqual([], loaded_data)

  def test_load_fields(self) -> None:
    """Tests if loading fields works as expected."""
    task_group = [{'key1': 'value1', 'key2': 'value2'}]
    self.checkpointer.save_episodes(task_group, 'task_group')
    loaded_data = self.checkpointer.load(fields=['key1'])
    expected_data = [{'key1': 'value1'}]
    self.assertEqual(expected_data, loaded_data)


if __name__ == '__main__':
  absltest.main()
