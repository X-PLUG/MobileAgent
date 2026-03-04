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
from android_world.task_evals.common_validators import contacts_validators
from android_world.utils import contacts_utils
from android_world.utils import test_utils


class TestAddContact(test_utils.AdbEvalTestBase):
  """Tests for AddContact task evaluation."""

  def test_is_successful_when_contact_found(self):
    self.mock_list_contacts.return_value = [
        contacts_utils.Contact('Test Case', '1234'),
    ]

    env = mock.MagicMock()
    params = {'name': 'Test Case', 'number': '1234'}

    task = contacts_validators.AddContact(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)

  def test_is_not_successful_when_contact_not_found(self):
    self.mock_list_contacts.return_value = []

    env = mock.MagicMock()
    params = {'name': 'Test Case', 'number': '1234'}

    task = contacts_validators.AddContact(params)
    self.assertEqual(test_utils.perform_task(task, env), 0)


if __name__ == '__main__':
  absltest.main()
