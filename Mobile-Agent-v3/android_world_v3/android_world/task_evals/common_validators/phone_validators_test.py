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
from android_world.task_evals.common_validators import phone_validators
from android_world.utils import test_utils


class TestMakePhoneCall(test_utils.AdbEvalTestBase):

  def test_is_successful_offhook(self):
    self.mock_get_call_state.return_value = 'OFFHOOK'
    self.mock_dialer_with_phone_number.return_value = True

    env = mock.MagicMock()
    params = {'phone_number': '1234567890'}
    task = phone_validators.MakeCall(params)

    self.assertEqual(test_utils.perform_task(task, env), 1)

  def test_is_successful_not_offhook(self):
    self.mock_get_call_state.return_value = 'IDLE'
    self.mock_dialer_with_phone_number.return_value = True

    env = mock.MagicMock()
    params = {'phone_number': '1234567890'}
    task = phone_validators.MakeCall(params)

    self.assertEqual(test_utils.perform_task(task, env), 0)

  def test_is_successful_wrong_number(self):
    self.mock_get_call_state.return_value = 'OFFHOOK'
    self.mock_dialer_with_phone_number.return_value = False

    env = mock.MagicMock()
    params = {'phone_number': '1234567890'}
    task = phone_validators.MakeCall(params)

    self.assertEqual(test_utils.perform_task(task, env), 0)

  def test_generate_random_params(self):
    random_params = phone_validators.MakeCall.generate_random_params()
    self.assertIn('phone_number', random_params)
    self.assertIsInstance(random_params['phone_number'], str)


if __name__ == '__main__':
  absltest.main()
