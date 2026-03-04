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

import random
from unittest import mock
from absl.testing import absltest
from android_world.task_evals.single import phone
from android_world.task_evals.utils import user_data_generation
from android_world.utils import test_utils


class MarkorPhoneTest(test_utils.AdbEvalTestBase):

  @mock.patch.object(random, "choice", autospec=True)
  @mock.patch.object(user_data_generation, "generate_apartments", autospec=True)
  def test_generate_random_params(
      self, mock_generate_apartments, mock_random_choice
  ):
    mock_candidates = {"John": ["1234567890"], "Doe": ["0987654321"]}
    mock_generate_apartments.return_value = mock_candidates
    mock_random_choice.side_effect = lambda x: x[0]  # Return first key

    result = phone.MarkorCallApartment.generate_random_params()

    # Verify
    expected_result = {"name": "John", "phone_number": "1234567890"}
    self.assertEqual(result, expected_result)

  def test_markor_phone_successful(self):
    self.mock_get_call_state.return_value = "OFFHOOK"
    self.mock_dialer_with_phone_number.return_value = True
    params = {"name": "apt1", "phone_number": "123"}
    task = phone.MarkorCallApartment(params)
    self.assertTrue(test_utils.perform_task(task, self.mock_env))
    self.mock_create_file.assert_called_once()


if __name__ == "__main__":
  absltest.main()
