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
from android_env import env_interface
from android_world.utils import fake_adb_responses
from android_world.utils import file_utils


class FakeAdbResponsesTest(absltest.TestCase):

  def test_create_check_directory_exists_true_response(self):
    env = mock.create_autospec(env_interface.AndroidEnvInterface)

    env.execute_adb_call.return_value = (
        fake_adb_responses.create_check_directory_exists_response(exists=True)
    )

    self.assertTrue(file_utils.check_directory_exists("path", env))

  def test_create_check_directory_exists_false_response(self):
    env = mock.create_autospec(env_interface.AndroidEnvInterface)

    env.execute_adb_call.return_value = (
        fake_adb_responses.create_check_directory_exists_response(exists=False)
    )

    self.assertFalse(file_utils.check_directory_exists("path", env))

  def test_create_check_file_or_folder_exists_responses_exists(self):
    base_path = "/sdcard/FunStuff"
    file_name = "jokes.txt"
    env = mock.create_autospec(env_interface.AndroidEnvInterface)

    env.execute_adb_call.side_effect = (
        fake_adb_responses.create_check_file_or_folder_exists_responses(
            file_name=file_name, base_path=base_path, exists=True
        )
    )

    self.assertTrue(
        file_utils.check_file_or_folder_exists(file_name, base_path, env)
    )

  def test_create_check_file_or_folder_exists_responses_not_exists(self):
    base_path = "/sdcard/FunStuff"
    file_name = "jokes.txt"
    env = mock.create_autospec(env_interface.AndroidEnvInterface)

    env.execute_adb_call.side_effect = (
        fake_adb_responses.create_check_file_or_folder_exists_responses(
            file_name=file_name, base_path=base_path, exists=False
        )
    )

    self.assertFalse(
        file_utils.check_file_or_folder_exists(file_name, base_path, env)
    )

  def test_create_check_file_or_folder_exists_responses_different_file(self):
    base_path = "/sdcard/FunStuff"
    env = mock.create_autospec(env_interface.AndroidEnvInterface)

    env.execute_adb_call.side_effect = (
        fake_adb_responses.create_check_file_or_folder_exists_responses(
            file_name="jokes.txt", base_path=base_path, exists=True
        )
    )

    self.assertFalse(
        file_utils.check_file_or_folder_exists("puns.txt", base_path, env)
    )


if __name__ == "__main__":
  absltest.main()
