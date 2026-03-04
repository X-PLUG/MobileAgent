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

import itertools
import tempfile
from unittest import mock

from absl.testing import absltest
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.single import simple_gallery_pro
from android_world.task_evals.utils import receipt_generator
from android_world.task_evals.utils import user_data_generation
from android_world.utils import app_snapshot
from android_world.utils import datetime_utils
from android_world.utils import fake_adb_responses
from android_world.utils import file_utils
from PIL import Image


def _touch_temp_file(file_name):
  """Creates an empty file in the /tmp/ directory.

  Args:
    file_name: The name of the file.
  """
  path = file_utils.convert_to_posix_path(tempfile.gettempdir(), file_name)
  with open(path, "w") as f:
    f.write("")


class SaveCopyOfReceiptTaskEvalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_clear_device_storage = self.enter_context(
        mock.patch.object(
            user_data_generation,
            "clear_device_storage",
            autospec=True,
        )
    )
    self.mock_receipt_generator = self.enter_context(
        mock.patch.object(
            receipt_generator,
            "create_receipt",
            return_value=(
                mock.create_autospec(Image.Image),
                "receipt_test.jpg",
            ),
        )
    )
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(
            app_snapshot,
            "restore_snapshot",
        )
    )
    self.mock_setup_datetime = self.enter_context(
        mock.patch.object(
            datetime_utils,
            "setup_datetime",
        )
    )

  def assertInitializes(
      self,
      eval_task: simple_gallery_pro.SaveCopyOfReceiptTaskEval,
      env: interface.AsyncEnv,
  ):
    _touch_temp_file(eval_task.params["file_name"])
    env.controller.execute_adb_call.side_effect = list(
        itertools.chain(
            fake_adb_responses.create_taskeval_initialize_responses(
                len(eval_task.app_names)
            ),
            fake_adb_responses.create_remove_files_responses(),
            fake_adb_responses.create_copy_to_device_responses(),
        )
    )
    eval_task.initialize_task(env)

  def test_generate_random_params_returns_file_name(self):
    random_params = (
        simple_gallery_pro.SaveCopyOfReceiptTaskEval.generate_random_params()
    )

    file_name = random_params["file_name"]
    self.assertIsNotNone(file_name)
    self.assertStartsWith(file_name, "receipt_")
    self.assertEndsWith(file_name, ".jpg")

  def test_is_successful_returns_one_if_file_is_present(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = simple_gallery_pro.SaveCopyOfReceiptTaskEval(
        simple_gallery_pro.SaveCopyOfReceiptTaskEval.generate_random_params()
    )
    self.assertInitializes(eval_task, env)
    env.controller.execute_adb_call.side_effect = (
        fake_adb_responses.create_check_file_or_folder_exists_responses(
            file_name=eval_task.params["file_name"],
            base_path=device_constants.DOWNLOAD_DATA,
            exists=True,
        )
    )

    self.assertEqual(eval_task.is_successful(env), 1.0)

  def test_is_successful_returns_zero_if_file_is_missing(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = simple_gallery_pro.SaveCopyOfReceiptTaskEval(
        simple_gallery_pro.SaveCopyOfReceiptTaskEval.generate_random_params()
    )
    self.assertInitializes(eval_task, env)
    env.controller.execute_adb_call.side_effect = (
        fake_adb_responses.create_check_file_or_folder_exists_responses(
            file_name=eval_task.params["file_name"],
            base_path=device_constants.DOWNLOAD_DATA,
            exists=False,
        )
    )

    self.assertEqual(eval_task.is_successful(env), 0.0)


if __name__ == "__main__":
  absltest.main()
