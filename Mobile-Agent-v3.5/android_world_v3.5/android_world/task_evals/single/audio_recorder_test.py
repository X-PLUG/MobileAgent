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

"""Tests for audio_recorder.py."""

import datetime
from unittest import mock

from absl.testing import absltest
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.single import audio_recorder
from android_world.utils import file_utils
from android_world.utils import test_utils


class AudioRecorderTest(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.mock_issue_generic_request = mock.patch.object(
        adb_utils, "issue_generic_request"
    ).start()
    self.mock_env = mock.MagicMock(spec=interface.AsyncEnv)

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_audio_recorder_is_successful(self):
    file1 = file_utils.FileWithMetadata(
        file_name="file_name1",
        full_path="/path/file_name1",
        file_size=1000,
        change_time=datetime.datetime.now(),
    )
    file2 = file_utils.FileWithMetadata(
        file_name="file_name2",
        full_path="/path/file_name2",
        file_size=1000,
        change_time=datetime.datetime.now(),
    )
    self.mock_get_file_list_with_metadata.return_value = [file1, file2]
    params = {}
    task = audio_recorder.AudioRecorderRecordAudio(params)
    task.initialize_task(self.mock_env)

    self.mock_get_file_list_with_metadata.return_value = [
        file1,
        file2,
        file_utils.FileWithMetadata(
            file_name="file_name3",
            full_path="/path/file_name3",
            file_size=1000,
            change_time=datetime.datetime.now(),
        ),
    ]
    result = task.is_successful(self.mock_env)
    self.assertEqual(result, 1)

  def test_audio_recorder_with_file_is_successful(self):
    self.mock_check_file_or_folder_exists.return_value = False
    params = {
        "file_name": "random_file_name",
        "text": "",  # Unused.
    }
    task = audio_recorder.AudioRecorderRecordAudioWithFileName(params)
    task.initialize_task(self.mock_env)

    self.mock_check_file_or_folder_exists.return_value = True
    result = task.is_successful(self.mock_env)
    self.assertEqual(result, 1)


if __name__ == "__main__":
  absltest.main()
