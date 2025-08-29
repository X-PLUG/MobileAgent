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

"""Tests for base evaluators."""

from unittest import mock
from absl.testing import absltest
from android_env.proto import adb_pb2
from android_world.task_evals.common_validators import file_validators
from android_world.utils import test_utils


class TestCreateFile(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.params = {
        "file_name": "my_note.md",
        "text": "Hello World",
    }

  def test_initialize(self):
    env = mock.MagicMock()
    task = file_validators.CreateFile(self.params, "/mock/data/path")
    task.initialize_task(env)

  def test_is_successful(self):
    self.mock_check_file_or_folder_exists.return_value = True

    mock_response_cat = adb_pb2.AdbResponse()
    mock_response_cat.generic.output = b"Hello World"
    self.mock_issue_generic_request.return_value = mock_response_cat

    env = mock.MagicMock()

    task = file_validators.CreateFile(self.params, "/mock/data/path")
    self.assertEqual(test_utils.perform_task(task, env.base_env), 1.0)

  def test_initialize_task_wrong_name(self):
    self.mock_issue_generic_request.response = (
        b"This is some other World. Not the same world."
    )
    self.mock_check_file_or_folder_exists.side_effect = [
        False,
        True,
    ]

    env = mock.MagicMock()

    task = file_validators.CreateFile(self.params, "/mock/data/path")
    self.assertFalse(test_utils.perform_task(task, env.base_env))


class TestDeleteFile(test_utils.AdbEvalTestBase):

  def test_is_successful(
      self,
  ):
    response_ls_deleted = adb_pb2.AdbResponse()
    response_ls_deleted.generic.output = b"another_note.md\n"
    self.mock_check_file_or_folder_exists.side_effect = [
        True,  # File exists.
        False,  # File doesn't exist.
    ]
    env = mock.MagicMock()
    params = {
        "file_name": "test_note.md",
        "noise_candidates": ["Noise Candidate"],
    }

    task = file_validators.DeleteFile(params, "/mock/data/path")

    self.assertEqual(test_utils.perform_task(task, env.base_env), 1.0)
    self.mock_create_file.assert_called()
    self.mock_create_random_files.assert_called()

  def test_is_successful_subfolder(self):
    # Create mock adb response for 'ls' command when note is deleted
    mock_response_ls_deleted = adb_pb2.AdbResponse()
    mock_response_ls_deleted.generic.output = b"another_note.md\n"
    self.mock_check_file_or_folder_exists.side_effect = [
        True,  # File exists.
        False,  # File doesn't exist.
    ]
    env = mock.MagicMock()
    params = {
        "file_name": "test_note.md",
        "subfolder": "a_folder",
        "noise_candidates": ["Noise Candidate"],
    }

    task = file_validators.DeleteFile(params, "/mock/data/path")

    self.assertEqual(test_utils.perform_task(task, env.base_env), 1.0)
    self.mock_create_file.assert_called()
    self.mock_create_random_files.assert_called()

  def test_is_not_successful(self):
    # Create mock adb response for 'ls' command when note still exists
    mock_response_ls_still_exists = adb_pb2.AdbResponse()
    mock_response_ls_still_exists.generic.output = (
        b"test_note.md\nanother_note.md\n"
    )
    self.mock_check_file_or_folder_exists.side_effect = [
        True,  # File exists.
        True,  # File still exists.
    ]
    env = mock.MagicMock()
    params = {
        "file_name": "test_note.md",
        "noise_candidates": ["Noise Candidate"],
    }

    task = file_validators.DeleteFile(params, "/mock/data/path")

    self.assertFalse(test_utils.perform_task(task, env.base_env))
    self.mock_create_file.assert_called()
    self.mock_create_random_files.assert_called()


class TestMoveFile(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.params = {
        "file_name": "test_file.md",
        "source_folder": "Source",
        "destination_folder": "Destination",
        "noise_candidates": ["Noise Candidate"],
    }

  def test_is_successful(self):
    # Create mock adb response for 'ls' command when note is deleted
    mock_response_ls_deleted = adb_pb2.AdbResponse()
    mock_response_ls_deleted.generic.output = b"another_note.md\n"

    self.mock_check_file_or_folder_exists.side_effect = [
        True,  # Source file exists.
        False,  # Destination file does not exist.
        False,  # Source file does not exist.
        True,  # Destination file exists.
    ]
    test_utils.log_mock_calls(self.mock_check_file_or_folder_exists)

    env = mock.MagicMock()

    task = file_validators.MoveFile(self.params, "/mock/data/path")
    self.assertEqual(test_utils.perform_task(task, env.base_env), 1.0)

    # Assert that the mock functions were called
    self.mock_create_file.assert_called()
    self.mock_mkdir.assert_called()
    self.mock_create_random_files.assert_called()

  def test_is_not_successful(self):
    # Create mock adb response for 'ls' command when note still exists
    mock_response_ls_still_exists = adb_pb2.AdbResponse()
    mock_response_ls_still_exists.generic.output = (
        b"test_note.md\nanother_note.md\n"
    )

    self.mock_check_file_or_folder_exists.side_effect = [
        True,  # Source file exists.
        False,  # Destination file does not exist.
        True,  # Source file still exists.
        False,  # Destination file still does not exist.
    ]

    env = mock.MagicMock()

    task = file_validators.MoveFile(self.params, "/mock/data/path")
    self.assertFalse(test_utils.perform_task(task, env.base_env))

    # Assert that the mock functions were called
    self.mock_create_file.assert_called()
    self.mock_create_random_files.assert_called()
    self.mock_mkdir.assert_called()


if __name__ == "__main__":
  absltest.main()
