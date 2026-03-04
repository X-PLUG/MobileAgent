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

"""Utils for testing file util logic."""

import contextlib
import os
import shutil
import tempfile

from android_env import env_interface
from android_world.utils import file_utils


@contextlib.contextmanager
def mock_tmp_directory_from_device(
    device_path: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float | None,
):
  """Mocks `file_utils.tmp_directory_from_device` for unit testing."""
  del env, timeout_sec
  with tempfile.TemporaryDirectory() as tmp_dir:
    parent_dir = file_utils.convert_to_posix_path(
        tmp_dir, os.path.split(os.path.split(device_path)[0])[1]
    )
    try:
      shutil.copytree(device_path, parent_dir)
      yield parent_dir

    finally:
      shutil.rmtree(parent_dir)


def mock_copy_data_to_device(
    local_db_path: str,
    remote_db_path: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float | None = None,
):
  """Mocks the behavior of file_utils.copy_data_to_device for testing purposes.

  This mock function will copy the database file from a local directory to
  the simulated remote directory.

  Args:
    local_db_path: The path to the local SQLite database file.
    remote_db_path: The file path on the simulated remote device.
    env: The Android environment interface (unused in the mock).
    timeout_sec: Optional timeout in seconds (unused in the mock).
  """
  del env, timeout_sec
  os.makedirs(os.path.dirname(remote_db_path), exist_ok=True)
  shutil.copy(local_db_path, remote_db_path)


def mock_remove_files(directory: str, env: env_interface.AndroidEnvInterface):
  """Mocks the behavior of file_utils.remove_files for testing purposes.

  This mock function simulates removing all files in the specified directory.

  Args:
    directory: The directory path from which to remove files.
    env: The Android environment interface (unused in the mock).
  """
  del env
  for filename in os.listdir(directory):
    file_path = file_utils.convert_to_posix_path(directory, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)
    elif os.path.isdir(file_path):
      shutil.rmtree(file_path)
