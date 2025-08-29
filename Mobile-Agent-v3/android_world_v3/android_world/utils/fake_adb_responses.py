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

"""Functions for creating fake AdbResponse instances for use in tests.

If a test mocks AndroidEnvInterface, it may wish to test behavior that depends
on the mock returning a response to an adb call. This module provides functions
to construct these for common use cases.
"""

from android_env.proto import adb_pb2
from android_world.utils import file_utils


def create_successful_generic_response(output: str) -> adb_pb2.AdbResponse:
  return adb_pb2.AdbResponse(
      status=adb_pb2.AdbResponse.Status.OK,
      generic=adb_pb2.AdbResponse.GenericResponse(
          output=output.encode("utf-8")
      ),
  )


def create_get_wifi_enabled_response(is_enabled: bool) -> adb_pb2.AdbResponse:
  """Returns an AdbResponse for whether wifi is turned on.

  Args:
    is_enabled: If true, wifi is turned on.
  """
  return create_successful_generic_response("1" if is_enabled else "0")


def create_get_bluetooth_enabled_response(
    is_enabled: bool,
) -> adb_pb2.AdbResponse:
  """Returns an AdbResponse for whether bluetooth is turned on.

  Args:
    is_enabled: If true, wifi is turned on.
  """
  return create_successful_generic_response("1" if is_enabled else "0")


def create_get_activity_response(
    full_activity: str,
) -> adb_pb2.AdbResponse:
  """Returns an AdbResponse for the current visible Activity.

  Args:
    full_activity: The full component name of the activity.
  """
  return adb_pb2.AdbResponse(
      status=adb_pb2.AdbResponse.Status.OK,
      get_current_activity=adb_pb2.AdbResponse.GetCurrentActivityResponse(
          full_activity=full_activity
      ),
  )


def create_check_directory_exists_response(exists: bool) -> adb_pb2.AdbResponse:
  """Returns an AdbResponse saying the requested directory exists."""
  return create_successful_generic_response(
      "Exists" if exists else "Does not exist"
  )


def create_check_file_or_folder_exists_responses(
    file_name: str, base_path: str, exists: bool
) -> list[adb_pb2.AdbResponse]:
  """Returns a list of AdbResponses saying the requested file or folder exists.

  Multiple responses are returned as we first check if the directory exists.

  Args:
    file_name: The name of the file.
    base_path: The path to the directory containing the file.
    exists: If true, the responses say that the file exists.
  """
  if not exists:
    return [create_check_directory_exists_response(exists=False)]
  return [
      create_check_directory_exists_response(exists=True),
      create_successful_generic_response(
          file_utils.convert_to_posix_path(base_path, file_name) + "\n"
      ),
  ]


def create_taskeval_initialize_responses(
    number_of_apps: int,
) -> list[adb_pb2.AdbResponse]:
  """Returns a list of responses to handle the initialize logic in TaskEval."""
  # Two calls are used to set the time. Then a call per app is used to close the
  # apps listed as associated with the task.
  number_of_responses = 2 + number_of_apps
  return [
      create_successful_generic_response("")
      for _ in range(number_of_responses)
  ]


def create_remove_files_responses() -> list[adb_pb2.AdbResponse]:
  """Returns a list of AdbResponses saying the requested files are removed."""
  return [
      create_check_directory_exists_response(exists=True),
      create_successful_generic_response(""),
  ]


def create_copy_to_device_responses() -> list[adb_pb2.AdbResponse]:
  """Returns a list of AdbResponses saying the file was copied to the device."""
  return [
      create_check_directory_exists_response(exists=True),
      create_successful_generic_response(""),
  ]
