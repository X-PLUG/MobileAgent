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

"""Utils for file operations using adb."""

import contextlib
import dataclasses
import datetime
import os
import pathlib
import random
import shutil
import string
import tempfile
from typing import Iterator
from typing import Optional

from absl import logging
from android_env import env_interface
from android_env.components import errors
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.utils import fuzzy_match_lib


def get_local_tmp_directory() -> str:
  """Returns the local temporary directory path.

  Returns:
    str: The local temporary directory path.
  """
  return tempfile.gettempdir()


def convert_to_posix_path(*args):
  """Converts the given path to a posix path.

  It can also be used to join paths.

  Args:
    *args: The paths to join.

  Returns:
    str: The path in posix format.
  """
  return str(pathlib.Path(*args).as_posix())


# Local temporary location for files copied to or from the device.
TMP_LOCAL_LOCATION = convert_to_posix_path(
    get_local_tmp_directory(), "android_world"
)


@dataclasses.dataclass(frozen=True)
class FileWithMetadata:
  """File with its metadata like change time.

  Attributes:
    file_name: file name.
    full_path: file name with full path.
    file_size: file size in bytes.
    change_time: file change time (ctime).
  """

  file_name: str
  full_path: str
  file_size: int
  change_time: datetime.datetime


def remove_single_file(
    target: str,
    base_path: str,
    env: env_interface.AndroidEnvInterface,
) -> None:
  """Remove a file (specified by its full path) if exists.

  Args:
    target: Target file name.
    base_path: Base directory to search for
    env: The environment to use.
  """
  if check_directory_exists(base_path, env):
    file_list = get_file_list_with_metadata(base_path, env)
    if target in [file_info.file_name for file_info in file_list]:
      adb_utils.issue_generic_request(
          ["shell", "rm", "-r", convert_to_posix_path(base_path, target)],
          env,
      )
  else:
    logging.warn(
        "Base path %s does not exist, ignoring remove_single_file.", base_path
    )


def clear_directory(
    directory_path: str,
    env: env_interface.AndroidEnvInterface,
) -> None:
  """Removes all files in the folder; also checks if folder is not empty.

  Args:
    directory_path: Location to create the file.
    env: The environment to use.

  Raises:
    RuntimeError when directory exists a failure occured while deleting files.
  """
  if not check_directory_exists(directory_path, env):
    return

  # Check if the folder is empty
  res = adb_utils.issue_generic_request(
      ["shell", "ls", "-1", directory_path], env
  )
  folder_contents = res.generic.output.decode().replace("\r", "").strip()

  if folder_contents:
    adb_utils.check_ok(
        adb_utils.issue_generic_request(
            ["shell", "rm", "-r", f"{directory_path}/*"],
            env,
        ),
        f"Failed to clear directory {directory_path}.",
    )


def create_file(
    file_name: str,
    directory_path: str,
    env: env_interface.AndroidEnvInterface,
    content: str = "",
) -> str:
  """Creates a new file.

  Args:
    file_name: Name of file.
    directory_path: Location to create the file.
    env: The environment to use.
    content: The contents to write to the file. If nothing is provided, then
      random text will be added.

  Returns:
    Content of the created file.
  """
  if not content:
    content = "".join(
        random.choices(string.ascii_letters + string.digits, k=20)
    )
  # Escape quotes to avoid issues with writing them to file.
  content = content.replace("'", "'\"'\"'")
  mkdir(directory_path, env)
  adb_utils.issue_generic_request(
      [
          "shell",
          "echo",
          f"'{content}'",
          ">",
          f"{directory_path}/{file_name}",
      ],
      env,
  )
  return content


def mkdir(directory_path: str, env: env_interface.AndroidEnvInterface) -> None:
  """Makes a directory using adb.

  Args:
    directory_path: The location to make it.
    env: The environment.

  Raises:
    RuntimeError when directory could not be created.
  """
  adb_utils.check_ok(
      adb_utils.issue_generic_request(
          [
              "shell",
              "mkdir",
              "-p",
              directory_path,
          ],
          env,
      ),
      f"Failed to create directory {directory_path}.",
  )


def copy_dir(
    source_path: str, dest_path: str, env: env_interface.AndroidEnvInterface
):
  """Recursively copies from one directory to another on device.

  Args:
    source_path: Source directory path on device.
    dest_path: Destination directory path on device.
    env: The environment.

  Raises:
    RuntimeError when the contents of the source path directory can not be
    written to the destination path.
  """

  if not check_directory_exists(source_path, env):
    logging.warn(
        "Source directory %s does not exist, ignoring copy_dir.", source_path
    )
    return

  if not check_directory_exists(dest_path, env):
    mkdir(dest_path, env)  # RuntimeError raised if path exists as a file.

  adb_utils.check_ok(
      adb_utils.issue_generic_request(
          ["shell", "cp", "-a", f"{source_path}/.", f"{dest_path}/"], env
      ),
      f"Failure copying {source_path} directory to {dest_path}.",
  )


def check_file_or_folder_exists(
    target: str, base_path: str, env: env_interface.AndroidEnvInterface
) -> bool:
  """Recursively checks if a file or folder exists under the specified base path.

  Args:
      target: Name of the file or folder to search for.
      base_path: The directory path under which to search.
      env: The Android environment interface.

  Returns:
      bool: True if the file or folder exists, False otherwise.

  Raises:
    RuntimeError: When ADB does not correctly execute.
  """
  if not check_directory_exists(base_path, env):
    return False

  # List all files and folders recursively under the base path
  res = adb_utils.issue_generic_request(
      ["shell", "find", base_path, "-type", "f", "-o", "-type", "d"], env
  )

  if not res.status:
    raise RuntimeError("ADB command failed.")

  all_paths = set(res.generic.output.decode().replace("\r", "").split("\n"))

  full_target_path = convert_to_posix_path(base_path, target)
  return full_target_path in all_paths


def check_file_exists(
    path: str,
    env: env_interface.AndroidEnvInterface,
    bash_file_test: str = "-f",
) -> bool:
  """Check if a file exists.

  Args:
    path: The path to check.
    env: The environment.
    bash_file_test: Bash test string. Use "-f" for file, "-d" for directory, and
      "-e" for either.

  Returns:
    Whether the file exists.
  """
  bash_script = f"""
  if [ {bash_file_test} "{path}" ]; then
      echo "Exists"
  else
      echo "Does not exist"
  fi
  """
  response = adb_utils.issue_generic_request(["shell", bash_script], env)
  if "Exists" in response.generic.output.decode("utf-8"):
    return True
  elif "Does not exist" in response.generic.output.decode("utf-8"):
    return False
  else:
    raise errors.AdbControllerError("Unexpected output from file check")


def check_directory_exists(
    path: str, env: env_interface.AndroidEnvInterface
) -> bool:
  """Check if a directory exists.

  Args:
    path: The path to check.
    env: The environment.

  Returns:
    Whether the directory exists.
  """
  return check_file_exists(path, env, bash_file_test="-d")


@contextlib.contextmanager
def tmp_directory_from_device(
    device_path: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = None,
):
  """Copy a directory from the device to a local temporary directory using ADB.

  Args:
    device_path: The path of the directory on the Android device.
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operations.

  Yields:
    A temporary folder that contains files copied from the device that is
    automatically deleted after use.

  Raises:
    FileExistsError: If the temp directory already exists.
    FileNotFoundError: If the remote directory does not exist.
    RuntimeError: If there is an adb communication error.
  """
  tmp_directory = tempfile.mkdtemp()
  logging.info(
      "Copying %s directory to local tmp %s", device_path, tmp_directory
  )

  adb_utils.set_root_if_needed(env, timeout_sec)

  if not check_directory_exists(device_path, env):
    raise FileNotFoundError(f"{device_path} does not exist.")
  try:
    os.makedirs(tmp_directory, exist_ok=True)
    files = get_file_list_with_metadata(device_path, env, timeout_sec)
    for file in files:
      pull_response = env.execute_adb_call(
          adb_pb2.AdbRequest(
              pull=adb_pb2.AdbRequest.Pull(path=file.full_path),
              timeout_sec=timeout_sec,
          )
      )
      adb_utils.check_ok(pull_response)
      with open(
          convert_to_posix_path(tmp_directory, file.file_name), "wb"
      ) as f:
        f.write(pull_response.pull.content)

    yield tmp_directory

  finally:
    try:
      shutil.rmtree(tmp_directory)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error(
          "Failed to delete temporary directory: %s with error %s",
          tmp_directory,
          e,
      )


@contextlib.contextmanager
def tmp_file_from_device(
    device_file: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = None,
) -> Iterator[str]:
  """Copies a remote file to a local temporary file.

  Args:
    device_file: The path on the device pointing to a file.
    env: The environment.
    timeout_sec: A timeout for the ADB operations.

  Yields:
    The name of the local temporary file.

  Raises:
    FileNotFoundError: If device_file does not exist.
    RuntimeError: If there is an adb communication error.
  """
  head, tail = os.path.split(device_file)
  dir_and_file_name = convert_to_posix_path(os.path.basename(head), tail)
  local_file = convert_to_posix_path(TMP_LOCAL_LOCATION, dir_and_file_name)
  try:
    # Need root access to access many directories.
    adb_utils.set_root_if_needed(env, timeout_sec)

    if not check_file_exists(device_file, env):
      raise FileNotFoundError(f"{device_file} does not exist.")
    if not os.path.exists(os.path.dirname(local_file)):
      os.makedirs(os.path.dirname(local_file), exist_ok=True)
    pull_response = env.execute_adb_call(
        adb_pb2.AdbRequest(
            pull=adb_pb2.AdbRequest.Pull(path=device_file),
            timeout_sec=timeout_sec,
        )
    )
    adb_utils.check_ok(pull_response)

    with open(local_file, "wb") as f:
      f.write(pull_response.pull.content)

    yield local_file
  finally:
    os.remove(local_file)


def copy_file_to_device(
    local_file_path: str,
    remote_file_path: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = None,
) -> adb_pb2.AdbResponse:
  """Copies a local file to a remote file."""
  with open(local_file_path, "rb") as f:
    file_contents = f.read()
    push_request = adb_pb2.AdbRequest(
        push=adb_pb2.AdbRequest.Push(
            content=file_contents, path=remote_file_path
        ),
        timeout_sec=timeout_sec,
    )
  push_response = env.execute_adb_call(push_request)

  # ' and whitespace are special characters in adb commands that need to be
  # escaped.
  escaped_path = remote_file_path.replace(" ", r"\ ").replace("'", r"\'")

  adb_utils.issue_generic_request(
      ["shell", "chmod", "777", escaped_path], env
  )
  return push_response


def copy_data_to_device(
    local_path: str,
    remote_path: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = None,
) -> adb_pb2.AdbResponse:
  """Copy a file or directory to the device from the local file system using ADB.

  Args:
    local_path: The path of the file or directory on the local file system.
    remote_path: The destination path on the Android device.
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.

  Raises:
    FileNotFoundError: If the local file or directory does not exist. Or if
      remote path does not exist.
  """
  if not os.path.exists(local_path):
    raise FileNotFoundError(f"{local_path} does not exist.")
  response = adb_pb2.AdbResponse()
  if os.path.isfile(local_path):
    # If the file extension is different, remote_path is likely a directory.
    if os.path.splitext(local_path)[1] != os.path.splitext(remote_path)[1]:
      remote_path = convert_to_posix_path(
          remote_path, os.path.basename(local_path)
      )
    return copy_file_to_device(local_path, remote_path, env, timeout_sec)

  # Copying a directory over, push every file separately.
  for file_path in os.listdir(local_path):
    current_response = copy_file_to_device(
        convert_to_posix_path(local_path, file_path),
        convert_to_posix_path(remote_path, os.path.basename(file_path)),
        env,
        timeout_sec,
    )
    if current_response.status != adb_pb2.AdbResponse.OK:
      return current_response
    response = current_response

  return response


def get_file_list_with_metadata(
    directory_path: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = None,
) -> list[FileWithMetadata]:
  """Get the list of all (regular) files with metadata in a given directory.

  Right now we only list regular files in the given directory and only grab file
  name, full directory and change time in metadata.

  Args:
    directory_path: The directory to list all its files.
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A list of files with metadata.
  Raises:
    RuntimeError: If the input directory path is not valid or shell ls fails.
  """
  if not check_directory_exists(directory_path, env):
    raise RuntimeError(f"{directory_path} is not a valid directory.")
  # Run [adb shell ls] to list all files in the given directory.
  try:
    ls_response = adb_utils.issue_generic_request(
        f"shell ls {directory_path} -ll -au", env, timeout_sec
    )
    adb_utils.check_ok(ls_response, "Failed to list files in directory.")
    files = []
    # Each file (including links and directories) will be listed in format as
    # follows,
    #  -rw-rw---- 1 u0_a158 media_rw 0 2023-11-28 23:17:43.176000000 +0000 1.txt
    # We loop through all the files and collect regular files with metadata.
    for file_details in ls_response.generic.output.decode("utf-8").split("\n"):
      # In shell output, the first character is used to indicate file type and
      # "-" means the file is a regular file.
      if file_details.startswith("-"):
        parts = file_details.split(None, 8)
        if len(parts) < 9:
          raise RuntimeError(f"Failed to parse file details: {file_details}")

        file_name = parts[
            8
        ].strip()  # This will preserve spaces in the filename
        files.append(
            FileWithMetadata(
                file_name=file_name,
                full_path=convert_to_posix_path(directory_path, file_name),
                file_size=int(parts[4]),
                change_time=datetime.datetime.fromisoformat(
                    " ".join(parts[5:7])[:-3]
                ),
            )
        )
    return files
  except errors.AdbControllerError as e:
    print(e)
    raise RuntimeError("Failed to list files in directory.") from e


def check_file_content(
    file_full_path: str,
    content: str,
    env: env_interface.AndroidEnvInterface,
    exact_match: bool = False,
    timeout_sec: Optional[float] = None,
) -> bool:
  """Check if a file content equals a given string.

  Args:
    file_full_path: Full path to the file, will return False if file does not
      exist.
    content: The expected file content.
    env: The Android environment interface.
    exact_match: A boolean indicates whether we use exact match or fuzzy match.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    If the given file has the given content, will return False in the case of
    incorrect file path/file does not exist.
  """

  try:
    res = adb_utils.issue_generic_request(
        ["shell", "cat", file_full_path], env, timeout_sec
    )
    res_content = res.generic.output.decode().replace("\r", "")
    if exact_match:
      return res_content == content
    return fuzzy_match_lib.fuzzy_match(res_content.strip(), content)
  except errors.AdbControllerError as e:
    print(e)
    return False
