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

import datetime
import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.utils import file_utils


def create_file_with_contents(file_name: str, contents: bytes) -> str:
  with open(file_name, 'wb') as f:
    f.write(contents)


class FilesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_issue_generic_request = mock.patch.object(
        adb_utils, 'issue_generic_request'
    ).start()
    self.mock_env = mock.MagicMock()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_check_directory_exists(self):
    self.mock_issue_generic_request.return_value.generic.output.decode.return_value = (
        'Exists'
    )
    result = file_utils.check_directory_exists('/existing/path', self.mock_env)
    self.assertTrue(result)

    # Test case where directory does not exist
    self.mock_issue_generic_request.return_value.generic.output.decode.return_value = (
        'Does not exist'
    )
    result = file_utils.check_directory_exists(
        '/non/existing/path', self.mock_env
    )
    self.assertFalse(result)

  @mock.patch.object(os.path, 'exists')
  @mock.patch.object(file_utils, 'check_directory_exists')
  @mock.patch.object(shutil, 'rmtree')
  @mock.patch.object(tempfile, 'mkdtemp')
  def test_tmp_directory_from_device(
      self,
      mock_mkdtemp,
      mock_rmtree,
      mock_check_directory_exists,
      mock_path_exists,
  ):
    """Test if tmp_directory_from_device correctly copies a directory and handles exceptions."""
    mock_response = adb_pb2.AdbResponse(status=adb_pb2.AdbResponse.Status.OK)
    self.mock_env.execute_adb_call.return_value = mock_response
    mock_path_exists.return_value = False
    file_names = ['test1.txt', 'test2.txt']
    mock_check_directory_exists.return_value = True
    self.mock_issue_generic_request.return_value = adb_pb2.AdbResponse(
        status=adb_pb2.AdbResponse.Status.OK,
        generic=adb_pb2.AdbResponse.GenericResponse(
            output=bytes(
                '-rw-rw---- 1 u0_a158 media_rw 0 2023-11-28 23:17:43.176000000'
                f' +0000 {file_names[0]}\n'
                '-rw-rw---- 1 u0_a158 media_rw 0 2023-11-28 23:17:43.176000000'
                f' +0000 {file_names[1]}',
                'utf-8',
            )
        ),
    )

    tmp_local_directory = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), 'random', 'dir'
    )
    mock_mkdtemp.return_value = tmp_local_directory
    with file_utils.tmp_directory_from_device(
        '/remotedir', self.mock_env
    ) as tmp_directory:
      self.assertEqual(tmp_local_directory, tmp_directory)
      self.mock_env.execute_adb_call.assert_has_calls([
          mock.call(
              adb_pb2.AdbRequest(
                  pull=adb_pb2.AdbRequest.Pull(
                      path=file_utils.convert_to_posix_path(
                          '/remotedir/', file_name
                      )
                  ),
                  timeout_sec=None,
              )
          )
          for file_name in file_names
      ])
      self.assertCountEqual(os.listdir(tmp_directory), file_names)
      mock_rmtree.assert_not_called()
    mock_rmtree.assert_called_with(tmp_local_directory)

    # Test FileNotFoundError
    mock_path_exists.return_value = False
    mock_check_directory_exists.return_value = False
    with self.assertRaises(FileNotFoundError):
      with file_utils.tmp_directory_from_device(
          '/nonexistent/dir', self.mock_env
      ):
        pass

    # Test ADB RuntimeError
    mock_check_directory_exists.return_value = True
    self.mock_issue_generic_request.return_value.status = (
        adb_pb2.AdbResponse.ADB_ERROR
    )
    with self.assertRaises(RuntimeError):
      with file_utils.tmp_directory_from_device(
          '/remote/dir',
          self.mock_env,
      ):
        pass

  def test_copy_data_to_device_copies_file(self):
    """Test if copy_data_to_device correctly copies a single file."""
    file_contents = b'test file contents'
    mock_response = adb_pb2.AdbResponse(status=adb_pb2.AdbResponse.Status.OK)
    self.mock_env.execute_adb_call.return_value = mock_response
    temp_dir = tempfile.mkdtemp()
    file_name = 'file1.txt'
    create_file_with_contents(
        file_utils.convert_to_posix_path(temp_dir, file_name), file_contents
    )

    response = file_utils.copy_data_to_device(
        temp_dir, '/remote/dir', self.mock_env
    )
    self.mock_env.execute_adb_call.assert_has_calls(
        [
            mock.call(
                adb_pb2.AdbRequest(
                    push=adb_pb2.AdbRequest.Push(
                        content=file_contents,
                        path=file_utils.convert_to_posix_path(
                            '/remote/dir/', file_name
                        ),
                    ),
                    timeout_sec=None,
                )
            )
        ],
        any_order=True,
    )

    self.assertEqual(response, mock_response)

  def test_copy_data_to_device_copies_full_dir(self):
    """Test if copy_data_to_device correctly copies data from a directory."""
    file_contents = b'test file contents'
    mock_response = adb_pb2.AdbResponse(status=adb_pb2.AdbResponse.Status.OK)
    self.mock_env.execute_adb_call.return_value = mock_response
    temp_dir = tempfile.mkdtemp()
    file_names = ['file1.txt', 'file2.txt']
    for file_name in file_names:
      create_file_with_contents(
          file_utils.convert_to_posix_path(temp_dir, file_name), file_contents
      )

    response = file_utils.copy_data_to_device(
        temp_dir, '/remote/dir', self.mock_env
    )
    calls = []
    for file_name in file_names:
      calls.append(
          mock.call(
              adb_pb2.AdbRequest(
                  push=adb_pb2.AdbRequest.Push(
                      content=file_contents,
                      path=file_utils.convert_to_posix_path(
                          '/remote/dir/', file_name
                      ),
                  ),
                  timeout_sec=None,
              )
          )
      )
    self.mock_env.execute_adb_call.assert_has_calls(
        calls,
        any_order=True,
    )

    self.assertEqual(response, mock_response)

  def test_copy_data_to_device_file_not_found(self):
    """Test if copy_data_to_device handles errors."""
    # Test FileNotFoundError
    with self.assertRaises(FileNotFoundError):
      file_utils.copy_data_to_device(
          '/nonexistent/path', '/remote/dir', self.mock_env
      )

  @mock.patch.object(file_utils, 'check_directory_exists')
  def test_get_file_list_with_metadata(self, mock_check_directory_exists):
    mock_check_directory_exists.return_value = True
    self.mock_issue_generic_request.return_value = adb_pb2.AdbResponse(
        status=adb_pb2.AdbResponse.Status.OK,
        generic=adb_pb2.AdbResponse.GenericResponse(
            output=bytes(
                '-rw-rw---- 1 u0_a158 media_rw 0 2023-11-28 23:17:43.176000000'
                ' +0000 test.txt',
                'utf-8',
            )
        ),
    )
    file_list = file_utils.get_file_list_with_metadata(
        '/test_path', self.mock_env
    )
    self.mock_issue_generic_request.assert_called_with(
        'shell ls /test_path -ll -au', self.mock_env, None
    )
    self.assertLen(file_list, 1)
    self.assertEqual(file_list[0].file_name, 'test.txt')
    self.assertEqual(file_list[0].full_path, '/test_path/test.txt')
    self.assertEqual(
        file_list[0].change_time,
        datetime.datetime(2023, 11, 28, 23, 17, 43, 176000),
    )

  def test_check_file_content(self):
    self.mock_issue_generic_request.return_value = adb_pb2.AdbResponse(
        status=adb_pb2.AdbResponse.Status.OK,
        generic=adb_pb2.AdbResponse.GenericResponse(
            output=bytes(
                'test content.',
                'utf-8',
            )
        ),
    )

    res = file_utils.check_file_content(
        '/test_path/test_file', 'test content', self.mock_env
    )
    self.mock_issue_generic_request.assert_called_with(
        ['shell', 'cat', '/test_path/test_file'], self.mock_env, None
    )
    self.assertTrue(res)


if __name__ == '__main__':
  absltest.main()
