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

import time
from unittest import mock
from absl.testing import absltest
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.task_evals.composite import markor_sms
from android_world.task_evals.utils import user_data_generation
from android_world.utils import test_utils


class TestMarkorCreateNoteAndSms(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )
    self.unused_mock_user_data_generation = mock.patch.object(
        user_data_generation, 'clear_device_storage'
    ).start()

  def test_MarkorCreateNoteAndSms_is_successful(self):
    # From shell date +%s
    mock_response_time = adb_pb2.AdbResponse()
    mock_response_time.generic.output = '{}'.format(
        str(int(time.time()))
    ).encode()

    # Create mock adb response for 'cat' command
    mock_response_cat = adb_pb2.AdbResponse()
    mock_response_cat.generic.output = b'Hello World'

    # From shell date +%s
    mock_response_time = adb_pb2.AdbResponse()
    mock_response_time.generic.output = '{}'.format(
        str(int(time.time()))
    ).encode()

    # Make stale message.
    one_day_s = 24 * 60 * 60
    mock_response_sms0 = adb_pb2.AdbResponse()
    date0_ms = str(int((time.time() - one_day_s) * 1000))
    mock_response_sms0.generic.output = (
        'Row: 0, address=1234567890, body=Hello World, service_center=NULL,'
        ' date={}'.format(
            date0_ms
        ).encode()
    )

    # Successful message.
    mock_response_sms1 = adb_pb2.AdbResponse()
    date1_ms = str(int(time.time() * 1000))
    mock_response_sms1.generic.output = (
        'Row: 0, address=1234567890, body=Hello World, service_center=NULL,'
        ' date={}'.format(
            date1_ms
        ).encode()
    )

    self.mock_issue_generic_request.side_effect = [
        mock_response_time,
        mock_response_sms0,
        mock_response_cat,
        mock_response_sms1,
        mock_response_time,
    ]
    test_utils.log_mock_calls(self.mock_issue_generic_request)

    self.mock_check_file_or_folder_exists.return_value = True

    env = mock.MagicMock()
    params = {
        'file_name': 'my_note.md',
        'text': 'Hello World',
        'number': '1234567890',
    }

    task = markor_sms.MarkorCreateNoteAndSms(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.assertEqual(self.mock_execute_sql_command.call_count, 2)

  def test_MarkorCreateNoteAndSms_partial_success(self):
    # From shell date +%s
    mock_response_time = adb_pb2.AdbResponse()
    mock_response_time.generic.output = '{}'.format(
        str(int(time.time()))
    ).encode()

    # Create mock adb response for 'cat' command
    mock_response_cat = adb_pb2.AdbResponse()
    mock_response_cat.generic.output = b'Hello World'

    # From shell date +%s
    mock_response_time = adb_pb2.AdbResponse()
    mock_response_time.generic.output = '{}'.format(
        str(int(time.time()))
    ).encode()

    # Make stale message.
    one_day_s = 24 * 60 * 60
    mock_response_sms0 = adb_pb2.AdbResponse()
    date0_ms = str(int((time.time() - one_day_s) * 1000))
    mock_response_sms0.generic.output = (
        'Row: 0, address=1234567890, body=Hello World, service_center=NULL,'
        ' date={}'.format(
            date0_ms
        ).encode()
    )

    # No message found response.
    mock_response_sms1 = adb_pb2.AdbResponse()
    mock_response_sms1.generic.output = (
        'No result found.'.encode()
    )

    self.mock_issue_generic_request.side_effect = [
        mock_response_time,
        mock_response_sms0,
        mock_response_cat,
        mock_response_sms1,
        mock_response_time,
    ]
    test_utils.log_mock_calls(self.mock_issue_generic_request)

    self.mock_check_file_or_folder_exists.return_value = True

    env = mock.MagicMock()
    params = {
        'file_name': 'my_note.md',
        'text': 'Hello World',
        'number': '1234567890',
    }

    task = markor_sms.MarkorCreateNoteAndSms(params)
    self.assertEqual(test_utils.perform_task(task, env), 0.5)
    self.assertEqual(self.mock_execute_sql_command.call_count, 2)


if __name__ == '__main__':
  absltest.main()
