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
from android_world.task_evals.common_validators import sms_validators
from android_world.utils import test_utils


class TestSmsAreEqual(absltest.TestCase):

  def test_sms_are_equal(self):
    four_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    two_minutes_ago = int(time.time() * 1000) - 2 * 60 * 1000
    messages = [
        f'Row: 0 _id=1, address=1111, body=Hi, friend, date={four_minutes_ago}',
        f'Row: 1 _id=2, address=1111, body=Hi, friend, date={two_minutes_ago}',
    ]
    self.assertTrue(sms_validators.sms_are_equal(messages[0], messages[1]))

  def test_address_are_not_equal(self):
    four_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    messages = [
        f'Row: 0 _id=1, address=1111, body=Hi, friend, date={four_minutes_ago}',
        f'Row: 1 _id=2, address=1113, body=Hi, friend, date={four_minutes_ago}',
    ]
    self.assertFalse(sms_validators.sms_are_equal(messages[0], messages[1]))

  def test_body_are_not_equal(self):
    two_minutes_ago = int(time.time() * 1000) - 2 * 60 * 1000
    four_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    messages = [
        f'Row: 0 _id=1, address=1111, body=Hi, friend, date={two_minutes_ago}',
        f'Row: 1 _id=2, address=1111, body=Yo, friend, date={four_minutes_ago}',
    ]
    self.assertFalse(sms_validators.sms_are_equal(messages[0], messages[1]))


class TestMessageWasSent(absltest.TestCase):

  def test_valid_message(self):
    current_time = int(time.time() * 1000)
    # 5 minutes ago in milliseconds
    four_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    messages = [
        f'Row: 0 _id=1, address=1111, body=Hi, friend, date={four_minutes_ago}'
    ]
    self.assertTrue(
        sms_validators.was_sent(messages, '1111', 'Hi, friend', current_time, 5)
    )

  def test_expired_message(self):
    current_time = int(time.time() * 1000)
    # 10 minutes ago in milliseconds
    ten_minutes_ago = int(time.time() * 1000) - 10 * 60 * 1000
    messages = [f'Row: 0 _id=1, address=1111, body=Hi, date={ten_minutes_ago}']
    self.assertFalse(
        sms_validators.was_sent(messages, '1111', 'Hi', current_time, 5)
    )

  def test_invalid_address(self):
    current_time = int(time.time() * 1000)
    # 5 minutes ago in milliseconds
    five_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    messages = [f'Row: 0 _id=1, address=2222, body=Hi, date={five_minutes_ago}']
    self.assertFalse(
        sms_validators.was_sent(messages, '1111', 'Hi', current_time)
    )

  def test_invalid_body(self):
    current_time = int(time.time() * 1000)
    # 5 minutes ago in milliseconds
    five_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    messages = [
        f'Row: 0 _id=1, address=1111, body=Hello, date={five_minutes_ago}'
    ]
    self.assertFalse(
        sms_validators.was_sent(messages, '1111', 'Hi', current_time)
    )

  def test_fuzzy_matching(self):
    current_time = int(time.time() * 1000)
    # Assuming your fuzzy_match function returns True for 'Hi' and 'hi'
    # 5 minutes ago in milliseconds
    five_minutes_ago = int(time.time() * 1000) - 4 * 60 * 1000
    messages = [f'Row: 0 _id=1, address=1111, body=hi, date={five_minutes_ago}']
    self.assertTrue(
        sms_validators.was_sent(messages, '1111', 'Hi', current_time, 5)
    )


class TestMessagesSendTextMessage(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )

  def test_is_successful(self):
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
        mock_response_sms1,
        mock_response_time,
    ]
    test_utils.log_mock_calls(self.mock_issue_generic_request)

    env = mock.MagicMock()
    params = {'number': '1234567890', 'message': 'Hello World'}

    task = sms_validators.SimpleSMSSendSms(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)

    # Clear sms and threads tables.
    self.assertEqual(self.mock_execute_sql_command.call_count, 2)

  def test_initialize_task_message_already_sent(self):
    # From shell date +%s
    mock_response_time = adb_pb2.AdbResponse()
    mock_response_time.generic.output = '{}'.format(
        str(int(time.time()))
    ).encode()

    # Make stale message.
    one_s = 1
    mock_response_sms0 = adb_pb2.AdbResponse()
    date0_ms = str(int((time.time() - one_s) * 1000))
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
        mock_response_sms1,
    ]

    env = mock.MagicMock()

    params = {'number': '1234567890', 'message': 'Hello World'}

    task = sms_validators.SimpleSMSSendSms(params)
    with self.assertRaises(ValueError):
      task.initialize_task(env)
    # Clear sms and threads tables.
    self.assertEqual(self.mock_execute_sql_command.call_count, 2)


if __name__ == '__main__':
  absltest.main()
