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
from unittest import mock
import zoneinfo

from absl.testing import absltest
from absl.testing import parameterized
from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.utils import datetime_utils


@mock.patch.object(adb_utils, 'issue_generic_request')
class AdbDatetimeManagerTest(absltest.TestCase):

  @mock.patch.object(adb_utils, 'put_settings')
  def test_setup_datetime_environment(
      self, mock_put_settings, unused_mock_issue_generic_request
  ):
    env_mock = mock.create_autospec(env_interface.AndroidEnvInterface)

    datetime_utils.setup_datetime(env_mock)

    expected_calls = [
        mock.call(
            adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
            'auto_time',
            '0',
            env_mock,
        ),
        mock.call(
            adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
            'auto_time_zone',
            '0',
            env_mock,
        ),
        mock.call(
            adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
            'time_12_24',
            '24',
            env_mock,
        ),
    ]
    mock_put_settings.assert_has_calls(expected_calls, any_order=False)

  def test_advance_system_time(self, mock_issue_generic_request):
    env_mock = mock.create_autospec(env_interface.AndroidEnvInterface)
    mock_issue_generic_request.return_value = adb_pb2.AdbResponse(
        status=adb_pb2.AdbResponse.Status.OK,
        generic=adb_pb2.AdbResponse.GenericResponse(
            output=bytes('Sun Oct 15 17:04:16 UTC 2023\n', 'utf-8')
        ),
    )
    datetime_utils.advance_system_time(datetime.timedelta(hours=2), env_mock)
    mock_issue_generic_request.assert_has_calls([
        mock.call(['shell', 'date'], env_mock),
        mock.call(['shell', 'date', '1015190423.16'], env_mock),
    ])


class DateTimeUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'UTC_test',
          'UTC',
          datetime.datetime(
              2023, 11, 15, 23, 12, 0, tzinfo=zoneinfo.ZoneInfo('UTC')
          ),
      ),
      (
          'NYC_test',
          'America/New_York',
          datetime.datetime(
              2023,
              11,
              15,
              18,
              12,
              0,
              tzinfo=zoneinfo.ZoneInfo('America/New_York'),
          ),
      ),
  )
  def test_correct_conversion(
      self, timezone: str, expected_dt: datetime.datetime
  ):
    """Test if UNIX timestamps are correctly converted to localized datetime."""
    timestamp = 1700089920  # UST: Nov 15, 2023, 23:12:00
    result = datetime_utils.timestamp_to_localized_datetime(timestamp, timezone)
    self.assertEqual(result, expected_dt)

  def test_invalid_timezone(self):
    """Test behavior with invalid timezone strings."""
    timestamp = 1609459200
    with self.assertRaises(zoneinfo.ZoneInfoNotFoundError):
      datetime_utils.timestamp_to_localized_datetime(
          timestamp, 'invalid_timezone'
      )


if __name__ == '__main__':
  absltest.main()
