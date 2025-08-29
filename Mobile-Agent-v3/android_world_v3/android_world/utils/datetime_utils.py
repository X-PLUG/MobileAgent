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

"""Manages date and time settings on an Android device using ADB commands."""

import datetime
import enum
import random
import zoneinfo

from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.env import device_constants


def timestamp_to_localized_datetime(
    timestamp: int, timezone: str = device_constants.TIMEZONE
) -> datetime.datetime:
  """Converts a UNIX timestamp to a localized datetime object.

  Args:
    timestamp: The UNIX timestamp to convert.
    timezone: The timezone string to localize the datetime.

  Returns:
    A localized datetime object.
  """
  utc_dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
  localized_dt = utc_dt.astimezone(zoneinfo.ZoneInfo(timezone))
  return localized_dt


def _create_unix_ts(
    *,
    day: int,
    hour: int,
    month: int = device_constants.DT.month,
    year: int = device_constants.DT.year,
    timezone: str = device_constants.TIMEZONE,
) -> int:
  """Converts a year, month, day, and hour into a timestamp.

  Args:
    day: The day of the date.
    hour: The hour of the date.
    month: The month of the date.
    year: The year of the date.
    timezone: The timezone to use for the date. Defaults to
      device_constants.TIMEZONE.

  Returns:
    int: The timestamp corresponding to the input date and hour.
  """
  dt = datetime.datetime(year, month, day, hour)
  localized_dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(timezone))
  result = int(localized_dt.timestamp())
  return result


def create_random_october_2023_unix_ts(
    start_day: int = device_constants.DT.day,
    end_day: int = 31,
    start_hour: int = 0,
) -> int:
  """Creates a random Unix timestamp in October 2023, the time period the device is set to.

  Args:
    start_day: The day to start in the random range.
    end_day: The day to end in the random range.
    start_hour: The hour to start in the random range; hour will be [start_hour,
      31]

  Returns:
    Unix timestamp.
  """
  return _create_unix_ts(
      day=random.randint(start_day, end_day),
      hour=random.randint(start_hour, 23),
      month=device_constants.DT.month,
      year=device_constants.DT.year,
      timezone=device_constants.TIMEZONE,
  )


class Toggle(enum.Enum):
  ON = '1'
  OFF = '0'


def toggle_auto_settings(
    env: env_interface.AndroidEnvInterface, toggle: Toggle
) -> None:
  """Disables the automatic date, time, and timezone settings.

  This is to maintain benchmark consistency and prevent external time updates.

  Args:
    env: AndroidEnv instance.
    toggle: Whether to enable or disable the settings.
  """
  adb_utils.put_settings(
      adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
      'auto_time',
      toggle.value,
      env,
  )
  adb_utils.put_settings(
      adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
      'auto_time_zone',
      toggle.value,
      env,
  )


def setup_datetime(env: env_interface.AndroidEnvInterface) -> None:
  """Prepares the Android device's date and time settings for benchmarking.

  This function should be called once before starting the benchmark tests. It
  disables automatic date, time, and timezone updates and sets the device to a
  24-hour time format. The purpose is to create a consistent environment for
  reproducible results.

  Args:
    env: AndroidEnv instance.
  """
  adb_utils.set_root_if_needed(env)
  toggle_auto_settings(env, Toggle.OFF)
  _enable_24_hour_format(env)
  _set_timezone_to_utc(env)


def set_datetime(
    env: env_interface.AndroidEnvInterface, dt: datetime.datetime
) -> None:
  """Configures the specific date and time for each task in the benchmark.

  This function should be called at the beginning of every task in the benchmark
  to set a specific date and time, ensuring consistency across repeated runs of
  the same task.

  Args:
    env: AndroidEnv instance.
    dt: The datetime to set the device to.
  """
  adb_utils.set_root_if_needed(env)
  _set_datetime(env, dt)


def advance_system_time(
    delta: datetime.timedelta,
    env: env_interface.AndroidEnvInterface,
) -> None:
  """Advance system time by a given time delta.

  Args:
    delta: Specify the amount of time to add to current time.
    env: AndroidEnv instance.
  """
  # Get current system time by parsing the output of running adb shell date
  # which looks like "Sun Oct 15 17:04:16 UTC 2023".
  current_time = datetime.datetime.strptime(
      adb_utils.issue_generic_request(
          ['shell', 'date'], env
      ).generic.output.decode().strip(),
      '%a %b %d %H:%M:%S %Z %Y',
  )

  # Set new system time.
  adb_utils.issue_generic_request(
      ['shell', 'date', (current_time + delta).strftime('%m%d%H%M%y.%S')], env
  )


def _enable_24_hour_format(env: env_interface.AndroidEnvInterface) -> None:
  """Sets to 24-hour time format to be consistent and region-independent."""
  adb_utils.put_settings(
      adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
      'time_12_24',
      '24',
      env,
  )


def _set_timezone_to_utc(env: env_interface.AndroidEnvInterface) -> None:
  """Sets the Android device's timezone to UTC.

  Args:
      env: An instance of AndroidEnv interface.
  """
  adb_command = ['shell', 'service', 'call', 'alarm', '3', 's16', 'UTC']
  adb_utils.issue_generic_request(adb_command, env)


def _set_datetime(
    env: env_interface.AndroidEnvInterface, dt: datetime.datetime
) -> None:
  """Sets the date and time on the Android device."""
  adb_utils.issue_generic_request(
      ['shell', 'date', dt.strftime('%m%d%H%M%y.%S')], env
  )


def generate_random_datetime(
    window_size: datetime.timedelta = datetime.timedelta(days=14),
    window_center: datetime.datetime = device_constants.DT,
) -> datetime.datetime:
  """Generates a random datetime within the given window.

  The window that the generated datetime is taken from is centered on
  device_constants.DT (= today) and is of length window_size.

  Args:
    window_size: The window size to generate a random datetime for.
    window_center: The center of the window to generate a random datetime for.

  Returns:
    A random datetime within the specified window.
  """
  start = window_center - (window_size / 2)
  return start + datetime.timedelta(
      minutes=random.randrange(window_size.days * 24 * 60)
  )
