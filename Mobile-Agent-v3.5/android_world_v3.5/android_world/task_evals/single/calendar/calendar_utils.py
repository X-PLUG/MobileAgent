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

"""Utils for Simple Calendar Pro."""

from typing import Optional
from android_world.env import interface
from android_world.task_evals.single.calendar import events_generator
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import datetime_utils


DB_PATH = '/data/data/com.simplemobiletools.calendar.pro/databases/events.db'
EVENTS_TABLE = 'events'  # Table in events.db.
DB_KEY = 'id'


def clear_calendar_db(
    env: interface.AsyncEnv, timeout_sec: Optional[float] = None
) -> None:
  """Removes the calendar database on the device."""
  sqlite_utils.delete_all_rows_from_table(
      EVENTS_TABLE, DB_PATH, env, 'simple calendar pro'
  )
  try:
    sqlite_utils.get_rows_from_remote_device(
        EVENTS_TABLE,
        DB_PATH,
        sqlite_schema_utils.CalendarEvent,
        env,
        timeout_sec,
    )
  except ValueError as e:
    raise RuntimeError(
        'After clearing the old SQLite database, a new empty database was'
        ' not created.'
    ) from e


def add_events(
    events: list[sqlite_schema_utils.CalendarEvent],
    env: interface.AsyncEnv,
    timeout_sec: Optional[float] = None,
) -> None:
  """Adds an event to the Android calendar database using ADB.

  Performs a round trip: copies db over from device, adds event, then sends
  db back to device.

  Args:
      events: The list of Events to add to the database.
      env: The Android environment interface.
      timeout_sec: A timeout for the ADB operations.
  """
  sqlite_utils.insert_rows_to_remote_db(
      events,
      DB_KEY,
      EVENTS_TABLE,
      DB_PATH,
      'simple calendar pro',
      env,
      timeout_sec,
  )


def add_random_events(env: interface.AsyncEnv, n: int = 75) -> None:
  """Adds random events to calendar to increase task complexity."""
  events = [
      events_generator.generate_event(
          datetime_utils.create_random_october_2023_unix_ts(start_day=1)
      )
      for _ in range(n)
  ]
  add_events(events, env)


def generate_simple_calendar_weekly_repeat_rule(day_of_week: int) -> int:
  """Generates a weekly repeat rule based on the provided list of weekdays.

  This logic is specific to Simple Calendar Pro, where each day is represented
  by 2^(n-1), with n being the day's number (1 for Monday, 2 for Tuesday, etc.).

  Args:
    day_of_week: Day of week, where Monday is 1, Tuesday is 2, ..., Sunday is 7.

  Returns:
    The repeat rule as an integer.
  """
  if not (1 <= day_of_week <= 7):
    raise ValueError('Invalid day of the week. Must be in range 1-7.')
  return 1 << (day_of_week - 1)
