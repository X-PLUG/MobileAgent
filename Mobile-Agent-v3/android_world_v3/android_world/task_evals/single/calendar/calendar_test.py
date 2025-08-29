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

from collections.abc import Iterable
import random
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from android_world.env import adb_utils
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single.calendar import calendar
from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.single.calendar import events_generator
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import app_snapshot
from android_world.utils import datetime_utils
from android_world.utils import file_utils


def _sample_events(
    event: sqlite_schema_utils.CalendarEvent,
) -> Iterable[sqlite_schema_utils.CalendarEvent]:
  yield event
  while True:
    start = random.randint(0, 1_000_000)
    noise = sqlite_schema_utils.CalendarEvent(
        start_ts=60,
        end_ts=start + 60,
        title="nothing " + str(random.randint(0, 1_000_000)),
        description="noise" + str(random.randint(0, 1_000_000)),
    )
    yield random.choice([event, noise])


class CalendarEventTestSetup(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.MagicMock()

    self.mock_list_rows = mock.patch.object(
        sqlite_validators.SQLiteApp, "list_rows", return_value=[]
    ).start()
    self.mock_add_rows = mock.patch.object(
        sqlite_validators.SQLiteApp, "add_rows"
    ).start()
    self.mock_tmp_directory_from_device = mock.patch.object(
        file_utils, "tmp_directory_from_device"
    ).start()
    self.mock_issue_generic_request = mock.patch.object(
        adb_utils, "issue_generic_request"
    ).start()
    self.mock_remove_files = mock.patch.object(
        file_utils, "clear_directory"
    ).start()
    self.mock_clear_db = mock.patch.object(
        sqlite_validators.SQLiteApp, "_clear_db"
    ).start()
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, "restore_snapshot")
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()


class TestSimpleCalendarAddOneEvent(CalendarEventTestSetup):

  def test_is_successful(self):
    task = calendar.SimpleCalendarAddOneEvent(
        calendar.SimpleCalendarAddOneEvent.generate_random_params()
    )
    self.mock_list_rows.side_effect = [[], task.params["row_objects"]]
    task.initialize_task(self.env)
    result = task.is_successful(self.env)
    self.assertEqual(result, 1)

  def test_generate_random_params(self):
    params = calendar.SimpleCalendarAddOneEvent.generate_random_params()
    self.assertIn("year", params)
    self.assertIn("month", params)
    self.assertIn("day", params)
    self.assertIn("hour", params)
    self.assertIn("event_title", params)
    self.assertIn("event_description", params)
    self.assertIn("duration_mins", params)
    self.assertIn("row_objects", params)

  @parameterized.named_parameters(
      ("typical event", 2023, 10, 16, 8, "Meeting", "Discuss project"),
      (
          "end of month event",
          2023,
          10,
          31,
          15,
          "Halloween Party",
          "Costume party at office",
      ),
      (
          "early morning event",
          2023,
          10,
          17,
          6,
          "Workout",
          "Morning gym session",
      ),
      (
          "late night event",
          2023,
          10,
          20,
          22,
          "Movie Night",
          "Watching the latest movie release",
      ),
  )
  @mock.patch.object(events_generator, "generate_event")
  def test_goal(
      self, year, month, day, hour, title, description, mock_generate_event
  ):
    start_time = datetime_utils._create_unix_ts(
        day=day, hour=hour, month=month, year=year, timezone="UTC"
    )
    event = sqlite_schema_utils.CalendarEvent(
        start_ts=start_time,
        end_ts=start_time + 10 * 60,  # 10 minutes duration
        title=title,
        description=description,
    )
    mock_generate_event.side_effect = _sample_events(event)

    params = calendar.SimpleCalendarAddOneEvent.generate_random_params()
    task = calendar.SimpleCalendarAddOneEvent(params)

    expected_goal = (
        "In Simple Calendar Pro, create a calendar event on"
        f" {year}-{month:02d}-{day:02d} at {hour}h with the title '{title}' and"
        f" the description '{description}'. The event should last for 10 mins."
    )
    self.assertEqual(task.goal, expected_goal)


class TestSimpleCalendarAddOneEventRelativeDay(CalendarEventTestSetup):

  @parameterized.named_parameters(
      ("start of range", 16, "Monday"), ("end of range", 21, "Saturday")
  )
  @mock.patch.object(events_generator, "generate_event")
  def test_goal_and_schema(self, day, day_of_week, mock_generate_event):
    start_time = datetime_utils._create_unix_ts(
        day=day,
        hour=8,
        month=10,
        year=2023,
        timezone="UTC",
    )
    event = sqlite_schema_utils.CalendarEvent(
        start_ts=start_time,
        end_ts=start_time + 10 * 60,
        title="A Title",
        description="A Description",
    )

    mock_generate_event.side_effect = _sample_events(event)

    params = (
        calendar.SimpleCalendarAddOneEventRelativeDay.generate_random_params()
    )
    task = calendar.SimpleCalendarAddOneEventRelativeDay(params)

    self.assertEqual(
        task.goal,
        "In Simple Calendar Pro, create a calendar event for this"
        f" {day_of_week} at 8h with the title 'A Title' and the description 'A"
        " Description'. The event should last for 10 mins.",
    )


class TestSimpleCalendarAddOneEventInTwoWeeks(CalendarEventTestSetup):

  def test_is_successful(self):
    param = calendar.SimpleCalendarDeleteOneEvent.generate_random_params()
    task = calendar.SimpleCalendarDeleteOneEvent(param)
    self.mock_list_rows.side_effect = [
        param["row_objects"] + param["noise_row_objects"],
        [],
    ]
    task.initialize_task(self.env)
    result = task.is_successful(self.env)
    self.assertEqual(result, 1)


class TestSimpleCalendarDeleteEventsOnRelativeDay(CalendarEventTestSetup):

  def test_generic_random_params(self):
    param = (
        calendar.SimpleCalendarDeleteEventsOnRelativeDay.generate_random_params()
    )
    events: list[sqlite_schema_utils.CalendarEvent] = cast(
        list[sqlite_schema_utils.CalendarEvent], param["row_objects"]
    ).copy()
    self.assertLen(events, 2)


class TestSimpleCalendarAddRepeatingEvent(CalendarEventTestSetup):

  @mock.patch.object(
      calendar_utils, "generate_simple_calendar_weekly_repeat_rule"
  )
  @mock.patch.object(events_generator, "generate_event")
  def test_generate_random_params_for_repeat_interval(
      self, mock_generate_event, mock_generate_rule
  ):
    mock_generate_rule.return_value = 2
    mock_generate_event.side_effect = _sample_events(
        sqlite_schema_utils.CalendarEvent(
            start_ts=datetime_utils.create_random_october_2023_unix_ts(),
            end_ts=datetime_utils.create_random_october_2023_unix_ts(),
            title="A Title",
            description="A Description",
            repeat_rule=0,
            repeat_interval=1,
        )
    )

    result = calendar.SimpleCalendarAddRepeatingEvent.generate_random_params()

    self.assertIn(result["row_objects"][0].repeat_rule, [0, 2])
    self.assertIn(
        result["row_objects"][0].repeat_interval,
        calendar._REPEAT_INTERVALS.values(),
    )


if __name__ == "__main__":
  absltest.main()
