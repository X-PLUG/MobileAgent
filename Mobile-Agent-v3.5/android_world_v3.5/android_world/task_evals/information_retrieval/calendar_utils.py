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

"""Information Retrieval utils for Simple Calendar Pro."""

import datetime
import random
import zoneinfo
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.single.calendar import calendar_utils as utils
from android_world.task_evals.single.calendar import events_generator
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import datetime_utils

TIME_FORMAT = '%H:%M'
DEFAULT_DURATION_S = 1800  # 30 minutes


def parse_duration(duration: str) -> int:
  """Converts a duration string into a seconds integer.

  Handles the following formats:
    %dh,%d hour(s), %dm, %d minutes

  This is not very robust and assumes the inputs are correctly formatted.
  Args:
    duration: The duration string to convert.

  Returns:
    int: The number of seconds corresponding to the input duration.
  """
  if 'h' in duration:
    return int(3600 * float(duration[: duration.find('h')].strip()))
  if 'm' in duration:
    return 60 * int(duration[: duration.find('m')].strip())

  raise ValueError(f'Invalid duration: {duration}')


def create_event_from_proto(
    event: state_pb2.Event,
) -> sqlite_schema_utils.CalendarEvent:
  """Creates an Event object from a state_pb2.Event proto."""
  start_unix_ts = convert_datetime_to_unix_ts(
      event.start_date, event.start_time
  )
  duration = DEFAULT_DURATION_S
  if event.HasField('duration'):
    duration = parse_duration(event.duration)
  end_ts = start_unix_ts + duration
  return sqlite_schema_utils.CalendarEvent(
      start_ts=start_unix_ts,
      end_ts=end_ts,
      title=event.title,
      location=event.location,
      description=event.description,
  )


def convert_str_to_datetime(
    date_str: str, time_str: str, tzinfo: zoneinfo.ZoneInfo | None = None
) -> datetime.datetime:
  """Converts a date string and a time string to a datetime.

  Handles the following formats for date_str:
    <month name> <day> <year>

  Handles the following formats for time_str:
    <24 hour format>:<minute> : e.g. 10:30, 15:00
    <12 hour format>:<minute><pm/am>: e.g. 10:30am, 10:00pm
    <12 hour format><pm/am> : e.g. 10am, 10pm

  Args:
    date_str: The date string to convert.
    time_str: The time string to convert.
    tzinfo: The timezone to use. If None, uses the default timezone.

  Returns:
    The datetime corresponding to the input date string.
  """
  dt = datetime_utils_ir.get_date(date_str)

  if dt is None:
    dt = datetime.date.fromtimestamp(device_constants.DT.timestamp())
  time_dt = datetime.time(hour=0)
  if time_str:
    time_dt = datetime_utils_ir.parse_time(time_str)
  hour = time_dt.hour
  minute = time_dt.minute

  localized_dt = datetime.datetime(
      dt.year,
      dt.month,
      dt.day,
      hour,
      minute,
      tzinfo=tzinfo if tzinfo else zoneinfo.ZoneInfo(device_constants.TIMEZONE),
  )
  return localized_dt


def convert_datetime_to_unix_ts(date_str: str, time_str: str) -> int:
  """Converts a date string and a time string to a datetime.

  See convert_str_to_datetime for supported string formats.

  Args:
    date_str: The date string to convert.
    time_str: The time string to convert.

  Returns:
    The UNIX timestamp corresponding to the input date and time strings.
  """
  return int(convert_str_to_datetime(date_str, time_str).timestamp())


def setup_task_state(
    relevant_state: state_pb2.Calendar,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
    env: interface.AsyncEnv,
) -> None:
  """Initializes the calendar app with initial state + other random events.

  This is specifically for information retrieval tasks. Other tasks
  should be initialized using SQLiteApp.

  Args:
    relevant_state: The initial, required Calendar state to set up the app with.
    exclusion_conditions: A list of conditions that constrain the extra random
      events.
    env: The android environment instance.
  """
  utils.clear_calendar_db(env)
  events = []
  for event in relevant_state.events:
    events.append(create_event_from_proto(event))
  events += [generate_random_event(exclusion_conditions) for _ in range(75)]
  random.shuffle(events)
  utils.add_events(events, env)


def generate_random_event(
    exclusion_conditions: list[task_pb2.ExclusionCondition],
    max_retries: int = 100,
) -> sqlite_schema_utils.CalendarEvent:
  """Generates a random event with the given exclusion conditions."""
  attempt = 0
  while attempt < max_retries:
    event = state_pb2.Event()
    random_datetime = datetime_utils.generate_random_datetime(
        window_size=datetime.timedelta(days=30)
    )
    event.start_date = random_datetime.date().strftime(
        datetime_utils_ir.DATE_FORMAT
    )
    event.start_time = random_datetime.time().strftime(TIME_FORMAT)
    random_duration = random.choice([15, 30, 45, 60])
    event.duration = '{} m'.format(random_duration)
    event.title = events_generator.generate_event_title()
    event.description = events_generator.generate_event_description()
    if check_event_conditions(event, exclusion_conditions):
      start_ts = int(random_datetime.timestamp())
      end_ts = start_ts + random_duration * 60
      return sqlite_schema_utils.CalendarEvent(
          start_ts=start_ts,
          end_ts=end_ts,
          title=event.title,
          description=event.description,
      )
    attempt += 1
  raise ValueError(
      'Failed to create a random event that meets the exclusion conditions'
  )


def check_event_conditions(
    event: state_pb2.Event,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
) -> bool:
  """Evaluates the specified event against a set of exclusion conditions.

  An event is considered eligible if it does not satisfy all of the conditions
  specified in the exclusion_conditions list. Each condition is checked against
  various fields of the event such as start date, start time, and title. The
  event is eligible if not all of these conditions are met, ensuring it doesn't
  fall under any exclusion criteria defined.

  Args:
    event: The event to check
    exclusion_conditions: All the conditions the event will be checked against,
      if they are all met, this event should be excluded and does not meet the
      conditions.

  Returns:
    A bool, True if the event does not meet all of the exclusion conditions,
    False
      otherwise.
  """
  if not exclusion_conditions:
    return True
  # Keeps track of whether an exclusion condition is met.
  all_conditions_met = True

  # If both start_date and start_time are specified in the exclusion conditions
  # as exact matches, use them together as a single datetime.
  fields_to_conditions = {
      condition.field: condition
      for condition in exclusion_conditions
      if condition.operation == task_pb2.ExclusionCondition.Operation.EQUAL_TO
  }
  if (
      'start_date' in fields_to_conditions.keys()
      and 'start_time' in fields_to_conditions.keys()
  ):
    condition_date = datetime_utils_ir.get_date(
        fields_to_conditions['start_date'].value
    )
    condition_time = datetime_utils_ir.parse_time(
        fields_to_conditions['start_time'].value
    )
    condition_datetime = datetime.datetime.combine(
        condition_date, condition_time
    )
  else:
    condition_datetime = None
  for condition in exclusion_conditions:
    start_date = datetime_utils_ir.get_date(event.start_date)
    start_time = datetime.datetime.strptime(
        event.start_time, TIME_FORMAT
    ).time()
    start_datetime = datetime.datetime.combine(start_date, start_time)
    end_datetime = start_datetime + datetime.timedelta(
        seconds=parse_duration(event.duration)
    )
    if condition.field == 'start_date':
      condition_value = (
          condition_datetime
          if condition_datetime
          else datetime_utils_ir.get_date(condition.value)
      )
      event_start_value = start_datetime if condition_datetime else start_date
      event_end_value = (
          end_datetime if condition_datetime else end_datetime.date()
      )

      if condition.operation == task_pb2.ExclusionCondition.Operation.EQUAL_TO:

        # Checks that no date between start and end overlaps with the
        # condition value.
        overlaps = proto_utils.compare(
            event_start_value,
            task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
            condition_value,
        ) and proto_utils.compare(
            event_end_value,
            task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
            condition_value,
        )
      else:
        # Checks if the whole span of the event meets the exclusion condition in
        # case the event spans multiple days.
        overlaps = proto_utils.compare(
            event_start_value,
            condition.operation,
            condition_value,
        ) or proto_utils.compare(
            event_end_value,
            condition.operation,
            condition_value,
        )
      all_conditions_met = all_conditions_met and overlaps

    elif condition.field == 'start_time':
      condition_value = (
          condition_datetime
          if condition_datetime
          else datetime_utils_ir.parse_time(condition.value)
      )
      event_start_value = start_datetime if condition_datetime else start_time
      event_end_value = (
          end_datetime if condition_datetime else end_datetime.time()
      )

      if condition.operation == task_pb2.ExclusionCondition.Operation.EQUAL_TO:
        # Checks that no time between start and end time overlaps with the
        # condition value.
        overlaps = proto_utils.compare(
            event_start_value,
            task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
            condition_value,
        ) and proto_utils.compare(
            event_end_value,
            task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
            condition_value,
        )

      else:
        # Checks if the whole span of the event meets the exclusion condition.
        overlaps = proto_utils.compare(
            event_start_value,
            condition.operation,
            condition_value,
        ) or proto_utils.compare(
            event_end_value,
            condition.operation,
            condition_value,
        )
      all_conditions_met = all_conditions_met and overlaps

    elif condition.field == 'title':
      all_conditions_met = all_conditions_met and proto_utils.compare(
          event.title.lower(), condition.operation, condition.value.lower()
      )

  return not all_conditions_met
