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

"""Information Retrieval utils for datetime."""

import datetime
import random
from android_world.env import device_constants

DATE_FORMAT = '%B %d %Y'


def get_date(date_str: str) -> datetime.date:
  return datetime.datetime.strptime(date_str, DATE_FORMAT).date()


def _generate_nl_date_options(date_str: str) -> list[str]:
  """Lists all options for a natural language way of expressing date.

  Possible options include:
    - today, tomorrow, yesterday if they apply
    - <day of week> if the day is within a week in the future or in the past.
    - 'this <day of week>' if the day is within a week in the future.
    - 'the <day of week> after next' if it applies.
    - <month name> <day>
    - <month name> <day> <year>

  Args:
    date_str: The date to rephrase in a natural language formats.

  Returns:
    A list of strings representing the date in a natural way.
  """
  date = get_date(date_str)
  options = [date.strftime('%B %d'), date.strftime(DATE_FORMAT)]
  if date == device_constants.DT.date():
    options.append('today')
  if date == device_constants.DT.date() + datetime.timedelta(days=1):
    options.append('tomorrow')
  if date == device_constants.DT.date() - datetime.timedelta(days=1):
    options.append('yesterday')
  if date > device_constants.DT.date():
    day_name = date.strftime('%A')
    if date - device_constants.DT.date() <= datetime.timedelta(days=7):
      options.append(day_name)
      options.append('this {}'.format(day_name))
    elif date - device_constants.DT.date() <= datetime.timedelta(days=14):
      options.append('the {} after next'.format(day_name))
  if date < device_constants.DT.date():
    day_name = date.strftime('%A')
    if device_constants.DT.date() - date <= datetime.timedelta(days=7):
      options.append(day_name)
  return options


def generate_reworded_date(date_str: str) -> str:
  """Randomly generates a natural language way of expressing date.

  Uses the following options:
    - today, tomorrow, yesterday if they apply
    - <day of week> if the day is within a week in the future or in the past.
    - 'this <day of week>' if the day is within a week in the future.
    - 'the <day of week> after next' if it applies.
    - <month name> <day>
    - <month name> <day> <year>

  Args:
    date_str: The date to rephrase in a natural language format.

  Returns:
    A string representing the date in a natural way.
  """

  options = _generate_nl_date_options(date_str)
  return random.choice(options)


def parse_time(time_str: str) -> datetime.time:
  """Parse a time string into a datetime object using multiple formats.

  The following formats are handled:
    <24 hour format>:<minute> : e.g. 10:00, 15:00
    <12 hour format>:<minute><pm/am>: e.g. 10:00am, 10:00pm
    <12 hour format><pm/am> : e.g. 10am, 10pm

  Args:
    time_str: The string representation of the time.

  Returns:
    A datetime.time object representing the time.

  Raises:
    ValueError: If the time string does not match any of the expected formats.
  """
  time_formats = ('%H:%M', '%I:%M%p', '%I%p')
  for fmt in time_formats:
    try:
      dt = datetime.datetime.strptime(time_str, fmt)
      return datetime.time(hour=dt.hour, minute=dt.minute)
    except ValueError:
      pass
  raise ValueError(f"Time string '{time_str}' does not match any known format.")
