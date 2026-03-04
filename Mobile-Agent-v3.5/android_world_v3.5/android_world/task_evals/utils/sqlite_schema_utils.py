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

"""Utilities for creating and processing rows in a SQLite database."""

import dataclasses
import datetime
import textwrap
from typing import Any, Callable, ClassVar, Optional, TypeVar
import uuid
from android_world.env import device_constants
from android_world.utils import datetime_utils

_YESTERDAY = device_constants.DT - datetime.timedelta(days=1)


@dataclasses.dataclass(frozen=True)
class SQLiteRow:
  """Base class for representing a row in a SQLite database table.

  Subclasses should define attributes corresponding to the table columns.
  """

  def to_csv_row(self, fields: list[str]) -> str:
    """Generates a CSV string representation of this instance.

    Args:
      fields: The fields of this instance to include in the CSV output.

    Returns:
      A string representing the CSV row for this instance.
    """
    return '|'.join(str(getattr(self, field, '')) for field in fields)

  def to_text_block(self, description_key: str, fields: list[str]) -> str:
    """Generates a text block representation of this instance.

    Args:
      description_key: The key for the main description/title of the text block.
      fields: The fields of this instance to include in the text block.

    Returns:
      A string representing the text block for this instance.
    """
    # Fetch the description/title.
    description = getattr(self, description_key, '')
    text_block = f'{description_key}: {description}\n'

    # Append additional fields.
    for field in fields:
      value = getattr(self, field, '')
      text_block += f' {field}: {value}\n'
    return text_block


def get_text_representation_of_rows(
    rows: list[SQLiteRow],
    fields: list[str],
    format_type: str = 'csv',
    description_key: str | None = None,
    wrap_width: int | None = None,
) -> str:
  """Formats a list of dataclass instances into a CSV string or a series of text blocks.

  Args:
    rows: A list of SQLiteRow instances.
    fields: The fields to include from each instance.
    format_type: The output format ('csv' or 'text_block').
    description_key: Key for the main description/title in text block format
      (required if format_type is 'text_block').
    wrap_width: If provided wrap text to be this width.

  Returns:
    A string representing the formatted output for the list of instances.
  """
  if format_type == 'csv':
    header = '|'.join(fields)
    rows = [
        '|'.join(str(getattr(instance, field, '')) for field in fields)
        for instance in rows
    ]
    return header + '\n' + '\n'.join(rows)
  elif format_type == 'text_block':
    blocks = []
    for instance in rows:
      if not description_key:
        raise ValueError('description_key is required for text block format')
      description = getattr(instance, description_key, '')
      text_block = f'{instance.__class__.__name__}: {description}\n'
      for field in fields:
        if field == description_key:
          continue
        value = getattr(instance, field, '')
        if wrap_width is not None:
          value = '\n'.join(textwrap.wrap(value, wrap_width))
        text_block += f' {field}: {value}\n'
      blocks.append(text_block)
    return '\n'.join(blocks)
  else:
    raise ValueError(
        "Invalid format_type specified. Choose 'csv' or 'text_block'."
    )


RowType = TypeVar('RowType', bound=SQLiteRow)


@dataclasses.dataclass(frozen=True)
class GenericRow(SQLiteRow):
  """Holds a row from an arbitrary database."""

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __getitem__(self, key):
    return self.__dict__[key]

  def __setitem__(self, key, value):
    raise TypeError('GenericRow is immutable')

  def __iter__(self):
    return iter(self.__dict__)

  def __len__(self):
    return len(self.__dict__)


@dataclasses.dataclass(frozen=True)
class CalendarEvent(SQLiteRow):
  """Represents a calendar event from the Simple Calendar Pro database."""

  start_ts: int
  end_ts: int
  title: str
  location: str = ''
  description: str = ''
  repeat_interval: int = 0
  repeat_rule: int = 0

  # Currently unused. We fill in with default values.
  reminder_1_minutes: int = -1
  reminder_2_minutes: int = -1
  reminder_3_minutes: int = -1
  reminder_1_type: int = 0
  reminder_2_type: int = 0
  reminder_3_type: int = 0
  repeat_limit: int = 0
  repetition_exceptions: str = '[]'
  attendees: str = ''
  import_id: str = ''
  time_zone: str = device_constants.TIMEZONE
  flags: int = 0
  event_type: int = 1
  parent_id: int = 0
  last_updated: int = 0
  source: str = 'imported-ics'
  availability: int = 0
  color: int = 0
  type: int = 0

  # Events in the database get an ID, due to autoincrement. Events initialized
  # independent on the DB do not need an ID.
  id: int = -1

  @property
  def duration_mins(self) -> int:
    if (self.end_ts - self.start_ts) % 60 != 0:
      raise ValueError('Duration should be even number of minutes.')
    return (self.end_ts - self.start_ts) // 60

  @property
  def start_datetime(self) -> datetime.datetime:
    """Python datetime object for the start time."""
    return datetime_utils.timestamp_to_localized_datetime(
        self.start_ts, timezone=device_constants.TIMEZONE
    )

  @property
  def end_datetime(self) -> datetime.datetime:
    """Python datetime object for the end time."""
    return datetime_utils.timestamp_to_localized_datetime(
        self.end_ts, timezone=device_constants.TIMEZONE
    )


@dataclasses.dataclass(frozen=True)
class Recipe(SQLiteRow):
  """Dataclass for a recipe in the Broccoli app."""

  title: str
  description: str = ''
  servings: str = ''
  preparationTime: str = ''  # pylint: disable=invalid-name
  source: str = ''
  ingredients: str = ''
  directions: str = ''
  favorite: int = 0

  imageName: str = ''  # pylint: disable=invalid-name

  # Auto-incremented primary key, default to -1 when not retrieved from the
  # database
  recipeId: int = -1  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class Expense(SQLiteRow):
  """Dataclass for an expense record."""

  name: str
  amount: int
  category: int = 0
  note: Optional[str] = ''
  created_date: int = 0
  modified_date: int = 0

  # Auto-incremented primary key, default to -1 when not retrieved from the
  # database
  expense_id: int = -1
  category_id_to_name: ClassVar[dict[int, str]] = {
      1: 'Others',
      2: 'Income',
      3: 'Food',
      4: 'Housing',
      5: 'Social',
      6: 'Entertainment',
      7: 'Transportation',
      8: 'Clothes',
      9: 'Health Care',
      10: 'Education',
      11: 'Donation',
  }

  @property
  def amount_dollars(self) -> str:
    return f'${self.amount / 100}'

  @property
  def category_name(self) -> str:
    return self.category_id_to_name[self.category]


@dataclasses.dataclass(frozen=True)
class PlaylistInfo(SQLiteRow):
  """Represents a playlist and metadata in VLC or similar media apps."""

  playlist_name: str
  media_file_name: str
  order_in_playlist: int
  duration_ms: int | None = None


# pylint: disable=invalid-name
@dataclasses.dataclass(frozen=True)
class Task(SQLiteRow):
  """Dataclass for a task in the application."""

  title: str
  importance: int = 0
  dueDate: int = 0
  hideUntil: int = 0
  created: int = 0
  modified: int = 0
  completed: int = 0
  deleted: int = 0
  notes: str | None = None
  estimatedSeconds: int = 0
  elapsedSeconds: int = 0
  timerStart: int = 0
  notificationFlags: int = 0
  lastNotified: int = 0
  recurrence: str | None = None
  repeat_from: int = 0
  calendarUri: str | None = None
  remoteId: str = ''
  collapsed: int = 0
  parent: int = 0
  order: int | None = None
  read_only: int = 0
  # pylint: enable=invalid-name

  # Auto-incremented primary key, default to -1 when not retrieved from the
  # database
  _id: int = -1


@dataclasses.dataclass(frozen=True)
class OsmAndMapMarker(SQLiteRow):
  """Dataclass for an OsmAnd app db row representing a map marker."""

  marker_id: str = ''
  marker_lat: float = -1.0
  marker_lon: float = -1.0
  marker_description: str = ''
  marker_active: int = 0
  marker_added: int = 0
  marker_visited: int = 0
  group_name: str = ''
  group_key: str = ''
  marker_color: int = 0
  marker_next_key: str = ''
  marker_disabled: int = 0
  marker_selected: int = 0
  marker_map_object_name: str = ''
  title: str = ''


@dataclasses.dataclass(frozen=True)
class SportsActivity(SQLiteRow):
  """Represents a row from the "track" table in OpenTracks.

  Note: These default values are provided for ease of use, but some of them
  should be set before uploading to table.
  """

  name: str
  description: str = ''
  category: str = ''
  # Should be equal to category, seems to simply set the activity icon in
  # the activity list.
  activity_type: str = ''
  starttime: int = 0
  stoptime: int = 0
  numpoints: int = 0
  totaldistance: float = 0.0  # Meters.
  # Milliseconds
  totaltime: int = 0
  movingtime: int = 0
  # All speed in meters per second.
  # If it doesn't match the given distance over time, the app recalculates them.
  # If they are set to 0, the app defaults to mph (instead of min/mile).
  avgspeed: float = 0.0
  avgmovingspeed: float = 0.0
  maxspeed: float = 0.0
  minelevation: float = 0.0
  maxelevation: float = 0.0
  elevationgain: float = 0.0
  icon: Optional[str] = None
  uuid: bytes = dataclasses.field(default_factory=lambda: uuid.uuid4().bytes)
  elevationloss: float = 0.0
  starttime_offset: int = 0

  # Auto-incremented primary key, default to -1 when not retrieved from the
  # database
  _id: int = -1


@dataclasses.dataclass(frozen=True)
class JoplinNormalizedNote(SQLiteRow):
  """Represents a row from the "notes_normalized" table in Joplin.

  Notes need to be added to this table for them to be searchable.
  """

  parent_id: str = ''
  title: str = ''
  body: str = ''
  latitude: float = 0.0
  longitude: float = 0.0
  altitude: float = 0.0
  source_url: str = ''
  is_todo: int = 0
  todo_due: int = 0
  todo_completed: int = 0
  user_created_time: int = 0
  user_updated_time: int = 0

  id: str = ''


@dataclasses.dataclass(frozen=True)
class JoplinNote(SQLiteRow):
  """Represents a row from the "notes" table in Joplin."""

  parent_id: str = ''
  title: str = ''
  body: str = ''
  created_time: int = int(_YESTERDAY.timestamp() * 1000)
  updated_time: int = int(_YESTERDAY.timestamp() * 1000)
  is_conflict: int = 0
  latitude: float = 0.0
  longitude: float = 0.0
  altitude: float = 0.0
  author: str = ''
  source_url: str = ''
  is_todo: int = 0
  todo_due: int = 0
  todo_completed: int = 0
  source: str = ''
  source_application: str = ''
  application_data: str = ''
  order: float = 0.0
  user_created_time: int = int(_YESTERDAY.timestamp() * 1000)
  user_updated_time: int = int(_YESTERDAY.timestamp() * 1000)
  encryption_cipher_text: str = ''
  encryption_applied: int = 0
  markup_language: int = 1
  is_shared: int = 0
  share_id: str = ''
  conflict_original_id: str = ''
  master_key_id: str = ''
  user_data: str = ''

  id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)


@dataclasses.dataclass(frozen=True)
class JoplinFolder(SQLiteRow):
  """Represents a row from "folder" table in Joplin."""

  title: str
  id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
  created_time: int = int(_YESTERDAY.timestamp() * 1000)
  updated_time: int = int(_YESTERDAY.timestamp() * 1000)
  user_created_time: int = int(_YESTERDAY.timestamp() * 1000)
  user_updated_time: int = int(_YESTERDAY.timestamp() * 1000)
  deleted_time: int = 0
  encryption_cipher_text: str = ''
  encryption_applied: int = 0
  parent_id: str = ''
  is_shared: int = 0
  share_id: str = ''
  master_key_id: str = ''
  icon: str = ''
  user_data: str = ''


def insert_into_db(
    data_object: SQLiteRow,
    table_name: str,
    exclude_key: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
  """Generates an SQL INSERT command to add a new row to the specified table.

  Args:
      data_object: An object representing the data to be added.
      table_name: Name of the table to insert data into.
      exclude_key: Typically, the ID key which is auto-incrementing, so we do
        not add it; the db will create it.

  Returns:
      A tuple containing the SQL INSERT command and the values to be inserted.
  """
  fields = []
  for field in dataclasses.fields(data_object):
    if exclude_key is not None and field.name == exclude_key:
      continue
    fields.append(field)
  column_names = ', '.join(f'"{field.name}"' for field in fields)
  placeholders = ', '.join('?' * len(fields))

  insert_command = (
      f'INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})'
  )
  values = tuple(getattr(data_object, field.name) for field in fields)

  return insert_command, values


def _is_candidate_equal_to_any_result(
    candidate: Any, result: list[Any]
) -> bool:
  """Private function to check if a candidate is equal to any of the objects in result."""
  for existing in result:
    if all(
        getattr(candidate, field.name) == getattr(existing, field.name)
        for field in dataclasses.fields(candidate)
    ):
      return True
  return False


def get_random_items(
    n: int,
    generate_item_fn: Callable[[], RowType],
    replacement: bool = False,
    filter_fn: Optional[Callable[[RowType], bool]] = None,
) -> list[RowType]:
  """Generates a list of random items, optionally filtering and avoiding duplicates.

  Args:
      n: The number of items to generate.
      generate_item_fn: Function to generate a single random item.
      replacement: Whether to allow replacement (duplicates) in the returned
        list.
      filter_fn: Optional function to filter items. If None, all items are
        accepted.

  Returns:
      A list of randomly generated items.
  """
  if not filter_fn:
    filter_fn = lambda _: True
  result = []
  i = 0
  while len(result) < n:
    candidate = generate_item_fn()
    i += 1
    if i > 10_000:
      raise ValueError(
          'Something went wrong: generation exhaused. There are total of'
          f" {len(result)} items created; couldn't generate {n} items."
      )
    if not filter_fn(candidate):
      continue
    if replacement:
      result.append(candidate)
    elif not _is_candidate_equal_to_any_result(candidate, result):
      result.append(candidate)
  return result
