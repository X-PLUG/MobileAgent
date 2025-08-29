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

"""Evaluators for Simple Calendar Pro.

They look at the underlying state of the sqlite database.
"""

from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils


def validate_event_removal_integrity(
    before: list[sqlite_schema_utils.CalendarEvent],
    after: list[sqlite_schema_utils.CalendarEvent],
    event_ids: list[int],
) -> bool:
  """Validates that events have been removed from the event list.

  See `sqlite_evaluators.validate_rows_removal_integrity` for details.

  Args:
    before: State of the events before removal, as a list of event tuples.
    after: State of the events after attempted removal, as a list of event
      tuples.
    event_ids: IDs of the events expected to be removed.

  Returns:
    True if specified events are removed and the integrity of the event list is
    maintained; False if any specified events are not removed, if any
    non-specified events are missing, or if new events have been added.
  """
  return sqlite_validators.validate_rows_removal_integrity(
      before, after, event_ids, 'id'
  )


def validate_event_addition_integrity(
    before: list[sqlite_schema_utils.CalendarEvent],
    after: list[sqlite_schema_utils.CalendarEvent],
    reference_events: list[sqlite_schema_utils.CalendarEvent],
    extras_compare: list[str] | None = None,
) -> bool:
  """Validates that specific events have been added correctly without side effects.

  By default, checks the following fields:
    - start_ts
    - end_ts
    - title  # Uses fuzzy match.
    - location  # Uses fuzzy match.
    - description  # Uses fuzzy match.

  Additional fields can be checked with `extras_compare`.

  Args:
      before: The state of the events before the addition.
      after: The state of the events after the attempted addition.
      reference_events: A list of events that are expected to be added.
      extras_compare: Additional fields to compare, if any.

  Returns:
      bool: True if the events were added correctly and other events remained
      unaltered. False otherwise.
  """

  # Default fields to compare
  compare_fields = [
      'start_ts',
      'end_ts',
      'title',
      'location',
      'description',
  ]
  free_form_fields = ['title', 'location', 'description']
  if extras_compare:
    compare_fields += extras_compare
  return sqlite_validators.validate_rows_addition_integrity(
      before, after, reference_events, compare_fields, free_form_fields
  )
