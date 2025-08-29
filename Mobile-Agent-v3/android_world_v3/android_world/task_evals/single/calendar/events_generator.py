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

"""Module for generating realistic calendar events.

It includes functions to generate event titles and descriptions using
predefined lists of titles, names, subjects, actions, and additional notes.
"""

import random
from android_world.task_evals.utils import sqlite_schema_utils


# Titles and Subjects for the event title generation
TITLES_PREFIXES = [
    'Meeting with',
    'Call with',
    'Workshop on',
    'Appointment for',
    'Catch up on',
    'Review session for',
]
NAMES = ['Alice', 'Bob', 'the Team', 'HR', 'Dr. Smith', 'Marketing']
SUBJECTS = ['Project X', 'Annual Report', 'Budget Planning', 'Campaign']

# Actions and Subjects Descriptions for the event description generation
ACTIONS = [
    'discuss',
    'finalize',
    'plan',
    'celebrate',
    'prepare for',
    'review',
    'explore',
    'understand',
    'organize',
    'strategize about',
]
SUBJECTS_DESCRIPTIONS = [
    'upcoming project milestones',
    'marketing strategies',
    'annual budget',
    'product launch',
    'team roles',
    'client feedback',
    'contract details',
    'software updates',
    'business objectives',
]
ADDITIONAL_NOTES = [
    'Please bring relevant documents.',
    'Remember to confirm attendance.',
    "Let's be punctual.",
    'Looking forward to productive discussions.',
    'Snacks will be provided.',
]


def generate_event(start_time: int) -> sqlite_schema_utils.CalendarEvent:
  """Generates a realistic calendar event.

  Args:
    start_time: The time to start the event. A Unix timestamp

  Returns:
    The event with random parameters.
  """
  end_time = start_time + (random.choice([15, 30, 45, 60]) * 60)
  return sqlite_schema_utils.CalendarEvent(
      start_ts=start_time,
      end_ts=end_time,
      title=generate_event_title(),
      description=generate_event_description(),
  )


def generate_event_title() -> str:
  """Generates a realistic event title."""
  title = random.choice(TITLES_PREFIXES)

  if 'with' in title:
    title += f' {random.choice(NAMES)}'
  else:
    title += f' {random.choice(SUBJECTS)}'

  return title


def generate_event_description() -> str:
  """Generates a realistic event description."""
  description = (
      'We will'
      f' {random.choice(ACTIONS)} {random.choice(SUBJECTS_DESCRIPTIONS)}.'
  )

  if random.choice([False, True]):
    description += f' {random.choice(ADDITIONAL_NOTES)}'

  return description
