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

import sqlite3
from absl.testing import absltest
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_test_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import datetime_utils


def remove_event_by_event_id(db_path: str, event_id: int):
  """Remove an event by its ID."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute('DELETE FROM events WHERE id=?', (event_id,))
  conn.commit()
  conn.close()


def add_event_to_db(db_path: str, event: sqlite_schema_utils.CalendarEvent):
  """Adds a new event to the database."""
  conn = sqlite3.connect(db_path)

  insert_command, values = sqlite_schema_utils.insert_into_db(
      event, 'events', 'id'
  )
  cursor = conn.cursor()
  cursor.execute(insert_command, values)

  conn.commit()
  conn.close()


def _validate_event_addition_integrity(
    before: list[sqlite_schema_utils.CalendarEvent],
    after: list[sqlite_schema_utils.CalendarEvent],
    reference_events: list[sqlite_schema_utils.CalendarEvent],
) -> bool:
  return sqlite_validators.validate_rows_addition_integrity(
      before,
      after,
      reference_events,
      [
          'start_ts',
          'end_ts',
          'title',
          'location',
          'description',
      ],
      ['title', 'location', 'description'],
  )


class TestRemoveEvent(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_db_path = sqlite_test_utils.setup_test_db()

  def test_single_event_removed_correctly(self):
    event_id = 1
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    remove_event_by_event_id(self.test_db_path, event_id)
    post_removal_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertTrue(
        sqlite_validators.validate_rows_removal_integrity(
            initial_state, post_removal_state, [event_id], 'id'
        )
    )

  def test_single_event_not_removed(self):
    event_id = 1
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    post_removal_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertFalse(
        sqlite_validators.validate_rows_removal_integrity(
            initial_state, post_removal_state, [event_id], 'id'
        )
    )

  def test_wrong_single_event_removed(self):
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    remove_event_by_event_id(self.test_db_path, 1)
    post_removal_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertFalse(
        sqlite_validators.validate_rows_removal_integrity(
            initial_state, post_removal_state, [2], 'id'
        )
    )

  def test_multiple_events_removed_correctly(self):
    event_ids = [2, 3]
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    for event_id in event_ids:
      remove_event_by_event_id(self.test_db_path, event_id)
    post_removal_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertTrue(
        sqlite_validators.validate_rows_removal_integrity(
            initial_state, post_removal_state, event_ids, 'id'
        )
    )

  def test_multiple_events_not_removed(self):
    event_ids = [2, 3]
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    post_removal_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertFalse(
        sqlite_validators.validate_rows_removal_integrity(
            initial_state, post_removal_state, event_ids, 'id'
        )
    )

  def test_remove_event_with_side_effects(self):
    # Test case: Remove events 4 and 5 but check only for event 4.
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )

    # Remove events 4 and 5
    remove_event_by_event_id(self.test_db_path, 4)
    remove_event_by_event_id(self.test_db_path, 5)

    post_removal_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )

    # Verify that while event 4 was removed, 5 was also removed meaning
    # there was an unintentional side-effect.
    self.assertFalse(
        sqlite_validators.validate_rows_removal_integrity(
            initial_state,
            post_removal_state,
            [4],
            'id',
        )
    )

  def test_event_not_in_before(self):
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    with self.assertRaises(ValueError):
      sqlite_validators.validate_rows_removal_integrity(
          initial_state, initial_state, [-999], 'id'
      )


class TestAddEvent(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_db_path = sqlite_test_utils.setup_test_db()

  def test_single_event_added_correctly(self):
    new_event = sqlite_schema_utils.CalendarEvent(
        start_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=9
        ),
        end_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=10
        ),
        title='Coffee',
        location='Cafe',
        description='Coffee with Alex',
    )

    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    add_event_to_db(self.test_db_path, new_event)
    post_addition_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )

    self.assertTrue(
        _validate_event_addition_integrity(
            initial_state, post_addition_state, [new_event]
        )
    )

  def test_multiple_events_added_correctly(self):
    new_events = [
        sqlite_schema_utils.CalendarEvent(
            start_ts=datetime_utils._create_unix_ts(
                year=2023, month=10, day=6, hour=11
            ),
            end_ts=datetime_utils._create_unix_ts(
                year=2023, month=10, day=6, hour=12
            ),
            title='Lunch',
            location='Restaurant',
            description='Lunch with Bob',
        ),
        sqlite_schema_utils.CalendarEvent(
            start_ts=datetime_utils._create_unix_ts(
                year=2023, month=10, day=7, hour=14
            ),
            end_ts=datetime_utils._create_unix_ts(
                year=2023, month=10, day=7, hour=15
            ),
            title='Meeting',
            location='Office',
            description='Project meeting',
        ),
    ]

    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    for event in new_events:
      add_event_to_db(self.test_db_path, event)
    post_addition_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )

    self.assertTrue(
        _validate_event_addition_integrity(
            initial_state, post_addition_state, new_events
        )
    )

  def test_no_event_added(self):
    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    post_addition_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertTrue(
        _validate_event_addition_integrity(
            initial_state, post_addition_state, []
        )
    )

  def test_wrong_event_added(self):
    event1 = sqlite_schema_utils.CalendarEvent(
        start_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=9
        ),
        end_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=10
        ),
        title='Coffee',
        location='Cafe',
        description='Coffee with Alex',
    )
    event2 = sqlite_schema_utils.CalendarEvent(
        start_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=9
        ),
        end_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=10
        ),
        title='Lunch',
        location='Eatery',
        description='Lunch with Joe',
    )

    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    add_event_to_db(self.test_db_path, event1)
    post_addition_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    self.assertFalse(
        _validate_event_addition_integrity(
            initial_state, post_addition_state, [event2]
        )
    )

  def test_add_duplicate_event(self):
    new_event = sqlite_schema_utils.CalendarEvent(
        start_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=9
        ),
        end_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=10
        ),
        title='Coffee',
        location='Cafe',
        description='Coffee with Alex',
    )

    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    add_event_to_db(self.test_db_path, new_event)
    # Add the same event again
    add_event_to_db(self.test_db_path, new_event)
    post_addition_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )

    # We expect this to fail if the event was added twice.
    self.assertFalse(
        _validate_event_addition_integrity(
            initial_state, post_addition_state, [new_event]
        )
    )

  def test_add_event_with_side_effects(self):
    event1 = sqlite_schema_utils.CalendarEvent(
        start_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=9
        ),
        end_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=10
        ),
        title='Coffee',
        location='Cafe',
        description='Coffee with Alex',
    )
    event2 = sqlite_schema_utils.CalendarEvent(
        start_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=9
        ),
        end_ts=datetime_utils._create_unix_ts(
            year=2023, month=10, day=6, hour=10
        ),
        title='Lunch',
        location='Eatery',
        description='Lunch with Joe',
    )

    initial_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )
    add_event_to_db(self.test_db_path, event1)
    add_event_to_db(self.test_db_path, event2)
    post_addition_state = sqlite_utils.execute_query(
        'SELECT * FROM events;',
        self.test_db_path,
        sqlite_schema_utils.CalendarEvent,
    )

    # We expect this to fail, since we added both event1 and event2.
    self.assertFalse(
        _validate_event_addition_integrity(
            initial_state, post_addition_state, [event1]
        )
    )


class TestVerifyPlaylist(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device_playlist_rows = [
        sqlite_schema_utils.PlaylistInfo('Summer Hits', 'song1.mp3', 0),
        sqlite_schema_utils.PlaylistInfo('Summer Hits', 'song2.mp3', 1),
        sqlite_schema_utils.PlaylistInfo('Summer Hits', 'song3.mp3', 2),
    ]
    self.candidate_playlist_name = 'Summer Hits'
    self.candidate_files = ['song1.mp3', 'song2.mp3', 'song3.mp3']

  def test_playlist_matches(self):
    result = sqlite_validators.verify_playlist(
        self.device_playlist_rows,
        self.candidate_playlist_name,
        self.candidate_files,
    )
    self.assertTrue(result)

  def test_playlist_does_not_match_due_to_order(self):
    self.candidate_files = [
        'song1.mp3',
        'song3.mp3',
        'song2.mp3',
    ]
    result = sqlite_validators.verify_playlist(
        self.device_playlist_rows,
        self.candidate_playlist_name,
        self.candidate_files,
    )
    self.assertFalse(result)

  def test_playlist_does_not_match_due_to_name(self):
    self.candidate_playlist_name = 'Winter Hits'
    result = sqlite_validators.verify_playlist(
        self.device_playlist_rows,
        self.candidate_playlist_name,
        self.candidate_files,
    )
    self.assertFalse(result)

  def test_empty_device_playlist(self):
    result = sqlite_validators.verify_playlist(
        [], self.candidate_playlist_name, self.candidate_files
    )
    self.assertFalse(result)

  def test_empty_candidate_files(self):
    result = sqlite_validators.verify_playlist(
        self.device_playlist_rows, self.candidate_playlist_name, []
    )
    self.assertFalse(result)


if __name__ == '__main__':
  absltest.main()
