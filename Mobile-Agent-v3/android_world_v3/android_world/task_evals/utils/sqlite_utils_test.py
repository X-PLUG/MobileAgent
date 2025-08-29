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

import os
import sqlite3
from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_env.wrappers import a11y_grpc_wrapper
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import interface
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_test_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import file_test_utils
from android_world.utils import file_utils


class SqliteUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.remote_db_path = sqlite_test_utils.setup_test_db()
    self.table_name = 'events'
    self.async_env_mock = mock.create_autospec(interface.AsyncEnv)
    self.android_env_mock = mock.create_autospec(
        env_interface.AndroidEnvInterface
    )
    self.enter_context(
        mock.patch.object(
            a11y_grpc_wrapper,
            'A11yGrpcWrapper',
            instance=True,
        )
    )
    self.controller = android_world_controller.AndroidWorldController(
        self.android_env_mock
    )
    self.async_env_mock.controller = self.controller
    self.row_type = sqlite_schema_utils.CalendarEvent

    self.mock_copy_db = self.enter_context(
        mock.patch.object(
            file_utils,
            'tmp_directory_from_device',
            side_effect=file_test_utils.mock_tmp_directory_from_device,
        )
    )
    self.mock_copy_data_to_device = self.enter_context(
        mock.patch.object(
            file_utils,
            'copy_data_to_device',
            side_effect=file_test_utils.mock_copy_data_to_device,
        )
    )
    self.mock_remove_files = self.enter_context(
        mock.patch.object(
            file_utils,
            'clear_directory',
            side_effect=file_test_utils.mock_remove_files,
        )
    )

  def test_get_rows_from_remote_device_success(self):
    expected_rows = sqlite_test_utils.get_db_rows()

    result = sqlite_utils.get_rows_from_remote_device(
        self.table_name, self.remote_db_path, self.row_type, self.async_env_mock
    )

    self.assertEqual(result, expected_rows)
    self.mock_copy_db.assert_called_once_with(
        os.path.dirname(self.remote_db_path), self.controller.env, None
    )

  @mock.patch.object(sqlite_utils, 'execute_query', autospec=True)
  def test_get_rows_from_remote_device_with_retries(self, mock_query_rows):
    mock_query_rows.side_effect = [
        sqlite3.OperationalError,
        sqlite_test_utils.get_db_rows(),
    ]

    result = sqlite_utils.get_rows_from_remote_device(
        self.table_name, self.remote_db_path, self.row_type, self.async_env_mock
    )

    self.assertEqual(result, sqlite_test_utils.get_db_rows())
    self.assertEqual(mock_query_rows.call_count, 2)

  @mock.patch.object(sqlite_utils, 'execute_query', autospec=True)
  def test_get_rows_from_remote_device_failure(self, mock_query_rows):
    mock_query_rows.side_effect = sqlite3.OperationalError

    with self.assertRaises(ValueError):
      sqlite_utils.get_rows_from_remote_device(
          self.table_name,
          self.remote_db_path,
          self.row_type,
          self.async_env_mock,
      )

  @mock.patch.object(adb_utils, 'close_app', autospec=True)
  def test_insert_rows_to_remote_db(self, mock_close_app):
    new_row = sqlite_schema_utils.CalendarEvent(
        start_ts=1672707600,
        end_ts=1672714800,
        title='A new row',
        location='location is here',
        description='',
        id=6,
    )

    sqlite_utils.insert_rows_to_remote_db(
        [new_row],
        'id',
        'events',
        self.remote_db_path,
        'TestApp',
        self.async_env_mock,
    )

    mock_close_app.assert_called_once_with('TestApp', self.controller)
    retrieved = sqlite_utils.get_rows_from_remote_device(
        self.table_name, self.remote_db_path, self.row_type, self.async_env_mock
    )
    original_rows = sqlite_test_utils.get_db_rows()
    self.assertEqual(retrieved, original_rows + [new_row])


if __name__ == '__main__':
  absltest.main()
