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

import dataclasses
import os
import sqlite3
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from android_env import env_interface
from android_env.wrappers import a11y_grpc_wrapper
from android_world.env import android_world_controller
from android_world.env import interface
from android_world.env.setup_device import apps
from android_world.task_evals.single import vlc
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import app_snapshot
from android_world.utils import file_test_utils
from android_world.utils import file_utils


@dataclasses.dataclass
class Playlist:
  name: str
  files: list[str]  # List of filenames for media files in the playlist


def _set_state_of_db(test_db_path: str, playlists: list[Playlist]):
  """Inserts playlists and their media files into the mock database."""

  if os.path.exists(test_db_path):
    os.remove(test_db_path)
  conn = sqlite3.connect(test_db_path)
  cursor = conn.cursor()

  # Create tables
  cursor.executescript("""
          CREATE TABLE Playlist(id_playlist INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);
          CREATE TABLE Media(id_media INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT UNIQUE);
          CREATE TABLE PlaylistMediaRelation(media_id INTEGER, playlist_id INTEGER, position INTEGER);
      """)

  for playlist in playlists:
    # Insert the playlist
    cursor.execute('INSERT INTO Playlist(name) VALUES (?)', (playlist.name,))
    playlist_id = cursor.lastrowid

    for filename in playlist.files:
      cursor.execute(
          """
      INSERT INTO Media(filename) VALUES (?)
      ON CONFLICT(filename) DO UPDATE SET filename = excluded.filename
      """,
          (filename,),
      )
      media_id = cursor.lastrowid
      # Relate media file to playlist
      cursor.execute(
          'INSERT INTO PlaylistMediaRelation(media_id, playlist_id, position)'
          ' VALUES (?, ?, ?)',
          (media_id, playlist_id, playlist.files.index(filename)),
      )

  conn.commit()
  conn.close()


class VlcTestBase(parameterized.TestCase):

  def setUp(self):
    """Set up the test environment and mock database."""
    super().setUp()
    self.env_mock = mock.create_autospec(interface.AsyncAndroidEnv)
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
    self.env_mock.controller = self.controller

    temp_dir = tempfile.mkdtemp()
    self.test_db_path = file_utils.convert_to_posix_path(
        temp_dir, 'app_db/vlc_media.db'
    )
    os.makedirs(
        file_utils.convert_to_posix_path(
            os.path.dirname(self.test_db_path), 'app_db'
        ),
        exist_ok=True,
    )

    vlc._DB_PATH = self.test_db_path

    # Mock file and SQLite utility functions
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
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, 'restore_snapshot')
    )

  def tearDown(self):
    super().tearDown()
    vlc._DB_PATH = '/data/data/org.videolan.vlc/app_db/vlc_media.db'


class VlcUtilsTest(VlcTestBase):

  def test_get_playlist_info(self):
    """Test fetching playlist information."""
    expected_info = [
        sqlite_schema_utils.PlaylistInfo(
            playlist_name='Test Playlist',
            media_file_name='test_media.mp4',
            order_in_playlist=0,
        )
    ]

    _set_state_of_db(
        self.test_db_path, [Playlist('Test Playlist', ['test_media.mp4'])]
    )
    result = vlc._get_playlist_file_info(self.env_mock)
    self.assertEqual(result, expected_info)


class VlcTaskEvalsTestBase(VlcTestBase):

  def setUp(self):
    super().setUp()
    self.mock_write_video_file_to_device = self.enter_context(
        mock.patch.object(
            user_data_generation, 'write_video_file_to_device', autospec=True
        )
    )
    self.mock_generate_random_string = self.enter_context(
        mock.patch.object(
            user_data_generation, 'generate_random_string', autospec=True
        )
    )
    self.mock_remove_files = self.enter_context(
        mock.patch.object(file_utils, 'clear_directory', autospec=True)
    )
    _set_state_of_db(
        self.test_db_path,
        [
            Playlist(
                'Stale Playlist',
                ['stale_test_media.mp4', 'stale_test_media2.mp4'],
            )
        ],
    )


class VlcCreatePlaylist(VlcTaskEvalsTestBase):

  def test_goal(self):
    params = {
        'playlist_name': 'Test Playlist',
        'files': ['test_media.mp4, test_media2.mp4'],
    }

    instance = vlc.VlcCreatePlaylist(params)

    self.assertEqual(
        instance.goal,
        'Create a playlist titled "Test Playlist" with the following files in'
        ' VLC (located in Internal Memory/VLCVideos), in order: test_media.mp4,'
        ' test_media2.mp4',
    )

  def test_initialize_task(self):
    params = {
        'playlist_name': 'Test Playlist',
        'files': ['test_media.mp4', 'test_media2.mp4'],
        'noise_files': ['noise_media.mp4'],
    }
    self.mock_generate_random_string.side_effect = ['hello', 'world', 'test']

    instance = vlc.VlcCreatePlaylist(params)
    instance.initialize_task(self.env_mock)

    expected_calls = [
        mock.call(
            'test_media.mp4',
            apps.VlcApp.videos_path,
            self.env_mock,
            messages=['hello'],
            fps=1,
            message_display_time=mock.ANY,
        ),
        mock.call(
            'test_media2.mp4',
            apps.VlcApp.videos_path,
            self.env_mock,
            messages=['world'],
            fps=1,
            message_display_time=mock.ANY,
        ),
        mock.call(
            'noise_media.mp4',
            apps.VlcApp.videos_path,
            self.env_mock,
            messages=['test'],
            fps=1,
            message_display_time=mock.ANY,
        ),
    ]

    self.mock_write_video_file_to_device.assert_has_calls(
        expected_calls, any_order=False
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no fuzzy',
          playlist_name='Test Playlist',
      ),
      dict(
          testcase_name='fuzzy1',
          playlist_name='test playlist',
      ),
      dict(
          testcase_name='fuzzy2',
          playlist_name='test playlis',
      ),
  )
  def test_is_successful(self, playlist_name: str):
    params = {
        'playlist_name': 'Test Playlist',
        'files': ['test_media.mp4', 'test_media2.mp4', 'test_media3.mp4'],
        'noise_files': [],
    }
    instance = vlc.VlcCreatePlaylist(params)
    instance.initialize_task(self.env_mock)
    self.assertFalse(instance.is_successful(env=self.env_mock))
    _set_state_of_db(
        self.test_db_path,
        [
            Playlist(
                playlist_name,
                ['test_media.mp4', 'test_media2.mp4', 'test_media3.mp4'],
            )
        ],
    )

    self.assertTrue(instance.is_successful(env=self.env_mock))
    instance.tear_down(self.env_mock)

  def test_is_successful_fails(self):
    params = {
        'playlist_name': 'Test Playlist',
        'files': ['test_media.mp4', 'test_media2.mp4', 'test_media3.mp4'],
        'noise_files': [],
    }
    instance = vlc.VlcCreatePlaylist(params)
    instance.initialize_task(self.env_mock)
    self.assertFalse(instance.is_successful(env=self.env_mock))
    _set_state_of_db(
        self.test_db_path,
        [
            Playlist(
                'Test Playlist',
                ['test_media.mp4', 'test_media2.mp4'],
            )
        ],
    )

    self.assertFalse(instance.is_successful(env=self.env_mock))
    instance.tear_down(self.env_mock)


class TestCreateTwoPlaylists(VlcTaskEvalsTestBase):

  def test_goal(self):
    params = {
        'playlist_name1': 'Playlist One',
        'files1': ['file1.mp3', 'file2.mp3'],
        'noise_files1': ['noise_file1.mp3', 'noise_file2.mp3'],
        'playlist_name2': 'Playlist Two',
        'files2': ['file3.mp3', 'file4.mp3'],
        'noise_files2': ['noise_file3.mp3', 'noise_file4.mp3'],
    }
    instance = vlc.VlcCreateTwoPlaylists(params)

    expected_goal = (
        'Create a playlist titled "Playlist One" with the following files in'
        ' VLC (located in Internal Memory/VLCVideos), in order: file1.mp3,'
        ' file2.mp3. And then, create a playlist titled "Playlist Two" with the'
        ' following files in VLC, in order: file3.mp3, file4.mp3.'
    )
    self.assertEqual(instance.goal, expected_goal)

  def test_initialize_task(self):
    params = {
        'playlist_name1': 'Playlist One',
        'files1': ['file1.mp3', 'file2.mp3'],
        'noise_files1': ['noise_file1.mp3'],
        'playlist_name2': 'Playlist Two',
        'files2': ['file3.mp3', 'file4.mp3'],
        'noise_files2': ['noise_file2.mp3'],
    }
    self.mock_generate_random_string.side_effect = [
        'random1',
        'random2',
        'noise1',
        'random3',
        'random4',
        'noise2',
    ]

    instance = vlc.VlcCreateTwoPlaylists(params)
    instance.initialize_task(self.env_mock)

    expected_calls = [
        mock.call(
            filename,
            apps.VlcApp.videos_path,
            self.env_mock,
            messages=[mock.ANY],
            fps=1,
            message_display_time=mock.ANY,
        )
        for filename in [
            'file1.mp3',
            'file2.mp3',
            'noise_file1.mp3',
            'file3.mp3',
            'file4.mp3',
            'noise_file2.mp3',
        ]
    ]
    self.mock_write_video_file_to_device.assert_has_calls(expected_calls)

  def test_is_successful(self):
    params = {
        'playlist_name1': 'Playlist One',
        'files1': ['file1.mp3', 'file2.mp3'],
        'noise_files1': [],
        'playlist_name2': 'Playlist Two',
        'files2': ['file3.mp3', 'file4.mp3'],
        'noise_files2': [],
    }
    create_two_playlists_task = vlc.VlcCreateTwoPlaylists(params)
    create_two_playlists_task.initialize_task(self.env_mock)
    self.assertFalse(create_two_playlists_task.is_successful(self.env_mock))
    _set_state_of_db(
        self.test_db_path,
        [
            Playlist('Playlist One', ['file1.mp3', 'file2.mp3']),
            Playlist('Playlist Two', ['file3.mp3', 'file4.mp3']),
        ],
    )

    self.assertTrue(create_two_playlists_task.is_successful(self.env_mock))
    create_two_playlists_task.tear_down(self.env_mock)

  def test_is_successful_fail(self):
    params = {
        'playlist_name1': 'Playlist One',
        'files1': ['file1.mp3', 'file2.mp3', 'file3.mp3'],
        'noise_files1': [],
        'playlist_name2': 'Playlist Two',
        'files2': ['file3.mp3'],
        'noise_files2': [],
    }
    create_two_playlists_task = vlc.VlcCreateTwoPlaylists(params)
    create_two_playlists_task.initialize_task(self.env_mock)
    self.assertFalse(create_two_playlists_task.is_successful(self.env_mock))
    _set_state_of_db(
        self.test_db_path,
        [
            Playlist('Playlist One', ['file1.mp3', 'file2.mp3']),
            Playlist('Playlist Two', ['file3.mp3', 'file4.mp3']),
        ],
    )

    self.assertFalse(create_two_playlists_task.is_successful(self.env_mock))
    create_two_playlists_task.tear_down(self.env_mock)

  @parameterized.named_parameters(
      dict(
          testcase_name='partial_one',
          playlist=Playlist('Playlist One', ['file1.mp3', 'file2.mp3']),
      ),
      dict(
          testcase_name='partial_two',
          playlist=Playlist('Playlist Two', ['file3.mp3', 'file4.mp3']),
      ),
  )
  def test_is_successful_partial(self, playlist: Playlist):
    params = {
        'playlist_name1': 'Playlist One',
        'files1': ['file1.mp3', 'file2.mp3'],
        'noise_files1': [],
        'playlist_name2': 'Playlist Two',
        'files2': ['file3.mp3', 'file4.mp3'],
        'noise_files2': [],
    }
    create_two_playlists_task = vlc.VlcCreateTwoPlaylists(params)
    create_two_playlists_task.initialize_task(self.env_mock)
    self.assertFalse(create_two_playlists_task.is_successful(self.env_mock))
    _set_state_of_db(
        self.test_db_path,
        [playlist],
    )

    self.assertEqual(
        create_two_playlists_task.is_successful(self.env_mock), 0.5
    )
    create_two_playlists_task.tear_down(self.env_mock)


if __name__ == '__main__':
  absltest.main()
