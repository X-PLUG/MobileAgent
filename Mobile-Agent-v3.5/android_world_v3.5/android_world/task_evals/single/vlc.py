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

"""Tasks for VLC player."""

import os
import random
from typing import Any
from android_world.env import interface
from android_world.env.setup_device import apps
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils

_DB_PATH = '/data/data/org.videolan.vlc/app_db/vlc_media.db'
_APP_NAME = 'vlc'


def _get_playlist_info_query() -> str:
  """Gets query for fetching playlists and their associated files."""
  return """
    SELECT
      Playlist.name AS playlist_name,
      Media.filename AS media_file_name,
      PlaylistMediaRelation.position AS order_in_playlist
    FROM
      PlaylistMediaRelation
    INNER JOIN Playlist ON Playlist.id_playlist = PlaylistMediaRelation.playlist_id
    INNER JOIN Media ON Media.id_media = PlaylistMediaRelation.media_id
    ORDER BY
      Playlist.name,
      PlaylistMediaRelation.position;
    """


def _clear_playlist_dbs(env: interface.AsyncEnv) -> None:
  """Clears all DBs related to playlists."""
  sqlite_utils.delete_all_rows_from_table('Playlist', _DB_PATH, env, _APP_NAME)
  sqlite_utils.delete_all_rows_from_table('Media', _DB_PATH, env, _APP_NAME)
  sqlite_utils.delete_all_rows_from_table(
      'PlaylistMediaRelation', _DB_PATH, env, _APP_NAME
  )


def _get_playlist_file_info(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.PlaylistInfo]:
  """Executes join query to fetch playlist file info."""
  with env.controller.pull_file(_DB_PATH, timeout_sec=3) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_DB_PATH)[1]
    )
    return sqlite_utils.execute_query(
        _get_playlist_info_query(),
        local_db_path,
        sqlite_schema_utils.PlaylistInfo,
    )


class _VLC(task_eval.TaskEval):

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    file_utils.clear_directory(apps.VlcApp.videos_path, env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_internal_storage(env)
    file_utils.clear_directory(apps.VlcApp.videos_path, env.controller)


class VlcCreatePlaylist(_VLC):
  """Task to create a playlist in VLC."""

  app_names = ['vlc']
  complexity = 2.8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist titled "{playlist_name}" with the following files'
        f' in VLC (located in Internal Memory/VLCVideos), in order: {files}'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = _get_playlist_file_info(env)
    return float(
        sqlite_validators.verify_playlist(
            actual, self.params['playlist_name'], self.params['files']
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }


class VlcCreateTwoPlaylists(task_eval.TaskEval):
  """Task to create two playlists in VLC."""

  app_names = ['vlc']
  complexity = 4.8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name1': {'type': 'string'},
          'files1': {
              'type': 'array',
              'items': {'type': 'string'},
          },
          'playlist_name2': {'type': 'string'},
          'files2': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name1', 'files1', 'playlist_name2', 'files2'],
  }
  template = ''  # Directly use goal.

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.task1_params = {
        'playlist_name': params['playlist_name1'],
        'files': params['files1'],
        'noise_files': params['noise_files1'],
    }
    self.task2_params = {
        'playlist_name': params['playlist_name2'],
        'files': params['files2'],
        'noise_files': params['noise_files2'],
    }
    self.task1 = VlcCreatePlaylist(self.task1_params)
    self.task2 = VlcCreatePlaylist(self.task2_params)

  @property
  def goal(self) -> str:
    goal1 = (
        f'Create a playlist titled "{self.params["playlist_name1"]}" with the'
        ' following files in VLC (located in Internal Memory/VLCVideos), in'
        f' order: {", ".join(self.params["files1"])}'
    )
    goal2 = (
        f'create a playlist titled "{self.params["playlist_name2"]}" with the'
        f' following files in VLC, in order: {", ".join(self.params["files2"])}'
    )
    return f'{goal1}. And then, {goal2}.'

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    self.task1.initialize_task(env)
    self.task2.setup_files(env)  # Don't want to clear db.

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.task1.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (self.task1.is_successful(env) + self.task2.is_successful(env)) / 2

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist1_params = VlcCreatePlaylist.generate_random_params()
    playlist2_params = VlcCreatePlaylist.generate_random_params()
    return {
        'playlist_name1': playlist1_params['playlist_name'],
        'files1': playlist1_params['files'],
        'noise_files1': playlist1_params['noise_files'],
        'playlist_name2': playlist2_params['playlist_name'],
        'files2': playlist2_params['files'],
        'noise_files2': playlist2_params['noise_files'],
    }


#### Synthetic data ############################################################


def generate_file_name() -> str:
  """Generates a more realistic and descriptive video file name."""
  prefixes = [
      'clip',
      'footage',
      'scene',
      'recording',
      'highlight',
      'moment',
      'episode',
  ]
  suffixes = [
      '',
      'HD',
      '4K',
      'raw',
      'export',
  ]
  prefix = random.choice(prefixes)
  suffix = random.choice(suffixes)
  num = str(random.randint(1, 99))
  name = f'{prefix}_{num}_{suffix}.mp4'
  return user_data_generation.generate_modified_file_name(name)


def _generate_playlist_name() -> str:
  """Generates realistic and descriptive playlist names."""
  themes = [
      'Adventure',
      'Comedy',
      'Daily Routines',
      'Documentary Insights',
      'Epic Moments',
      'Family Gatherings',
      'Fitness Challenges',
      'Gaming Sessions',
      'How To',
      'Mystery and Thrills',
      'Recipe Collection',
      'Road Trips',
      'Summer Highlights',
      'Tech Reviews',
      'Travel Guide',
      'Ultimate Fails',
  ]
  qualifiers = [
      'Essentials',
      'Favorites',
      'Marathon',
      'Playlist',
      'Series',
      'Specials',
      'Ultimate Collection',
  ]
  # Select a random theme and qualifier
  theme = random.choice(themes)
  qualifier = random.choice(qualifiers)
  # Form the playlist name
  return f'{theme} {qualifier}'
