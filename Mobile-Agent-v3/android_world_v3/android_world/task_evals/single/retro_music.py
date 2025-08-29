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

"""Tasks for Retro Music app."""

import dataclasses
import os
import random
from typing import Any
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


_APP_NAME = 'retro music'
_PLAYLIST_DB_PATH = (
    '/data/data/code.name.monkey.retromusic/databases/playlist.db'
)
_PLAYBACK_DB_PATH = (
    '/data/data/code.name.monkey.retromusic/databases/music_playback_state.db'
)


def _get_playlist_info_query() -> str:
  """Gets query for fetching playlists and their associated files."""
  # Note: Since we are using a window function, this requires sqlite 3.25.0.
  return """
    SELECT
        pe.playlist_name AS playlist_name,
        se.title AS media_file_name,
        se.duration AS duration_ms,
        ROW_NUMBER() OVER (
            PARTITION BY pe.playlist_name
            ORDER BY se.song_key
        ) - 1 AS order_in_playlist
    FROM
        PlaylistEntity pe
        JOIN SongEntity se ON pe.playlist_id = se.playlist_creator_id
    ORDER BY
        pe.playlist_name,
        order_in_playlist;
    """


def _get_playlist_data(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.PlaylistInfo]:
  """Executes join query to fetch playlist file info."""
  with env.controller.pull_file(
      _PLAYLIST_DB_PATH, timeout_sec=3
  ) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_PLAYLIST_DB_PATH)[1]
    )
    return sqlite_utils.execute_query(
        _get_playlist_info_query(),
        local_db_path,
        sqlite_schema_utils.PlaylistInfo,
    )


def _get_playing_queue(env: interface.AsyncEnv) -> list[str]:
  """Executes join query to fetch playlist file info."""

  @dataclasses.dataclass(frozen=True)
  class Queue(sqlite_schema_utils.SQLiteRow):
    title: str

  with env.controller.pull_file(
      _PLAYBACK_DB_PATH, timeout_sec=3
  ) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_PLAYBACK_DB_PATH)[1]
    )
    result = sqlite_utils.execute_query(
        'SELECT title from playing_queue;',
        local_db_path,
        Queue,
    )
    return [r.title for r in result]


def _clear_playlist_dbs(env: interface.AsyncEnv) -> None:
  """Clears all DBs related to playlists."""
  sqlite_utils.delete_all_rows_from_table(
      'PlaylistEntity', _PLAYLIST_DB_PATH, env, _APP_NAME
  )
  sqlite_utils.delete_all_rows_from_table(
      'SongEntity', _PLAYLIST_DB_PATH, env, _APP_NAME
  )


def _scan_music_directory(env: interface.AsyncEnv):
  """Scans the music directory to update the media store."""
  action = 'android.intent.action.MEDIA_SCANNER_SCAN_FILE'
  data_uri = 'file:///storage/emulated/0/Music'
  adb_utils.send_android_intent(
      command='broadcast', action=action, env=env.controller, data_uri=data_uri
  )
  adb_utils.close_app('retro music', env.controller)


class RetroCreatePlaylist(task_eval.TaskEval):
  """Task to create a playlist in Retro Music."""

  app_names = ['retro music']
  complexity = 2.4
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
  template = ''  # Directly use goal.

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist in Retro Music titled "{playlist_name}" with the'
        f' following songs, in order: {names}'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    _clear_playlist_dbs(env)

    for file in self.params['files'] + self.params['noise_files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(3 * 60 * 1000, 5 * 60 * 1000),
      )
    _scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = _get_playlist_data(env)
    return int(
        sqlite_validators.verify_playlist(
            actual,
            self.params['playlist_name'],
            [f.split('.')[0] for f in self.params['files']],
        )
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_internal_storage(env)
    _clear_playlist_dbs(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]
    num_files = random.randint(2, 5)
    files, noise_files = files[0:num_files], files[num_files:]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': noise_files,
    }


class RetroPlayingQueue(RetroCreatePlaylist):
  """Task to create a playing queue in Retro Music."""

  complexity = 3.2

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    return (
        f'Add the following songs, in order, {names} to my playing queue in'
        ' Retro music.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    queue = _get_playing_queue(env)
    expected = [f.split('.')[0] for f in self.params['files']]
    return int(queue == expected)


class RetroSavePlaylist(RetroCreatePlaylist):
  """Task to create a playlist and save it in Retro Music."""

  complexity = 5

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist in Retro Music titled "{playlist_name}" with the'
        f' following songs, in order: {names}. Then export the playlist to the'
        ' Downloads directory on the device.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    playlist_exists = file_utils.check_file_exists(
        file_utils.convert_to_posix_path(
            device_constants.DOWNLOAD_DATA,
            self.params['playlist_name'] + '.m3u',
        ),
        env.controller,
    )

    return (super().is_successful(env) + int(playlist_exists)) / 2.0


def _generate_list_with_sum(n, m):
  """Generates a list of m integers with sum n."""
  random_numbers = [random.randint(0, n) for _ in range(m - 1)]
  random_numbers.sort()
  random_numbers.insert(0, 0)
  random_numbers.append(n)
  result = [random_numbers[i + 1] - random_numbers[i] for i in range(m)]
  return result


class RetroPlaylistDuration(RetroCreatePlaylist):
  """Task to create a playlist with a specific duration in Retro Music."""

  app_names = ['retro music']
  complexity = 3

  @property
  def goal(self) -> str:
    return (
        'Create a playlist in Retro Music titled'
        f' "{self.params["playlist_name"]}" with a duration between 45 and 50'
        ' minutes using the provided songs.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    _clear_playlist_dbs(env)

    # Guarantee there is an answer.
    durations = _generate_list_with_sum(
        int(47.5 * 60 * 1000), len(self.params['files'])
    )
    for file, duration in zip(self.params['files'], durations):
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=duration,
      )

    for file in self.params['noise_files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(3 * 60 * 1000, 5 * 60 * 1000),
      )
    _scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    songs = _get_playlist_data(env)
    total_ms = 0
    for song in songs:
      if song.playlist_name != self.params['playlist_name']:
        return False
      total_ms += song.duration_ms
    return float(45 * 60 * 1000 <= total_ms <= 50 * 60 * 1000)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]
    num_files = random.randint(9, 11)
    files, noise_files = files[0:num_files], files[num_files:]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': noise_files,
    }


_SONGS = [
    'My Heart is Yours',
    'Endless Summer',
    'Whispering Wind',
    'Lost in the Echo',
    'Chasing Shadows',
    'Night Drive',
    'Echoes of Silence',
    'Bright Lights',
    'Moments',
    'Forever Young',
    'Rising Sun',
    'Silent Dreams',
    'City of Stars',
    'Moonlight Sonata',
    'Through the Storm',
    'Return to Paradise',
    'Voices in the Hall',
    'Under the Sky',
    "Dreamer's Awake",
    'Serenity Now',
    'Falling Feathers',
    'Orbiting Stars',
    'Reflections',
    'Beyond the Horizon',
    'Golden Days',
    'Twilight Calling',
    'Heartbeat Away',
    'Mystic Journey',
    'Hidden Paths',
    'Distant Memories',
    'Path to Zenith',
    'Eternal Flame',
    'Shadows of Time',
    'Whispers of the Past',
    'Waves of Change',
]


def _generate_playlist_name() -> str:
  """Generates a diverse and creative playlist name."""
  themes = [
      'Chill Beats',
      'Morning Vibes',
      'Workout Energy',
      'Study Sessions',
      'Golden Oldies',
      'Indie Gems',
      'Rock Anthems',
      'Electronic Waves',
      'Jazz Classics',
      'Hip Hop Hits',
      'Country Roads',
      'Classical Moods',
      'Pop Essentials',
      'Latin Grooves',
      'Reggae Rhythms',
      'Soulful Sounds',
      'Blues Vibes',
      'Metal Mayhem',
      'Party Mix',
      'Tranquil Tunes',
      'R&B Favorites',
      'Folk Inspirations',
      'Disco Nights',
      'Global Beats',
      'Sleepytime Songs',
      'Acoustic Sessions',
      'Vintage Vinyl',
      'Instrumental Study',
      'Coffeehouse Jazz',
      'Rainy Day Relax',
      'Gym Pump Up',
      'Retro Pop Hits',
      'Indie Rock Roadtrip',
      'Electronic Chillout',
      'Classical Concentration',
      'Jazz Lounge',
      'Hip Hop Bangers',
      'Country Classics',
      'Classical Opera Highlights',
      'Pop Punk Power',
      'Latin Dance Party',
      'Reggae Sunsplash',
      'Soul Classics',
      'Blues Break',
      'Party Starters',
      'Tranquil Ambient',
      'R&B Grooves',
      'Folk Favourites',
      'Disco Fever',
      'World Music Tour',
  ]
  identifier = random.randint(1, 999)

  theme = random.choice(themes)
  return f'{theme} {identifier}'
