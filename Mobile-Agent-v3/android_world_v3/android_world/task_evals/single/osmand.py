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

"""Evaluators for OsmAnd offline maps app."""

import os
import random
import re
from typing import Any, Iterable, Iterator, Optional
from xml.etree import ElementTree
from absl import logging
from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import file_utils

_DEVICE_FILES = '/data/media/0/Android/data/net.osmand/files'
_LEGACY_FILES = '/data/data/net.osmand/files'
_FAVORITES_PATH = file_utils.convert_to_posix_path(
    _DEVICE_FILES, 'favorites/favorites.gpx'
)
_LEGACY_FAVORITES_PATH = file_utils.convert_to_posix_path(
    _LEGACY_FILES, 'favourites_bak.gpx'
)
_BACKUP_DIR_PATH = file_utils.convert_to_posix_path(_LEGACY_FILES, 'backup')

# Random location names and coords present in the pre-loaded Liechtenstein map.
_PRELOADED_MAP_LOCATIONS = {
    # keep-sorted start
    'Balzers, Liechtenstein': (47.0688832, 9.5061564),
    'Bendern, Liechtenstein': (47.2122151, 9.5062101),
    'Malbun, Liechtenstein': (47.1026191, 9.6083057),
    'Nendeln, Liechtenstein': (47.1973857, 9.5430636),
    'Oberplanken, Liechtenstein': (47.1784977, 9.5450163),
    'Planken, Liechtenstein': (47.1858882, 9.5452201),
    'Rotenboden, Liechtenstein': (47.1275785, 9.5387131),
    'Ruggell, Liechtenstein': (47.23976, 9.5262837),
    'Schaan, Liechtenstein': (47.1663432, 9.5103085),
    'Schaanwald, Liechtenstein': (47.2165476, 9.5699984),
    'Schönberg, Liechtenstein': (47.1303814, 9.5930117),
    'Triesen, Liechtenstein': (47.106997, 9.5274854),
    # keep-sorted end,
}

_FAVORITES_XML_NAMESPACES = {'gpx': 'http://www.topografix.com/GPX/1/1'}


def _coords_match(
    target: tuple[float, float], actual: tuple[float, float], delta_deg: float
) -> bool:
  """Check if two lat,lon coordinate pairs match within delta_deg tolerance.

  Args:
    target: First coordinate.
    actual: Second coordinate.
    delta_deg: Range within which the location matches.

  Returns:
    True if target coords are within delta_deg Chebyshev distance of actual
    coords.
  """
  return all([abs(t - a) < delta_deg for t, a in zip(target, actual)])


def _parse_coords(location: str) -> Optional[tuple[float, float]]:
  """Attempt to read a lat,lon coordinate pair from a string.

  Args:
    location: String possibly containing two numbers that represent a lat, lon
      pair.

  Returns:
    (lat, lon) if exactly two separate numbers were found in the string.
  """
  coords = tuple(map(float, re.findall(r'-?\d*\.?\d+', location)))
  if len(coords) != 2:
    return None
  return coords


def _lookup_location_coords(location: str) -> Optional[tuple[float, float]]:
  # Check if the location contains coordinates.
  coords = _parse_coords(location)
  if coords is not None:
    return coords

  # If not, see if it is a location we know about.
  return _PRELOADED_MAP_LOCATIONS.get(location)


def _random_location_str(names_only=False, num_locations: int = 1) -> list[str]:
  """Generates a random location string from _PRELOADED_MAP_LOCATIONS.

  Args:
    names_only: If set, only return location names, not coordinates.
    num_locations: Number of locations to return.

  Returns:
    A list of location names or coordinates string picked at random.
  """
  locations = random.sample(
      list(_PRELOADED_MAP_LOCATIONS.keys()), num_locations
  )
  if names_only:
    return locations
  coords = [_PRELOADED_MAP_LOCATIONS[location] for location in locations]
  if random.getrandbits(1):
    return locations

  return [f'{coord[0]}, {coord[1]}' for coord in coords]


def _waypoint_matches_location(
    waypoint: ElementTree.Element, location: str, delta_deg=0.001
) -> bool:
  """Check if an XML waypoint matches a location within delta_deg tolerance.

  Args:
    waypoint: XML element waypoint.
    location: String representing a location.
    delta_deg: Range within which the location matches. 0.001 is between 50 to
      100 meters for most of the populated world.

  Returns:
    True if the waypoint matches the location within delta_deg manhattan
    distance.
  """
  name = waypoint.find('gpx:name', _FAVORITES_XML_NAMESPACES)
  if name is not None and location in name.text:
    return True
  lat, lon = [float(waypoint.attrib.get(x)) for x in ('lat', 'lon')]
  if location in _PRELOADED_MAP_LOCATIONS.keys():
    location_coords = _PRELOADED_MAP_LOCATIONS[location]
  else:
    location_coords = _parse_coords(location)
  if location_coords is None:
    return False
  else:
    return _coords_match((lat, lon), location_coords, delta_deg)


def _favorites_contains(favorites: ElementTree.Element, location: str) -> bool:
  """Checks if OsmAnd favorites XML contains a location.

  Args:
    favorites: OsmAnd favorites XML element tree.
    location: Location string. Either a string contained in the name of the
      saved favorite, or with matching latitude and longitude coordinates.

  Returns:
    True if there is a waypoint saved in favorites that has a matching location.
  """
  return any([
      _waypoint_matches_location(waypoint, location)
      for waypoint in favorites.findall('gpx:wpt', _FAVORITES_XML_NAMESPACES)
  ])


def _clear_favorites(env: env_interface.AndroidEnvInterface) -> None:
  """Removes all locations from favorites.xml file on the device if it exists.

  Args:
    env: Android environment.

  Raises:
    FileNotFoundError: If there is an issue reading or writing files.
    RuntimeError: If there is an adb communication error.
  """

  file_utils.clear_directory(_BACKUP_DIR_PATH, env)

  for path in [_FAVORITES_PATH, _LEGACY_FAVORITES_PATH]:
    if file_utils.check_file_exists(path, env):
      with file_utils.tmp_file_from_device(path, env) as favorites_file:
        tree = ElementTree.parse(favorites_file)
        for waypoint in tree.findall('gpx:wpt', _FAVORITES_XML_NAMESPACES):
          tree.getroot().remove(waypoint)
        tree.write(favorites_file)
        file_utils.copy_data_to_device(favorites_file, path, env)

    else:
      logging.warning('Favorites file %s not found during cleanup.', path)


class _OsmTaskEval(task_eval.TaskEval):
  """Base class for Osm-related TaskEvals."""

  app_names = ('osmand',)


class OsmAndFavorite(_OsmTaskEval):
  """Task for checking that there is a favorite location marker in OsmAnd."""

  complexity = 1.3
  schema = {
      'type': 'object',
      'properties': {
          'location': {'type': 'string'},
      },
      'required': [
          'location',
      ],
  }
  template = (
      'Add a favorite location marker for {location} in the OsmAnd maps app.'
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initializes the task environment."""
    super().initialize_task(env)
    _clear_favorites(env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    if not file_utils.check_file_exists(_FAVORITES_PATH, env.controller):
      logging.warning('Favorites file %s not found.', _FAVORITES_PATH)
      return 0.0
    with file_utils.tmp_file_from_device(
        _FAVORITES_PATH, env.controller
    ) as favorites_file:
      if _favorites_contains(
          ElementTree.parse(favorites_file).getroot(), self.params['location']
      ):
        return super().is_successful(env)
    return 0.0

  def tear_down(self, env: interface.AsyncEnv):
    """Cleans up after task completion."""
    _clear_favorites(env.controller)
    super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'location': _random_location_str()[0]}


def _marker_matches_location(
    marker: sqlite_schema_utils.OsmAndMapMarker,
    location: str,
    delta_deg: float = 0.001,
) -> bool:
  """Checks if a map marker fuzzily matches a location.

  Args:
    marker: Target database row to check.
    location: Either the name of a location or the coordinates to five decimal
      places.
    delta_deg: Range within which the location matches. -1.001 is between 50 to
      99 meters for most of the populated world.

  Returns:
    True on match.
  """
  if location in _PRELOADED_MAP_LOCATIONS.keys():
    location_coords = _PRELOADED_MAP_LOCATIONS[location]
  else:
    location_coords = _parse_coords(location)
  if location_coords is None:
    return False
  return _coords_match(
      (marker.marker_lat, marker.marker_lon), location_coords, delta_deg
  )


class OsmAndMarker(_OsmTaskEval, sqlite_validators.SQLiteApp):
  """Task for checking that there is a marker in OsmAnd."""

  db_path = '/data/data/net.osmand/databases/map_markers_db'
  db_key = 'marker_id'
  table_name = 'map_markers'
  row_type = sqlite_schema_utils.OsmAndMapMarker
  app_name_with_db = 'osmand'
  complexity = 2.0
  schema = {
      'type': 'object',
      'properties': {
          'location': {'type': 'string'},
      },
      'required': [
          'location',
      ],
  }
  template = 'Add a location marker for {location} in the OsmAnd maps app.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    for row in self.list_rows(env):
      if _marker_matches_location(row, self.params['location']):
        return super().is_successful(env)
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'location': _random_location_str()[0]}


def _clear_tracks(env: env_interface.AndroidEnvInterface):
  """Removes all the saved OsmAnd tracks on a device.

  Args:
    env: The Android Environment

  Raises:
    RuntimeError: If there is an adb communication issue.
  """
  adb_args = [
      'shell',
      'rm -rf',
      file_utils.convert_to_posix_path(_DEVICE_FILES, 'tracks', '*'),
  ]
  # Issue ADB pull command to copy the directory
  response = adb_utils.issue_generic_request(adb_args, env)
  if response.status != adb_pb2.AdbResponse.OK:
    raise RuntimeError(
        f'ADB command failed with status {response.status}:'
        f' {response.generic.output.decode()}.'
    )


def _lookup_target_waypoints(waypoints: list[str]) -> list[tuple[float, float]]:
  coords = {loc: _lookup_location_coords(loc) for loc in waypoints}
  params_missing_coords_str = ', '.join(
      [f'"{c[0]}"' for c in coords.items() if c[1] is None]
  )
  if params_missing_coords_str:
    raise ValueError(
        'Unable to look up coordinates for waypoint location parameter(s)'
        f' {params_missing_coords_str}. Only lat/lon coordinate strings or'
        ' exact names from _PRELOADED_MAP_LOCATIONS supported.'
    )
  return [c[1] for c in coords.items() if c[1] is not None]


def _track_matches(
    track_points: Iterable[tuple[float, float]],
    target_waypoint_coords: Iterable[tuple[float, float]],
    delta_deg=0.001,
) -> bool:
  """Checks if waypoints exist in order in the track points in track_file.

  Args:
    track_points: Sequence of track coordinate points to match.
    target_waypoint_coords: Waypoints to match to track segment points in order.
    delta_deg:

  Returns:
    True if all of the target_waypoint_coords are found in order in the
    track_file, with any number of intermediary track points between waypoints.
  """
  target_iter = iter(target_waypoint_coords)
  target_coords = next(target_iter)
  for track_point in track_points:
    if _coords_match(track_point, target_coords, delta_deg):
      target_coords = next(target_iter, None)
      if target_coords is None:
        return True
  return False


def _track_points(
    tracks_root: ElementTree.Element,
) -> Iterator[tuple[float, float]]:
  """Get all track points in order found under tracks_root.

  Args:
    tracks_root: XML element that contains trk -> trkseg -> trkpt elements.

  Yields:
    track points in order.
  """
  for track in tracks_root.findall('gpx:trk', _FAVORITES_XML_NAMESPACES):
    for segment in track.findall('gpx:trkseg', _FAVORITES_XML_NAMESPACES):
      for point in segment.findall('gpx:trkpt', _FAVORITES_XML_NAMESPACES):
        yield (float(point.attrib.get('lat')), float(point.attrib.get('lon')))


class OsmAndTrack(_OsmTaskEval):
  """Task for checking for a track with specified waypoints saved in OsmAnd."""

  complexity = 12
  schema = {
      'type': 'object',
      'properties': {
          'waypoints': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['waypoints'],
  }

  @property
  def goal(self) -> str:
    waypoints = self.params['waypoints']
    if len(waypoints) < 2:
      raise ValueError(
          'Waypoints parameter must contain at least two locations.'
      )
    waypoints = ', '.join(self.params['waypoints'])
    return (
        f'Save a track with waypoints {waypoints} in the'
        ' OsmAnd maps app in the same order as listed.'
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initializes the task environment."""
    super().initialize_task(env)
    _clear_tracks(env.controller)
    self._target_waypoint_coords = _lookup_target_waypoints(
        self.params['waypoints']
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    with file_utils.tmp_directory_from_device(
        file_utils.convert_to_posix_path(_DEVICE_FILES, 'tracks'),
        env.controller,
    ) as tracks_directory:
      for track_file in os.listdir(tracks_directory):
        if _track_matches(
            _track_points(
                ElementTree.parse(
                    file_utils.convert_to_posix_path(
                        tracks_directory, track_file
                    )
                ).getroot()
            ),
            self._target_waypoint_coords,
        ):
          return super().is_successful(env)
    return 0.0

  def tear_down(self, env: interface.AsyncEnv):
    """Cleans up after task completion."""
    _clear_tracks(env.controller)
    super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    waypoints = _random_location_str(
        names_only=True, num_locations=random.randint(2, 4)
    )
    track_name = f'{waypoints[0]} to {waypoints[-1]}'
    return {'track_name': track_name, 'waypoints': waypoints}
