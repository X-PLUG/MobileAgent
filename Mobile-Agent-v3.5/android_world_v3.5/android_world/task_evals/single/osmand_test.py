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

import random
from typing import Any
from xml.etree import ElementTree

from absl.testing import absltest
from android_world.task_evals.single import osmand
from android_world.task_evals.utils import sqlite_schema_utils


class TestOsmAndFavorite(absltest.TestCase):

  def test_empty_favorites_contains_nothing(self):
    favorites = ElementTree.fromstring("""
        <gpx xmlns="http://www.topografix.com/GPX/1/1">
        </gpx>
        """)

    name_is_contained = osmand._favorites_contains(favorites, "Triesen")
    coords_are_contained = osmand._favorites_contains(
        favorites, "47.1069970, 9.5274854"
    )

    self.assertFalse(name_is_contained)
    self.assertFalse(coords_are_contained)

  def test_favorites_contains_name(self):
    favorites = ElementTree.fromstring("""
        <gpx xmlns="http://www.topografix.com/GPX/1/1">
          <wpt lat="47.1069970" lon="9.5274854"><name>Triesen</name></wpt>
        </gpx>
        """)

    name_is_contained = osmand._favorites_contains(favorites, "Triesen")

    self.assertTrue(name_is_contained)

  def test_favorites_contains_coords(self):
    favorites = ElementTree.fromstring("""
        <gpx xmlns="http://www.topografix.com/GPX/1/1">
          <wpt lat="47.1069970" lon="9.5274854"><name>Triesen</name></wpt>
        </gpx>
        """)

    coords_are_contained = osmand._favorites_contains(
        favorites, "47.1069970, 9.5274854"
    )

    self.assertTrue(coords_are_contained)

  def test_random_favorite_location(self):
    # Observed locations given the set PRNG seed value below. The actual sampled
    # random locations may change and break this test but be stochastically
    # equivalent. In those cases, manually check that the actual samples are
    # as expected and copy them here.
    observed_random_location_samples = (
        "Rotenboden, Liechtenstein\n"
        "47.1275785, 9.5387131\n"
        "Oberplanken, Liechtenstein\n"
        "47.1663432, 9.5103085\n"
        "Rotenboden, Liechtenstein\n"
        "Oberplanken, Liechtenstein\n"
        "47.23976, 9.5262837\n"
        "Schaanwald, Liechtenstein\n"
        "Nendeln, Liechtenstein\n"
        "47.1026191, 9.6083057"
    )

    def location_param(params: dict[str, Any]) -> str:
      self.assertSameElements(params.keys(), ("location",))
      return params["location"]

    # Grab the first 10 random parameter sets. By setting the seed the result
    # is made repeatable.
    random.seed(0)
    random_location_samples = "\n".join([
        location_param(osmand.OsmAndFavorite.generate_random_params())
        for _ in range(10)
    ])

    self.assertSequenceEqual(
        random_location_samples, observed_random_location_samples
    )


class TestOsmAndMarker(absltest.TestCase):

  def test_empty_marker_matches_nothing(self):
    marker = sqlite_schema_utils.OsmAndMapMarker()

    name_match = osmand._marker_matches_location(marker, "Triesen")
    coords_match = osmand._marker_matches_location(
        marker, "47.1069970, 9.5274854"
    )

    self.assertFalse(name_match)
    self.assertFalse(coords_match)

  def test_marker_matches_name(self):
    marker = sqlite_schema_utils.OsmAndMapMarker(
        marker_lat=47.1069970, marker_lon=9.5274854
    )

    name_match = osmand._marker_matches_location(
        marker, "Triesen, Liechtenstein"
    )

    self.assertTrue(name_match)

  def test_marker_matches_coords(self):
    marker = sqlite_schema_utils.OsmAndMapMarker(
        marker_lat=47.1069970, marker_lon=9.5274854
    )

    coords_match = osmand._marker_matches_location(
        marker, "47.1069970, 9.5274854"
    )

    self.assertTrue(coords_match)


class TestOsmAndTrack(absltest.TestCase):

  def test_goal_with_too_few_waypoint_params(self):
    single_waypoint_params = {
        "waypoints": ["test waypoint 1"],
    }

    track_eval = osmand.OsmAndTrack(params=single_waypoint_params)

    with self.assertRaises(ValueError):
      _ = track_eval.goal

  def test_goal_with_valid_params(self):
    valid_params = {
        "waypoints": ["test waypoint 1", "test waypoint 2", "test waypoint 3"],
    }

    track_eval = osmand.OsmAndTrack(params=valid_params)

    self.assertEqual(
        track_eval.goal,
        "Save a track with waypoints test waypoint 1, test waypoint 2, test"
        " waypoint 3 in the OsmAnd maps app in the same order as listed.",
    )

  def test_lookup_unknown_target_waypoint(self):
    waypoint_name_not_in_preloaded_map_locations = "Obock"

    self.assertIsNone(
        osmand._lookup_location_coords(
            waypoint_name_not_in_preloaded_map_locations
        )
    )

  def test_lookup_known_waypoint(self):
    known_waypoint, known_waypoint_coords = next(
        iter(osmand._PRELOADED_MAP_LOCATIONS.items())
    )

    self.assertEqual(
        known_waypoint_coords, osmand._lookup_location_coords(known_waypoint)
    )

  def test_lookup_coords_just_returns_coords(self):
    self.assertEqual(
        (-1.234, 56.7), osmand._lookup_location_coords("-1.234, 56.7")
    )

  def test_lookup_target_waypoints_throws_on_unknown(self):
    known_waypoint = next(iter(osmand._PRELOADED_MAP_LOCATIONS))
    coords_waypoint = "-1.234, 56.7"
    waypoint_name_not_in_preloaded_map_locations = "Obock"

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Unable to look up coordinates for waypoint location parameter(s)"
        ' "Obock". Only lat/lon coordinate strings or exact names from'
        " _PRELOADED_MAP_LOCATIONS supported.",
    ):
      osmand._lookup_target_waypoints([
          known_waypoint,
          coords_waypoint,
          waypoint_name_not_in_preloaded_map_locations,
      ])

  def test_track_points_parsing(self):
    track_xml = ElementTree.fromstring("""
        <gpx xmlns="http://www.topografix.com/GPX/1/1">
          <trk>
            <trkseg>
              <trkpt lat="47.0687992" lon="9.5061564"/>
            </trkseg>
          </trk>
        </gpx>
        """)

    self.assertSequenceEqual(
        [(47.0687992, 9.5061564)], list(osmand._track_points(track_xml))
    )

  def test_track_doesnt_match_missing_waypoint(self):
    track_points = ((0, 1), (2, 3))
    self.assertFalse(osmand._track_matches(track_points, [(4, 5)]))

  def test_track_matches_one_to_one(self):
    track_points = ((0, 1), (2, 3))
    self.assertTrue(osmand._track_matches(track_points, track_points))

  def test_track_doesnt_match_out_of_order(self):
    track_points = ((0, 1), (2, 3))
    reverse_order_waypoints = ((2, 3), (0, 1))
    self.assertFalse(
        osmand._track_matches(track_points, reverse_order_waypoints)
    )

  def test_track_matches_interleaved_waypoints(self):
    track_points = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    interleaved_waypoints = ((2, 3), (8, 9))
    self.assertTrue(osmand._track_matches(track_points, interleaved_waypoints))


if __name__ == "__main__":
  absltest.main()
