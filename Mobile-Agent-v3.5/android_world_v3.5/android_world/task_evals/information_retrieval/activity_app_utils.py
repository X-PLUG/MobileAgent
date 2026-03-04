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

"""Utils for Open Tracks sports activity tracker.

App available at github.com/OpenTracksApp/OpenTracks.
"""

import datetime
import random
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.information_retrieval import calendar_utils
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import datetime_utils

_PRIMARY_KEY = '_id'
_TABLE = 'tracks'
_DB_PATH = '/data/data/de.dennisguse.opentracks/databases/database.db'
_APP_NAME = 'activity tracker'

_MILES_TO_METERS = 1609.34


def setup_task_state(
    relevant_state: state_pb2.SportsActivityApp,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
    env: interface.AsyncEnv,
) -> None:
  clear_db(env)
  activities = []
  for activity in relevant_state.sports_activities:
    activities.append(_create_activity_from_proto(activity))
  activities += _generate_random_activities(20, exclusion_conditions)
  random.shuffle(activities)
  _add_activities(activities, env)


def _distance_rounding_error_conversion(value: float) -> float:
  # OpenTracks stores distance in meters but displays in miles or feet.
  # To make sure that we don't lose too much precision in the conversion,
  # we first convert to miles, rounded to 2 decimal places (which is what is
  # displayed in the app) then back to rounded meters and set that value
  # both in the proto and for the app.
  return round(round(value * (1.0 / _MILES_TO_METERS), 2) * _MILES_TO_METERS)


def _create_activity_from_proto(
    activity: state_pb2.SportsActivity,
) -> sqlite_schema_utils.SportsActivity:
  """Creates a SportsActivity object from a state_pb2.SportsActivity proto."""
  start_unix_ts = (
      calendar_utils.convert_datetime_to_unix_ts(
          activity.start_date, activity.start_time
      )
      * 1000
  )

  total_distance = _distance_rounding_error_conversion(
      float(activity.total_distance)
  )
  activity.total_distance = str(total_distance)
  # Duration is in minutes, we need time to be in milliseconds.
  stop_unix_ts = start_unix_ts + int(float(activity.duration) * 60 * 1000)
  total_time = stop_unix_ts - start_unix_ts
  description = activity.description if activity.HasField('description') else ''
  category = activity.category if activity.HasField('category') else ''
  avg_speed = float(activity.total_distance) / (
      (stop_unix_ts - start_unix_ts) * 1000
  )
  return sqlite_schema_utils.SportsActivity(
      name=activity.name,
      category=category,
      activity_type=category,
      description=description,
      totaldistance=float(activity.total_distance),
      starttime=start_unix_ts,
      stoptime=stop_unix_ts,
      totaltime=total_time,
      movingtime=total_time,
      avgspeed=avg_speed,
      avgmovingspeed=avg_speed,
      elevationgain=(
          int(activity.elevation_gain)
          if activity.HasField('elevation_gain')
          else 0
      ),
      elevationloss=(
          int(activity.elevation_loss)
          if activity.HasField('elevation_loss')
          else 0
      ),
  )


def clear_db(env: interface.AsyncEnv) -> None:
  """Clears the task database."""
  sqlite_utils.delete_all_rows_from_table(_TABLE, _DB_PATH, env, _APP_NAME)
  adb_utils.close_app(_APP_NAME, env.controller)  # Register changes.


def _add_activities(
    rows: list[sqlite_schema_utils.SportsActivity],
    env: interface.AsyncEnv,
) -> None:
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      _PRIMARY_KEY,
      _TABLE,
      _DB_PATH,
      _APP_NAME,
      env,
  )


def list_rows(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.SportsActivity]:
  return sqlite_utils.get_rows_from_remote_device(
      _TABLE,
      _DB_PATH,
      sqlite_schema_utils.SportsActivity,
      env,
  )


def _generate_random_activities(
    num_activities: int,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
) -> list[sqlite_schema_utils.Task]:
  """Generates random tasks with the given exclusion conditions."""
  return sqlite_schema_utils.get_random_items(
      num_activities,
      generate_item_fn=_generate_random_activity,
      filter_fn=lambda x: _check_activity_conditions(x, exclusion_conditions),
  )


def _generate_random_activity():
  """Generates a single random sqlite_schema_utils.SportsActivity object."""
  new_activity = state_pb2.SportsActivity()
  new_activity.category = random.choice(
      list(_CATEGORY_TO_ACTIVITY_NAMES.keys())
  )
  new_activity.name = random.choice(
      _CATEGORY_TO_ACTIVITY_NAMES[new_activity.category]
  )
  # Make sure that the start date is in the past
  random_start_datetime = datetime_utils.generate_random_datetime(
      window_center=device_constants.DT - datetime.timedelta(days=7)
  )
  new_activity.start_date = random_start_datetime.date().strftime(
      datetime_utils_ir.DATE_FORMAT
  )
  new_activity.start_time = random_start_datetime.time().strftime('%H:%M')

  random_duration = datetime.timedelta(minutes=random.randrange(1, 60 * 5))
  new_activity.duration = '{}'.format(int(random_duration.seconds / 60))
  new_activity.total_distance = str(random.randint(0, 20000))
  new_activity.elevation_gain = str(random.randint(0, 500))
  new_activity.elevation_loss = str(random.randint(0, 500))
  return _create_activity_from_proto(new_activity)


def _check_activity_conditions(
    activity: sqlite_schema_utils.SportsActivity,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
) -> bool:
  """Evaluates the specified task against a set of exclusion conditions.

  A activity is considered eligible if it does not satisfy all of the conditions
  specified in the exclusion_conditions list. Each condition is checked against
  various fields of the activity. The activity is eligible if not all of these
  conditions are met, ensuring it doesn't fall under any exclusion criteria
  defined.

  Args:
    activity: The activity to check.
    exclusion_conditions: All the conditions the activity will be checked
      against, if they are all met, this activity should be excluded and does
      not meet the conditions.

  Returns:
    A bool, True if the activity does not meet all of the exclusion conditions,
    False otherwise.
  """
  if not exclusion_conditions:
    return True
  # Keeps track of whether an exclusion condition is met.
  all_conditions_met = True
  for condition in exclusion_conditions:
    if condition.field == 'start_date':
      condition_value = datetime_utils_ir.get_date(condition.value)
      start_datetime = datetime_utils.timestamp_to_localized_datetime(
          int(activity.starttime / 1000)
      )

      all_conditions_met = all_conditions_met and proto_utils.compare(
          start_datetime.date(), condition.operation, condition_value
      )
    elif condition.field == 'category':
      all_conditions_met = all_conditions_met and proto_utils.compare(
          activity.category.lower(),
          condition.operation,
          condition.value.lower(),
      )
    elif condition.field == 'total_distance':
      all_conditions_met = all_conditions_met and proto_utils.compare(
          activity.totaldistance,
          condition.operation,
          float(_distance_rounding_error_conversion(float(condition.value))),
      )

  return not all_conditions_met


_CATEGORY_TO_ACTIVITY_NAMES = {
    'biking': [
        'Scenic Cycling',
        'Bike Tour',
        'Trail Ride',
        'Bicycle Adventure',
        'Pedal Excursion',
        'Cycle Trip',
        'Off-road Cycling',
        'Mountain Biking',
        'Road Cycling',
        'Bike Expedition',
    ],
    'running': [
        'Trail Run',
        'Road Run',
        'Jogging',
        'Running Adventure',
        'Sprint Session',
        'Trail Jog',
        'Long Distance Run',
        'Morning Run',
        'Evening Run',
        'Running Challenge',
    ],
    'hiking': [
        'Trail Hike',
        'Mountain Trek',
        'Hiking Excursion',
        'Nature Walk',
        'Outdoor Hike',
        'Scenic Hike',
        'Mountain Hike',
        'Wilderness Hike',
        'Hill Walk',
        'Nature Hike',
    ],
    'swimming': [
        'Pool Swim',
        'Open Water Swim',
        'Swim Workout',
        'Swim Session',
        'Swimming Adventure',
        'Lap Swim',
        'Swim Training',
        'Water Exercise',
        'Swimming Excursion',
        'Swim Challenge',
    ],
    'walking': [
        'Neighborhood Walk',
        'Evening Stroll',
        'Morning Walk',
        'Urban Walk',
        'Nature Walk',
        'City Stroll',
        'Trail Walk',
        'Casual Walk',
        'Power Walk',
        'Brisk Walk',
    ],
    'skiing': [
        'Ski Trip',
        'Slope Session',
        'Alpine Skiing',
        'Downhill Skiing',
        'Cross-country Skiing',
        'Backcountry Skiing',
        'Powder Day',
        'Snow Adventure',
        'Winter Skiing',
        'Ski Expedition',
    ],
    'snowboarding': [
        'Snowboard Trip',
        'Freestyle Session',
        'Slope Riding',
        'Snowboard Adventure',
        'Snowboarding Excursion',
        'Powder Ride',
        'Mountain Snowboarding',
        'Backcountry Snowboarding',
        'Snowboard Challenge',
        'Snowboard Trek',
    ],
    'kayaking': [
        'Kayak Tour',
        'Paddling Adventure',
        'Kayak Excursion',
        'River Paddle',
        'Lake Kayaking',
        'Sea Kayaking',
        'Kayak Expedition',
        'Waterway Paddle',
        'Kayak Journey',
        'Paddle Outing',
    ],
    'rowing': [
        'Rowing Adventure',
        'Crew Session',
        'Lake Rowing',
        'River Rowing',
        'Rowing Excursion',
        'Rowing Challenge',
        'Rowing Workout',
        'Rowing Trek',
        'Rowing Expedition',
        'Sculling Session',
    ],
    'sailing': [
        'Sailing Trip',
        'Sailing Adventure',
        'Sailboat Ride',
        'Boat Tour',
        'Yacht Sailing',
        'Sailing Excursion',
        'Nautical Adventure',
        'Sailing Expedition',
        'Windward Sailing',
        'Sailboat Expedition',
    ],
    'skateboarding': [
        'Skateboard Session',
        'Urban Skate',
        'Skatepark Session',
        'Skateboard Adventure',
        'Street Skateboarding',
        'Skateboarding Excursion',
        'Skateboard Trek',
        'Skateboard Ride',
        'Skateboard Challenge',
        'Skateboard Exploration',
    ],
    'surfing': [
        'Surf Session',
        'Beach Surfing',
        'Wave Riding',
        'Surfing Adventure',
        'Surfing Excursion',
        'Surfboard Session',
        'Surfing Challenge',
        'Surf Trek',
        'Surfing Expedition',
        'Wave Exploration',
    ],
    'climbing': [
        'Rock Climbing',
        'Indoor Climbing',
        'Bouldering',
        'Climbing Adventure',
        'Climbing Excursion',
        'Crag Climbing',
        'Mountain Climbing',
        'Rock Climbing Challenge',
        'Climbing Session',
        'Climbing Expedition',
    ],
    'mountain biking': [
        'Mountain Bike Ride',
        'Trail Biking',
        'MTB Adventure',
        'Mountain Biking Excursion',
        'Off-road Biking',
        'Dirt Biking',
        'Mountain Bike Expedition',
        'Singletrack Session',
        'MTB Trek',
        'Trail Riding',
    ],
    'road biking': [
        'Road Bike Ride',
        'Cycling Adventure',
        'Road Cycling',
        'Long Distance Ride',
        'Bike Touring',
        'Road Bike Session',
        'Countryside Cycling',
        'City Cycling',
        'Bike Commute',
        'Road Riding',
    ],
    'trail running': [
        'Trail Run',
        'Off-road Running',
        'Trail Jog',
        'Trail Running Adventure',
        'Trail Run Session',
        'Trail Running Challenge',
        'Long Distance Trail Run',
        'Trail Run Expedition',
        'Trail Running Trek',
        'Off-road Run',
    ],
    'trail hiking': [
        'Trail Hike',
        'Hiking Adventure',
        'Nature Hike',
        'Mountain Trail Hike',
        'Scenic Hiking',
        'Trail Trek',
        'Trail Walking',
        'Off-road Hike',
        'Trail Hiking Excursion',
        'Hiking Expedition',
    ],
    'cycling': [
        'Scenic Cycling',
        'Bike Tour',
        'Trail Ride',
        'Bicycle Adventure',
        'Pedal Excursion',
        'Cycle Trip',
        'Off-road Cycling',
        'Mountain Biking',
        'Road Cycling',
        'Bike Expedition',
    ],
    'paddling': [
        'Boat Tour',
        'Paddle Adventure',
        'River Paddle',
        'Lake Paddling',
        'Sea Paddling',
        'Boating Expedition',
        'Waterway Paddle',
        'Paddle Journey',
        'Paddle Outing',
        'Canoeing Adventure',
    ],
}
