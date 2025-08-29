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

import datetime
import zoneinfo
from absl.testing import absltest
from absl.testing import parameterized
from android_world.env import device_constants
from android_world.task_evals.information_retrieval import datetime_utils


class DatetimeUtilsTest(parameterized.TestCase):

  def test_give_me_a_name(self):
    pass

  @parameterized.parameters([
      ('October 15', 'October 15 2023'),
      ('October 15 2023', 'October 15 2023'),
      ('October 20 2023', 'October 20 2023'),
      ('today', 'October 25 2023'),
      ('tomorrow', 'October 26 2023'),
      ('Thursday', 'October 26 2023'),
      ('Friday', 'October 27 2023'),
      ('Saturday', 'October 28 2023'),
      ('Sunday', 'October 29 2023'),
      ('Monday', 'October 30 2023'),
      ('Tuesday', 'October 31 2023'),
      ('Wednesday', 'November 1 2023'),
      ('this Thursday', 'October 26 2023'),
      ('this Friday', 'October 27 2023'),
      ('this Saturday', 'October 28 2023'),
      ('this Sunday', 'October 29 2023'),
      ('this Monday', 'October 30 2023'),
      ('this Tuesday', 'October 31 2023'),
      ('this Wednesday', 'November 1 2023'),
      ('the Thursday after next', 'November 2 2023'),
      ('the Friday after next', 'November 3 2023'),
      ('the Saturday after next', 'November 4 2023'),
      ('the Sunday after next', 'November 5 2023'),
      ('the Monday after next', 'November 6 2023'),
      ('the Tuesday after next', 'November 7 2023'),
      ('the Wednesday after next', 'November 8 2023'),
  ])
  def test_generate_reworded_date(self, expected_rewording: str, date: str):
    original_today = device_constants.DT
    device_constants.DT = datetime.datetime(
        2023, 10, 25, 15, 34, 0, tzinfo=zoneinfo.ZoneInfo('UTC')
    )
    result = datetime_utils._generate_nl_date_options(date)
    self.assertContainsSubset([expected_rewording], result)
    device_constants.DT = original_today

  @parameterized.parameters([
      ('2:30pm', datetime.time(hour=14, minute=30)),
      ('14:30', datetime.time(hour=14, minute=30)),
      ('2am', datetime.time(hour=2, minute=00)),
  ])
  def test_parse_time(self, time_string: str, expected: datetime.time):
    result = datetime_utils.parse_time(time_string)
    self.assertEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
