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

from absl.testing import absltest
from absl.testing import parameterized
from android_world.task_evals.single.calendar import calendar_utils


class TestTimestampToLocalDatetime(parameterized.TestCase):

  @parameterized.parameters([
      ('Sunday', 7, 64),
      ('Monday', 1, 1),
      ('Tuesday', 2, 2),
      ('Wednesday', 3, 4),
      ('Thursday', 4, 8),
      ('Friday', 5, 16),
      ('Saturday', 6, 32),
  ])
  def test_valid_days(self, name: str, day_of_week: int, expected: int):
    result = calendar_utils.generate_simple_calendar_weekly_repeat_rule(
        day_of_week
    )
    self.assertEqual(result, expected, f'Test failed for {name}')


if __name__ == '__main__':
  absltest.main()
