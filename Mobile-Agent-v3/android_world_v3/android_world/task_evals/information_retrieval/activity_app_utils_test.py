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
from android_world.task_evals.information_retrieval import activity_app_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2


class ActivityAppUtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      (
          state_pb2.SportsActivity(
              start_date='October 15 2023',
              start_time='12:30',
              category='running',
              name='Morning Run',
              total_distance='10000',
              duration='60',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 15 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          False,
      ),
      (
          state_pb2.SportsActivity(
              start_date='October 15 2023',
              start_time='12:30',
              category='running',
              name='Morning Run',
              total_distance='10000',
              duration='60',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          True,
      ),
      (
          state_pb2.SportsActivity(
              start_date='October 15 2023',
              start_time='12:30',
              category='running',
              name='Morning Run',
              total_distance='10000',
              duration='60',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='category',
                  value='running',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          False,
      ),
      (
          state_pb2.SportsActivity(
              start_date='October 22 2023',
              start_time='12:30',
              category='running',
              name='Morning Run',
              total_distance='10000',
              duration='60',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN,
              ),
              task_pb2.ExclusionCondition(
                  field='category',
                  value='running',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          True,
      ),
      (
          state_pb2.SportsActivity(
              start_date='October 22 2023',
              start_time='12:30',
              category='running',
              name='Morning Run',
              total_distance='10000',
              duration='60',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='total_distance',
                  value='10000',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              ),
          ],
          False,
      ),
  ])
  def test_check_task_conditions(
      self,
      activity_proto: state_pb2.SportsActivity,
      exclusion_conditions: list[task_pb2.ExclusionCondition],
      expected_value: bool,
  ):
    activity = activity_app_utils._create_activity_from_proto(activity_proto)
    self.assertEqual(
        activity_app_utils._check_activity_conditions(
            activity, exclusion_conditions
        ),
        expected_value,
    )


if __name__ == '__main__':
  absltest.main()
