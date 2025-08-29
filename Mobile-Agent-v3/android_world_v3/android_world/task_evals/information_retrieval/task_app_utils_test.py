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
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2


class TaskAppUtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      (
          state_pb2.TasksAppTask(
              due_date='October 15 2023',
              due_time='12:30',
              importance='2',
              title='Meeting',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='due_date',
                  value='October 15 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          False,
      ),
      (
          state_pb2.TasksAppTask(
              completed_date='October 22 2023',
              due_date='October 25 2023',
              importance='1',
              title='Meeting',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='completed_date',
                  value='October 24 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          True,
      ),
      (
          state_pb2.TasksAppTask(
              hide_until_date='October 22 2023',
              hide_until_time='12:30',
              importance='3',
              title='Meeting',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='hide_until_time',
                  value='12:30pm',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          False,
      ),
      (
          state_pb2.TasksAppTask(
              due_date='October 22 2023',
              due_time='12:30',
              importance='1',
              title='Meeting with David',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='due_date',
                  value='October 24 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN,
              ),
              task_pb2.ExclusionCondition(
                  field='title',
                  value='David',
                  operation=task_pb2.ExclusionCondition.Operation.CONTAINS,
              ),
          ],
          True,
      ),
      (
          state_pb2.TasksAppTask(
              completed_date='October 22 2023',
              importance='0',
              title='Meeting with David',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='title',
                  value='Jane',
                  operation=task_pb2.ExclusionCondition.Operation.CONTAINS,
              )
          ],
          True,
      ),
      (
          state_pb2.TasksAppTask(
              due_date='October 15 2023',
              due_time='12:30',
              importance='2',
              title='Meeting',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='due_date',
                  value='October 15 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              )
          ],
          False,
      ),
      (
          state_pb2.TasksAppTask(
              hide_until_date='October 22 2023',
              hide_until_time='12:30',
              importance='2',
              title='Meeting',
          ),
          [
              task_pb2.ExclusionCondition(
                  field='importance',
                  value='3',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              )
          ],
          True,
      ),
  ])
  def test_check_task_conditions(
      self,
      tasks_app_task: state_pb2.TasksAppTask,
      exclusion_conditions: list[task_pb2.ExclusionCondition],
      expected_value: bool,
  ):
    task = task_app_utils.create_task_from_proto(tasks_app_task)
    self.assertEqual(
        task_app_utils.check_task_conditions(task, exclusion_conditions),
        expected_value,
    )


if __name__ == '__main__':
  absltest.main()
