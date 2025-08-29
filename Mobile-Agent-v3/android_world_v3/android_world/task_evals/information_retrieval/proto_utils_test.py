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
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.utils import sqlite_schema_utils


class ProtoUtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      (
          [
              state_pb2.Event(
                  start_date='{month} {day} 2023',
                  duration='{duration}',
                  location='{location}',
                  description='{description}',
              )
          ],
          {
              'month': 'October',
              'day': '1',
              'extra': 'extra',
              'duration': '60m',
              'location': 'Mountain View',
              'description': 'Meeting',
          },
          [],
          [
              state_pb2.Event(
                  start_date='October 1 2023',
                  duration='60m',
                  location='Mountain View',
                  description='Meeting',
              )
          ],
      ),
      (
          [
              state_pb2.Event(
                  start_date='{date}',
                  duration='{duration}',
                  start_time='{time}',
              ),
              state_pb2.Event(
                  start_date='{date_without_replacement}',
                  duration='{duration_without_replacement}',
                  start_time='{time_without_replacement}',
              ),
          ],
          {
              'date': 'next Tuesday',
              'time': '12:30',
              'duration': '1h',
          },
          [
              task_pb2.TaskParams(
                  name='date', possible_values=['next Tuesday', 'today']
              ),
              task_pb2.TaskParams(
                  name='time', possible_values=['12:30', '5pm']
              ),
              task_pb2.TaskParams(
                  name='duration', possible_values=['15m', '1h']
              ),
          ],
          [
              state_pb2.Event(
                  start_date='next Tuesday',
                  duration='1h',
                  start_time='12:30',
              ),
              state_pb2.Event(
                  start_date='today',
                  duration='15m',
                  start_time='5pm',
              ),
          ],
      ),
  ])
  def test_format_calendar_state_with_params(
      self,
      events: list[state_pb2.Event],
      chosen_params: dict[str, Any],
      all_params: list[task_pb2.TaskParams],
      expected_events: list[sqlite_schema_utils.CalendarEvent],
  ):
    calendar = state_pb2.Calendar(events=events)
    state = state_pb2.State(calendar=calendar)
    expected_all_params = []
    for param in all_params:
      expected_param = task_pb2.TaskParams()
      expected_param.CopyFrom(param)
      expected_all_params.append(expected_param)
    proto_utils.format_state_with_params(state, chosen_params, all_params)

    self.assertEqual(
        list(state.calendar.events),
        expected_events,
    )
    # Make sure all_params hasn't been modified.
    self.assertEqual(
        all_params,
        expected_all_params,
    )

  @parameterized.parameters([
      (
          [
              state_pb2.TasksAppTask(
                  due_date='{month} {day} 2023',
                  importance='{importance}',
                  title='{title}',
              )
          ],
          {
              'month': 'October',
              'day': '1',
              'extra': 'extra',
              'importance': '3',
              'title': 'Unimportant task',
          },
          [],
          [
              state_pb2.TasksAppTask(
                  due_date='October 1 2023',
                  importance='3',
                  title='Unimportant task',
              )
          ],
      ),
      (
          [
              state_pb2.TasksAppTask(
                  completed_date='{date}',
                  title='{title}',
                  importance='{importance}',
              ),
              state_pb2.TasksAppTask(
                  completed_date='{date_without_replacement}',
                  title='{title_without_replacement}',
                  importance='{importance_without_replacement}',
              ),
          ],
          {
              'date': 'next Tuesday',
              'title': 'Pay Rent',
              'importance': '0',
          },
          [
              task_pb2.TaskParams(
                  name='date', possible_values=['next Tuesday', 'today']
              ),
              task_pb2.TaskParams(
                  name='title', possible_values=['Pay Rent', 'Unimportant task']
              ),
              task_pb2.TaskParams(
                  name='importance', possible_values=['0', '3']
              ),
          ],
          [
              state_pb2.TasksAppTask(
                  completed_date='next Tuesday',
                  importance='0',
                  title='Pay Rent',
              ),
              state_pb2.TasksAppTask(
                  completed_date='today',
                  importance='3',
                  title='Unimportant task',
              ),
          ],
      ),
  ])
  def test_format_tasks_app_state_with_params(
      self,
      tasks_app_tasks: list[state_pb2.TasksAppTask],
      chosen_params: dict[str, Any],
      all_params: list[task_pb2.TaskParams],
      expected_tasks: list[sqlite_schema_utils.Task],
  ):
    tasks_app = state_pb2.TasksApp(tasks_app_tasks=tasks_app_tasks)
    state = state_pb2.State(tasks_app=tasks_app)
    expected_all_params = []
    for param in all_params:
      expected_param = task_pb2.TaskParams()
      expected_param.CopyFrom(param)
      expected_all_params.append(expected_param)
    proto_utils.format_state_with_params(state, chosen_params, all_params)

    self.assertEqual(
        list(state.tasks_app.tasks_app_tasks),
        expected_tasks,
    )
    self.assertEqual(all_params, expected_all_params)

  def test_format_initial_calendar_state_with_params(self):
    params = {
        'month': 'October',
        'day': '1',
        'extra': 'extra',
        'duration': '60m',
        'location': 'Mountain View',
        'description': 'Meeting',
    }

    event = state_pb2.Event()
    event.start_date = '{month} {day} 2023'
    event.duration = '{duration}'
    event.location = '{location}'
    event.description = '{description}'

    expected_event = state_pb2.Event()
    expected_event.start_date = 'October 1 2023'
    expected_event.duration = '60m'
    expected_event.location = 'Mountain View'
    expected_event.description = 'Meeting'

    exclusion_condition = task_pb2.ExclusionCondition()
    exclusion_condition.operation = task_pb2.ExclusionCondition.EQUAL_TO
    exclusion_condition.field = 'start_date'
    exclusion_condition.value = '{month} {day}'
    expected_exclusion_condition = task_pb2.ExclusionCondition()
    expected_exclusion_condition.operation = (
        task_pb2.ExclusionCondition.EQUAL_TO
    )
    expected_exclusion_condition.field = 'start_date'
    expected_exclusion_condition.value = 'October 1'
    calendar = state_pb2.Calendar(events=[event, event, event])
    relevant_state = task_pb2.RelevantState(
        state=state_pb2.State(calendar=calendar),
        exclusion_conditions=[exclusion_condition],
    )
    proto_utils.format_relevant_state_with_params(relevant_state, params, [])

    self.assertEqual(
        list(relevant_state.state.calendar.events),
        [expected_event, expected_event, expected_event],
    )
    self.assertEqual(
        list(relevant_state.exclusion_conditions),
        [expected_exclusion_condition],
    )

  def test_format_initial_tasks_app_state_with_params(self):
    params = {
        'month': 'October',
        'day': '1',
        'extra': 'extra',
        'importance': '0',
        'title': 'Pay Rent',
    }

    tasks_app_task = state_pb2.TasksAppTask()
    tasks_app_task.due_date = '{month} {day} 2023'
    tasks_app_task.title = '{title}'
    tasks_app_task.importance = '{importance}'

    expected_task = state_pb2.TasksAppTask()
    expected_task.due_date = 'October 1 2023'
    expected_task.title = 'Pay Rent'
    expected_task.importance = '0'

    exclusion_condition = task_pb2.ExclusionCondition()
    exclusion_condition.operation = task_pb2.ExclusionCondition.EQUAL_TO
    exclusion_condition.field = 'due_date'
    exclusion_condition.value = '{month} {day}'
    expected_exclusion_condition = task_pb2.ExclusionCondition()
    expected_exclusion_condition.operation = (
        task_pb2.ExclusionCondition.EQUAL_TO
    )
    expected_exclusion_condition.field = 'due_date'
    expected_exclusion_condition.value = 'October 1'
    tasks_app = state_pb2.TasksApp(
        tasks_app_tasks=[tasks_app_task, tasks_app_task, tasks_app_task]
    )
    relevant_state = task_pb2.RelevantState(
        state=state_pb2.State(tasks_app=tasks_app),
        exclusion_conditions=[exclusion_condition],
    )
    proto_utils.format_relevant_state_with_params(relevant_state, params, [])

    self.assertEqual(
        list(relevant_state.state.tasks_app.tasks_app_tasks),
        [expected_task, expected_task, expected_task],
    )
    self.assertEqual(
        list(relevant_state.exclusion_conditions),
        [expected_exclusion_condition],
    )

  @parameterized.parameters([
      (
          task_pb2.Task(
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='{month} {day} 2023',
                                  duration='{duration}',
                                  location='{location}',
                                  description='{description}',
                              )
                          ]
                      )
                  ),
                  exclusion_conditions=[
                      task_pb2.ExclusionCondition(
                          operation=task_pb2.ExclusionCondition.EQUAL_TO,
                          field='start_date',
                          value='{month} {day} 2023',
                      )
                  ],
              ),
              task_params=[
                  task_pb2.TaskParams(
                      name='month', possible_values=['October', 'November']
                  ),
                  task_pb2.TaskParams(name='day', possible_values=['1', '2']),
                  task_pb2.TaskParams(
                      name='duration', possible_values=['60m', '1h']
                  ),
              ],
          ),
          {
              'month': 'October',
              'day': '1',
              'extra': 'extra',
              'duration': '60m',
              'location': 'Mountain View',
              'description': 'Meeting',
          },
          task_pb2.Task(
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 1 2023',
                                  duration='60m',
                                  location='Mountain View',
                                  description='Meeting',
                              )
                          ]
                      )
                  ),
                  exclusion_conditions=[
                      task_pb2.ExclusionCondition(
                          operation=task_pb2.ExclusionCondition.EQUAL_TO,
                          field='start_date',
                          value='October 1 2023',
                      )
                  ],
              ),
              task_params=[
                  task_pb2.TaskParams(
                      name='month', possible_values=['October', 'November']
                  ),
                  task_pb2.TaskParams(name='day', possible_values=['1', '2']),
                  task_pb2.TaskParams(
                      name='duration', possible_values=['60m', '1h']
                  ),
              ],
          ),
      ),
      (
          task_pb2.Task(
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='{month} {day} 2023',
                                  duration='{duration}',
                                  location='{location}',
                                  description='{description}',
                              ),
                              state_pb2.Event(
                                  start_date='{month_without_replacement}',
                                  duration='{duration_without_replacement}',
                                  location='{location_without_replacement}',
                              ),
                          ]
                      )
                  ),
                  exclusion_conditions=[
                      task_pb2.ExclusionCondition(
                          operation=task_pb2.ExclusionCondition.EQUAL_TO,
                          field='start_date',
                          value='{month} {day} 2023',
                      )
                  ],
              ),
              task_params=[
                  task_pb2.TaskParams(
                      name='month',
                      possible_values=['{CURRENT_MONTH}', '{NEXT_MONTH}'],
                  ),
                  task_pb2.TaskParams(name='day', possible_values=['1', '2']),
                  task_pb2.TaskParams(
                      name='duration', possible_values=['60m', '30m']
                  ),
                  task_pb2.TaskParams(
                      name='location',
                      possible_values=['Mountain View', 'London'],
                  ),
              ],
          ),
          {
              'month': '{CURRENT_MONTH}',
              'CURRENT_MONTH': 'October',
              'day': '1',
              'extra': 'extra',
              'duration': '60m',
              'location': 'Mountain View',
              'description': 'Meeting',
              'NEXT_MONTH': 'November',
          },
          task_pb2.Task(
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 1 2023',
                                  duration='60m',
                                  location='Mountain View',
                                  description='Meeting',
                              ),
                              state_pb2.Event(
                                  start_date='November',
                                  duration='30m',
                                  location='London',
                              ),
                          ]
                      )
                  ),
                  exclusion_conditions=[
                      task_pb2.ExclusionCondition(
                          operation=task_pb2.ExclusionCondition.EQUAL_TO,
                          field='start_date',
                          value='October 1 2023',
                      )
                  ],
              ),
              task_params=[
                  task_pb2.TaskParams(
                      name='month', possible_values=['October', 'November']
                  ),
                  task_pb2.TaskParams(name='day', possible_values=['1', '2']),
                  task_pb2.TaskParams(
                      name='duration', possible_values=['60m', '30m']
                  ),
                  task_pb2.TaskParams(
                      name='location',
                      possible_values=['Mountain View', 'London'],
                  ),
              ],
          ),
      ),
  ])
  def test_format_calendar_proto_with_params(
      self,
      task: task_pb2.Task,
      chosen_params: dict[str, Any],
      expected_proto: task_pb2.Task,
  ):
    proto_utils.initialize_proto(task, chosen_params)

    self.assertEqual(
        task,
        expected_proto,
    )

  @parameterized.parameters([
      (
          datetime.date(2023, 10, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.EQUAL_TO,
          True,
      ),
      (
          datetime.date(2023, 11, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.EQUAL_TO,
          False,
      ),
      (
          datetime.time(8, 54),
          datetime.time(20, 15),
          task_pb2.ExclusionCondition.Operation.EQUAL_TO,
          False,
      ),
      (
          'Meeting with David',
          'David',
          task_pb2.ExclusionCondition.Operation.CONTAINS,
          True,
      ),
      (
          'Meeting with David',
          'John',
          task_pb2.ExclusionCondition.Operation.CONTAINS,
          False,
      ),
      (
          datetime.date(2023, 11, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.GREATER_THAN,
          True,
      ),
      (
          datetime.date(2023, 10, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.GREATER_THAN,
          False,
      ),
      (
          datetime.time(8, 54),
          datetime.time(20, 15),
          task_pb2.ExclusionCondition.Operation.GREATER_THAN,
          False,
      ),
      (
          datetime.date(2023, 11, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
          True,
      ),
      (
          datetime.date(2023, 10, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
          True,
      ),
      (
          datetime.time(8, 54),
          datetime.time(20, 15),
          task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
          False,
      ),
      (
          datetime.date(2023, 11, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.LESS_THAN,
          False,
      ),
      (
          datetime.date(2023, 10, 10),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.LESS_THAN,
          True,
      ),
      (
          datetime.time(8, 54),
          datetime.time(20, 15),
          task_pb2.ExclusionCondition.Operation.LESS_THAN,
          True,
      ),
      (
          datetime.date(2023, 11, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
          False,
      ),
      (
          datetime.date(2023, 10, 15),
          datetime.date(2023, 10, 15),
          task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
          True,
      ),
      (
          datetime.time(8, 54),
          datetime.time(20, 15),
          task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
          True,
      ),
      (0, 3, task_pb2.ExclusionCondition.Operation.EQUAL_TO, False),
  ])
  def test_compare(
      self,
      field_value: Any,
      conditional_value: Any,
      operator: task_pb2.ExclusionCondition.Operation,
      expected_value: bool,
  ):
    self.assertEqual(
        proto_utils.compare(field_value, operator, conditional_value),
        expected_value,
    )

  @parameterized.parameters([
      (
          state_pb2.State(
              calendar=state_pb2.Calendar(
                  events=[
                      state_pb2.Event(
                          start_date='October 15 2023',
                          start_time='12:30',
                          duration='30m',
                          location='library',
                      ),
                      state_pb2.Event(
                          start_date='October 16 2023',
                          start_time='15:30',
                          duration='30m',
                          location='meeting room A',
                      ),
                  ]
              )
          ),
          'start_date',
          ['October 15 2023', 'October 16 2023'],
      ),
      (
          state_pb2.State(
              tasks_app=state_pb2.TasksApp(
                  tasks_app_tasks=[
                      state_pb2.TasksAppTask(
                          due_date='October 17 2023',
                          importance='2',
                          title='Chores',
                      ),
                      state_pb2.TasksAppTask(
                          due_date='October 18 2023',
                          importance='0',
                          title='Important task',
                      ),
                  ]
              ),
          ),
          'title',
          ['Chores', 'Important task'],
      ),
  ])
  def test_get_field_values(
      self,
      message: state_pb2.State,
      field_name: str,
      expected_values: list[Any],
  ):
    self.assertEqual(
        list(proto_utils._get_field_values(message, field_name)),
        expected_values,
    )

  @parameterized.parameters([
      (
          task_pb2.Task(
              name='test_task',
              prompt='Test calendar',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='12:30',
                                  duration='30m',
                                  location='library',
                                  title='Meet with Sam',
                              ),
                              state_pb2.Event(
                                  start_date='October 16 2023',
                                  start_time='15:30',
                                  duration='30m',
                                  title='Presentation',
                              ),
                          ]
                      ),
                  )
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              operation=task_pb2.FieldTransformation.Operation.IDENTITY,
                              field_name='title',
                          ),
                          match_type=task_pb2.Expectation.MatchType.STRING_MATCH,
                      ),
                  ]
              ),
          ),
          ['Meet with Sam', 'Presentation'],
      ),
      (
          task_pb2.Task(
              name='test_task',
              prompt='Test tasks_app',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      tasks_app=state_pb2.TasksApp(
                          tasks_app_tasks=[
                              state_pb2.TasksAppTask(
                                  due_date='October 17 2023',
                                  importance='2',
                                  title='Chores',
                              ),
                              state_pb2.TasksAppTask(
                                  due_date='October 18 2023',
                                  importance='0',
                                  title='Important task',
                              ),
                          ]
                      )
                  )
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              operation=task_pb2.FieldTransformation.Operation.COUNT,
                              field_name='tasks_app_tasks',
                          )
                      ),
                  ]
              ),
          ),
          [2],
      ),
      (
          task_pb2.Task(
              name='test_task',
              prompt='Test sports app',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      sports_activity_app=state_pb2.SportsActivityApp(
                          sports_activities=[
                              state_pb2.SportsActivity(
                                  total_distance='200',
                                  name='Morning Run',
                                  category='running',
                              ),
                              state_pb2.SportsActivity(
                                  total_distance='500',
                                  name='Morning Run',
                                  category='running',
                              ),
                          ]
                      )
                  )
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              operation=task_pb2.FieldTransformation.Operation.SUM,
                              field_name='total_distance',
                          )
                      ),
                  ]
              ),
          ),
          [700],
      ),
      (
          task_pb2.Task(
              name='test_task',
              prompt='Test tasks_app',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      tasks_app=state_pb2.TasksApp(
                          tasks_app_tasks=[
                              state_pb2.TasksAppTask(
                                  due_date='October 17 2023',
                                  due_time='12:30',
                                  importance='2',
                                  title='Chores',
                              ),
                              state_pb2.TasksAppTask(
                                  due_date='October 18 2023',
                                  due_time='8:30',
                                  importance='0',
                                  title='Important task',
                              ),
                          ]
                      )
                  )
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              operation=task_pb2.FieldTransformation.Operation.IDENTITY,
                              field_name='due_date',
                          ),
                          match_type=task_pb2.Expectation.MatchType.DATE_MATCH,
                      ),
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              field_name='due_time',
                          ),
                          match_type=task_pb2.Expectation.MatchType.TIME_MATCH,
                      ),
                  ]
              ),
          ),
          [
              datetime.datetime(
                  year=2023, month=10, day=17, hour=12, minute=30
              ),
              datetime.datetime(year=2023, month=10, day=18, hour=8, minute=30),
          ],
      ),
  ])
  def test_get_expected_answer(
      self,
      task: task_pb2.Task,
      expected_values: list[Any],
  ):
    self.assertEqual(proto_utils.get_expected_answer(task), expected_values)

  @parameterized.parameters([
      (
          'Meet with Sam',
          task_pb2.Task(
              name='test_task',
              prompt='Test task',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='10am',
                                  title='Meet with Sam',
                              ),
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='10am',
                                  title='Dentist appointment',
                              ),
                          ]
                      ),
                  ),
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              field_name='title',
                              operation=task_pb2.FieldTransformation.Operation.IDENTITY,
                          ),
                          match_type=task_pb2.Expectation.MatchType.STRING_MATCH,
                      ),
                  ]
              ),
          ),
          False,
      ),
      (
          'Dentist appointment, Meet with Sam',
          task_pb2.Task(
              name='test_task',
              prompt='Test task',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='10am',
                                  title='Meet with Sam',
                              ),
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='10am',
                                  title='Dentist appointment',
                              ),
                          ]
                      ),
                  ),
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              field_name='title',
                              operation=task_pb2.FieldTransformation.Operation.IDENTITY,
                          ),
                          match_type=task_pb2.Expectation.MatchType.STRING_MATCH,
                      ),
                  ]
              ),
          ),
          True,
      ),
      (
          'Meet with Sam October 15 2023, Presentation October 16 2023',
          task_pb2.Task(
              name='test_task',
              prompt='Test calendar',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='12:30',
                                  duration='30m',
                                  location='library',
                                  title='Meet with Sam',
                              ),
                              state_pb2.Event(
                                  start_date='October 16 2023',
                                  start_time='15:30',
                                  duration='30m',
                                  title='Presentation',
                              ),
                          ]
                      ),
                  )
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              field_name='title',
                              operation=task_pb2.FieldTransformation.Operation.IDENTITY,
                          ),
                          match_type=task_pb2.Expectation.MatchType.STRING_MATCH,
                      ),
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              field_name='start_date',
                              operation=task_pb2.FieldTransformation.Operation.IDENTITY,
                          ),
                          match_type=task_pb2.Expectation.MatchType.DATE_MATCH,
                      ),
                  ]
              ),
          ),
          ValueError(
              "Unsupported combined match types: ['STRING_MATCH', 'DATE_MATCH']"
          ),
      ),
      (
          'You had 2 events totaling 60 minutes.',
          task_pb2.Task(
              name='test_task',
              prompt='Test calendar',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      calendar=state_pb2.Calendar(
                          events=[
                              state_pb2.Event(
                                  start_date='October 15 2023',
                                  start_time='12:30',
                                  duration='30',
                                  location='library',
                                  title='Meet with Sam',
                              ),
                              state_pb2.Event(
                                  start_date='October 16 2023',
                                  start_time='15:30',
                                  duration='30',
                                  title='Presentation',
                              ),
                          ]
                      ),
                  )
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              operation=task_pb2.FieldTransformation.Operation.SUM,
                              field_name='duration',
                          )
                      ),
                  ]
              ),
          ),
          ValueError('Answer given in the incorrect format.'),
      ),
      (
          '10000',
          task_pb2.Task(
              name='test_task',
              prompt='Test task',
              relevant_state=task_pb2.RelevantState(
                  state=state_pb2.State(
                      sports_activity_app=state_pb2.SportsActivityApp(
                          sports_activities=[
                              state_pb2.SportsActivity(
                                  start_date='October 15 2023',
                                  name='Workout',
                                  total_distance='9895',
                              ),
                              state_pb2.SportsActivity(
                                  start_date='October 10 2023',
                                  name='Workout 2',
                                  total_distance='100',
                              ),
                          ]
                      ),
                  ),
              ),
              success_criteria=task_pb2.SuccessCriteria(
                  expectations=[
                      task_pb2.Expectation(
                          field_transformation=task_pb2.FieldTransformation(
                              field_name='total_distance',
                              operation=task_pb2.FieldTransformation.Operation.SUM,
                          ),
                          match_type=task_pb2.Expectation.MatchType.NUMBER_MATCH,
                          tolerance=10,
                      ),
                  ]
              ),
          ),
          True,
      ),
  ])
  def test_check_agent_answer(
      self, agent_answer, task, expected: bool | Exception
  ):
    if isinstance(expected, bool):
      got = proto_utils.check_agent_answer(agent_answer, task)
      self.assertEqual(expected, got)
    else:
      with self.assertRaises(type(expected)) as exception:
        proto_utils.check_agent_answer(agent_answer, task)
      self.assertEqual(exception.exception.args, expected.args)


if __name__ == '__main__':
  absltest.main()
