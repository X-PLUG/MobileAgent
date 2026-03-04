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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from android_world.task_evals.information_retrieval import calendar_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.utils import sqlite_schema_utils


class TestTimestampToLocalDatetime(parameterized.TestCase):

  @parameterized.parameters([
      ('30m', 30 * 60),
      ('45 minutes', 45 * 60),
      ('45 mins', 45 * 60),
      ('1 hour', 3600),
      ('1.5 hours', 5400),
      ('2h', 3600 * 2),
  ])
  def test_parse_duration(self, duration_string: str, expected: int):
    result = calendar_utils.parse_duration(duration_string)
    self.assertEqual(result, expected)

  def test_convert_date_to_unix_ts(self):
    new_date = 'October 16 2023'
    new_time = '2pm'
    self.assertEqual(
        calendar_utils.convert_datetime_to_unix_ts(new_date, new_time),
        1697464800,
    )

  def test_create_event_from_proto(self):
    event = state_pb2.Event(
        start_date='October 15 2023',
        start_time='2:30pm',
        duration='30 minutes',
        description='Description',
        title='Meeting',
    )

    converted_event = calendar_utils.create_event_from_proto(event)
    expected_event = sqlite_schema_utils.CalendarEvent(
        start_ts=1697380200,  # October 15, 2023 14:30
        end_ts=1697382000,  # October 15, 2023, 15:00
        title='Meeting',
        description='Description',
    )
    self.assertEqual(converted_event, expected_event)

  @mock.patch.object(random, 'choice')
  @mock.patch.object(random, 'randrange')
  def test_generate_random_event(self, mock_randrange, mock_choice):
    expected_event = sqlite_schema_utils.CalendarEvent(
        start_ts=1697556840,  # October 17, 2023 15:34
        end_ts=1697558640,  # October 17, 2023, 16:04
        title='Call with Nick',
        description="We will organize Jess' birthday party.",
    )

    # 1st event mock calls:
    # Mock date will be 2 days after current date (= October 17, 15:34)
    date_call1 = 24 * 60 * 17
    duration_call1 = 30
    title_call1 = 'Meeting with'
    title_name_call1 = 'Dan'
    description_verb_call1 = 'discuss'
    description_object_call1 = 'annual budget'

    # 2nd event mock calls:
    date_call2 = 24 * 60 * 17
    duration_call2 = 30
    title_call2 = 'Call with'
    title_name_call2 = 'Nick'
    description_verb_call2 = 'organize'
    description_object_call2 = "Jess' birthday party"
    mock_randrange.side_effect = [date_call1, date_call2]
    mock_choice.side_effect = [
        duration_call1,
        title_call1,
        title_name_call1,
        description_verb_call1,
        description_object_call1,
        False,
        duration_call2,
        title_call2,
        title_name_call2,
        description_verb_call2,
        description_object_call2,
        False,
    ]

    exclusion_conditions = [
        task_pb2.ExclusionCondition(
            field='title',
            value=title_name_call1,
            operation=task_pb2.ExclusionCondition.Operation.CONTAINS,
        ),
    ]
    event = calendar_utils.generate_random_event(exclusion_conditions)
    self.assertEqual(event, expected_event)

  @parameterized.named_parameters(
      dict(
          testcase_name=(
              'multiple exclusion conditions: start_date exact match, event'
              ' time overlap - is excluded'
          ),
          event=state_pb2.Event(
              start_date='October 5 2023',
              start_time='12:30',
              duration='60m',
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 5 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:45',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'multiple exclusion conditions: start_date before exclusion'
              ' condition, event time overlap - not excluded'
          ),
          event=state_pb2.Event(
              start_date='October 4 2023',
              start_time='12:30',
              duration='60m',
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 5 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:45',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=True,
      ),
      dict(
          testcase_name=(
              'multiple exclusion conditions: start_date and time no overlap -'
              ' not excluded'
          ),
          event=state_pb2.Event(
              start_date='October 6 2023',
              start_time='12:30',
              duration='48h',  # end date is October 8 2023
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 5 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:30',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=True,
      ),
      dict(
          testcase_name=(
              'event overlaps start_date exclusion condition - excluded'
          ),
          event=state_pb2.Event(
              start_date='October 4 2023',
              start_time='12:30',
              duration='48h',  # end date is October 6 2023
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 5 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'event overlaps start_time exclusion condition with no excluded'
              ' start_date - excluded'
          ),
          event=state_pb2.Event(
              start_date='October 4 2023',
              start_time='12:30',
              duration='30m',
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:45',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'event overlaps start_date and start_time exclusion conditions -'
              ' excluded'
          ),
          event=state_pb2.Event(
              start_date='October 4 2023',
              start_time='12:30',
              duration='48h',  # end date is October 6 2023
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 5 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:45',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=False,
      ),
      dict(
          testcase_name='equal_to exclusion condition not excluded',
          event=state_pb2.Event(
              start_date='October 15 2023',
              start_time='12:30',
              duration='60m',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          expected_value=True,
      ),
      dict(
          testcase_name='less_than exclusion condition is excluded',
          event=state_pb2.Event(
              start_date='October 15 2023',
              start_time='12:30',
              duration='60m',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.LESS_THAN,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'multiple exclusion conditions: start_date less_than matches,'
              ' start_time does not match - not excluded'
          ),
          event=state_pb2.Event(
              start_date='October 15 2023',
              start_time='12:30',
              duration='60m',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.LESS_THAN,
              ),
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='14:00',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=True,
      ),
      dict(
          testcase_name='contains exclusion condition is excluded',
          event=state_pb2.Event(
              start_date='October 15 2023',
              start_time='12:30',
              duration='60m',
              title='Meeting with John',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='title',
                  value='John',
                  operation=task_pb2.ExclusionCondition.Operation.CONTAINS,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'multiple exclusion conditions: only one applies - not excluded'
          ),
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='60m',
              title='test title',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN,
              ),
              task_pb2.ExclusionCondition(
                  field='title',
                  value='test title',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              ),
          ],
          expected_value=True,
      ),
      dict(
          testcase_name='single condition: EQUAL_TO start_date - excluded',
          event=state_pb2.Event(
              start_date='October 15 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 15 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name='single condition: EQUAL_TO start_date - not excluded',
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 24 2023',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          expected_value=True,
      ),
      dict(
          testcase_name='single condition: EQUAL_TO start_time - excluded',
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:30pm',
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'multiple conditions: CONTAINS title met GREATER_THAN date not'
              ' met - not excluded'
          ),
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting with David',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 24 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN,
              ),
              task_pb2.ExclusionCondition(
                  field='title',
                  value='David',
                  operation=task_pb2.ExclusionCondition.Operation.CONTAINS,
              ),
          ],
          expected_value=True,
      ),
      dict(
          testcase_name='single condition: title contains - not excluded',
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting with David',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='title',
                  value='Jane',
                  operation=task_pb2.ExclusionCondition.Operation.CONTAINS,
              )
          ],
          expected_value=True,
      ),
      dict(
          testcase_name=(
              'single condition: greater_than_or_equal_to start_date - excluded'
          ),
          event=state_pb2.Event(
              start_date='October 15 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 15 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'single condition: GREATER_THAN_OR_EQUAL_TO start_date - not'
              ' excluded'
          ),
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 24 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              )
          ],
          expected_value=True,
      ),
      dict(
          testcase_name=(
              'single condition: LESS_THAN_OR_EQUAL_TO - not excluded'
          ),
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:30',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:00pm',
                  operation=task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
              )
          ],
          expected_value=True,
      ),
      dict(
          testcase_name=(
              'single condition: GREATER_THAN_OR_EQUAL_TO start_time - excluded'
          ),
          event=state_pb2.Event(
              start_date='October 22 2023',
              start_time='12:00',
              duration='30m',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_time',
                  value='12:30pm',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name=(
              'single condition: GREATER_THAN_OR_EQUAL_TO overlap at day'
              ' boundary - excluded'
          ),
          event=state_pb2.Event(
              start_date='October 21 2023',
              start_time='23:59',
              duration='1h',
              location='Mountain View',
              title='Meeting',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  field='start_date',
                  value='October 22 2023',
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
              )
          ],
          expected_value=False,
      ),
      dict(
          testcase_name='multiple conditions: time range - not excluded',
          event=state_pb2.Event(
              start_date='October 18 2023',
              start_time='00:45',
              duration='15 m',
              description=(
                  'We will celebrate upcoming project milestones. Looking'
                  ' forward to productive discussions.'
              ),
              title='Call with HR',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                  field='start_date',
                  value='October 27 2023',
              ),
              task_pb2.ExclusionCondition(
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
                  field='start_time',
                  value='11:00am',
              ),
              task_pb2.ExclusionCondition(
                  operation=task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
                  field='start_time',
                  value='8pm',
              ),
          ],
          expected_value=True,
      ),
      dict(
          testcase_name='multiple conditions: time range - excluded',
          event=state_pb2.Event(
              start_date='October 17 2023',
              start_time='07:31',
              duration='15 m',
              description='We will celebrate contract details.',
              title='Catch up on Campaign',
          ),
          exclusion_conditions=[
              task_pb2.ExclusionCondition(
                  operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                  field='start_date',
                  value='October 17 2023',
              ),
              task_pb2.ExclusionCondition(
                  operation=task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO,
                  field='start_time',
                  value='7:45am',
              ),
              task_pb2.ExclusionCondition(
                  operation=task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO,
                  field='start_time',
                  value='8pm',
              ),
          ],
          expected_value=False,
      ),
  )
  def test_check_event_conditions(
      self,
      event: state_pb2.Event,
      exclusion_conditions: list[task_pb2.ExclusionCondition],
      expected_value: bool,
  ):
    self.assertEqual(
        calendar_utils.check_event_conditions(event, exclusion_conditions),
        expected_value,
    )


if __name__ == '__main__':
  absltest.main()
