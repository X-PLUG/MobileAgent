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

"""Utils for manipulating the task and initialization protos."""

from collections.abc import Iterator
import datetime
import random
import re
from typing import Any, TypeVar

from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.utils import fuzzy_match_lib
from google.protobuf import message

ExpectedAnswer = TypeVar(
    'ExpectedAnswer',
    str,
    datetime.datetime,
    datetime.date,
    datetime.time,
    float,
    int,
)

AppData = TypeVar(
    'AppData', state_pb2.Event, state_pb2.TasksAppTask, state_pb2.SportsActivity
)

FieldMessage = TypeVar(
    'FieldMessage',
    state_pb2.Event,
    task_pb2.Expectation,
    state_pb2.TasksAppTask,
    state_pb2.SportsActivity,
    task_pb2.ExclusionCondition,
)


def _combine_date_and_time(
    answer1: ExpectedAnswer, answer2: ExpectedAnswer
) -> str | datetime.datetime:
  """Combines two expectations into a single answer.

  Combines them in the following ways:
   - one of the inputs is a date and the other is a time - will output a
   datetime
   - all other combinations will be combined as a string with a single space
   between the two.

  Args:
    answer1: The first answer to be merged.
    answer2: The second answer to be merged

  Returns:
    The merged result, either a datetime or a string.
  """
  if isinstance(answer1, datetime.date) and isinstance(answer2, datetime.time):
    return datetime.datetime(
        answer1.year, answer1.month, answer1.day, answer2.hour, answer2.minute
    )
  elif isinstance(answer2, datetime.date) and isinstance(
      answer1, datetime.time
  ):
    return datetime.datetime(
        answer2.year, answer2.month, answer2.day, answer1.hour, answer1.minute
    )
  else:
    raise ValueError(f'Unsupported combination: {answer1} and {answer2}')


def _check_match_types(
    match_types: list[task_pb2.Expectation.MatchType],
) -> None:
  """Checks if the match types are supported."""

  if len(match_types) == 1 or not match_types:
    return
  if len(match_types) > 2:
    raise ValueError(
        'Unsupported combined match types: {}'.format([
            task_pb2.Expectation.MatchType.Name(match_type)
            for match_type in match_types
        ])
    )
  if set(match_types) != set((
      task_pb2.Expectation.MatchType.DATE_MATCH,
      task_pb2.Expectation.MatchType.TIME_MATCH,
  )):
    raise ValueError(
        'Unsupported combined match types: {}'.format([
            task_pb2.Expectation.MatchType.Name(match_type)
            for match_type in match_types
        ])
    )


def _cast_answers_to_type(
    match_types: list[task_pb2.Expectation.MatchType], answers: list[str]
) -> list[ExpectedAnswer]:
  if not match_types:
    return answers
  match match_types:
    case [task_pb2.Expectation.MatchType.STRING_MATCH]:
      return [str(answer) for answer in answers]
    case [task_pb2.Expectation.MatchType.NUMBER_MATCH]:
      return [float(answer) for answer in answers]
    case [task_pb2.Expectation.MatchType.DATE_MATCH]:
      return [
          datetime.datetime.strptime(
              answer, datetime_utils_ir.DATE_FORMAT
          ).date()
          for answer in answers
      ]
    case [task_pb2.Expectation.MatchType.TIME_MATCH]:
      return [
          datetime.datetime.strptime(answer, '%H:%M').time()
          for answer in answers
      ]
    case [
        task_pb2.Expectation.MatchType.DATE_MATCH,
        task_pb2.Expectation.MatchType.TIME_MATCH,
    ]:
      return [
          datetime.datetime.strptime(
              answer, datetime_utils_ir.DATE_FORMAT + ' %H:%M'
          )
          for answer in answers
      ]
    case _:
      raise ValueError(f'Unsupported match types: {match_types}')


def check_agent_answer(agent_answer: str, task: task_pb2.Task) -> bool:
  """Checks if the agent answer matches the task's expectations."""
  # If there are multiple answers, they are separated by commas
  answers = [answer.strip() for answer in agent_answer.split(',')]
  match_types = list(
      map(
          lambda expectation: expectation.match_type,
          task.success_criteria.expectations,
      )
  )
  _check_match_types(match_types)

  try:
    type_cast_answers = _cast_answers_to_type(match_types, answers)
  except ValueError as e:
    raise ValueError('Answer given in the incorrect format.') from e

  expected_answers = get_expected_answer(task)
  comparator = lambda x, y: x == y
  if task_pb2.Expectation.MatchType.STRING_MATCH in match_types:
    comparator = fuzzy_match_lib.fuzzy_match
  elif (
      task_pb2.Expectation.MatchType.NUMBER_MATCH in match_types
      and task.success_criteria.expectations[0].HasField('tolerance')
  ):
    comparator = (
        lambda x, y: abs(x - y)
        < task.success_criteria.expectations[0].tolerance
    )
  if len(type_cast_answers) != len(expected_answers):
    return False
  return all(
      any(comparator(x, y) for y in expected_answers) for x in type_cast_answers
  )


def get_expected_answer(
    task: task_pb2.Task,
) -> list[ExpectedAnswer]:
  """Gets the expected answer from the task's success criteria."""
  expected_answers = []
  for expectation in task.success_criteria.expectations:
    if expectation.HasField('expected_value'):
      return _cast_answers_to_type(
          [expectation.match_type], [expectation.expected_value]
      )
    field_transformation = expectation.field_transformation
    field_values = _get_field_values(
        task.relevant_state.state, field_transformation.field_name
    )
    expected_answer = []
    # SUM and COUNT are of type NUMBER_MATCH so handle those first.
    if (
        field_transformation.operation
        == task_pb2.FieldTransformation.Operation.SUM
    ):
      return [sum((float(value) for value in field_values))]
    elif (
        field_transformation.operation
        == task_pb2.FieldTransformation.Operation.COUNT
    ):
      return [len(list(field_values))]
    elif expectation.match_type == task_pb2.Expectation.MatchType.STRING_MATCH:
      return list(field_values)
    elif expectation.match_type == task_pb2.Expectation.MatchType.NUMBER_MATCH:
      return [float(value) for value in field_values]
    elif expectation.match_type == task_pb2.Expectation.MatchType.DATE_MATCH:
      expected_answer.extend([
          datetime.datetime.strptime(
              value, datetime_utils_ir.DATE_FORMAT
          ).date()
          for value in field_values
      ])
    elif expectation.match_type == task_pb2.Expectation.MatchType.TIME_MATCH:
      expected_answer.extend(
          [datetime_utils_ir.parse_time(value) for value in field_values]
      )
    if not expected_answers:
      expected_answers.extend(expected_answer)
    else:
      expected_answers = [
          _combine_date_and_time(answer1, answer2)
          for answer1, answer2 in zip(expected_answers, expected_answer)
      ]
  return expected_answers


def _get_field_values(proto: message.Message, field_name: str) -> Iterator[Any]:
  """Gets the values for the given field_name from a proto."""
  for field, _ in proto.ListFields():
    field_value = getattr(proto, field.name)
    is_repeated_field = not isinstance(
        field_value, message.Message
    ) and not isinstance(field_value, str)
    if field.name == field_name:
      if is_repeated_field:
        for value in field_value:
          yield value
      else:
        yield field_value
    elif isinstance(field_value, message.Message):
      yield from _get_field_values(field_value, field_name)
    elif is_repeated_field:
      for element in field_value:
        yield from _get_field_values(element, field_name)


def _remove_used_params(
    used_params: dict[str, Any], all_params: list[task_pb2.TaskParams]
) -> None:
  """Removes the used params from the list of params."""
  for index, param in enumerate(all_params):
    if (
        param.name not in used_params
        or used_params[param.name] not in param.possible_values
    ):
      continue
    used_value = used_params[param.name]
    new_param = task_pb2.TaskParams()
    new_param.CopyFrom(param)
    new_param.possible_values.remove(used_value)
    all_params[index] = new_param


def format_state_with_params(
    state: state_pb2.State,
    task_params: dict[str, Any],
    all_params: list[task_pb2.TaskParams],
) -> None:
  """Formats the state with the task params and all_params if necessary."""
  # Make a copy of the list so that the caller's copy isn't affected.
  unused_params = all_params.copy()
  _remove_used_params(task_params, unused_params)
  for field, _ in state.ListFields():
    app_proto: (
        state_pb2.Calendar
        | state_pb2.TasksAppTask
        | state_pb2.SportsActivityApp
    ) = getattr(state, field.name)
    for app_field, _ in app_proto.ListFields():
      if app_field.name == 'app_name':
        continue
      app_data_list = getattr(app_proto, app_field.name)
      for app_data in app_data_list:
        for app_data_field, _ in app_data.ListFields():
          _format_field_if_exists(
              app_data, app_data_field.name, task_params, unused_params
          )


def format_relevant_state_with_params(
    relevant_state: task_pb2.RelevantState,
    task_params: dict[str, Any],
    all_params: list[task_pb2.TaskParams],
) -> None:
  unused_params = all_params.copy()
  _remove_used_params(task_params, unused_params)
  format_state_with_params(relevant_state.state, task_params, unused_params)
  for condition in relevant_state.exclusion_conditions:
    _format_field_if_exists(condition, 'value', task_params, unused_params)


def _format_params_with_params(
    task_params: list[task_pb2.TaskParams], params: dict[str, Any]
):
  for task_param in task_params:
    for index, possible_value in enumerate(task_param.possible_values):
      task_param.possible_values[index] = possible_value.format(**params)
  for param_name, param_value in params.items():
    if isinstance(param_value, str):
      params[param_name] = param_value.format(**params)


def initialize_proto(task: task_pb2.Task, task_params: dict[str, Any]):
  _format_params_with_params(list(task.task_params), task_params)
  _format_success_criteria_with_params(task.success_criteria, task_params)
  format_relevant_state_with_params(
      task.relevant_state, task_params, list(task.task_params)
  )


def _format_success_criteria_with_params(
    success_criteria: task_pb2.SuccessCriteria, task_params: dict[str, Any]
):
  for expectation in success_criteria.expectations:
    if expectation.HasField('expected_value'):
      _format_field_if_exists(expectation, 'expected_value', task_params, [])


def _format_field_if_exists(
    proto: FieldMessage,
    field_name: str,
    task_params: dict[str, Any],
    unused_params: list[task_pb2.TaskParams],
):
  """Formats the field if it exists with the params.

  Formats each field with task_params. Additionaly, if the field has a param
  with '_without_replacement' in its name, it will pick parameters from
  unused_params to format it. These picked parameter values will then be
  removed from the unused_params list.

  Args:
    proto: The proto whose field will be formatted.
    field_name: The name of the field to format.
    task_params: The task's parameters to format the field with.
    unused_params: Extra list of parameters to chose from if task_params does
      not fully format the field.
  """
  if proto.HasField(field_name):
    if '_without_replacement}' in str(getattr(proto, field_name)):
      _format_without_replacement(proto, field_name, unused_params)
    else:
      setattr(
          proto,
          field_name,
          getattr(proto, field_name).format(**task_params),
      )


def _format_without_replacement(
    proto: FieldMessage,
    field_name: str,
    unused_params: list[task_pb2.TaskParams],
):
  """Handles field formatting when the param name contains '_without_replacement'.

  The field's parameter value will be chosen from the unused_params list and
  that value will then be removed as a possible value from that list.

  Args:
    proto: The proto whose field will be formatted.
    field_name: The name of the field to format.
    unused_params: A list of TaskParams containing possible values that have not
      yet been used for other field formatting.
  """
  field_value = getattr(proto, field_name)
  # Get the names of the parameter:
  without_replacement_params = [
      param_name[1 : param_name.find('_without_replacement')]
      for param_name in re.findall(r'\{.+?\}', field_value)
      if param_name.endswith('without_replacement}')
  ]
  for param_name in without_replacement_params:
    original_param_name = param_name + '_without_replacement'
    new_value = None
    for task_param in unused_params:
      if task_param.name == param_name:
        new_value = random.choice(list(task_param.possible_values))
        _remove_used_params({task_param.name: new_value}, unused_params)
        break

    setattr(
        proto,
        field_name,
        getattr(proto, field_name).format(**{original_param_name: new_value}),
    )


_T = TypeVar('_T')


def compare(
    field_value: _T,
    operator: task_pb2.ExclusionCondition.Operation,
    comparison_value: _T,
) -> bool:
  """Compares the field value against the comparison value using the operator."""
  if operator == task_pb2.ExclusionCondition.Operation.EQUAL_TO:
    return field_value == comparison_value
  elif operator == task_pb2.ExclusionCondition.Operation.GREATER_THAN:
    return field_value > comparison_value
  elif (
      operator == task_pb2.ExclusionCondition.Operation.GREATER_THAN_OR_EQUAL_TO
  ):
    return field_value >= comparison_value
  elif operator == task_pb2.ExclusionCondition.Operation.LESS_THAN:
    return field_value < comparison_value
  elif operator == task_pb2.ExclusionCondition.Operation.LESS_THAN_OR_EQUAL_TO:
    return field_value <= comparison_value
  elif operator == task_pb2.ExclusionCondition.Operation.CONTAINS:
    return comparison_value in str(field_value)
  else:
    raise ValueError(f'Unsupported operator: {operator}')
