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

"""Information retrieval registry; it dynamically creates tasks.

Information retrieval tasks are defined in a textproto file. For each task in
the proto,
we dynamically create a new task with the name of the task in the class name.
"""

import os
import random
from typing import Any, Generic, Type, TypeVar
from android_world.task_evals.information_retrieval import information_retrieval
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.utils import file_utils
from google.protobuf import text_format

TaskType = TypeVar('TaskType', bound=information_retrieval.InformationRetrieval)

_DEFAULT_COMPLEXITY = 1.0
_COMPLEXITY_OVERRIDES = {
    'SportsTrackerActivitiesOnDate': 2,
    'SportsTrackerActivityDuration': 1.2,
    'SportsTrackerTotalDistanceForCategoryOverInterval': 2.2,
    'SportsTrackerTotalDurationForCategoryThisWeek': 1.6,
    'TasksDueNextWeek': 1.2,
}


class InformationRetrievalRegistry(Generic[TaskType]):
  """Information retrieval registry; it dynamically creates tasks."""

  @property
  def registry(
      self,
  ) -> dict[str, TaskType]:
    return self._task_registry

  def _read_tasks(self) -> task_pb2.Tasks:
    proto = task_pb2.Tasks()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = file_utils.convert_to_posix_path(
        script_dir, 'proto', 'tasks.textproto'
    )
    with open(local_path, 'r') as f:
      textproto_content = f.read()
    text_format.Merge(textproto_content, proto)
    return proto

  def __init__(
      self,
      filename: str | None = None,
      task_type: Type[TaskType] = information_retrieval.InformationRetrieval,
  ):
    self._task_registry: dict[str, TaskType] = {}
    self.filename = filename
    self.task_type = task_type
    raw_tasks = self._read_tasks()
    for raw_task in raw_tasks.tasks:
      task_class = self._build_task_class(raw_task)
      self._task_registry[raw_task.name] = task_class

  def _build_task_class(
      self,
      task_proto: task_pb2.Task,
  ) -> TaskType:
    """Dynamically builds and returns a new subclass of InformationRetrieval.

    This function creates a subclass of InformationRetrieval from the task.

    Args:
      task_proto: The task proto defining the class to be created.

    Returns:
      A subclass of InformationRetrieval that is dynamically created.
    """

    @classmethod
    def generate_random_params(cls) -> dict[str, Any]:  # pylint:disable=unused-argument
      params = {}
      for task_param in task_proto.task_params:
        params[task_param.name] = random.choice(
            list(task_param.possible_values)
        )
      return params

    @property
    def task_template(self) -> task_pb2.Task:  # pylint:disable=unused-argument
      return task_proto

    return type(
        task_proto.name,
        (self.task_type,),
        {
            'generate_random_params': generate_random_params,
            'task_template': task_template,
            'complexity': _COMPLEXITY_OVERRIDES.get(
                task_proto.name, _DEFAULT_COMPLEXITY
            ),
        },
    )
