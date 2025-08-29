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

"""A generic task eval with tunable template and apps.

Example usage:
  # Create new task class.
  task_type = generic.create_task_type(
      "NewTask", {"instruction": "test_instruction", "app_names": ("chrome",)}
  )

  # Create new task with assigned params.
  task = task_type(task_type.generate_random_params())

"""

from typing import Any

from android_world.env import interface
from android_world.task_evals import task_eval


def create_task_type(name, params=None):
  if params is None:
    params = {}
  return type(name, (GenericTaskEval,), {'instance_params': params})


class GenericTaskEval(task_eval.TaskEval):
  """A generic task eval with tunable template and apps."""

  app_names = ()
  complexity = -1

  instance_params = {}

  template = ''

  schema = {}

  def __init__(self, instance_params: dict[str, Any]):
    """Initialize the task with given params.

    Args:
      instance_params: The parameters to initialize the task with, including
        instruction and app names.
    """
    super().__init__(instance_params)
    self.template = instance_params.get('instruction', {})
    if 'app_names' in  instance_params:
      self.app_names = instance_params['app_names']
    print(f'App names: {self.app_names}')

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """By defualt, no reward is given."""
    return 0.0

  @classmethod
  def set_instance_params(cls, instance_params: dict[str, Any]):
    """Set the task params."""
    cls.instance_params = instance_params

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return cls.instance_params
