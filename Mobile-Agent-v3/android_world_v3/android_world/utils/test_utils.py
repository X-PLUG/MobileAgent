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

"""Mocks for agents."""

import random
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import phone_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import app_snapshot
from android_world.utils import contacts_utils
from android_world.utils import datetime_utils
from android_world.utils import file_utils
import numpy as np


class FakeCurrentStateEval(task_eval.TaskEval):
  """Fake current state eval for testing."""

  app_names = tuple()
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {},
  }
  template = 'Current state eval'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'Number': random.randint(0, 100_000), 'seed': 123}


class FakeAdbEval(task_eval.TaskEval):
  """Fake adb eval for testing."""

  app_names = tuple()
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {},
  }
  template = 'ADB eval'
  name = 'FakeAdbEval'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'Number': random.randint(0, 100_000), 'seed': 123}


class FakeMiniWobTask(FakeAdbEval):

  @property
  def goal(self) -> str:
    if not self.initialized:
      raise ValueError(
          'MiniWoB task must be initialized using initialize_task '
          'before the goal can be retrieved.'
      )
    return super().goal


class AdbEvalTestBase(absltest.TestCase):
  """A base test case class that mocks commonly used ADB utilities and Android interactions."""

  def setUp(self):
    super().setUp()
    self.mock_env = FakeAsyncEnv()
    self.mock_get_call_state = mock.patch.object(
        adb_utils, 'get_call_state'
    ).start()
    self.mock_get_current_activity = mock.patch.object(
        adb_utils, 'get_current_activity'
    ).start()
    self.mock_forest_to_ui_elements = mock.patch.object(
        representation_utils, 'forest_to_ui_elements'
    ).start()
    self.mock_get_a11y_tree = mock.patch.object(
        android_world_controller, 'get_a11y_tree'
    ).start()
    self.mock_dialer_with_phone_number = mock.patch.object(
        phone_validators, 'check_if_dialer_with_phone_number'
    ).start()
    self.mock_close_app = mock.patch.object(adb_utils, 'close_app').start()
    self.mock_close_recents = mock.patch.object(
        adb_utils, 'close_recents'
    ).start()
    self.mock_issue_generic_request = mock.patch.object(
        adb_utils, 'issue_generic_request'
    ).start()
    self.mock_execute_sql_command = mock.patch.object(
        adb_utils, 'execute_sql_command'
    ).start()
    self.mock_check_file_or_folder_exists = mock.patch.object(
        file_utils, 'check_file_or_folder_exists'
    ).start()
    self.mock_check_file_exists = mock.patch.object(
        file_utils, 'check_file_exists'
    ).start()
    self.mock_mkdir = mock.patch.object(file_utils, 'mkdir').start()
    self.mock_create_file = mock.patch.object(file_utils, 'create_file').start()
    self.mock_remove_single_file = mock.patch.object(
        file_utils, 'remove_single_file'
    ).start()
    self.mock_remove_files = mock.patch.object(
        file_utils, 'clear_directory'
    ).start()
    self.mock_copy_dir = mock.patch.object(file_utils, 'copy_dir').start()
    self.mock_create_random_files = mock.patch.object(
        user_data_generation, 'generate_noise_files'
    ).start()
    self.mock_get_file_list_with_metadata = mock.patch.object(
        file_utils, 'get_file_list_with_metadata'
    ).start()
    self.mock_get_clipboard_contents = mock.patch.object(
        adb_utils, 'get_clipboard_contents'
    ).start()
    self.mock_set_clipboard_contents = mock.patch.object(
        adb_utils, 'set_clipboard_contents'
    ).start()
    self.mock_toggle_airplane_mode = mock.patch.object(
        adb_utils, 'toggle_airplane_mode'
    ).start()
    self.mock_get_logical_screen_size = mock.patch.object(
        adb_utils, 'get_logical_screen_size'
    ).start()
    self.mock_get_physical_frame_boundary = mock.patch.object(
        adb_utils, 'get_physical_frame_boundary'
    ).start()
    self.mock_get_orientation = mock.patch.object(
        adb_utils, 'get_orientation'
    ).start()
    self.mock_set_datetime = mock.patch.object(
        datetime_utils, 'set_datetime'
    ).start()
    self.mock_set_datetime = mock.patch.object(
        datetime_utils, 'setup_datetime'
    ).start()
    self.mock_advance_system_time = mock.patch.object(
        datetime_utils, 'advance_system_time'
    ).start()
    self.mock_add_contact = mock.patch.object(
        contacts_utils, 'add_contact'
    ).start()
    self.mock_list_contacts = mock.patch.object(
        contacts_utils, 'list_contacts'
    ).start()
    self.mock_clear_contacts = mock.patch.object(
        contacts_utils, 'clear_contacts'
    ).start()
    self.mock_sleep = mock.patch.object(time, 'sleep').start()
    self.mock_restore_snapshot = mock.patch.object(
        app_snapshot, 'restore_snapshot'
    ).start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()


def log_mock_calls(mock_obj: Any):
  """Logs mock calls; useful for debugging."""
  original_side_effect = mock_obj.side_effect

  def side_effect(*args, **kwargs):
    print(f'Called with args={args}, kwargs={kwargs}')
    if callable(original_side_effect):
      return original_side_effect(*args, **kwargs)
    return next(original_side_effect)

  mock_obj.side_effect = side_effect


def perform_task(task: task_eval.TaskEval, env: interface.AsyncEnv) -> float:
  """Runs the setup, is_successful, and teardown and returns the result."""
  task.initialize_task(env)
  result = task.is_successful(env)
  task.tear_down(env)
  return result


class FakeAsyncEnv(interface.AsyncAndroidEnv):
  """Fake environment for testing."""

  def __init__(self):
    self._reset_called = True
    self._controller = mock.create_autospec(
        android_world_controller.AndroidWorldController, instance=True
    )

  @property
  def controller(self) -> android_world_controller.AndroidWorldController:
    return self._controller

  def reset(self, go_home: bool = False) -> interface.State:
    return interface.State(
        pixels=(np.random.rand(10, 10, 3) * 255).astype(np.uint8),
        forest=None,
        ui_elements=[],
    )

  def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
    return interface.State(
        pixels=(np.random.rand(10, 10, 3) * 255).astype(np.uint8),
        forest=mock.MagicMock(),
        ui_elements=[],
    )

  def execute_action(self, action: json_action.JSONAction):
    del action

  def run_adb_command(self, command: str) -> adb_pb2.AdbResponse:
    del command
    return adb_pb2.AdbResponse()

  @property
  def foreground_activity_name(self) -> str:
    return 'MockActivity'

  @property
  def screen_size(self) -> tuple[int, int]:
    return (100, 100)

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return (100, 100)
