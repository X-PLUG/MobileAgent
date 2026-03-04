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

from typing import Any
from unittest import mock
from absl.testing import absltest
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.utils import test_utils


class MockTaskEval(task_eval.TaskEval):

  @property
  def complexity(self) -> int:
    return 1

  @property
  def app_names(self) -> tuple[str, ...]:
    return ("MockApp",)

  @property
  def schema(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
        },
        "required": ["param1"],
    }

  @property
  def template(self) -> str:
    return "Mock task with {param1}"

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"param1": "value1"}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0


class TestTaskEval(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.params = {"param1": "test"}
    self.mock_env = mock.create_autospec(interface.AsyncEnv)
    self.scripted_task = MockTaskEval(self.params)

  def test_initialization(self):
    self.assertFalse(self.scripted_task.initialized)
    self.scripted_task.initialize_task(self.mock_env)
    self.mock_set_datetime.assert_called_once()
    self.assertTrue(self.scripted_task.initialized)

  def test_initialize_already_initialized(self):
    self.scripted_task.initialize_task(self.mock_env)
    with self.assertRaises(RuntimeError):
      self.scripted_task.initialize_task(self.mock_env)

  def test_is_successful_not_initialized(self):
    with self.assertRaises(RuntimeError):
      self.scripted_task.is_successful(self.mock_env)

  def test_name_property(self):
    self.assertEqual(self.scripted_task.name, "MockTaskEval")

  def test_goal_property(self):
    self.assertEqual(self.scripted_task.goal, "Mock task with test")

  def test_tear_down(self):
    self.scripted_task.tear_down(self.mock_env)
    self.mock_close_recents.assert_called_once()


if __name__ == "__main__":
  absltest.main()
