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

"""Abstract base class representing a Mini World of Bits (MiniWoB) task in Android."""

import abc
import time

from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval

_APP_NAME = "com.google.androidenv.miniwob"
_MAIN_ACTIVITY = f"{_APP_NAME}/{_APP_NAME}.app.MainActivity"


def _extract_data(
    action: str, env: env_interface.AndroidEnvInterface
) -> str | None:
  """Issues broadcast and extracts data with retries."""
  for _ in range(3):
    raw = adb_utils.send_android_intent("broadcast", action, env)
    result = adb_utils.extract_broadcast_data(
        raw.generic.output.decode("utf-8")
    )
    if result is not None:
      return result
    time.sleep(1)  # App still needs to load.
  return None


def _get_episode_utterance(env: env_interface.AndroidEnvInterface) -> str:
  """Gets the utterance for the current episode by querying MiniWob Android app."""
  utterance = _extract_data(f"{_APP_NAME}.app.GET_UTTERANCE_ACTION", env)
  if utterance is None:
    raise ValueError("Could not extract utterance; something went wrong.")
  return utterance


def get_episode_reward(env: env_interface.AndroidEnvInterface) -> float:
  """Gets the reward for the current episode by querying MiniWob Android app."""
  reward = _extract_data(f"{_APP_NAME}.app.GET_REWARD_ACTION", env)
  if reward is None:
    raise ValueError("Could not extract reward; something went wrong.")
  if not reward:  # Episode is not terminated.
    return 0.0
  return float(int(reward))


def is_episode_terminated(env: interface.AsyncEnv) -> bool:
  """Checks if the current episode is terminated."""
  return get_episode_reward(env.controller.env) != 0.0


class MiniWoBTask(task_eval.TaskEval, abc.ABC):
  """Abstract base class representing a Mini World of Bits (MiniWoB) task in Android.

  This class serves as a lighweight template for creating specific task
  instances within the MiniWoB framework, integrating with the Android
  environment for task execution.

  Each MiniWoBTask is characterized by a unique task name and an utterance
  that is dynamically generated and populated during task initialization.
  The utterance, generated through MiniWoB's JavaScript logic, provides the
  instructions for the task to be executed on the Android device. The evaluation
  logic is also providd by MiniWoB.
  """

  start_on_home_screen = False

  schema = {
      "type": "object",
      "properties": {
          "task_name": {"type": "string"},
          # This is filled in in `initialize_task`. The utterance is generated
          # by MiniWoB JavaScript logic. Once task is initialized, we fill it
          # in by querying the MiniWoB app on Android.
          "utterance": {"type": "string"},
      },
      "required": ["task_name"],
  }
  template = (
      "Follow the instructions shown on the top of the screen: {utterance}"
  )
  app_names = (_APP_NAME,)
  complexity = 3

  @property
  def goal(self) -> str:
    if not self.initialized:
      raise ValueError(
          "MiniWoB task must be initialized using initialize_task "
          "before the goal can be retrieved."
      )
    return super().goal

  def _initialize_apps(self, env: interface.AsyncEnv) -> None:
    """Initializes the MiniWoB apps."""

  def initialize_task(self, env: interface.AsyncEnv):
    """Initializes the MiniWoB task.

    Configures the task, i.e. loads HTML file for given task name. Starts the
    episode.

    Args:
      env: AndroidEnv instance.
    """
    super().initialize_task(env)
    task_name = self.params["task_name"]
    task_config = f'{{"task":"{task_name}"}}'
    adb_utils.start_activity(
        _MAIN_ACTIVITY,
        ["--es", "RL_TASK_APP_CONFIG", f"'{task_config}'"],
        env.controller,
    )
    time.sleep(1)
    # Reset and start the task.
    adb_utils.start_activity(
        _MAIN_ACTIVITY, ["--ez", "reset", "true"], env.controller
    )
    adb_utils.start_activity(
        _MAIN_ACTIVITY, ["--ez", "step", "true"], env.controller
    )
    self._params["utterance"] = _get_episode_utterance(env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return float(get_episode_reward(env.controller) == 1.0)
