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

"""Task wrapper for screen variation robustness tests."""

import time
from typing import Any

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval


def generate_screen_variation_wrapper(
    base_task: type[task_eval.TaskEval],
    screen_width: int,
    screen_height: int,
    screen_orientation: str,
    params: dict[str, Any],
    screen_config_name: str,
) -> type[task_eval.TaskEval]:
  """Generate a wrapper for a given task for the screen variation experiment.

  Args:
    base_task: The base task to run the experiment.
    screen_width: The width for the new resolution.
    screen_height: The height for the new resolution.
    screen_orientation: The orientation for the experiment.
    params: The fixed parameter for the experiment.
    screen_config_name: The experiment name suffix.

  Returns:
    A wrapper class for the given task with the given screen config.
  """

  class ScreenVariation(base_task):
    """A wrapper class for screen variation robustness experiments."""

    width = screen_width
    height = screen_height
    orientation = screen_orientation
    config_name = screen_config_name

    def initialize_task(self, env: interface.AsyncEnv):
      super().initialize_task(env)
      # Go back to home screen with a reset.
      env.reset(True)
      adb_utils.set_screen_size(self.width, self.height, env.controller)
      # It has been observed that without this pause, the following orientation
      # change will not work.
      time.sleep(2)
      # Task starts from the home screen and the following orientation change
      # will take effect for the next app opened but expired after closing.
      adb_utils.change_orientation(self.orientation, env.controller)

    @property
    def name(self) -> str:
      return base_task.__name__ + '_' + self.config_name

    @classmethod
    def generate_random_params(cls):
      return params

  return ScreenVariation


SCREEN_MODIFIERS = {
    'NormalPortrait': {
        'width': 1080,
        'height': 2400,
        'orientation': 'portrait',
    },
    'NormalLandscape': {
        'width': 1080,
        'height': 2400,
        'orientation': 'landscape',
    },
    'LowResPortrait': {'width': 720, 'height': 1520, 'orientation': 'portrait'},
    'LowResLandscape': {
        'width': 720,
        'height': 1520,
        'orientation': 'landscape',
    },
    'HighResPortrait': {
        'width': 1600,
        'height': 2560,
        'orientation': 'portrait',
    },
}
