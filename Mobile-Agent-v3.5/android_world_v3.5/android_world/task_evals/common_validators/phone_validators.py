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

"""Logic to validate a phone call is made to a specific number."""

import random
import re
from typing import Any

from absl import logging
from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals import task_eval


def check_if_dialer_with_phone_number(
    ui_elements: list[representation_utils.UIElement],
    current_activity: str,
    *,
    expected_number: str,
) -> bool:
  """Check if the correct number is dialed based on UI elements.

  Args:
    ui_elements: List of UIElement objects representing the UI.
    current_activity: The current Android activity.
    expected_number: The expected dialed phone number as a string.

  Returns:
    True if the expected number is found, False otherwise.
  """
  if not current_activity.startswith("com.google.android.dialer"):
    return False
  for element in ui_elements:
    if element.text:
      cleaned_element_text = re.sub(r"[^\d]", "", element.text)
      cleaned_expected_number = re.sub(r"[^\d]", "", expected_number)

      if cleaned_element_text == cleaned_expected_number:
        return True
  return False


def clear_phone_state(env: env_interface.AndroidEnvInterface) -> None:
  """Clears phone log and ends any active call."""
  adb_utils.end_call_if_active(env)
  adb_utils.clear_android_emulator_call_log(env)


class MakeCall(task_eval.TaskEval):
  """Task to make a phone call to a specific number.

  The class uses ADB commands to interact with the Android environment,
  specifically the dialer application. It can check the call state and ensures
  that the phone number dialed matches the expected number. NOTE: It also checks
  the correct number is on the call screen. This cannot be done using adb so we
  have to fall back to UI element checks.
  """

  app_names = ("dialer",)
  complexity = 1
  schema = {
      "type": "object",
      "properties": {
          "phone_number": {"type": "string"},
      },
      "required": ["phone_number"],
  }
  template = ""

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)
    self.phone_number = params["phone_number"]

  def _called_correct_number(self, env: interface.AsyncEnv) -> bool:
    ui_elements = env.get_state().ui_elements
    current_activity = adb_utils.get_current_activity(env.controller)[0]
    return check_if_dialer_with_phone_number(
        expected_number=self.phone_number,
        ui_elements=ui_elements,
        current_activity=current_activity,
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Clears call history."""
    super().initialize_task(env)
    clear_phone_state(env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    call_state = adb_utils.get_call_state(env.controller)
    if call_state != "OFFHOOK":
      logging.info("Not dialed. Call state: %s", call_state)
      return 0.0
    if not self._called_correct_number(env):
      logging.info("Dialed a number, but not correct number")
      return 0.0
    return 1.0

  def tear_down(self, env: interface.AsyncEnv):
    """Maybe ends call and clears call history."""
    super().tear_down(env)
    clear_phone_state(env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    random_phone_number = "555" + "".join(random.choices("0123456789", k=7))
    return {"phone_number": random_phone_number}
