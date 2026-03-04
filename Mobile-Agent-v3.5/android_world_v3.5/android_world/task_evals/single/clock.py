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

"""Tasks for the clock app."""

import random
from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals import task_eval


def _is_stopwatch_running(
    ui_elements: list[representation_utils.UIElement],
    current_activity: str,
) -> bool:
  """Checks if current screen is stopwatch running."""
  if "DeskClock" not in current_activity:
    return False
  pause_present = False
  lap_present = False

  for element in ui_elements:
    if element.content_description == "Pause":
      pause_present = True
    elif element.content_description == "Lap":
      lap_present = True
  return pause_present and lap_present


def _is_stopwatch_paused(
    ui_elements: list[representation_utils.UIElement],
    current_activity: str,
) -> bool:
  """Checks if current screen is stopwatch paused."""
  if "DeskClock" not in current_activity:
    return False
  start_present = False
  n_stopwatch = 0

  for element in ui_elements:
    if element.content_description == "Start":
      start_present = True
    elif (
        element.content_description == "Stopwatch"
        or element.text == "Stopwatch"
    ):
      n_stopwatch += 1
  logging.info("Start present: %s", start_present)
  logging.info("Stopwatch: %d", n_stopwatch)
  return start_present and n_stopwatch >= 2


def _is_timer_set(
    ui_elements: list[representation_utils.UIElement],
    current_activity: str,
    *,
    hours: int,
    minutes: int,
    seconds: int,
) -> bool:
  """Determines if a timer is set.

  Args:
    ui_elements: A list of UI elements representing the interface components
      within the "DeskClock" activity.
    current_activity: The name of the current activity within the UI.
    hours: The number of hours to check for in the timer.
    minutes: The number of minutes to check for in the timer.
    seconds: The number of seconds to check for in the timer.

  Returns:
    True if the timer is set with the specified hours, minutes, and seconds
      within the "DeskClock" activity; False otherwise.
  """
  if "DeskClock" not in current_activity:
    return False
  text_format = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
  content_desc_format = f"{hours} hours, {minutes} minutes, {seconds} seconds"

  for element in ui_elements:
    if (
        element.text == text_format
        or element.content_description == content_desc_format
    ):
      return True

  return False


def _close_clock_app(env: interface.AsyncEnv):
  """Closes the clock app."""
  adb_utils.clear_app_data(
      adb_utils.extract_package_name(adb_utils.get_adb_activity("clock")),
      env.controller,
  )


class _ClockEval(task_eval.TaskEval):
  """Base class for clock tasks."""

  app_names = ("clock",)

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _close_clock_app(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _close_clock_app(env)


class ClockTimerEntry(_ClockEval):
  """Task for checking if timer is set (but not started)."""

  complexity = 1
  schema = {
      "type": "object",
      "properties": {
          "hours": {"type": "integer"},
          "minutes": {"type": "integer"},
          "seconds": {"type": "integer"},
      },
      "required": ["hours", "minutes", "seconds"],
  }
  template = (
      "Create a timer with {hours} hours, {minutes} minutes, and {seconds}"
      " seconds. Do not start the timer."
  )

  def is_successful(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    current_activity = adb_utils.get_current_activity(env.controller)[0]
    return (
        1.0
        if _is_timer_set(
            ui_elements=ui_elements,
            current_activity=current_activity,
            hours=self._params["hours"],
            minutes=self._params["minutes"],
            seconds=self._params["seconds"],
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)

    params = {
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
    }

    return params


class ClockStopWatchPausedVerify(_ClockEval):
  """Task for checking if stop watch is paused.

  Precondition: The stopwatch is already paused at 00:00:00.

  There is not programmatic way to control the stopwatch. However, the app can
  be forced cleared, effectively resetting and pausing the watch. Hence, there
  is only a "Verify" version of this task.
  """

  complexity = 1
  schema = {
      "type": "object",
      "properties": {},
  }
  template = "Pause the stopwatch."

  def is_successful(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    current_activity = adb_utils.get_current_activity(env.controller)[0]
    return (
        1.0
        if _is_stopwatch_paused(
            ui_elements=ui_elements,
            current_activity=current_activity,
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class ClockStopWatchRunning(_ClockEval):
  """Task for checking if stop watch is paused.

  Precondition: The stopwatch is already paused at 00:00:00.

  There is no programmatic way to control the stopwatch. However, the app can be
  forced cleared, effectively resetting and pausing the watch.
  """

  complexity = 1
  schema = {
      "type": "object",
      "properties": {},
  }
  template = "Run the stopwatch."

  def is_successful(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    super().is_successful(env)
    ui_elements = env.get_state().ui_elements
    current_activity = adb_utils.get_current_activity(env.controller)[0]
    return (
        1.0
        if _is_stopwatch_running(
            ui_elements=ui_elements,
            current_activity=current_activity,
        )
        else 0.0
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
