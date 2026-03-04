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

"""Environment interface for real-time interaction Android."""

import abc
import dataclasses
import time
from typing import Any, Optional, Self

from absl import logging
from android_env.components import action_type
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import json_action
from android_world.env import representation_utils
import dm_env
import numpy as np


def _get_no_op_action() -> dict[str, Any]:
  """Creates a no-op action; used to retrieve screen & UI tree."""
  return {
      'action_type': np.array(action_type.ActionType.LIFT, dtype=np.int32),
      'touch_position': np.array((0.0, 0.0)),
  }


@dataclasses.dataclass(frozen=True)
class State:
  """State of the Android environment.

  Attributes:
    pixels: RGB array of current screen.
    forest: Raw UI forest; see android_world_controller.py for more info.
    ui_elements: Processed children and stateful UI elements extracted from
      forest.
    auxiliaries: Additional information about the state.
  """

  pixels: np.ndarray
  forest: Any
  ui_elements: list[representation_utils.UIElement]
  auxiliaries: dict[str, Any] | None = None

  @classmethod
  def create_and_infer_elements(
      cls,
      pixels: np.ndarray,
      forest: Any,
      screen_size: Optional[tuple[int, int]] = None,
  ) -> Self:
    """Creates a new instance, inferring UI elements from the forest."""

    elements = representation_utils.forest_to_ui_elements(
        forest, screen_size=screen_size
    )
    return cls(pixels, forest, elements)


class AsyncEnv(abc.ABC):
  """Interface for interacting with a real-time Android device.

  Computing environments, such as Android, run in real-time, independently of
  the agent interacting with it. All observations and actions are asynchronous
  and OS does not pause when providing observations or when accepting actions.
  Changes from action execution may take some time to appear.
  """

  @property
  @abc.abstractmethod
  def controller(self) -> android_world_controller.AndroidWorldController:
    """Returns the controller for the environment."""

  @abc.abstractmethod
  def reset(self, go_home: bool = False) -> State:
    """Go home on reset.

    Args:
      go_home: Whether to go home during the reset.
    """

  @abc.abstractmethod
  def get_state(self, wait_to_stabilize: bool = False) -> State:
    """Gets the state of the environment; i.e., screenshot & UI tree.

    In practice this will usually be called after executing an action. Logic
    should be implemented, perhaps a simple time.sleep, to ensure the
    environment updates after the action.

    Args:
      wait_to_stabilize: Whether to wait for the screen to stabilize before
        returning state.

    Returns:
      Observation containing RGB array of screen, the accessibility forest,
        and UI elements derived from the forest. See android_world_controller.py
        for
        more detail.
    """

  def display_message(self, message: str, header: str = '') -> None:
    """Displays a message on the screen."""

  @abc.abstractmethod
  def ask_question(
      self, question: str, timeout_seconds: float = -1.0
  ) -> str | None:
    """Asks a question to a hypothetical user in the environment.

    Common uses are to ask a question to clarify the user-provided goal, to ask
    for help when the agent is stuck, or when there is ambiguity in the current
    screen.

    Args:
      question: The question to ask the user.
      timeout_seconds: The timeout in seconds to wait for a response. If
        negative, then wait indefinitely.

    Returns:
      The response from the user or None if the user did not answer within the
      timeout.
    """

  @abc.abstractmethod
  def execute_action(self, action: json_action.JSONAction) -> None:
    """Executes action on the environment."""

  @property
  @abc.abstractmethod
  def foreground_activity_name(self) -> str:
    """Returns the activity name of the app currently opened in foreground."""

  @property
  @abc.abstractmethod
  def device_screen_size(self) -> tuple[int, int]:
    """Returns the screen size of the environment in pixels: (width, height)."""

  @property
  @abc.abstractmethod
  def logical_screen_size(self) -> tuple[int, int]:
    """Retrieves the logical screen size of the Android device.

    While the physical size is a fixed attribute of the display, the logical
    size is flexible and varies based on system settings such as the orientation
    or if the resolution is changed.

    Returns: The (width, height) in pixels, denoting the logical dimensions of
    the screen. Width and height values are aligned with the device's current
    orientation, meaning width is always logical horizontal direction (like in
    the landscape orientation width will be the physical vertical direction).
    """

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the environment."""

  @property
  @abc.abstractmethod
  def interaction_cache(self) -> str:
    """Returns the interaction cache of the environment."""

  @abc.abstractmethod
  def hide_automation_ui(self) -> None:
    """Hides any UI, such as screen coordinates,."""

  @property
  @abc.abstractmethod
  def orientation(self) -> int:
    """Returns the orientation of the environment.

    Returns: 0 for portrait, 1 for landscape, 2 for reverse portrait,
    3 for reverse landscape.
    """

  @property
  @abc.abstractmethod
  def physical_frame_boundary(self) -> tuple[int, int, int, int]:
    """Returns the physical frame boundary of the environment.

    Returns: First two integers are the coordinates for top left corner, last
    two are for lower right corner. All coordinates are given in portrait
    orientation.
    """


def _process_timestep(timestep: dm_env.TimeStep) -> State:
  """Parses timestep observation and returns State."""
  return State(
      pixels=timestep.observation['pixels'],
      forest=timestep.observation[
          android_world_controller.OBSERVATION_KEY_FOREST
      ],
      ui_elements=timestep.observation[
          android_world_controller.OBSERVATION_KEY_UI_ELEMENTS
      ],
      auxiliaries={},
  )


class AsyncAndroidEnv(AsyncEnv):
  """Async environment interface using AndroidEnv to communicate with device."""

  interaction_cache = ''

  def __init__(
      self, controller: android_world_controller.AndroidWorldController
  ):
    self._controller = controller
    self._prior_state = None
    # Variable used to temporarily save interactions between agent and user.
    # Like when agent use answer action to answer user questions, we
    # use this to save the agent response. Or later on when agent has the
    # ability to ask user question, user's answer will be saved here as well.
    self.interaction_cache = ''

  @property
  def controller(self) -> android_world_controller.AndroidWorldController:
    return self._controller

  def reset(self, go_home: bool = False) -> State:
    if go_home:
      adb_utils.press_home_button(self.controller)
    self.interaction_cache = ''

    return _process_timestep(self.controller.reset())

  def _get_state(self):
    return _process_timestep(self.controller.step(_get_no_op_action()))

  def _get_stable_state(
      self,
      stability_threshold: int = 3,
      sleep_duration: float = 0.5,
      timeout: float = 6.0,
  ) -> State:
    """Checks if the UI elements remain stable over a number of checks and returns the state.

    Args:
        stability_threshold: Number of consecutive checks where UI elements must
          remain the same to consider UI stable.
        sleep_duration: Minimum time in seconds between each check.
        timeout: Maximum time in seconds to wait for UI to become stable before
          giving up.

    Returns:
        The current state of the UI if stability is achieved within the timeout.
    """
    if not self._prior_state:
      self._prior_state = self._get_state()
    if stability_threshold <= 0:
      raise ValueError('Stability threshold must be a positive integer.')

    stable_checks = 1
    start_time = time.time()
    deadline = start_time + timeout

    while stable_checks < stability_threshold and time.time() < deadline:
      iteration_start_time = time.time()
      current_state = self._get_state()

      if self._prior_state.ui_elements == current_state.ui_elements:
        stable_checks += 1
        if stable_checks == stability_threshold:
          break  # Exit early if stability is achieved.
      else:
        stable_checks = 1  # Reset if any change is detected
        self._prior_state = current_state

      elapsed_time = time.time() - iteration_start_time
      remaining_sleep = sleep_duration - elapsed_time
      if remaining_sleep > 0:
        sleep_time = min(remaining_sleep, deadline - time.time())
        if sleep_time > 0:
          time.sleep(sleep_time)
      # If remaining_sleep <= 0, proceed immediately to the next iteration

    return current_state  # pylint: disable=undefined-variable

  def get_state(self, wait_to_stabilize: bool = False) -> State:
    if wait_to_stabilize:
      return self._get_stable_state()
    return self._get_state()

  def execute_action(self, action: json_action.JSONAction) -> None:
    if action.action_type == json_action.ANSWER:
      self.interaction_cache = action.text
      if action.text:
        self.display_message(action.text, header='Agent answered:')
      return
    if action.action_type == json_action.STATUS:
      # Do nothing if it is a termination action.
      return
    state = self.get_state(wait_to_stabilize=False)
    actuation.execute_adb_action(
        action,
        state.ui_elements,
        self.logical_screen_size,
        self.controller,
    )

  def hide_automation_ui(self) -> None:
    """Hides the coordinates on screen."""
    adb_utils.issue_generic_request(
        'shell settings put system pointer_location 0', self.controller
    )

  def display_message(self, message: str, header: str = '') -> None:
    adb_utils.send_android_intent(
        command='broadcast',
        action='com.example.ACTION_UPDATE_OVERLAY',
        env=self.controller,
        extras={'task_type_string': header, 'goal_string': message},
    )

  def ask_question(
      self, question: str, timeout_seconds: float = -1.0
  ) -> str | None:
    raise NotImplementedError('ask_question is not implemented.')

  @property
  def foreground_activity_name(self) -> str:
    activity = adb_utils.get_current_activity(self.controller)[0]
    if activity:
      return activity
    else:
      return ''

  @property
  def device_screen_size(self) -> tuple[int, int]:
    return self.controller.device_screen_size

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return adb_utils.get_logical_screen_size(self.controller)

  def close(self) -> None:
    try:
      self.controller.close()
    except:  # pylint: disable=bare-except
      logging.warning('Failed to close controller. Continuing.')

  @property
  def orientation(self) -> int:
    return adb_utils.get_orientation(self.controller)

  @property
  def physical_frame_boundary(self) -> tuple[int, int, int, int]:
    return adb_utils.get_physical_frame_boundary(self.controller)
