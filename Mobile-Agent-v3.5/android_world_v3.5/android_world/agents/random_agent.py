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

"""Random agent for testing purposes."""

import random
import string
from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


def _generate_random_action(
    screen_size: tuple[int, int]
) -> json_action.JSONAction:
  """Generates a random action with a bias towards 'click' action.

  Args:
    screen_size: A tuple (width, height) representing the screen size.

  Returns:
    A dictionary representing the random action.
  """
  scroll_directions = ['up', 'down', 'left', 'right']
  text_characters = string.ascii_letters + string.digits

  # Define action weights.
  action_weights = {
      json_action.CLICK: 0.5,  # Higher weight for 'click'
      json_action.DOUBLE_TAP: 0.05,
      json_action.SCROLL: 0.05,
      json_action.SWIPE: 0.05,
      json_action.NAVIGATE_HOME: 0.05,
      json_action.NAVIGATE_BACK: 0.05,
      json_action.KEYBOARD_ENTER: 0.05,
      json_action.WAIT: 0.05,
      json_action.INPUT_TEXT: 0.05,
  }

  # Select a random action type.
  action_type = random.choices(
      list(action_weights.keys()), weights=list(action_weights.values()), k=1
  )[0]

  # Generate action details based on action type.
  action_details = {'action_type': action_type}

  if action_type in [
      json_action.CLICK,
      json_action.DOUBLE_TAP,
      json_action.SWIPE,
      json_action.INPUT_TEXT,
  ]:
    action_details['x'] = random.randint(0, screen_size[0] - 1)
    action_details['y'] = random.randint(0, screen_size[1] - 1)
    if action_type == json_action.INPUT_TEXT:
      action_details['text'] = ''.join(
          random.choices(text_characters, k=10)
      )  # Random text of length 10
    elif action_type == json_action.SWIPE:
      action_details['direction'] = random.choice(scroll_directions)
  elif action_type == json_action.SCROLL:
    action_details['direction'] = random.choice(scroll_directions)

  return json_action.JSONAction(**action_details)


class RandomAgent(base_agent.EnvironmentInteractingAgent):
  """A random agent interaction loop for testing purposes."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = 'RandomAgent',
      verbose: bool = False,
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      name: The agent name.
      verbose: True if the grounder should produce verbose updates.
    """
    super().__init__(env, name)
    self._verbose = verbose

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """See base class."""
    state = self.get_post_transition_state()
    action = _generate_random_action(self.env.device_screen_size)
    self.env.execute_action(action)
    if self._verbose:
      print(action)
    step_data = {
        'raw_screenshot': state.pixels,
        'ui_elements': state.ui_elements,
    }
    done = False
    return base_agent.AgentInteractionResult(
        done,
        step_data,
    )
