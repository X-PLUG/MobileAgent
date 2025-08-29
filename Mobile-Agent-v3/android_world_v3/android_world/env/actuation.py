# Copyright 2024 The android_world Authors.
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

"""Utilies for actuation."""

import copy
import logging
import time
from typing import Any
from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.agents import new_json_action as json_action
from android_world.env import representation_utils


def execute_adb_action(
    action: json_action.JSONAction,
    screen_elements: list[Any],  # list[UIElement]
    screen_size: tuple[int, int],
    env: env_interface.AndroidEnvInterface,
) -> None:
  """Execute an action based on a JSONAction object.

  Args:
      action: JSONAction object containing the action to be executed.
      screen_elements: List of UI elements on the screen.
      screen_size: The (width, height) of the screen.
      env: The environment to execute the action in.
  """
  # 
  if action.action_type in ['click', 'double_tap', 'long_press']:
    # import pdb; pdb.set_trace()
    idx = action.index
    x = action.x
    y = action.y
    if idx is not None:
      if idx < 0 or idx >= len(screen_elements):
        raise ValueError(
            f'Invalid element index: {idx}, must be between 0 and'
            f' {len(screen_elements)-1}.'
        )
      element = screen_elements[idx]
      if element.bbox_pixels is None:
        raise ValueError('Bbox is not present on element.')
      x, y = element.bbox_pixels.center
      x, y = int(x), int(y)
      if action.action_type == 'click':
        adb_utils.tap_screen(x, y, env)
      elif action.action_type == 'double_tap':
        adb_utils.double_tap(x, y, env)
      else:
        adb_utils.long_press(x, y, env)
    elif x is not None and y is not None:
      x, y = int(x), int(y)
      if action.action_type == 'click':
        adb_utils.tap_screen(x, y, env)
      elif action.action_type == 'double_tap':
        adb_utils.double_tap(x, y, env)
      else:
        adb_utils.long_press(x, y, env)
    else:
      raise ValueError(f'Invalid click action: {action}')

  elif action.action_type == 'input_text':
    # import pdb; pdb.set_trace()
    text = action.text
    if text:
      if action.index is not None or (
          action.x is not None and action.y is not None
      ):
        # First focus on enter text UI element.
        # click_action = copy.deepcopy(action)
        # click_action.action_type = 'click'
        # execute_adb_action(click_action, screen_elements, screen_size, env)
        # time.sleep(1.0)
        adb_utils.long_press(action.x, action.y, env) # TODO
        time.sleep(1.0)  
        adb_utils.long_press(1000, 2030, env)

        time.sleep(1.0)
      adb_utils.type_text(text, env, timeout_sec=10)
      adb_utils.press_enter_button(env)
    else:
      logging.warning(
          'Input_text action indicated, but no text provided. No '
          'action will be executed.'
      )

  elif action.action_type == 'keyboard_enter':
    adb_utils.press_enter_button(env)

  elif action.action_type == 'navigate_home':
    adb_utils.press_home_button(env)

  elif action.action_type == 'navigate_back':
    adb_utils.press_back_button(env)

  elif action.action_type == 'press_keyboard':
    adb_utils.press_keyboard_generic(action.keycode, env)
  elif action.action_type == 'drag_and_drop':
    if action.touch_xy is not None and action.lift_xy is not None:
      command = adb_utils.generate_drag_and_drop_command(
          action.touch_xy[0],
          action.touch_xy[1],
          action.lift_xy[0],
          action.lift_xy[1],
          4000,
      )
      adb_utils.issue_generic_request(command, env)
    else:
      logging.warning(
          'Drag and drop action indicated, but no coordinates provided. No '
          'action will be executed.'
      )
  elif action.action_type == 'scroll':

    screen_width, screen_height = screen_size
    if action.index:
      x_min, y_min, x_max, y_max = (
          max(screen_elements[action.index].bbox_pixels.x_min, 0),
          max(screen_elements[action.index].bbox_pixels.y_min, 0),
          min(screen_elements[action.index].bbox_pixels.x_max, screen_width),
          min(screen_elements[action.index].bbox_pixels.y_max, screen_height),
      )
    else:
      x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)

    start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    direction = action.direction
    if direction == 'down':
      end_x, end_y = (x_min + x_max) // 2, y_min
    elif direction == 'up':
      end_x, end_y = (x_min + x_max) // 2, y_max
    elif direction == 'right':
      end_x, end_y = x_min, (y_min + y_max) // 2
    elif direction == 'left':
      end_x, end_y = x_max, (y_min + y_max) // 2
    else:
      print('Invalid direction')
      return
    command = adb_utils.generate_swipe_command(
        int(start_x), int(start_y), int(end_x), int(end_y)
    )
    adb_utils.issue_generic_request(command, env)

  elif action.action_type == 'swipe':  # Inverse of scroll.
    start_x, start_y, end_x, end_y = action.direction
    command = adb_utils.generate_swipe_command(
        int(start_x), int(start_y), int(end_x), int(end_y), 500
    )
    adb_utils.issue_generic_request(command, env)

  elif action.action_type == 'open_app':
    app_name = action.app_name
    if app_name:
      adb_utils.launch_app(app_name, env)
    else:
      raise ValueError('No app name provided')

  elif action.action_type == 'wait':
    time.sleep(1.0)

  elif action.action_type == 'launch_adb_activity':
    if action.activity_nickname == 'app_drawer':
      adb_utils.press_home_button(env)
      time.sleep(1.0)
      start_x, start_y = int(screen_size[0] / 2), int(screen_size[1] * 0.9)
      end_x = start_x
      end_y = int(0.3 * screen_size[1])
      request = adb_utils.generate_swipe_command(start_x, start_y, end_x, end_y)
      adb_utils.issue_generic_request(request, env)
    elif action.activity_nickname == 'quick_settings':
      start_x, start_y = int(screen_size[0] / 2), 30
      end_x = start_x
      end_y = int(0.3 * screen_size[1])
      request = adb_utils.generate_swipe_command(
          start_x, start_y, end_x, end_y, duration_ms=10
      )
      adb_utils.issue_generic_request(request, env)
  elif action.action_type == 'change_orientation':
    adb_utils.change_orientation(action.orientation, env)
  elif action.action_type == json_action.UNKNOWN:
    print('Unknown action type; no action will be executed. Try again...')
  else:
    print('Invalid action type')
    
  time.sleep(2)


def find_and_click_element(
    element_text: str,
    env: android_world_controller.AndroidWorldController,
    case_sensitive: bool = False,
):
  """Identifies element with element_text and clicks it.

  Args:
    element_text: Text of the UI element to click on.
    env: The Android env instance.
    case_sensitive: Whether to use case sensitivity when determining which UI
      element to tap.
  """
  # Find text.
  action = _wait_and_find_click_element(element_text, env, case_sensitive)

  ui_elements = env.get_ui_elements()
  screen_size = (0, 0)  # Unused, but required.
  execute_adb_action(action, ui_elements, screen_size, env)


def _wait_and_find_click_element(
    target_text: str,
    env: android_world_controller.AndroidWorldController,
    case_sensitive: bool,
    dist_threshold: int = 1,  # Allow one character difference.
) -> json_action.JSONAction:
  """Wait for the screen to update until "element_text" appears."""
  ui_elements = env.get_ui_elements()
  element, distance = _find_target_element(
      ui_elements, target_text, case_sensitive
  )
  start = time.time()
  current = time.time()
  while current - start < 10:
    if distance <= dist_threshold:
      return json_action.JSONAction(action_type='click', index=element)
    ui_elements = env.get_ui_elements()
    element, distance = _find_target_element(
        ui_elements, target_text, case_sensitive
    )
    current = time.time()
  raise ValueError(f'Target text "{target_text}" not found.')


def _find_target_element(
    ui_elements: list[representation_utils.UIElement],
    target_text: str,
    case_sensitive: bool,
) -> tuple[int, int]:
  """Determine the UI element with the closest match to target_text, by looking at the `text` and `content_description` of each UI element."""
  best_match_index = -1
  lowest_distance = int(1e9)

  for i, element in enumerate(ui_elements):
    for attr in [element.text, element.content_description]:
      if attr is not None:
        if case_sensitive:
          distance = _levenshtein_distance(target_text, attr)
        else:
          distance = _levenshtein_distance(target_text.lower(), attr.lower())
        if distance < lowest_distance:
          lowest_distance = distance
          best_match_index = i

  return (best_match_index, lowest_distance)


def _levenshtein_distance(s1: str, s2: str) -> int:
  """Compute the Levenshtein distance between two strings."""
  if len(s1) < len(s2):
    s1, s2 = s2, s1

  if not s2:
    return len(s1)

  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1
      deletions = current_row[j] + 1
      substitutions = previous_row[j] + (c1 != c2)
      current_row.append(min(insertions, deletions, substitutions))
    previous_row = current_row

  return previous_row[-1]
