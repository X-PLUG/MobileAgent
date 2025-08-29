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

"""Represents an action for Android interaction, parsed from a JSON format."""

import dataclasses
import json
from typing import Optional


_JSON_SEPARATORS = (',', ':')

ANSWER = 'answer'
CLICK = 'click'
DOUBLE_TAP = 'double_tap'
INPUT_TEXT = 'input_text'
KEYBOARD_ENTER = 'keyboard_enter'
LONG_PRESS = 'long_press'
NAVIGATE_BACK = 'navigate_back'
NAVIGATE_HOME = 'navigate_home'
OPEN_APP = 'open_app'
SCROLL = 'scroll'
STATUS = 'status'
SWIPE = 'swipe'
UNKNOWN = 'unknown'
WAIT = 'wait'

_ACTION_TYPES = (
    CLICK,
    DOUBLE_TAP,
    SCROLL,
    SWIPE,
    INPUT_TEXT,
    NAVIGATE_HOME,
    NAVIGATE_BACK,
    KEYBOARD_ENTER,
    OPEN_APP,
    STATUS,
    WAIT,
    LONG_PRESS,
    ANSWER,
    UNKNOWN,
)

_SCROLL_DIRECTIONS = ('left', 'right', 'down', 'up')

# Keys of JSON action.
ACTION_TYPE = 'action_type'
INDEX = 'index'
X = 'x'
Y = 'y'
TEXT = 'text'
DIRECTION = 'direction'
APP_NAME = 'app_name'
GOAL_STATUS = 'goal_status'

ACTION_KEYS = [
    ACTION_TYPE,
    INDEX,
    X,
    Y,
    TEXT,
    DIRECTION,
    APP_NAME,
    GOAL_STATUS,
]


@dataclasses.dataclass()
class JSONAction:
  """Represents a parsed JSON action.

  # Example
  result_json = {'action_type': 'click', 'x': %d, 'y': %d}
  action = JSONAction(**result_json)

  Attributes:
    action_type: The action type.
    index: The index to click, if action is a click. Either an index or a <x, y>
      should be provided. See x, y attributes below.
    x: The x position to click, if the action is a click.
    y: The y position to click, if the action is a click.
    text: The text to type, if action is type.
    direction: The direction to scroll, if action is scroll.
    goal_status: If the status is a 'status' type, indicates the status of the
      goal.
    app_name: The app name to launch, if the action type is 'open_app'.
    keycode: Keycode actions are necessary for an agent to interact with complex
      UI elements (like large textareas) that can't be accessed or controlled by
      simply taping, ensuring precise control over navigation and selection in
      the interface.
    clear_text: Whether to clear the text field before typing.
  """

  action_type: Optional[str] = None
  index: Optional[str | int] = None
  x: Optional[int] = None
  y: Optional[int] = None
  text: Optional[str] = None
  direction: Optional[str] = None
  goal_status: Optional[str] = None
  app_name: Optional[str] = None
  keycode: Optional[str] = None
  clear_text: Optional[bool] = None

  def __post_init__(self):
    if self.action_type not in _ACTION_TYPES:
      raise ValueError(f'Invalid action type: {self.action_type}')
    if self.index is not None:
      self.index = int(self.index)
      if self.x is not None or self.y is not None:
        raise ValueError('Either an index or a <x, y> should be provided.')
    if self.direction and self.direction not in _SCROLL_DIRECTIONS:
      raise ValueError(f'Invalid scroll direction: {self.direction}')
    if self.text is not None and not isinstance(self.text, str):
      self.text = str(self.text)
    if self.keycode is not None and not self.keycode.startswith('KEYCODE_'):
      raise ValueError(f'Invalid keycode: {self.keycode}')

  def __repr__(self) -> str:
    properties = []
    for key, value in self.__dict__.items():
      if value is not None:
        if isinstance(value, float):
          value = f'{value:.3f}'
        properties.append(f'{key}={value!r}')
    return f"JSONAction({', '.join(properties)})"

  def __eq__(self, other):
    if isinstance(other, JSONAction):
      return _compare_actions(self, other)
    return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def json_str(self) -> str:
    non_null = {}
    for key, value in self.__dict__.items():
      if value is not None:
        non_null[key] = value
    return json.dumps(non_null, separators=_JSON_SEPARATORS)


def _compare_actions(a: JSONAction, b: JSONAction) -> bool:
  """Compares two JSONActions.

  Args:
    a: The first action.
    b: The second action.

  Returns:
    If the actions are equal.
  """
  # Ignore cases.
  if a.app_name is not None and b.app_name is not None:
    app_name_match = a.app_name.lower() == b.app_name.lower()
  else:
    app_name_match = a.app_name == b.app_name

  if a.text is not None and b.text is not None:
    text_match = a.text.lower() == b.text.lower()
  else:
    text_match = a.text == b.text

  # Compare the non-metadata fields.
  return (
      app_name_match
      and text_match
      and a.action_type == b.action_type
      and a.index == b.index
      and a.x == b.x
      and a.y == b.y
      and a.keycode == b.keycode
      and a.direction == b.direction
      and a.goal_status == b.goal_status
  )
