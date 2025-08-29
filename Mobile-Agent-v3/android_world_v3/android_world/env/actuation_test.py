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

import copy
import time
from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import json_action
from android_world.env import representation_utils


@mock.patch.object(time, 'sleep')
@mock.patch.object(actuation, '_find_target_element')
@mock.patch.object(android_world_controller, 'get_a11y_tree')
@mock.patch.object(representation_utils, 'forest_to_ui_elements')
class TestWaitAndFindClickElement(absltest.TestCase):

  def test_element_found_immediately(
      self,
      unused_mock_representation_utils,
      unused_mock_get_a11y_tree,
      mock_create,
      mock_sleep,
  ):
    """Test when the element is found immediately."""
    mock_create.return_value = (0, 0)
    mock_sleep.side_effect = [0, 1]
    action = actuation._wait_and_find_click_element(
        'target', mock.MagicMock(), case_sensitive=True
    )
    self.assertEqual(
        action, json_action.JSONAction(action_type='click', index=0)
    )

  def test_element_not_found_within_timeout(
      self,
      unused_mock_representation_utils,
      unused_mock_get_a11y_tree,
      mock_create,
      mock_sleep,
  ):
    """Test when the element is not found within the timeout period."""
    mock_create.return_value = (-1, float('inf'))
    mock_sleep.side_effect = (
        0,
        11,
    )  # Simulating 11 seconds have passed
    with self.assertRaises(ValueError):
      actuation._wait_and_find_click_element(
          'target', mock.MagicMock(), case_sensitive=True
      )


class TestCreateReferredClickAction(absltest.TestCase):

  def test_empty_ui_elements(self):
    """Test with no UI elements."""
    self.assertEqual(
        actuation._find_target_element([], 'target', case_sensitive=True),
        (-1, int(1e9)),
    )

  def test_single_exact_match(self):
    """Test with one UI element that is an exact match."""
    ui_elements = [
        representation_utils.UIElement(text='target', content_description='')
    ]
    self.assertEqual(
        actuation._find_target_element(
            ui_elements, 'target', case_sensitive=True
        ),
        (0, 0),
    )

  def test_multiple_elements_with_closest_match(self):
    """Test with multiple elements where one is the closest match."""
    ui_elements = [
        representation_utils.UIElement(text='targ', content_description=''),
        representation_utils.UIElement(text='', content_description='targetX'),
        representation_utils.UIElement(text='target', content_description=''),
    ]
    self.assertEqual(
        actuation._find_target_element(
            ui_elements, 'target', case_sensitive=True
        ),
        (2, 0),
    )

  def test_no_exact_match(self):
    """Test with no exact matching elements."""
    ui_elements = [
        representation_utils.UIElement(text='no match', content_description='')
    ]
    _, distance = actuation._find_target_element(
        ui_elements, 'target', case_sensitive=True
    )
    self.assertGreater(distance, 0)


class ExecuteAdbActionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(spec=env_interface.AndroidEnvInterface)
    self.screen_elements = [
        representation_utils.UIElement(
            bbox_pixels=representation_utils.BoundingBox(
                x_min=0, x_max=50, y_min=0, y_max=60
            )
        )
    ]
    self.screen_size = (100, 100)

  def test_click_by_index(self):
    action = json_action.JSONAction(action_type='click', index=0)
    with mock.patch.object(adb_utils, 'tap_screen') as mock_tap_screen:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_tap_screen.assert_called_once_with(25, 30, self.mock_env)

  def test_click_by_coordinates(self):
    action = json_action.JSONAction(action_type='click', x=50, y=50)
    with mock.patch.object(adb_utils, 'tap_screen') as mock_tap_screen:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_tap_screen.assert_called_once_with(50, 50, self.mock_env)

  def test_click_by_coordinate_floats(self):
    action = json_action.JSONAction(action_type='click', x=50.2, y=50.3)
    with mock.patch.object(adb_utils, 'tap_screen') as mock_tap_screen:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_tap_screen.assert_called_once_with(50, 50, self.mock_env)

  def test_input_text(self):
    action = json_action.JSONAction(
        action_type='input_text', text='test input', x=50, y=50
    )
    click_action = copy.deepcopy(action)
    click_action.action_type = 'click'
    with (
        mock.patch.object(adb_utils, 'tap_screen') as mock_tap_screen,
        mock.patch.object(adb_utils, 'type_text') as mock_type_text,
        mock.patch.object(
            adb_utils, 'press_enter_button'
        ) as mock_press_enter_button,
        mock.patch.object(
            adb_utils, 'issue_generic_request'
        ) as mock_issue_generic_request,
    ):
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_tap_screen.assert_called_once_with(50, 50, self.mock_env)
      mock_type_text.assert_called_once_with(
          'test input', self.mock_env, timeout_sec=10
      )
      mock_press_enter_button.assert_called_once_with(self.mock_env)
      mock_issue_generic_request.assert_not_called()

  def test_input_text_with_clear_text(self):
    action = json_action.JSONAction(
        action_type='input_text', text='test input', x=50, y=50, clear_text=True
    )
    with (
        mock.patch.object(
            adb_utils, 'issue_generic_request'
        ) as mock_issue_generic_request,
    ):
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_issue_generic_request.assert_called_once_with(
          [
              'shell',
              'input',
              'keycombination',
              '113',
              '29',
              '&&',
              'input',
              'keyevent',
              '67',
          ],
          self.mock_env,
      )

  def test_scroll(self):
    action = json_action.JSONAction(action_type='scroll', direction='down')
    with (
        mock.patch.object(
            adb_utils, 'generate_swipe_command'
        ) as mock_generate_swipe_command,
        mock.patch.object(
            adb_utils, 'issue_generic_request'
        ) as mock_issue_generic_request,
    ):
      mock_generate_swipe_command.return_value = 'command'
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_generate_swipe_command.assert_called_once_with(50, 50, 50, 0)
      mock_issue_generic_request.assert_called_once_with(
          'command', self.mock_env
      )

  def test_swipe(self):
    action = json_action.JSONAction(action_type='swipe', direction='up')
    with (
        mock.patch.object(
            adb_utils, 'generate_swipe_command'
        ) as mock_generate_swipe_command,
        mock.patch.object(
            adb_utils, 'issue_generic_request'
        ) as mock_issue_generic_request,
    ):
      mock_generate_swipe_command.return_value = 'command'
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_generate_swipe_command.assert_called_once_with(50, 100, 50, 0, 500)
      mock_issue_generic_request.assert_called_once_with(
          'command', self.mock_env
      )

  def test_open_app(self):
    action = json_action.JSONAction(action_type='open_app', app_name='test app')
    with mock.patch.object(adb_utils, 'launch_app') as mock_launch_app:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_launch_app.assert_called_once_with('test app', self.mock_env)

  def test_double_tap(self):
    action = json_action.JSONAction(action_type='double_tap', index=0)
    with mock.patch.object(adb_utils, 'double_tap') as mock_double_tap:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_double_tap.assert_called_once_with(25, 30, self.mock_env)

  def test_long_press(self):
    action = json_action.JSONAction(action_type='long_press', index=0)
    with mock.patch.object(adb_utils, 'long_press') as mock_long_press:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_long_press.assert_called_once_with(25, 30, self.mock_env)

  def test_keyboard_enter(self):
    action = json_action.JSONAction(action_type='keyboard_enter')
    with mock.patch.object(
        adb_utils, 'press_enter_button'
    ) as mock_press_enter_button:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_press_enter_button.assert_called_once_with(self.mock_env)

  def test_navigate_home(self):
    action = json_action.JSONAction(action_type='navigate_home')
    with mock.patch.object(
        adb_utils, 'press_home_button'
    ) as mock_press_home_button:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_press_home_button.assert_called_once_with(self.mock_env)

  def test_navigate_back(self):
    action = json_action.JSONAction(action_type='navigate_back')
    with mock.patch.object(
        adb_utils, 'press_back_button'
    ) as mock_press_back_button:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_press_back_button.assert_called_once_with(self.mock_env)

  def test_wait(self):
    action = json_action.JSONAction(action_type='wait')
    with mock.patch.object(time, 'sleep') as mock_sleep:
      actuation.execute_adb_action(
          action, self.screen_elements, self.screen_size, self.mock_env
      )
      mock_sleep.assert_called_once_with(1.0)

  def test_unknown_action(self):
    action = json_action.JSONAction(action_type=json_action.UNKNOWN)
    actuation.execute_adb_action(
        action, self.screen_elements, self.screen_size, self.mock_env
    )


if __name__ == '__main__':
  absltest.main()
