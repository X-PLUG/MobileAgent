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

from unittest import mock
from absl.testing import absltest
from android_world.agents import random_agent
from android_world.env import actuation
from android_world.utils import test_utils


class TestGenerateRandomAction(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.screen_size = (1080, 1920)

  def test_action_type_is_valid(self):
    action = random_agent._generate_random_action(self.screen_size)
    self.assertIn(
        action.action_type,
        [
            'click',
            'double_tap',
            'scroll',
            'swipe',
            'navigate_home',
            'navigate_back',
            'wait',
            'input_text',
            'keyboard_enter',
        ],
    )

  def test_coordinates_within_bounds_for_click(self):
    for _ in range(100):
      action = random_agent._generate_random_action(self.screen_size)
      if action.action_type in ['click', 'double_tap', 'swipe']:
        self.assertGreaterEqual(action.x, 0)
        self.assertLess(action.x, self.screen_size[0])
        self.assertGreaterEqual(action.y, 0)
        self.assertLess(action.y, self.screen_size[1])

  def test_text_generated_for_input_text(self):
    for _ in range(100):
      action = random_agent._generate_random_action(self.screen_size)
      if action.action_type == 'input_text':
        self.assertIsInstance(action.text, str)
        self.assertLen(action.text, 10)
        self.assertGreaterEqual(action.x, 0)
        self.assertLess(action.x, self.screen_size[0])
        self.assertGreaterEqual(action.y, 0)
        self.assertLess(action.y, self.screen_size[1])

  def test_direction_valid_for_scroll(self):
    for _ in range(100):
      action = random_agent._generate_random_action(self.screen_size)
      if action.action_type == 'scroll':
        self.assertIn(action.direction, ['up', 'down', 'left', 'right'])


class RandomAgentInteractionTest(absltest.TestCase):

  @mock.patch.object(actuation, 'execute_adb_action')
  def test_step_method(self, mock_execute_adb_action):
    env = test_utils.FakeAsyncEnv()
    agent = random_agent.RandomAgent(env, verbose=True)
    mock_execute_adb_action.return_value = None

    goal = 'do something'
    step_data = agent.step(goal)

    self.assertIn('raw_screenshot', step_data.data)
    self.assertIn('ui_elements', step_data.data)


if __name__ == '__main__':
  absltest.main()
