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

from typing import Any
from unittest import mock
from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import m3a
from android_world.env import adb_utils
from android_world.utils import test_utils
import numpy as np


class MockMultimodalLlmWrapper(infer.MultimodalLlmWrapper):
  """Mock multimodal LLM wrapper for testing."""

  def __init__(self, mock_responses: list[tuple[str, Any]]):
    self.mock_responses = mock_responses
    self.index = 0

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Any]:
    if self.index < len(self.mock_responses):
      index = self.index
      self.index += 1
      return self.mock_responses[index][0], None, self.mock_responses[index][1]
    else:
      return infer.ERROR_CALLING_LLM, None, None


class M3AInteractionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_orientation = mock.patch.object(
        adb_utils,
        'get_orientation',
    ).start()
    self.mock_get_physical_frame_boundary = mock.patch.object(
        adb_utils,
        'get_physical_frame_boundary',
    ).start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_step_method_with_completion(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        'test raw response',
    )])
    self.mock_get_orientation.return_value = 0
    self.mock_get_physical_frame_boundary.return_value = [0, 0, 100, 100]
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step_data = agent.step(goal)
    self.assertTrue(step_data.done)

  def test_step_method_with_invalid_action_output(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        'Output in incorrect format.',
        'test raw response',
    )])
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step_data = agent.step(goal)

    self.assertFalse(step_data.done)
    self.assertIn(
        'Output for action selection is not in the correct format',
        step_data.data['summary'],
    )

  def test_history_recording(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([
        (
            (
                "Reason: answer question.\nAction: {'action_type': 'answer',"
                " 'text': 'fake answer.'}"
            ),
            'test raw response',
        ),
        (
            'fake summary',
            'test raw response',
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            'test raw response',
        ),
    ])
    self.mock_get_orientation.side_effect = [0, 0, 0]
    self.mock_get_physical_frame_boundary.side_effect = [
        [0, 0, 100, 100],
        [0, 0, 100, 100],
        [0, 0, 100, 100],
    ]
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)
    self.assertIn('fake summary', step1_data.data['summary'])

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)


if __name__ == '__main__':
  absltest.main()
