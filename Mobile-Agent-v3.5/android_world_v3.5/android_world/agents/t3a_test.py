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
from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import t3a
from android_world.utils import test_utils


class MockLlmWrapper(infer.LlmWrapper):
  """Mock LLM wrapper for testing."""

  def __init__(self, mock_responses: list[tuple[str, Any]]):
    self.mock_responses = mock_responses
    self.index = 0

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Any]:
    if self.index < len(self.mock_responses):
      index = self.index
      self.index += 1
      return self.mock_responses[index][0], None, self.mock_responses[index][1]
    else:
      return infer.ERROR_CALLING_LLM, None, None


class T3AInteractionTest(absltest.TestCase):

  def test_step_method_with_completion(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent = t3a.T3A(env, mock_llm)

    goal = "do something"
    step_data = agent.step(goal)

    self.assertTrue(step_data.done)

  def test_history_recording(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([
        (
            (
                "Reason: completed.\nAction: {'action_type': 'answer',"
                " 'text': 'mock_response'}"
            ),
            "fake_response_1",
        ),
        (
            "fake_summary",
            "fake_response_1",
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            "fake_response_2",
        ),
    ])
    agent = t3a.T3A(env, mock_llm)

    goal = "do something"
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)


if __name__ == "__main__":
  absltest.main()
