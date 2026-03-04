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
from android_world import constants
from android_world import episode_runner
from android_world.agents import base_agent
from android_world.env import interface


class FakeEnvironmentInteractingAgent(base_agent.EnvironmentInteractingAgent):

  def __init__(
      self,
      env: interface.AsyncAndroidEnv,
      name: str,
      return_done: bool = False,
      return_data: dict[str, Any] | None = None,
  ):
    super().__init__(env, name)
    self.return_done = return_done
    self.return_data = return_data
    self.call_count = 0
    if return_data is None:
      self.return_data = {}

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    self.call_count += 1
    return base_agent.AgentInteractionResult(
        done=self.return_done, data=self.return_data
    )


class EpisodeRunnerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.create_autospec(interface.AsyncAndroidEnv)

  @mock.patch.object(base_agent, 'EnvironmentInteractingAgent')
  def test_max_steps_reached(self, mock_agent_class):
    mock_agent = FakeEnvironmentInteractingAgent(self.env, 'fake_agent')
    mock_agent_class.return_value = mock_agent

    result = episode_runner.run_episode('test_goal', mock_agent, max_n_steps=2)

    self.assertFalse(result.done)
    self.assertLen(result.step_data[constants.STEP_NUMBER], 2)
    self.assertEqual(mock_agent.call_count, 2)

  @mock.patch.object(base_agent, 'EnvironmentInteractingAgent')
  def test_termination_fn_early_termination(self, mock_agent_class):
    mock_agent = FakeEnvironmentInteractingAgent(self.env, 'fake_agent')
    mock_agent_class.return_value = mock_agent

    def termination_fn(env):
      del env
      return True

    result = episode_runner.run_episode(
        'test_goal', mock_agent, termination_fn=termination_fn
    )

    self.assertTrue(result.done)
    self.assertEqual(mock_agent.call_count, 1)
    self.assertLen(result.step_data[constants.STEP_NUMBER], 1)

  @mock.patch.object(base_agent, 'EnvironmentInteractingAgent')
  def test_start_on_home_screen(self, mock_agent_class):
    mock_agent = FakeEnvironmentInteractingAgent(self.env, 'fake_agent')
    mock_agent_class.return_value = mock_agent

    episode_runner.run_episode(
        'test_goal', mock_agent, start_on_home_screen=True
    )

    mock_agent.env.reset.assert_called_with(go_home=True)


if __name__ == '__main__':
  absltest.main()
