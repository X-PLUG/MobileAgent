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

import time
from unittest import mock
from absl.testing import absltest
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.miniwob import miniwob_base


@mock.patch.object(adb_utils, "extract_broadcast_data")
@mock.patch.object(adb_utils, "send_android_intent")
class Testintents(absltest.TestCase):

  def test_get_episode_utterance_valid(self, mock_send_intent, mock_extract):
    mock_env = mock.Mock()
    mock_send_intent.return_value = mock.Mock(
        generic=mock.Mock(output=b"encoded data")
    )
    mock_extract.return_value = "test utterance"

    actual_utterance = miniwob_base._get_episode_utterance(mock_env)
    self.assertEqual(actual_utterance, "test utterance")
    mock_send_intent.assert_called_with(
        "broadcast",
        f"{miniwob_base._APP_NAME}.app.GET_UTTERANCE_ACTION",
        mock_env,
    )
    mock_extract.assert_called_once()

  def test_get_episode_utterance_invalid(self, mock_send_intent, mock_extract):
    mock_env = mock.Mock()
    mock_send_intent.return_value = mock.Mock(
        generic=mock.Mock(output=b"encoded data")
    )
    mock_extract.return_value = None

    with self.assertRaises(ValueError):
      miniwob_base._get_episode_utterance(mock_env)

  def test_get_episode_reward_success(self, mock_send_intent, mock_extract):
    mock_env = mock.Mock()
    mock_send_intent.return_value = mock.Mock(
        generic=mock.Mock(output=b"encoded reward")
    )
    mock_extract.return_value = "1"

    reward = miniwob_base.get_episode_reward(mock_env)
    self.assertEqual(reward, 1)
    mock_send_intent.assert_called_with(
        "broadcast",
        f"{miniwob_base._APP_NAME}.app.GET_REWARD_ACTION",
        mock_env,
    )
    mock_extract.assert_called_once()

  def test_get_episode_reward_not_terminated(
      self, mock_send_intent, mock_extract
  ):
    mock_env = mock.Mock()
    mock_send_intent.return_value = mock.Mock(
        generic=mock.Mock(output=b"encoded reward")
    )
    mock_extract.return_value = ""

    reward = miniwob_base.get_episode_reward(mock_env)
    self.assertEqual(reward, 0)
    mock_send_intent.assert_called_with(
        "broadcast",
        f"{miniwob_base._APP_NAME}.app.GET_REWARD_ACTION",
        mock_env,
    )
    mock_extract.assert_called_once()

  def test_get_episode_reward_failure(self, mock_send_intent, mock_extract):
    mock_env = mock.Mock()
    mock_send_intent.return_value = mock.Mock(
        generic=mock.Mock(output=b"encoded reward")
    )
    mock_extract.return_value = None

    with self.assertRaises(ValueError):
      miniwob_base.get_episode_reward(mock_env)


class TestableMiniWoBTaskForTest(miniwob_base.MiniWoBTask):

  @classmethod
  def generate_random_params(cls):
    return {"task_name": "test_task"}


@mock.patch.object(adb_utils, "start_activity")
@mock.patch.object(time, "sleep")
@mock.patch.object(miniwob_base, "_get_episode_utterance")
class TestMiniWoBTask(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(interface.AsyncEnv)
    self.params = {"task_name": "test_task"}
    self.mock_task = TestableMiniWoBTaskForTest(
        TestableMiniWoBTaskForTest.generate_random_params()
    )

  def test_initialize_task(
      self, mock_get_utterance, unused_mock_sleep, mock_start_activity
  ):
    mock_get_utterance.return_value = "test utterance"

    self.mock_task.initialize_task(self.mock_env)

    expected_calls = [
        mock.call(
            miniwob_base._MAIN_ACTIVITY,
            ["--es", "RL_TASK_APP_CONFIG", '\'{"task":"test_task"}\''],
            self.mock_env.controller,
        ),
        mock.call(
            miniwob_base._MAIN_ACTIVITY,
            ["--ez", "reset", "true"],
            self.mock_env.controller,
        ),
        mock.call(
            miniwob_base._MAIN_ACTIVITY,
            ["--ez", "step", "true"],
            self.mock_env.controller,
        ),
    ]
    mock_start_activity.assert_has_calls(expected_calls, any_order=False)
    self.assertIn("utterance", self.mock_task.params)
    self.assertEqual(self.mock_task.params["utterance"], "test utterance")

  @mock.patch.object(miniwob_base, "get_episode_reward")
  def test_is_successful(
      self,
      mock_get_reward,
      unused_mock_get_utterance,
      unused_mock_sleep,
      unused_mock_start_activity,
  ):
    self.mock_task.initialize_task(self.mock_env)
    mock_get_reward.return_value = 1
    self.assertEqual(self.mock_task.is_successful(self.mock_env), 1)

    mock_get_reward.return_value = False
    self.assertEqual(self.mock_task.is_successful(self.mock_env), 0)


if __name__ == "__main__":
  absltest.main()
