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
from android_world.env import interface
from android_world.task_evals.composite import system
from android_world.task_evals.single import system as single_system
from android_world.utils import app_snapshot
from android_world.utils import fake_adb_responses


class TurnOnWifiAndOpenAppTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, "restore_snapshot").start()
    )

  def test_generate_random_params(self):
    params = system.TurnOnWifiAndOpenApp.generate_random_params()

    self.assertIn(
        params["app_name"], single_system._APP_NAME_TO_PACKAGE_NAME.keys()
    )

  def test_is_successful_returns_0_if_wifi_off_and_incorrect_app_is_open(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOnWifiAndOpenApp({"app_name": "clock"})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=False),
        fake_adb_responses.create_get_activity_response(
            "com.google.android.apps.maps/.Inbox"
        ),
    ]

    self.assertEqual(eval_task.is_successful(env), 0)

  def test_is_successful_returns_0_5_if_wifi_is_off(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOnWifiAndOpenApp({"app_name": "settings"})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=False),
        fake_adb_responses.create_get_activity_response(
            "com.android.settings/com.android.settings.Settings"
        ),
    ]

    self.assertEqual(eval_task.is_successful(env), 0.5)

  def test_is_successful_returns_0_5_if_incorrect_app_is_open(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOnWifiAndOpenApp({"app_name": "settings"})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=True),
        fake_adb_responses.create_get_activity_response(
            "com.google.page1/.Page1"
        ),
    ]

    self.assertEqual(eval_task.is_successful(env), 0.5)

  def test_is_successful_returns_1_if_wifi_is_on_and_app_is_open(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOnWifiAndOpenApp({"app_name": "settings"})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=True),
        fake_adb_responses.create_get_activity_response(
            "com.android.settings/com.android.settings.Settings"
        ),
    ]

    self.assertEqual(eval_task.is_successful(env), 1.0)


class TurnOffWifiAndTurnOnBluetoothTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, "restore_snapshot")
    )

  def test_is_successful_returns_0_if_wifi_is_on_and_bluetooth_is_off(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOffWifiAndTurnOnBluetooth({})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=True),
        fake_adb_responses.create_get_bluetooth_enabled_response(
            is_enabled=False
        ),
    ]
    self.assertEqual(eval_task.is_successful(env), 0.0)

  def test_is_successful_returns_0_5_if_wifi_is_off_and_bluetooth_is_on(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOffWifiAndTurnOnBluetooth({})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=True),
        fake_adb_responses.create_get_bluetooth_enabled_response(
            is_enabled=True
        ),
    ]
    self.assertEqual(eval_task.is_successful(env), 0.5)

  def test_is_successful_returns_0_5_if_wifi_is_on_and_bluetooth_is_off(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOffWifiAndTurnOnBluetooth({})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=False),
        fake_adb_responses.create_get_bluetooth_enabled_response(
            is_enabled=False
        ),
    ]
    self.assertEqual(eval_task.is_successful(env), 0.5)

  def test_is_successful_returns_1_if_wifi_is_off_and_bluetooth_is_on(self):
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task = system.TurnOffWifiAndTurnOnBluetooth({})
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.side_effect = [
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=False),
        fake_adb_responses.create_get_bluetooth_enabled_response(
            is_enabled=True
        ),
    ]
    self.assertEqual(eval_task.is_successful(env), 1.0)


if __name__ == "__main__":
  absltest.main()
