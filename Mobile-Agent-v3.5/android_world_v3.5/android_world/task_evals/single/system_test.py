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
from android_world.task_evals.single import system
from android_world.utils import app_snapshot
from android_world.utils import fake_adb_responses
from android_world.utils import test_utils


class SystemWifiTurnOnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, 'restore_snapshot')
    )

  def test_is_successful_returns_1_if_wifi_enabled(self):
    eval_task = system.SystemWifiTurnOn(params={'on_or_off': 'on'})
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.return_value = (
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=True)
    )

    self.assertEqual(eval_task.is_successful(env), 1.0)

  def test_is_successful_returns_0_if_wifi_disabled(self):
    eval_task = system.SystemWifiTurnOn(params={'on_or_off': 'on'})
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.return_value = (
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=False)
    )

    self.assertEqual(eval_task.is_successful(env), 0.0)


class SystemWifiTurnOffTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, 'restore_snapshot')
    )

  def test_is_successful_returns_1_if_wifi_disabled(self):
    eval_task = system.SystemWifiTurnOn(params={'on_or_off': 'off'})
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.return_value = (
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=False)
    )

    self.assertEqual(eval_task.is_successful(env), 1.0)

  def test_is_successful_returns_0_if_wifi_enabled(self):
    eval_task = system.SystemWifiTurnOn(params={'on_or_off': 'off'})
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.return_value = (
        fake_adb_responses.create_get_wifi_enabled_response(is_enabled=True)
    )

    self.assertEqual(eval_task.is_successful(env), 0.0)


class TestSystemCopyToClipboard(test_utils.AdbEvalTestBase):

  def test_is_successful(self):
    # Setup
    self.mock_get_clipboard_contents.return_value = (
        '1234 Elm St, Springfield, IL'
    )

    env = mock.MagicMock()
    params = {'clipboard_content': '1234 Elm St, Springfield, IL'}

    # Instantiate task and check success
    task = system.SystemCopyToClipboard(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.assertEqual(self.mock_set_clipboard_contents.call_count, 2)
    self.assertEqual(self.mock_get_clipboard_contents.call_count, 1)

  def test_is_successful_fuzzy_match(self):
    # Setup
    self.mock_get_clipboard_contents.return_value = (
        '1234 Elm Street, Springfield, IL'
    )

    env = mock.MagicMock()
    params = {'clipboard_content': '1234 Elm St, Springfield, IL'}

    # Instantiate task and check success
    task = system.SystemCopyToClipboard(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.assertEqual(self.mock_set_clipboard_contents.call_count, 2)
    self.assertEqual(self.mock_get_clipboard_contents.call_count, 1)

  def test_is_not_successful(self):
    # Setup
    self.mock_get_clipboard_contents.return_value = (
        '5678 Oak St, Springfield, IL'
    )

    env = mock.MagicMock()
    params = {'clipboard_content': '1234 Elm St, Springfield, IL'}

    # Instantiate task and check failure
    task = system.SystemCopyToClipboard(params)
    task.initialize_task(env)
    self.assertEqual(task.is_successful(env), 0)
    task.tear_down(env)
    self.assertEqual(self.mock_set_clipboard_contents.call_count, 2)
    self.assertEqual(self.mock_get_clipboard_contents.call_count, 1)

  def test_initialized_called_twice(self):
    # Setup
    self.mock_get_clipboard_contents.return_value = (
        '5678 Oak St, Springfield, IL'
    )

    env = mock.MagicMock()
    params = {'clipboard_content': '1234 Elm St, Springfield, IL'}

    # Instantiate task and check failure
    task = system.SystemCopyToClipboard(params)
    task.initialize_task(env)
    with self.assertRaisesRegex(RuntimeError, 'already called.'):
      task.initialize_task(env)


class SystemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, 'restore_snapshot')
    )

  def test_parse_component_name_normalizes_components(self):
    absolute_component_name = system.parse_component_name(
        'com.android.settings/com.android.settings.Settings'
    )
    relative_component_name = system.parse_component_name(
        'com.android.settings/.Settings'
    )
    self.assertEqual(absolute_component_name, relative_component_name)

  def test_parse_component_name_parses_package_name(self):
    component_name = system.parse_component_name(
        'com.android.settings/com.android.settings.Settings'
    )

    self.assertEqual(component_name.package_name, 'com.android.settings')

  def test_parse_component_name_parses_class_name(self):
    component_name = system.parse_component_name(
        'com.android.settings/com.android.settings.Settings'
    )

    self.assertEqual(component_name.class_name, '.Settings')

  def test_generate_random_params(self):
    params = system.OpenAppTaskEval.generate_random_params()

    self.assertIn(params['app_name'], system._APP_NAME_TO_PACKAGE_NAME.keys())

  def test_is_successful_returns_0_for_bad_package(self):
    eval_task = system.OpenAppTaskEval({'app_name': 'settings'})
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.return_value = (
        fake_adb_responses.create_get_activity_response(
            'com.google.gmail/com.google.gmail.Inbox'
        )
    )

    score = eval_task.is_successful(env)

    self.assertEqual(score, 0.0)

  def test_is_successful_returns_1_for_good_package(self):
    eval_task = system.OpenAppTaskEval({'app_name': 'settings'})
    env = mock.create_autospec(interface.AsyncEnv)
    eval_task.initialize_task(env)
    env.controller.execute_adb_call.return_value = (
        fake_adb_responses.create_get_activity_response(
            'com.android.settings/com.android.settings.Settings'
        )
    )

    score = eval_task.is_successful(env)

    self.assertEqual(score, 1.0)


if __name__ == '__main__':
  absltest.main()
