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
from android_env.components import errors
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import tools
from android_world.env.setup_device import apps
from android_world.env.setup_device import setup
from android_world.utils import app_snapshot


class GetAppListToSetupTest(absltest.TestCase):

  def test_get_app_list_to_setup_none(self):
    self.assertIsNone(setup.get_app_list_to_setup(None))

  def test_get_app_list_to_setup_with_valid_ids(self):
    task_ids = ["ClockCreateTimer", "ContactsSearchContact"]
    expected_apps = (apps.ClockApp, apps.ContactsApp)
    self.assertCountEqual(setup.get_app_list_to_setup(task_ids), expected_apps)

  def test_get_app_list_to_setup_with_mixed_ids(self):
    task_ids = ["ClockCreateTimer", "InvalidTask", "DialerCallNumber"]
    expected_apps = (apps.ClockApp, apps.DialerApp)
    self.assertCountEqual(setup.get_app_list_to_setup(task_ids), expected_apps)

  def test_get_app_list_to_setup_with_space_in_app_name(self):
    task_ids = ["AudioRecorderRecordAudio"]
    expected_apps = (apps.AudioRecorder,)
    self.assertCountEqual(setup.get_app_list_to_setup(task_ids), expected_apps)

  def test_get_app_list_to_setup_with_pascal_case_conversion(self):
    task_ids = ["SimpleCalendarProCreateEvent"]
    expected_apps = (apps.SimpleCalendarProApp,)
    self.assertCountEqual(setup.get_app_list_to_setup(task_ids), expected_apps)


class SetupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_issue_generic_request = self.enter_context(
        mock.patch.object(adb_utils, "issue_generic_request")
    )

  @mock.patch.object(tools, "AndroidToolController")
  @mock.patch.object(setup, "download_and_install_apk")
  @mock.patch.object(app_snapshot, "save_snapshot")
  def test_setup_apps(self, mock_save_snapshot, mock_install_apk, unused_tools):
    env = mock.create_autospec(interface.AsyncEnv)
    mock_app_setups = {
        app_class: mock.patch.object(app_class, "setup").start()
        for app_class in setup._APPS
    }

    setup.setup_apps(env)

    for app_class in setup._APPS:
      if app_class.apk_names:  # 1P apps do not have APKs.
        mock_install_apk.assert_any_call(
            app_class.apk_names[0], env.controller.env
        )
      mock_app_setups[app_class].assert_any_call(env)
      mock_save_snapshot.assert_any_call(app_class.app_name, env.controller)


class _App(apps.AppSetup):

  def __init__(self, apk_names, app_name):
    self.apk_names = apk_names
    self.app_name = app_name


class InstallApksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.create_autospec(interface.AsyncEnv)
    self.mockdownload_and_install_apk = self.enter_context(
        mock.patch.object(setup, "download_and_install_apk")
    )
    self.apps = [
        _App(apk_names=["apk1", "apk2"], app_name="App1"),
        _App(apk_names=[], app_name="App2"),  # No APKs
        _App(apk_names=["apk3"], app_name="App3"),
    ]
    setup._APPS = self.apps

  def test_install_all_apks_success(self):
    self.mockdownload_and_install_apk.return_value = None

    for app in self.apps:
      setup.maybe_install_app(app, self.env)

    expected_calls = [
        mock.call("apk1", self.env.controller.env),
        mock.call("apk3", self.env.controller.env),
    ]
    self.mockdownload_and_install_apk.assert_has_calls(
        expected_calls, any_order=True
    )

  def test_install_all_apks_success_with_fallback(self):
    def side_effect(apk_name, env):
      del env
      if apk_name == "apk1":
        raise errors.AdbControllerError
      return None

    self.mockdownload_and_install_apk.side_effect = side_effect

    for app in self.apps:
      setup.maybe_install_app(app, self.env)

    expected_calls = [
        mock.call("apk1", self.env.controller.env),
        mock.call("apk2", self.env.controller.env),
        mock.call("apk3", self.env.controller.env),
    ]
    self.mockdownload_and_install_apk.assert_has_calls(expected_calls)


if __name__ == "__main__":
  absltest.main()
