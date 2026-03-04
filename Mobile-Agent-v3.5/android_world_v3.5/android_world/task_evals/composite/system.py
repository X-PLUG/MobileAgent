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

"""Composite tasks using Android Operating System actions."""

from typing import Any

from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.single import system
from android_world.task_evals.utils import schema


class TurnOnWifiAndOpenApp(task_eval.TaskEval):
  """Evals the agent opening an app after turning on Wifi."""

  app_names = ("settings",)
  complexity = 2
  schema = schema.create([schema.string("app_name", is_required=True)])

  template = "Turn on Wifi, then open the {app_name} app"

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    self.turn_on_wifi_task = system.SystemWifiTurnOn(params={"on_or_off": "on"})
    self.turn_on_wifi_task.initialize_task(env)
    self.open_app_task = system.OpenAppTaskEval(
        params={
            "app_name": self.params["app_name"],
        }
    )
    self.open_app_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    wifi_score = self.turn_on_wifi_task.is_successful(env)
    open_app_score = self.open_app_task.is_successful(env)
    return (wifi_score + open_app_score) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.open_app_task.tear_down(env)
    self.turn_on_wifi_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return system.OpenAppTaskEval.generate_random_params()


class TurnOffWifiAndTurnOnBluetooth(task_eval.TaskEval):
  """Evals the agent turning off WiFi and enabling bluetooth."""

  app_names = ("settings",)
  complexity = 2

  # No parameters.
  schema = schema.create([])

  template = "Turn off WiFi, then enable bluetooth"

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    self.turn_off_wifi_task = system.SystemWifiTurnOff(
        params={"on_or_off": "off"}
    )
    self.turn_off_wifi_task.initialize_task(env)
    self.turn_on_bluetooth_task = system.SystemBluetoothTurnOn(
        params={"on_or_off": "on"}
    )
    self.turn_on_bluetooth_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    wifi_score = self.turn_off_wifi_task.is_successful(env)
    bluetooth_score = self.turn_on_bluetooth_task.is_successful(env)
    return (wifi_score + bluetooth_score) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.turn_on_bluetooth_task.tear_down(env)
    self.turn_off_wifi_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}
