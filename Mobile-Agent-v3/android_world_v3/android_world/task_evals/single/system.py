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

"""Tasks for general system tasks like interacting with settings."""

import dataclasses
import random
from typing import Any

from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.utils import fuzzy_match_lib
import immutabledict


class _SystemBrightnessToggle(task_eval.TaskEval):
  """Task for checking that the screen brightness has been set to {max_or_min}."""

  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'max_or_min': {'type': 'string', 'enum': ['max', 'min']}},
      'required': ['max_or_min'],
  }
  template = 'Turn brightness to the {max_or_min} value.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'system', 'screen_brightness'],
        env.controller,
    )
    brightness_level = int(res.generic.output.decode().strip())

    if self.params['max_or_min'] == 'max':
      return 1.0 if brightness_level == 255 else 0.0
    else:
      return 1.0 if brightness_level == 1 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'max_or_min': 'max' if random.choice([True, False]) else 'min'}


class SystemBrightnessMinVerify(_SystemBrightnessToggle):
  """Task for verifying that the screen brightness is already at minimum.

  Precondition: Screen brightness is at minimum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('min', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'min'}


class SystemBrightnessMaxVerify(_SystemBrightnessToggle):
  """Task for verifying that the screen brightness is already at maximum.

  Precondition: Screen brightness is at maximum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('max', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'max'}


class SystemBrightnessMin(_SystemBrightnessToggle):
  """Task for ensuring that the screen brightness is set to minimum.

  Precondition: Screen brightness is not at minimum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('max', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'min'}


class SystemBrightnessMax(_SystemBrightnessToggle):
  """Task for ensuring that the screen brightness is set to maximum.

  Precondition: Screen brightness is not at maximum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('min', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'max'}


class _SystemWifiToggle(task_eval.TaskEval):
  """Task for checking that WiFi has been turned {on_or_off}."""

  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'on_or_off': {'type': 'string', 'enum': ['on', 'off']}},
      'required': ['on_or_off'],
  }
  template = 'Turn wifi {on_or_off}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'global', 'wifi_on'], env.controller
    )
    wifi_status = res.generic.output.decode().strip()

    if self.params['on_or_off'] == 'on':
      # WiFi is on when the value is either 1 or 2. If Airplane mode is on, and
      # WiFi is on, it will be "2".
      return 1.0 if wifi_status in ['1', '2'] else 0.0
    else:
      # WiFi is off when the value is 0.
      return 1.0 if wifi_status == '0' else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on' if random.choice([True, False]) else 'off'}


class SystemWifiTurnOffVerify(_SystemWifiToggle):
  """Task for verifying that WiFi is already turned off.

  Precondition: WiFi is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemWifiTurnOnVerify(_SystemWifiToggle):
  """Task for verifying that WiFi is already turned on.

  Precondition: WiFi is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class SystemWifiTurnOff(_SystemWifiToggle):
  """Task for ensuring that WiFi is turned off.

  Precondition: WiFi is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemWifiTurnOn(_SystemWifiToggle):
  """Task for ensuring that WiFi is turned on.

  Precondition: WiFi is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class _SystemBluetoothToggle(task_eval.TaskEval):
  """Task for checking that Bluetooth has been turned {on_or_off}."""

  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'on_or_off': {'type': 'string', 'enum': ['on', 'off']}},
      'required': ['on_or_off'],
  }
  template = 'Turn bluetooth {on_or_off}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'global', 'bluetooth_on'], env.controller
    )
    bluetooth_status = res.generic.output.decode().strip()
    expected_status = '1' if self.params['on_or_off'] == 'on' else '0'
    return 1.0 if bluetooth_status == expected_status else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'on_or_off': 'on' if random.choice([True, False]) else 'off'}


class SystemBluetoothTurnOffVerify(_SystemBluetoothToggle):
  """Task for verifying that Bluetooth is already turned off.

  Precondition: Bluetooth is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemBluetoothTurnOnVerify(_SystemBluetoothToggle):
  """Task for verifying that Bluetooth is already turned on.

  Precondition: Bluetooth is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class SystemBluetoothTurnOff(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned off.

  Precondition: Bluetooth is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemBluetoothTurnOn(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned on.

  Precondition: Bluetooth is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class SystemCopyToClipboard(task_eval.TaskEval):
  """Task for verifying that the correct params are copied to the clipboard."""

  app_names = ('clipper',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {
          'clipboard_content': {'type': 'string'},
      },
      'required': ['clipboard_content'],
  }

  template = 'Copy the following text to the clipboard: {clipboard_content}'

  def __init__(self, params: dict[str, Any]):
    """Initialize the task with given params."""
    super().__init__(params)
    self.clipboard_content = params['clipboard_content']

  def _clear_clipboard(self, env: interface.AsyncEnv) -> None:
    # Use a unique string to set the clipboard contents.
    adb_utils.set_clipboard_contents('~~~RESET~~~', env.controller)

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self._clear_clipboard(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the clipboard content matches the expected content."""
    actual_clipboard_content = adb_utils.get_clipboard_contents(env.controller)
    return (
        1.0
        if fuzzy_match_lib.fuzzy_match(
            self.clipboard_content, actual_clipboard_content
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self._clear_clipboard(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        'clipboard_content': random.choice([
            '1234 Elm St, Springfield, IL',
            'Acme Corp, Suite 200',
            'john.doe@example.com',
            "Jane's Flower Shop",
            'Call me at 555-1234',
            'Order No: A123456',
            'Reservation under: Jane',
            'Discount code: SAVE20',
            'Membership ID: XYZ789',
            'Invoice #98765',
            'Tracking #: 1Z204E2A',
            'Transaction ID: abc123',
            '9876 Pine Ave, Riverside, CA',
            'Global Tech, Floor 3',
            'jane.smith@example.com',
            "Mike's Grocery Store",
            'Text me at 555-6789',
            'Order No: B654321',
            'Reservation under: Mike',
            'Promo code: DEAL30',
            'Membership ID: ABC123',
            'Invoice #54321',
            'Tracking #: 3H488Y2B',
            'Transaction ID: def456',
            '2554 Oak Street, Boston, MA',
            'Innovate Inc, Room 10',
            'alex.jordan@example.net',
            "Sara's Bakery",
            'Reach out at 555-9101',
            'Order No: C987654',
            'Reservation under: Sara',
            'Coupon code: OFF50',
            'Membership ID: LMN456',
            'Invoice #32198',
            'Tracking #: 5K672F4C',
            'Transaction ID: ghi789',
        ])
    }


@dataclasses.dataclass(frozen=True)
class _ComponentName:
  """Android identifier for an application component.

  Identifier for an application component - i.e., an Activity, a Service, a
  BroadcastReceiver, or a Content Provider. Encapsulates two pieces of
  information used to identify the component - the package name of the app it
  exists in, and the class name of the object within that app.
  """

  package_name: str
  class_name: str


def _normalize_class_name(package_name: str, class_name: str) -> str:
  """Normalizes a fully qualified class name to be relative to the package.

  Class names are strings, which can be fully qualified or relative to the
  app's package. This function normalizes a fully qualified class name to be
  relative, to make it easy to test two class names for equality.

      normalized_class_name = _normalize_class_name(
          'com.android.settings',
          'com.android.settings.Settings'
      )
      assert normalized_class_name == '.Settings'

  Args:
    package_name: The package name of the app.
    class_name: The name of the class.

  Returns:
    The class name, normalized to be relative if fully qualified.
  """
  if class_name.startswith(package_name):
    return class_name[len(package_name) :]
  return class_name


def parse_component_name(component_name: str) -> _ComponentName:
  """Parses a ComponentName from a string.

  Args:
    component_name: The string representation of the component name, e.g.
      'com.android.settings/com.android.settings.Settings'.

  Returns:
    The parsed ComponentName.
  Raises:
    ValueError: If called with an invalid string representation of a
      ComponentName.
  """
  parts = component_name.split('/')
  if len(parts) != 2:
    raise ValueError(
        'Badly formed component name: the package and class names must be '
        'separated by a single slash'
    )
  return _ComponentName(
      package_name=parts[0],
      class_name=_normalize_class_name(
          package_name=parts[0], class_name=parts[1]
      ),
  )


_APP_NAME_TO_PACKAGE_NAME = immutabledict.immutabledict({
    'camera': 'com.android.camera2',
    'clock': 'com.google.android.deskclock',
    'contacts': 'com.google.android.contacts',
    'settings': 'com.android.settings',
    'dialer': 'com.google.android.dialer',
})


class OpenAppTaskEval(task_eval.TaskEval):
  """Task eval for opening an app."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 1

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Open the {app_name} app. Clear any pop-ups that may appear by granting'
      ' all permissions that are required.'
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    active_activity, _ = adb_utils.get_current_activity(env.controller)
    expected_package_name = _APP_NAME_TO_PACKAGE_NAME[self.params['app_name']]
    if (
        parse_component_name(active_activity).package_name
        == expected_package_name
    ):
      return 1.0
    else:
      logging.info(
          'Expected %s to be active app but saw %s',
          expected_package_name,
          active_activity,
      )
      return 0.0
