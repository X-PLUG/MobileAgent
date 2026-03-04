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

"""API tools library for Android agents."""

import inspect
import json
import time
from typing import Optional, Union

from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.utils import contacts_utils


# When the compose message is pulled up, the send button has this as text for
# Simple SMS Messenger.
SIMPLE_SMS_SEND_TEXT = "SMS"
# For Google messaging app.
SMS_SEND_TEXT = "Send SMS"


class AndroidToolController:
  """Executes API tools on an Android device."""

  def __init__(
      self,
      env: android_world_controller.AndroidWorldController,
  ):
    """Initializes the controller with an Android environment instance.

    Args:
      env: The AndroidEnv interface to be used.
    """
    self._env = env

  def click_element(self, element_text: str):
    actuation.find_and_click_element(element_text, self._env)

  def open_web_page(self, url: str):
    """Open a web page in the default browser on an Android device.

    This function sends an intent to the Android system to open the specified
    URL.

    Args:
      url: The URL of the web page to open. E.g., http://www.google.com.
    """
    if not url.startswith("http://"):
      url = "http://" + url
    adb_command = ["shell", f"am start -a android.intent.action.VIEW -d {url}"]
    adb_utils.issue_generic_request(adb_command, self._env)

  def send_sms(
      self,
      phone_number: str,
      message: str,
  ):
    """Send an SMS to a specified phone number.

    This function sends an intent to the Android system to open the messaging
    app with the recipient's number and message pre-filled.

    Args:
      phone_number: The phone number to which the SMS should be sent.
      message: The pre-filled message text.
    """
    # Construct the Intent command
    intent_command = (
        "am start -a android.intent.action.SENDTO -d sms:{phone_number} "
        f'--es sms_body "{message}"'
    ).format(phone_number=phone_number)

    adb_command = ["shell", intent_command]
    adb_utils.issue_generic_request(adb_command, self._env)
    time.sleep(5.0)

    package_name = adb_utils.extract_package_name(
        adb_utils.get_current_activity(self._env)[0]
    )
    # Depending on what the default SMS app we need to click different buttons.
    if package_name == "com.google.android.apps.messaging":
      self.click_element(SMS_SEND_TEXT)
    elif package_name == "com.simplemobiletools.smsmessenger":
      self.click_element(SIMPLE_SMS_SEND_TEXT)
    else:
      raise ValueError(f"Messaging app not supported: {package_name}")

  def _gather_tool_details(
      self,
  ) -> dict[str, list[Optional[dict[str, Union[dict[str, str], str]]]]]:
    """Get the details and examples of usage for public APIs related to Android tools.

    Returns:
        A dictionary where the keys are API names and the values are lists of
        dictionaries containing the docstrings and usage examples.
    """
    return {
        "open_web_page": self._tool_info(
            self.open_web_page,
            [
                {"url": "http://www.google.com"},
                {"url": "http://www.example.com"},
            ],
        ),
        "send_sms": self._tool_info(
            self.send_sms,
            [
                {
                    "phone_number": "+123456789",
                    "message": "Hello, how are you?",
                },
                {
                    "phone_number": "+987654321",
                    "message": "Meeting rescheduled to 3 PM.",
                },
            ],
        ),
        "add_contact": self._tool_info(
            contacts_utils.add_contact,
            [
                {"name": "John Doe", "phone_number": "+123456789"},
                {"name": "Joe", "phone_number": "987654321"},
            ],
        ),
    }

  def _tool_info(
      self, method, example_args: list[dict[str, str]]
  ) -> list[Optional[dict[str, Union[dict[str, str], str]]]]:
    """Helper function to construct tool information and examples.

    Args:
        method: The method for which to gather information.
        example_args: A list of argument dictionaries for examples.

    Returns:
        A list containing the method's documentation and examples.
    """
    doc_info = {"doc": inspect.getdoc(method)}
    examples = [
        {"method": method.__name__, "args": args} for args in example_args
    ]
    return [doc_info, *examples]

  def display_tool_usage(self) -> str:
    """Format the tool information and examples into a user-friendly string.

    Returns:
        A string representing the available tools and their usage examples.
    """
    tools_info = self._gather_tool_details()
    formatted_info = ["Available Tools and Usage Examples:\n"]

    for tool_name, tool_details in tools_info.items():
      formatted_info.append(f"\nAPI: {tool_name}\n")
      formatted_info.append(f"Description: {tool_details[0]['doc']}\n")
      formatted_info.append("Examples:\n")
      for example in tool_details[1:]:
        formatted_info.append(f"  - JSON Request: {example}\n")

    return "".join(formatted_info)

  def handle_json_request(self, json_request: str):
    """Handle a JSON formatted request to use a tool.

    Args:
        json_request: A JSON string with the method and arguments.
    """
    request = json.loads(json_request)
    method_name = request["method"]
    args = request.get("args", {})

    if hasattr(self, method_name) and callable(getattr(self, method_name)):
      method = getattr(self, method_name)
      method(**args)
    else:
      raise ValueError(f"Method {method_name} not found.")
