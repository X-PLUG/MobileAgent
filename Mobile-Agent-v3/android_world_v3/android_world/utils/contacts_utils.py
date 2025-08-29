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

"""Utils for contacts operations using adb."""

import dataclasses
import re
import time
from typing import Iterator

from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import android_world_controller


def clean_phone_number(phone_number: str) -> str:
  """Removes all non-numeric characters from a phone number.

  Args:
    phone_number: The phone number to clean.

  Returns:
    The phone number with all non-numeric characters removed.
  """
  return re.sub(r"\D", "", phone_number)


def add_contact(
    name: str,
    phone_number: str,
    env: android_world_controller.AndroidWorldController,
    ui_delay_sec: float = 1.0,
):
  """Adds a contact with the specified name and phone number.

  This function sends an intent to the Android system to add a contact with
  the information pre-filled, clicks the "Save" button to create it, and then
  returns from the activity.

  Args:
    name: The name of the new contact
    phone_number: The phone number belonging to that contact.
    env: The android environment to add the contact to.
    ui_delay_sec: Delay between UI interactions. If this value is too low, the
      "save" button may be mis-clicked.
  """
  intent_command = (
      "am start -a android.intent.action.INSERT -t"
      f' vnd.android.cursor.dir/contact -e name "{name}" -e phone'
      f" {phone_number}"
  )

  adb_command = ["shell", intent_command]
  adb_utils.issue_generic_request(adb_command, env)
  time.sleep(ui_delay_sec)
  actuation.find_and_click_element("SAVE", env)
  time.sleep(ui_delay_sec)
  adb_utils.press_back_button(env)
  time.sleep(ui_delay_sec)


@dataclasses.dataclass(frozen=True)
class Contact:
  """Basic contact information."""
  name: str
  number: str


def list_contacts(
    env: android_world_controller.AndroidWorldController,
) -> list[Contact]:
  """Lists all contacts available in the Android environment.

  Args:
    env: Android environment to search for contacts.

  Returns:
    A list of all contact names and numbers present on the device.
  """
  intent_command = (
      "content query --uri content://contacts/phones/ --projection"
      " display_name:number"
  )
  adb_command = ["shell", intent_command]

  def parse(adb_output: str) -> Iterator[Contact]:
    for match in re.finditer(r"display_name=(.*), number=(.*)", adb_output):
      yield Contact(match.group(1), clean_phone_number(match.group(2)))

  return list(
      parse(
          adb_utils.issue_generic_request(
              adb_command, env
          ).generic.output.decode("utf-8")
      )
  )


def clear_contacts(env: android_world_controller.AndroidWorldController):
  """Clears all contacts on the device."""
  adb_utils.clear_app_data("com.android.providers.contacts", env)
