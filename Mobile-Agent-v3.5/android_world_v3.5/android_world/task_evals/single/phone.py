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

"""Tasks for making and receiving phone calls."""

import random
import time
from typing import Any
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.common_validators import phone_validators
from android_world.task_evals.common_validators import sms_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class MarkorCallApartment(phone_validators.MakeCall):
  """Represents a task that combines phone calling with a Markor note lookup.

  This task involves reading a phone number from a Markor note and making
  a phone call to the specified number. It checks whether the In-Call UI is
  displayed
  with options like 'Hold'.
  """

  app_names = ("markor",)
  complexity = 1
  schema = {
      "type": "object",
      "properties": {
          "name": {"type": "string"},
          "phone_number": {"type": "string"},
      },
      "required": ["phone_number"],
  }
  template = (
      "Call the number for the apartment name {name}. The number is in"
      " apartments.md file in Markor. Ensure the In-Call UI is displayed with"
      " options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    content = user_data_generation.dict_to_notes(
        user_data_generation.generate_apartments()
    )
    file_utils.create_file(
        "apartments.md", device_constants.MARKOR_DATA, env.controller, content
    )
    self.phone_number = self.params["phone_number"]

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    candidates = user_data_generation.generate_apartments()
    name = random.choice(list(candidates.keys()))
    number = candidates[name][0]
    return {
        "name": name,
        "phone_number": number,
    }


class PhoneMakeCall(phone_validators.MakeCall):
  """Task to make a phone call."""

  template = (
      "Make a phone call to the number {phone_number} and ensure the In-Call UI"
      " is displayed with options like 'Hold'."
  )


class PhoneReturnMissedCall(phone_validators.MakeCall):
  """Task to return a missed phone call.

  This task involves making a call back to a number that was missed and
  verifying that the In-Call UI appears with options like 'Hold'.
  """

  template = (
      "Return the call I just missed and ensure the In-Call UI is displayed"
      " with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.call_emulator(env.controller, self.phone_number)
    time.sleep(5)
    adb_utils.end_call_if_active(env.controller)


class PhoneRedialNumber(phone_validators.MakeCall):
  """Task to re-dial the last dialed number.

  This task involves re-dialing the last dialed number and verifying that the
  In-Call UI appears with options like 'Hold'.
  """

  template = (
      "Re-dial the number I was just talking to and ensure the In-Call UI is "
      "displayed with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.call_phone_number(env.controller, self.phone_number)
    time.sleep(5)
    adb_utils.end_call_if_active(env.controller)


class PhoneCallTextSender(phone_validators.MakeCall):
  """Task to call the sender of the most recent text message.

  This task involves making a call back to the number that most recently sent a
  text message and verifying that the In-Call UI appears with options like
  'Hold'.
  """

  template = (
      "Call the number that just texted me and ensure the In-Call UI is "
      "displayed with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    sms_validators.clear_sms_and_threads(env.controller)
    print(self.phone_number)
    adb_utils.text_emulator(
        env.controller, self.phone_number, "Hey give me a call"
    )


class PhoneAnswerCall(phone_validators.MakeCall):
  """Task to answer an incoming phone call.

  This task involves answering an incoming call from a specified number and
  verifying that the In-Call UI appears with options like 'Hold'.
  """

  template = (
      "Answer the incoming phone call and ensure the In-Call UI is displayed"
      " with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.call_emulator(env.controller, self.phone_number)
