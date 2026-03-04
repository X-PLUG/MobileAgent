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

"""Tasks for Simple SMS Messenger."""

import random
import time
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import tools
from android_world.task_evals.common_validators import phone_validators
from android_world.task_evals.common_validators import sms_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import contacts_utils


class SimpleSmsSend(sms_validators.SimpleSMSSendSms):
  """Task for checking an SMS was sent."""

  template = (
      "Send a text message using Simple SMS Messenger to {number} with message:"
      " {message}"
  )


class SimpleSmsSendAfterCall(sms_validators.SimpleSMSSendSms):
  """Task for checking an SMS was sent after a missed call.

  NOTE: This is currently disabled due to emulator flakiness with phone calls.
  """

  app_names = ("simple sms messenger", "phone")
  template = (
      "Send a text message using Simple SMS Messenger to the number I just"
      " missed a call from with message: {message}"
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    phone_validators.clear_phone_state(env.controller)
    adb_utils.call_emulator(env.controller, self.params["number"])
    time.sleep(5.0)
    adb_utils.end_call_if_active(env.controller)


class SimpleSmsReplyMostRecent(sms_validators.SimpleSMSSendSms):
  """Task for checking that a reply was sent to the most recent SMS."""

  template = (
      "Reply to the most recent text message using Simple SMS Messenger with"
      " message: {message}"
  )

  def _generate_non_goal_message(self):
    message = random.choice(sms_validators.SimpleSMSSendSms.messages)
    while message == self.params["message"]:
      message = random.choice(sms_validators.SimpleSMSSendSms.messages)
    return message

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    # Disable notifications so we don't have to wait for them to disappear
    # before running the task.
    adb_utils.disable_headsup_notifications(env.controller)

    for _ in range(random.randint(0, 5)):
      adb_utils.text_emulator(
          env.controller,
          user_data_generation.generate_random_number(),
          self._generate_non_goal_message(),
      )

    # Texts don't necessarily come in the same order as sent here, so pause here
    # to make sure the most recent text comes last.
    time.sleep(5)

    most_recent_message = self._generate_non_goal_message()
    adb_utils.text_emulator(
        env.controller,
        self.params["number"],
        most_recent_message,
    )

    # Need to pause to make sure re-enabling notifications happens after the
    # last text came in
    time.sleep(5)

    adb_utils.enable_headsup_notifications(env.controller)

    most_recent = sms_validators.parse_message(
        self._get_received_messages(env.controller)[0]
    )
    if (
        most_recent["address"] != self.params["number"]
        and most_recent["body"] != most_recent_message
    ):
      raise ValueError(
          "Unexpected initial state - most recent message is not what is"
          " expected."
      )


class SimpleSmsReply(sms_validators.SimpleSMSSendSms):
  """Task for checking a reply was sent."""

  complexity = 1.2
  template = "Reply to {number} with message: {message} in Simple SMS Messenger"

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.disable_headsup_notifications(env.controller)

    relevant_text_sent = False

    # Add a random number of texts, with the text we care about randomly
    # interspersed.
    for _ in range(random.randint(1, 5)):
      if not relevant_text_sent:
        if random.choice([True, False]):
          adb_utils.text_emulator(
              env.controller,
              self.params["number"],
              random.choice(sms_validators.SimpleSMSSendSms.messages),
          )
          relevant_text_sent = True

      adb_utils.text_emulator(
          env.controller,
          user_data_generation.generate_random_number(),
          random.choice(sms_validators.SimpleSMSSendSms.messages),
      )

    if not relevant_text_sent:
      adb_utils.text_emulator(
          env.controller,
          self.params["number"],
          random.choice(sms_validators.SimpleSMSSendSms.messages),
      )

    # Need to pause to make sure re-enabling notifications happens after the
    # last text came in
    time.sleep(0.5)
    adb_utils.enable_headsup_notifications(env.controller)


class SimpleSmsSendClipboardContent(sms_validators.SimpleSMSSendSms):
  """Task for checking that the clipboard contents were sent as an SMS."""

  app_names = ("simple sms messenger", "clipper")
  complexity = 1.2
  template = (
      "Send a message to {number} with the clipboard content in Simple SMS"
      " Messenger"
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_clipboard_contents(self.params["message"], env.controller)


class SimpleSmsSendReceivedAddress(sms_validators.SimpleSMSSendSms):
  """Task for checking that a received address is forward to someone else."""

  complexity = 1.8
  template = (
      "Text the address of the event to {name1} that {name2} just sent me in"
      " Simple SMS Messenger"
  )

  schema = {
      "type": "object",
      "properties": {
          "name1": {"type": "string"},
          "number": {"type": "string"},
          "name2": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["name1", "number", "name2", "message"],
  }

  addresses = [
      "123 Main St Girdwood, AK, 99587",
      "6 Elm St, Birmingham, AL, 35217",
      "789 E Oak St, Phoenix AZ 85006",
      "1011 S Maple St, Little Rock, AR, 72204",
      "1415 W Cedar Ave Denver, CO, 80223",
      "968 Spruce St, Hartford, CT, 06103",
      "1819 Birch Ct, Dover, DE, 19901",
      "2021 Poplar St, Atlanta, GA, 30340",
  ]

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    name1 = user_data_generation.generate_random_name()
    name2 = user_data_generation.generate_random_name(excluding=name1)

    return {
        "name1": name1,
        "number": user_data_generation.generate_random_number(),
        "name2": name2,
        "message": user_data_generation.generate_random_address(),
    }

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    adb_utils.disable_headsup_notifications(env.controller)
    super().initialize_task(env)

    name2_number = user_data_generation.generate_random_number()
    contacts_utils.add_contact(
        self.params["name1"], self.params["number"], env.controller
    )
    time.sleep(5.0)
    contacts_utils.add_contact(
        self.params["name2"], name2_number, env.controller
    )

    # Add text containing address from name2
    adb_utils.text_emulator(
        env.controller,
        name2_number,
        self.params["message"],
    )

    # Need to pause to make sure re-enabling notifications happens after the
    # text came in
    time.sleep(1)
    adb_utils.enable_headsup_notifications(env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    adb_utils.delete_contacts(env.controller)


class SimpleSmsResend(sms_validators.SimpleSMSSendSms):
  """Task for checking that a message was resent."""

  complexity = 1.2
  template = "Resend the message I just sent to {name} in Simple SMS Messenger"

  schema = {
      "type": "object",
      "properties": {
          "name": {"type": "string"},
          "number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["name", "number", "message"],
  }

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "name": user_data_generation.generate_random_name(),
        "number": user_data_generation.generate_random_number(),
        "message": random.choice(cls.messages),
    }

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    controller = tools.AndroidToolController(env.controller)
    adb_utils.disable_headsup_notifications(env.controller)
    super().initialize_task(env)

    contacts_utils.add_contact(
        self.params["name"], self.params["number"], env.controller
    )
    time.sleep(3.0)
    controller.send_sms(self.params["number"], self.params["message"])

    # Make sure conversation happens before the repeat message
    time.sleep(3.0)

    # Add text asking to repeat
    adb_utils.text_emulator(
        env.controller,
        self.params["number"],
        "Sorry, there was a glitch, what was the last message you sent me?",
    )

    # Need to pause to make sure re-enabling notifications happens after the
    # text came in
    time.sleep(1)
    adb_utils.enable_headsup_notifications(env.controller)
    self.before_messages = self.get_sent_messages(env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    after_messages = self.get_sent_messages(env.controller)
    if len(after_messages) != len(self.before_messages) + 1:
      return 0.0

    # New messages get added at index 0.
    return (
        1.0  # pylint:disable=g-long-ternary
        if sms_validators.sms_are_equal(
            after_messages[0], self.before_messages[-1]
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    adb_utils.delete_contacts(env.controller)
