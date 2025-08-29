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

"""Logic for validating an SMS has been sent."""

import random
import time

from absl import logging
from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.utils import user_data_generation
from android_world.utils import fuzzy_match_lib


def parse_message(row: str) -> dict[str, str]:
  """Parse a string representing a row of message data into a dictionary.

  The row should contain multiple key-value pairs separated by commas and an
  equal sign. The function specifically accounts for the 'body' field, which can
  contain commas, by handling it separately from other fields.

  Args:
    row (str): A string containing the row data, with key-value pairs separated
      by ",".

  Returns:
    A dictionary where the keys are the field names and the values are the
    respective field values from the row string.

  Example:
  >>> parse_message("Row: 0 _id=5, thread_id=5, body=Hello, World, read=1")
  {'Row': '0', '_id': '5', 'thread_id': '5', 'body': 'Hello, World', 'read':
  '1'}
  """
  parsed_dict = {}

  row = row.strip()
  body_start = row.find("body=")

  if body_start != -1:
    body_content = row[body_start + 5 :]
    next_equal_sign = body_content.find("=")
    if next_equal_sign != -1:
      comma_before_next_equal_sign = body_content.rfind(
          ", ", 0, next_equal_sign
      )
      body_content = body_content[:comma_before_next_equal_sign]
    parsed_dict["body"] = body_content
    row = row[:body_start] + row[body_start + 5 + len(body_content) :]

  parts = row.split(", ")

  for part in parts:
    if "=" in part:
      key, value = part.split("=", 1)
      parsed_dict[key.strip()] = value.strip()
    elif ":" in part:
      key, value = part.split(":", 1)
      parsed_dict[key.strip()] = value.strip()
  return parsed_dict


def _decode_messages_from_response(response: adb_pb2.AdbResponse) -> list[str]:
  """Decodes the ADB response into a list of messages."""
  if (
      response.generic.output.decode()
      .replace("\r", "")
      .startswith("No result found.")
  ):
    return []
  messages = response.generic.output.split(b"\nRow:")
  for i, m in enumerate(messages):
    if i > 0:
      messages[i] = b"Row:" + m
  return [m.decode() for m in messages]


def was_sent(
    messages: list[str],
    phone_number: str,
    body: str,
    current_time_ms: int,
    time_mins: int = 5,
) -> bool:
  """Checks if a message was sent within the last time_mins minutes.

  Example:
    Given the `messages` list as, which are from `adb shell content query --uri
    content://sms/sent`:
    [
      'Row: 0 _id=2, address=+1111, date=1693421073675, body=Yo',
      'Row: 1 _id=1, address=+1111, date=1693421026207, body=Hi'
    ]
    `message_was_sent(messages, "+1111", "Yo")` would return True if the
    current time is within 5 minutes of `date=1693421073675`

  Args:
    messages: A list of message records returned by ADB shell content query,
      each as a string.
    phone_number: The target phone number or address to check the message
      against.
    body: The message body text to check for.
    current_time_ms: The current time, used to determine message staleness.
    time_mins: The time window in minutes within which to look for the message.

  Returns:
    Whether is was sent or not.
  """
  n_minutes_ms = time_mins * 60 * 1000
  for message in messages:
    # Extract the relevant fields from the ADB query result
    fields = parse_message(message)
    try:
      # Number can contain spaces and dashes, remove before comparing.
      msg_number = fields["address"].replace("-", "").replace(" ", "")
      msg_body = fields["body"]
      msg_date = int(fields["date"])
    except KeyError as key_error:
      raise ValueError(
          "Could not find the address, body, and date fields for message:"
          f" {message}"
      ) from key_error

    if (
        msg_number == phone_number
        and fuzzy_match_lib.fuzzy_match(msg_body, body)
        and (current_time_ms - msg_date <= n_minutes_ms)
    ):
      return True
    elif msg_number == phone_number and fuzzy_match_lib.fuzzy_match(
        msg_body, body
    ):
      logging.info(
          "The message was sent, but was sent over %i ago.", n_minutes_ms
      )

  return False


def sms_are_equal(message1: str, message2: str) -> bool:
  """Checks if two messages are equal.

  A message is equal to another if its address and body fields are equal.
  Args:
   message1: The first message to compare
   message2: The second message to compare

  Returns:
    Whether the messages are equal or not.
  """
  # Extract the relevant fields from the ADB query result
  message1_fields = parse_message(message1)
  message2_fields = parse_message(message2)
  phone_number1 = message1_fields["address"].replace("-", "").replace(" ", "")
  phone_number2 = message2_fields["address"].replace("-", "").replace(" ", "")
  return phone_number1 == phone_number2 and fuzzy_match_lib.fuzzy_match(
      message1_fields["body"], message2_fields["body"]
  )


def clear_sms_and_threads(env: env_interface.AndroidEnvInterface) -> None:
  """Removes all messages from UI by clearing the sms and threads tables."""
  db_path = "/data/data/com.android.providers.telephony/databases/mmssms.db"
  adb_utils.execute_sql_command(db_path, "DELETE FROM sms;", env)
  adb_utils.execute_sql_command(db_path, "DELETE FROM threads;", env)


class SimpleSMSSendSms(task_eval.TaskEval):
  """Task for checking that a single text message has been sent to a specific number with a specific message.

  It checks the sms table in
  /data/data/com.android.providers.telephony/databases/mmssms.db.

  While this technique is app agnostic, the template task specifies Simple SMS
  Pro as the target messaging app instead the default Android messaging app.
  The Android messaging app UI does not immediately reflect db state changes. We
  use Simple SMS Messenger due to its reliable and immediate UI synchronization
  with direct SQLite `sms` table manipulations, eliminating the hidden caching
  issues observed in the default messaging app.
  """

  app_names = ("simple sms messenger",)
  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "number": {"type": "string"},
          "message": {"type": "string"},
      },
      "required": ["number", "message"],
  }
  template = ""

  messages = user_data_generation.RANDOM_SENTENCES

  def get_sent_messages(
      self, env: env_interface.AndroidEnvInterface
  ) -> list[str]:
    response = adb_utils.issue_generic_request(
        "shell content query --uri content://sms/sent".split(), env
    )
    return _decode_messages_from_response(response)

  def _get_received_messages(
      self, env: env_interface.AndroidEnvInterface
  ) -> list[str]:
    response = adb_utils.issue_generic_request(
        "shell content query --uri content://sms/inbox".split(), env
    )
    return _decode_messages_from_response(response)

  # Returns the time on the android env in milliseconds.
  def get_android_time(self, env: env_interface.AndroidEnvInterface) -> int:
    adb_output = adb_utils.issue_generic_request(
        ["shell", "date", "+%s"], env
    )  # Fetch UNIX timestamp from Android
    return int(adb_output.generic.output.strip()) * 1000

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_airplane_mode("off", env.controller)
    clear_sms_and_threads(env.controller)
    android_time = self.get_android_time(env.controller)

    messages = self.get_sent_messages(env.controller)
    time.sleep(5)
    logging.info("During initialize_task, messages: %s", messages)
    if was_sent(
        messages,
        phone_number=self.params["number"],
        body=self.params["message"],
        current_time_ms=android_time,
    ):
      raise ValueError(
          "Message has already been sent, evaluator is not currently able to"
          " dedup. Please wait some time, change the goal message, or decrease "
          "the time param in sms_was_sent."
      )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    messages = self.get_sent_messages(env.controller)
    time.sleep(5)
    logging.info("During is_successful, messages: %s", messages)
    sms_was_sent = was_sent(
        messages,
        phone_number=self.params["number"],
        body=self.params["message"],
        current_time_ms=self.get_android_time(env.controller),
    )
    in_correct_app = (
        adb_utils.extract_package_name(
            adb_utils.get_current_activity(env.controller)[0]
        )
        == "com.simplemobiletools.smsmessenger"
    )
    if _check_if_stuck_at_sending(env):
      raise ValueError(
          "Message could not be sent due to Android/emulator issue."
      )
    return 1.0 if sms_was_sent and in_correct_app else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    number = user_data_generation.generate_random_number()
    message = random.choice(SimpleSMSSendSms.messages)

    return {
        "number": number,
        "message": message,
    }


def _check_if_stuck_at_sending(env: interface.AsyncEnv) -> bool:
  """Checks if the app is stuck at the sending screen."""
  state = env.get_state()
  for element in state.ui_elements:
    if element.text is not None and element.text.startswith("Sending"):
      return True
  return False
