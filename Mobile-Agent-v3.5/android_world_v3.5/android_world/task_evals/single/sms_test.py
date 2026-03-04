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

import random
import time
from unittest import mock

from absl.testing import absltest
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.env import representation_utils
from android_world.env import tools
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sms_validators
from android_world.task_evals.single import sms
from android_world.task_evals.utils import user_data_generation
from android_world.utils import contacts_utils
from android_world.utils import test_utils


class TestMessagesSendTextMessage(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()

    # Mock parent-level methods
    self.mock_initialize_sms_task = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'initialize_task'
    ).start()
    self.mock_is_successful = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'is_successful'
    ).start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_is_successful(self):
    # SimpleSmsSend doesn't add anything to the parent class, so simply
    # check that parent functions have been called.
    env = mock.MagicMock()
    params = {'number': '1234567890', 'message': 'Hello World'}

    task = sms.SimpleSmsSend(params)
    self.mock_is_successful.return_value = True
    self.assertEqual(test_utils.perform_task(task, env), 1)

    # Check that parent's functions got called
    self.mock_initialize_sms_task.assert_called_once()
    self.mock_is_successful.assert_called_once()

  def test_initialize_task(self):
    env = mock.MagicMock()

    params = {'number': '1234567890', 'message': 'Hello World'}

    task = sms_validators.SimpleSMSSendSms(params)
    task.initialize_task(env)

    # Check that parent's function got called
    self.mock_initialize_sms_task.assert_called_once()


class TestSimpleSmsReplyMostRecent(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    mock_randint = mock.patch.object(random, 'randint').start()
    mock_randchoice = mock.patch.object(random, 'choice').start()

    # Mock parent-level methods
    self.mock_initialize_sms_task = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'initialize_task'
    ).start()
    mock_generate_number = mock.patch.object(
        user_data_generation, 'generate_random_number'
    ).start()
    self.mock_is_successful = mock.patch.object(
        task_eval.TaskEval, 'is_successful'
    ).start()
    self.mock_android_time = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_android_time'
    ).start()
    self.mock_android_time.return_value = int(time.time())
    self.mock_get_received_messages = mock.patch.object(
        sms_validators.SimpleSMSSendSms, '_get_received_messages'
    ).start()

    # Mock adb_utils methods
    self.mock_disable_notifications = mock.patch.object(
        adb_utils, 'disable_headsup_notifications'
    ).start()
    self.mock_enable_notifications = mock.patch.object(
        adb_utils, 'enable_headsup_notifications'
    ).start()
    self.mock_text_emulator = mock.patch.object(
        adb_utils, 'text_emulator'
    ).start()

    # Setup mocks
    self.random_number_1 = '+1212365478'
    self.random_number_2 = '+19876543210'
    self.most_recent_number = '1234567890'
    self.message_1 = 'Message 1'
    self.message_2 = 'Message 2'
    self.most_recent_message = 'Hello World'
    # Instantiate state with 2 unimportant text message
    mock_randint.return_value = 2
    mock_randchoice.side_effect = [
        self.message_1,
        self.message_2,
        self.most_recent_message,
    ]
    mock_generate_number.side_effect = [
        self.random_number_1,
        self.random_number_2,
    ]

    self.initial_state_messages = [
        # Most recent message
        'Row: 0, address={}, body={}, service_center=NULL, date={}'.format(
            self.most_recent_number,
            self.most_recent_message,
            str(int((time.time() + 120) * 1000)),
        ),
        'Row: 1, address={}, body={}, service_center=NULL, date={}'.format(
            self.random_number_1,
            self.message_1,
            str(int((time.time() + 60) * 1000)),
        ),
        'Row: 2, address={}, body={}, service_center=NULL, date={}'.format(
            self.random_number_2,
            self.message_2,
            str(int(time.time() * 1000)),
        ),
    ]
    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_initialize_task(self):
    env = mock.MagicMock()
    params = {'number': self.most_recent_number, 'message': 'New message'}
    self.mock_get_received_messages.return_value = self.initial_state_messages

    task = sms.SimpleSmsReplyMostRecent(params)
    task.initialize_task(env)
    self.mock_text_emulator.assert_has_calls([
        mock.call(env.controller, self.random_number_1, self.message_1),
        mock.call(env.controller, self.random_number_2, self.message_2),
        mock.call(
            env.controller, self.most_recent_number, self.most_recent_message
        ),
    ])
    self.mock_disable_notifications.assert_called_once()
    self.mock_enable_notifications.assert_called_once()

    self.mock_initialize_sms_task.assert_called_once()

  def test_is_successful(self):
    new_message = 'New message'
    mock_sent_message = adb_pb2.AdbResponse()
    date_ms = str(int(time.time() * 1000))
    mock_sent_message.generic.output = (
        'Row: 0, address={}, body={}, service_center=NULL, date={}'.format(
            self.most_recent_number, new_message, date_ms
        ).encode()
    )
    self.mock_issue_generic_request.side_effect = [mock_sent_message]
    self.mock_get_received_messages.return_value = self.initial_state_messages
    test_utils.log_mock_calls(self.mock_issue_generic_request)
    env = mock.MagicMock()
    params = {'number': self.most_recent_number, 'message': new_message}

    task = sms.SimpleSmsReplyMostRecent(params)

    self.assertEqual(test_utils.perform_task(task, env), 1)


class TestSimpleSmsReply(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    mock_randint = mock.patch.object(random, 'randint').start()
    mock_randchoice = mock.patch.object(random, 'choice').start()

    # Mock parent-level methods
    self.mock_initialize_sms_task = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'initialize_task'
    ).start()
    mock_generate_number = mock.patch.object(
        user_data_generation, 'generate_random_number'
    ).start()
    self.mock_is_successful = mock.patch.object(
        task_eval.TaskEval, 'is_successful'
    ).start()
    self.mock_android_time = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_android_time'
    ).start()
    self.mock_android_time.return_value = int(time.time())

    # Mock adb_utils methods
    self.mock_disable_notifications = mock.patch.object(
        adb_utils, 'disable_headsup_notifications'
    ).start()
    self.mock_enable_notifications = mock.patch.object(
        adb_utils, 'enable_headsup_notifications'
    ).start()
    self.mock_text_emulator = mock.patch.object(
        adb_utils, 'text_emulator'
    ).start()

    # Setup mocks
    self.random_number_1 = '+1212365478'
    self.random_number_2 = '+19876543210'
    self.relevant_number = '1234567890'
    self.message_1 = 'Message 1'
    self.message_2 = 'Message 2'
    self.relevant_message = 'Hello World'
    # Instantiate state with 2 unimportant text message
    mock_randint.return_value = 2
    # Relevant message will always be sent last
    mock_randchoice.side_effect = [
        False,
        self.message_1,
        False,
        self.message_2,
        self.relevant_message,
    ]
    mock_generate_number.side_effect = [
        self.random_number_1,
        self.random_number_2,
    ]
    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_initialize_task(self):
    env = mock.MagicMock()
    params = {'number': self.relevant_number, 'message': 'New message'}

    task = sms.SimpleSmsReply(params)
    task.initialize_task(env)
    self.mock_text_emulator.assert_has_calls([
        mock.call(env.controller, self.random_number_1, self.message_1),
        mock.call(env.controller, self.random_number_2, self.message_2),
        mock.call(env.controller, self.relevant_number, self.relevant_message),
    ])
    self.mock_disable_notifications.assert_called_once()
    self.mock_enable_notifications.assert_called_once()

    self.mock_initialize_sms_task.assert_called_once()

  def test_is_successful(self):
    new_message = 'New message'
    # Add successful message
    mock_sent_message = adb_pb2.AdbResponse()
    date_ms = str(int(time.time() * 1000))
    mock_sent_message.generic.output = (
        'Row: 0, address={}, body={}, service_center=NULL, date={}'.format(
            self.relevant_number, new_message, date_ms
        ).encode()
    )
    self.mock_issue_generic_request.side_effect = [mock_sent_message]

    test_utils.log_mock_calls(self.mock_issue_generic_request)

    env = mock.MagicMock()
    params = {'number': self.relevant_number, 'message': new_message}

    task = sms.SimpleSmsReply(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)


class TestSimpleSmsSendClipboardContent(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()

    # Mock parent-level methods
    self.mock_initialize_sms_task = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'initialize_task'
    ).start()
    self.mock_is_successful = mock.patch.object(
        task_eval.TaskEval, 'is_successful'
    ).start()
    self.mock_get_sent_messages = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_sent_messages'
    ).start()
    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_initialize_task(self):
    clipboard_contents = 'Hello World'

    env = mock.MagicMock()
    params = {'number': '1234567890', 'message': clipboard_contents}

    task = sms.SimpleSmsSendClipboardContent(params)
    task.initialize_task(env)
    self.mock_set_clipboard_contents.assert_called_with(
        clipboard_contents, env.controller
    )
    self.mock_initialize_sms_task.assert_called_once()

  def test_is_successful(self):
    clipboard_contents = 'Hello World'
    number = '1234567890'
    date_ms = str(int(time.time() * 1000))
    self.mock_get_sent_messages.side_effect = [
        # Expected message
        [
            'Row: 0, address={}, body={}, service_center=NULL, date={}'
            .format(number, clipboard_contents, date_ms)
        ],
    ]

    env = mock.MagicMock()
    params = {'number': '1234567890', 'message': clipboard_contents}

    task = sms.SimpleSmsSendClipboardContent(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.mock_set_clipboard_contents.assert_called_with(
        clipboard_contents, env.controller
    )


class TestSimpleSmsSendReceivedAddress(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()

    # Create a list of UIElement objects corresponding to the Save button.
    ui_elements = [
        representation_utils.UIElement(
            text='Save',
            bbox=representation_utils.BoundingBox(
                x_min=-10,
                x_max=-20,
                y_min=-50,
                y_max=-80,
            ),
        ),
    ]
    self.mock_forest_to_ui_elements.return_value = ui_elements

    # Mock parent-level methods
    self.mock_initialize_sms_task = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'initialize_task'
    ).start()
    mock_generate_number = mock.patch.object(
        user_data_generation, 'generate_random_number'
    ).start()
    self.mock_is_successful = mock.patch.object(
        task_eval.TaskEval, 'is_successful'
    ).start()
    self.mock_android_time = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_android_time'
    ).start()
    self.mock_get_sent_messages = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_sent_messages'
    ).start()

    # Mock controller methods
    self.mock_add_contact = mock.patch.object(
        contacts_utils, 'add_contact'
    ).start()

    # Mock adb_utils methods
    self.mock_disable_notifications = mock.patch.object(
        adb_utils, 'disable_headsup_notifications'
    ).start()
    self.mock_enable_notifications = mock.patch.object(
        adb_utils, 'enable_headsup_notifications'
    ).start()
    self.mock_text_emulator = mock.patch.object(
        adb_utils, 'text_emulator'
    ).start()
    self.mock_delete_contacts = mock.patch.object(
        adb_utils, 'delete_contacts'
    ).start()

    # Setup mocks
    self.mock_android_time.return_value = int(time.time())
    self.random_number = '1234567890'
    mock_generate_number.side_effect = [
        self.random_number,
    ]
    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_initialize_task(self):
    # Create contacts
    name1 = 'Jane Smith'
    name1_number = '1444554333'
    name2 = 'John Smith'

    env = mock.MagicMock()
    params = {
        'name1': name1,
        'number': name1_number,
        'name2': name2,
        'message': '100 Main Street',
    }

    task = sms.SimpleSmsSendReceivedAddress(params)
    task.initialize_task(env)
    self.mock_disable_notifications.assert_called_once()
    self.mock_initialize_sms_task.assert_called_once()
    self.mock_add_contact.assert_has_calls([
        mock.call(name1, name1_number, env.controller),
        mock.call(name2, self.random_number, env.controller),
    ])
    self.mock_text_emulator.assert_called_with(
        env.controller, self.random_number, '100 Main Street'
    )
    self.mock_enable_notifications.assert_called_once()

  def test_is_successful(self):
    # Create contacts
    name1 = 'Jane Smith'
    name1_number = '1444554333'
    name2 = 'John Smith'
    address = '100 Main Street, Seattle, WA'

    # Successful message.
    date_ms = str(int(time.time() * 1000))
    self.mock_get_sent_messages.side_effect = [
        [
            'Row: 0, address={}, body={}, service_center=NULL, date={}'.format(
                name1_number, address, date_ms
            )
        ],
    ]

    env = mock.MagicMock()
    params = {
        'name1': name1,
        'number': name1_number,
        'name2': name2,
        'message': address,
    }

    task = sms.SimpleSmsSendReceivedAddress(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.mock_delete_contacts.assert_called_once()

  def test_is_not_successful(self):
    # Create contacts
    name1 = 'Jane Smith'
    name1_number = '1444554333'
    name2 = 'John Smith'
    address = '100 Main Street, Seattle, WA'

    # Unsuccessful message - sent to the wrong number.
    date_ms = str(int(time.time() * 1000))
    self.mock_get_sent_messages.side_effect = [
        [
            'Row: 0, address={}, body={}, service_center=NULL, date={}'.format(
                self.random_number, address, date_ms
            )
        ],
    ]

    env = mock.MagicMock()
    params = {
        'name1': name1,
        'number': name1_number,
        'name2': name2,
        'message': address,
    }

    task = sms.SimpleSmsSendReceivedAddress(params)
    self.assertEqual(test_utils.perform_task(task, env), 0)
    self.mock_delete_contacts.assert_called_once()


class TestMessagesResendTextMessage(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.glitch_message = (
        'Sorry, there was a glitch, what was the last message you sent me?'
    )

    # UIElements corresponding to the contact Save button and SMS send button.
    ui_elements = [
        representation_utils.UIElement(
            text='Save',
            bbox=representation_utils.BoundingBox(
                x_min=-10,
                x_max=-20,
                y_min=-50,
                y_max=-80,
            ),
        ),
        representation_utils.UIElement(
            text='SMS',
            bbox=representation_utils.BoundingBox(
                x_min=-10,
                x_max=-20,
                y_min=-50,
                y_max=-80,
            ),
        ),
    ]
    self.mock_forest_to_ui_elements.return_value = ui_elements

    # Mock parent-level methods
    self.mock_initialize_sms_task = mock.patch.object(
        task_eval.TaskEval, 'initialize_task'
    ).start()
    self.mock_is_successful = mock.patch.object(
        task_eval.TaskEval, 'is_successful'
    ).start()
    self.mock_android_time = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_android_time'
    ).start()
    self.mock_get_sent_messages = mock.patch.object(
        sms_validators.SimpleSMSSendSms, 'get_sent_messages'
    ).start()

    # Mock controller methods
    self.mock_add_contact = mock.patch.object(
        contacts_utils, 'add_contact'
    ).start()
    self.mock_send_sms = mock.patch.object(
        tools.AndroidToolController, 'send_sms'
    ).start()

    # Mock adb_utils methods
    self.mock_disable_notifications = mock.patch.object(
        adb_utils, 'disable_headsup_notifications'
    ).start()
    self.mock_enable_notifications = mock.patch.object(
        adb_utils, 'enable_headsup_notifications'
    ).start()
    self.mock_text_emulator = mock.patch.object(
        adb_utils, 'text_emulator'
    ).start()
    self.mock_delete_contacts = mock.patch.object(
        adb_utils, 'delete_contacts'
    ).start()

    # Setup mocks
    self.mock_android_time.return_value = int(time.time() * 1000)

    self.extract_package_name = mock.patch.object(
        adb_utils, 'extract_package_name'
    ).start()
    self.extract_package_name.return_value = (
        'com.simplemobiletools.smsmessenger'
    )

  def test_is_successful(self):
    # Create contact
    name = 'Jane Smith'
    agent_number = '1444554333'
    recipient_number = '9876543210'

    initial_date_ms = str(int(time.time() * 1000))
    final_date_ms = str(int(time.time() * 1000) + 6000)

    resend_message = (
        'Row: 1, address={}, body={}, service_center=NULL, date={}'.format(
            recipient_number, self.glitch_message, initial_date_ms
        )
    )
    initial_message = (
        'Row: 2, address={}, body=Hello World, service_center=NULL, date={}'
        .format(agent_number, initial_date_ms)
    )
    expected_message = (
        'Row: 0, address={}, body=Hello World, service_center=NULL, date={}'
        .format(agent_number, final_date_ms)
    )
    self.mock_get_sent_messages.side_effect = [
        # Empty messages pre-initialization
        [],
        # Post initialization messages
        [resend_message, initial_message],
        [expected_message, resend_message, initial_message],
    ]

    env = mock.MagicMock()
    params = {
        'name': name,
        'number': agent_number,
        'message': 'Hello World',
    }

    task = sms.SimpleSmsResend(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)

  def test_initialize_task(self):
    # Create contact
    name = 'Jane Smith'
    number = '1444554333'
    message = '100 Main Street'

    env = mock.MagicMock()
    params = {
        'name': name,
        'number': number,
        'message': message,
    }

    task = sms.SimpleSmsResend(params)
    task.initialize_task(env)
    self.mock_disable_notifications.assert_called_once()
    self.mock_initialize_sms_task.assert_called_once()
    self.mock_add_contact.assert_called_with(name, number, env.controller)
    # Check that initial message was sent
    self.mock_send_sms.assert_called_with(number, message)
    # Check that resend message was sent
    self.mock_text_emulator.assert_called_with(
        env.controller, number, self.glitch_message
    )
    self.mock_enable_notifications.assert_called_once()


if __name__ == '__main__':
  absltest.main()
