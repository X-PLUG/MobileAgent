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

"""Tests for adb_utils."""

from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import adb_utils


class AdbTestSetup(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.patch.object(
        env_interface, 'AndroidEnvInterface', autospec=True
    ).start()
    self.mock_issue_generic_request = mock.patch.object(
        adb_utils, 'issue_generic_request', autospec=True
    ).start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()


class PhoneUtilsTest(AdbTestSetup):

  def test_get_call_state_idle(self):
    dumpsys_content = """last known state:
Phone Id=0
  mCallState=0
  mRingingCallState=0
  mForegroundCallState=0"""
    # Setup
    mock_dumpsys_response = adb_pb2.AdbResponse()
    mock_dumpsys_response.generic.output = dumpsys_content.encode('utf-8')
    self.mock_issue_generic_request.return_value = mock_dumpsys_response

    # Act
    result = adb_utils.get_call_state(self.mock_env)

    # Assert
    self.assertEqual(result, 'IDLE')

  def test_get_call_state_ringing(self):
    dumpsys_content = """last known state:
Phone Id=0
  mCallState=2
  mRingingCallState=0
  mForegroundCallState=0"""
    # Setup
    mock_dumpsys_response = adb_pb2.AdbResponse()
    mock_dumpsys_response.generic.output = dumpsys_content.encode('utf-8')
    self.mock_issue_generic_request.return_value = mock_dumpsys_response

    # Act
    result = adb_utils.get_call_state(self.mock_env)

    # Assert
    self.assertEqual(result, 'OFFHOOK')

  def test_call_emulator(self):
    mock_response = adb_pb2.AdbResponse()
    mock_response.generic.output = b'Success'
    self.mock_issue_generic_request.return_value = mock_response

    phone_number = '+123456789'
    result = adb_utils.call_emulator(self.mock_env, phone_number)

    self.assertEqual(result.generic.output.decode(), 'Success')

  @mock.patch.object(adb_utils, 'get_call_state', autospec=True)
  def test_end_call_if_active(self, mock_get_call_state):
    mock_get_call_state.return_value = 'OFFHOOK'
    adb_utils.end_call_if_active(self.mock_env)

    self.mock_issue_generic_request.assert_called()

  def test_clear_android_emulator_call_log(self):
    adb_utils.clear_android_emulator_call_log(self.mock_env)
    self.mock_issue_generic_request.assert_called()

  def test_call_phone_number(self):
    mock_response = adb_pb2.AdbResponse()
    mock_response.generic.output = b'Success'
    self.mock_issue_generic_request.return_value = mock_response

    phone_number = '123456789'
    result = adb_utils.call_phone_number(self.mock_env, phone_number)

    self.assertEqual(result.generic.output.decode(), 'Success')

  def test_text_emulator(self):
    mock_response = adb_pb2.AdbResponse()
    mock_response.generic.output = b'Success'
    self.mock_issue_generic_request.return_value = mock_response

    phone_number = '+123456789'
    message = 'Hello, world!'
    result = adb_utils.text_emulator(self.mock_env, phone_number, message)

    self.assertEqual(result.generic.output.decode(), 'Success')


class AdbSettingsTest(AdbTestSetup):

  def test_set_default_app(self):
    mock_response = adb_pb2.AdbResponse()
    mock_response.generic.output = b'Success'
    self.mock_issue_generic_request.return_value = mock_response

    setting_key = 'sms_default_application'
    package_name = 'com.example.app'
    result = adb_utils.set_default_app(self.mock_env, setting_key, package_name)

    self.assertEqual(result.generic.output.decode(), 'Success')

  def test_successful_put_operation(self):
    self.mock_env.execute_adb_call.return_value = adb_pb2.AdbResponse()

    # Execute the function
    response = adb_utils.put_settings(
        namespace=adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
        key='example_key',
        value='example_value',
        env=self.mock_env,
    )

    # Assertions
    self.assertIsInstance(response, adb_pb2.AdbResponse)
    self.mock_env.execute_adb_call.assert_called_once()

  def test_invalid_inputs_put_operation(self):
    self.mock_env.execute_adb_call.return_value = adb_pb2.AdbResponse()

    # Invalid namespace (non-enum value)
    with self.assertRaises(ValueError):
      adb_utils.put_settings(
          namespace='INVALID',  # This should be an enum, not a string
          key='example_key',
          value='example_value',
          env=self.mock_env,
      )

    # Empty key
    with self.assertRaises(ValueError):
      adb_utils.put_settings(
          namespace=adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
          key='',
          value='example_value',
          env=self.mock_env,
      )

    # Empty value
    with self.assertRaises(ValueError):
      adb_utils.put_settings(
          namespace=adb_pb2.AdbRequest.SettingsRequest.Namespace.SYSTEM,
          key='example_key',
          value='',
          env=self.mock_env,
      )


class AdbTypingTest(AdbTestSetup):

  def test_can_type_text(self):
    with mock.patch.object(
        env_interface.AndroidEnvInterface, 'execute_adb_call'
    ) as mock_execute_adb_call:
      mock_execute_adb_call.return_value = adb_pb2.AdbResponse(
          status=adb_pb2.AdbResponse.Status.OK
      )
      adb_utils.type_text('Type some\ntext', self.mock_env)
      expected_calls = [
          mock.call(
              adb_pb2.AdbRequest(
                  input_text=adb_pb2.AdbRequest.InputText(text='Type'),
                  timeout_sec=adb_utils._DEFAULT_TIMEOUT_SECS,
              )
          ),
          mock.call(
              adb_pb2.AdbRequest(
                  input_text=adb_pb2.AdbRequest.InputText(text='%s'),
                  timeout_sec=adb_utils._DEFAULT_TIMEOUT_SECS,
              )
          ),
          mock.call(
              adb_pb2.AdbRequest(
                  input_text=adb_pb2.AdbRequest.InputText(text='some'),
                  timeout_sec=adb_utils._DEFAULT_TIMEOUT_SECS,
              )
          ),
          mock.call(
              adb_pb2.AdbRequest(
                  press_button=adb_pb2.AdbRequest.PressButton(
                      button=adb_pb2.AdbRequest.PressButton.ENTER
                  ),
                  timeout_sec=adb_utils._DEFAULT_TIMEOUT_SECS,
              )
          ),
          mock.call(
              adb_pb2.AdbRequest(
                  input_text=adb_pb2.AdbRequest.InputText(text='text'),
                  timeout_sec=adb_utils._DEFAULT_TIMEOUT_SECS,
              )
          ),
      ]
      mock_execute_adb_call.assert_has_calls(expected_calls)

  def test_type_white_space(self):
    with mock.patch.object(
        env_interface.AndroidEnvInterface, 'execute_adb_call'
    ) as mock_execute_adb_call:
      mock_execute_adb_call.return_value = adb_pb2.AdbResponse(
          status=adb_pb2.AdbResponse.Status.OK
      )
      adb_utils.type_text(' ', self.mock_env)
      expected_calls = [
          mock.call(
              adb_pb2.AdbRequest(
                  input_text=adb_pb2.AdbRequest.InputText(text='%s'),
                  timeout_sec=adb_utils._DEFAULT_TIMEOUT_SECS,
              )
          ),
      ]
      mock_execute_adb_call.assert_has_calls(expected_calls)
      self.assertLen(expected_calls, mock_execute_adb_call.call_count)


class TestExtractBroadcastData(absltest.TestCase):

  def test_successful_data_extraction(self):
    raw_output = 'Broadcast completed: result=-1, data="Test data"\n'
    expected_result = 'Test data'
    result = adb_utils.extract_broadcast_data(raw_output)
    self.assertEqual(result, expected_result)

  def test_result_zero_returns_none(self):
    raw_output = 'Broadcast completed: result=0\n'
    result = adb_utils.extract_broadcast_data(raw_output)
    self.assertIsNone(result)

  def test_unexpected_output_raises_error(self):
    raw_output = 'Unexpected output format'
    with self.assertRaises(ValueError):
      adb_utils.extract_broadcast_data(raw_output)


class TestScreenUtils(absltest.TestCase):

  def test_parse_screen_size_response_success(self):
    """Test successful parsing of screen size from adb response."""
    response = 'Physical size: 1080x2400'
    expected_size = (1080, 2400)
    self.assertEqual(
        adb_utils._parse_screen_size_response(response), expected_size
    )

  def test_parse_screen_size_response_failure(self):
    """Test parsing failure when adb response is in an unexpected format."""
    response = 'Invalid response format'
    with self.assertRaises(ValueError) as context:
      adb_utils._parse_screen_size_response(response)
    self.assertIn(
        'Screen size information not found in adb response',
        str(context.exception),
    )


if __name__ == '__main__':
  absltest.main()
