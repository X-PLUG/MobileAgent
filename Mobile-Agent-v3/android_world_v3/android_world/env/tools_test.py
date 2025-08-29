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

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import tools


@mock.patch.object(adb_utils, "issue_generic_request")
class TestAndroidToolControllerOpenWebPage(absltest.TestCase):

  def test_open_web_page_with_http_prefix(self, mock_generic_request):
    """Test opening a web page with http prefix."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)

    controller = tools.AndroidToolController(mock_env)
    controller.open_web_page("http://www.example.com")

    mock_generic_request.assert_called_once_with(
        [
            "shell",
            "am start -a android.intent.action.VIEW -d http://www.example.com",
        ],
        mock_env,
    )

  def test_open_web_page_without_http_prefix(self, mock_generic_request):
    """Test opening a web page without http prefix."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)

    controller = tools.AndroidToolController(mock_env)
    controller.open_web_page("www.example.com")

    mock_generic_request.assert_called_once_with(
        [
            "shell",
            "am start -a android.intent.action.VIEW -d http://www.example.com",
        ],
        mock_env,
    )


class TestAndroidToolControllerSendSmsIntent(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="GoogleSMS",
          activity_name="com.google.android.apps.messaging/ActivityName",
          ui_element="Send SMS",
      ),
      dict(
          testcase_name="SimpleSMS",
          activity_name="com.simplemobiletools.smsmessenger/ActivityName",
          ui_element="SMS",
      ),
  )
  @mock.patch.object(adb_utils, "issue_generic_request")
  @mock.patch.object(tools.AndroidToolController, "click_element")
  @mock.patch.object(adb_utils, "get_current_activity")
  def test_send_sms(
      self,
      mock_get_current_activity,
      mock_click_element,
      mock_generic_request,
      activity_name,
      ui_element,
  ):
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    controller = tools.AndroidToolController(mock_env)
    mock_get_current_activity.return_value = (
        activity_name,
        None,
    )
    phone_number = "+123456789"
    message = "Hello, how are you?"

    controller.send_sms(phone_number, message)

    expected_adb_command = [
        "shell",
        (
            "am start -a android.intent.action.SENDTO -d"
            f' sms:{phone_number} --es sms_body "{message}"'
        ),
    ]
    mock_generic_request.assert_called_once_with(expected_adb_command, mock_env)
    mock_click_element.assert_called_once_with(ui_element)


@mock.patch.object(tools.AndroidToolController, "open_web_page", autospec=True)
@mock.patch.object(tools.AndroidToolController, "send_sms", autospec=True)
class TestAndroidToolControllerHandleJsonRequest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    self.controller = tools.AndroidToolController(self.mock_env)

  def test_handle_json_request_valid_method_open_web_page(
      self, unused_mock_send_sms, mock_open_web
  ):
    json_request = json.dumps(
        {"method": "open_web_page", "args": {"url": "http://www.example.com"}}
    )

    self.controller.handle_json_request(json_request)

    mock_open_web.assert_called_once_with(
        self.controller, url="http://www.example.com"
    )

  def test_handle_json_request_valid_method_send_sms_intent(
      self, mock_send_sms, unused_mock_open_web
  ):
    """Test handling a valid JSON request for sending an SMS intent."""
    json_request = json.dumps({
        "method": "send_sms",
        "args": {"phone_number": "+123456789", "message": "Hello"},
    })

    self.controller.handle_json_request(json_request)

    mock_send_sms.assert_called_once_with(
        self.controller, phone_number="+123456789", message="Hello"
    )

  def test_handle_json_request_invalid_method(
      self,
      unused_mock_send_sms,
      unused_mock_open_web,
  ):
    """Test handling a JSON request with an invalid method."""
    json_request = json.dumps({"method": "non_existent_method", "args": {}})
    with self.assertRaises(ValueError):
      self.controller.handle_json_request(json_request)


if __name__ == "__main__":
  absltest.main()
