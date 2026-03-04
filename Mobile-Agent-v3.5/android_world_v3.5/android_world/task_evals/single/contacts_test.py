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

from unittest import mock
from absl.testing import absltest
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals.single import contacts
from android_world.utils import test_utils


class ContactInfoIsEntered(absltest.TestCase):

  def test_contact_info_is_entered_valid_elements(self):
    ui_elements = [
        representation_utils.UIElement(text='John', hint_text='First name'),
        representation_utils.UIElement(text='Doe', hint_text='Last name'),
        representation_utils.UIElement(text='123-456-7890', hint_text='Phone'),
        representation_utils.UIElement(
            text='Work', content_description='Work Phone'
        ),
    ]

    self.assertTrue(
        contacts._contact_info_is_entered(
            'John',
            'Doe',
            '123-4567890',
            'Work',
            ui_elements,
        )
    )

  def test_contact_info_is_entered_missing_element(self):
    ui_elements = [
        representation_utils.UIElement(text='John', hint_text='First name'),
        representation_utils.UIElement(text='Doe', hint_text='Last name'),
        representation_utils.UIElement(text='1-234-567-890', hint_text='Phone'),
    ]

    self.assertFalse(
        contacts._contact_info_is_entered(
            'John', 'Doe', '1234-5678-90', 'Work', ui_elements
        )
    )

  def test_contact_info_is_entered_invalid_label(self):
    ui_elements = [
        representation_utils.UIElement(text='John', hint_text='First name'),
        representation_utils.UIElement(text='Doe', hint_text='Last name'),
        representation_utils.UIElement(text='1[234]567890', hint_text='Phone'),
        representation_utils.UIElement(
            text='Mobile', content_description='Label Phone'
        ),
    ]

    self.assertFalse(
        contacts._contact_info_is_entered(
            'John', 'Doe', '1234567890', 'Work', ui_elements
        )
    )


class ContactDraftTest(test_utils.AdbEvalTestBase):

  @mock.patch('android_world.env.representation_utils.forest_to_ui_elements')
  def test_contact_draft_is_successful(self, mock_forest_to_ui_elements):
    # Create an instance of ContactDraft.
    first = 'Jane'
    last = 'Smith'
    phone = '1-244-455-4333'
    phone_label = 'Work'
    contact_draft = contacts.ContactsNewContactDraft(
        {
            'first': first,
            'last': last,
            'phone': phone,
            'phone_label': phone_label,
        },
    )

    # Create a list of UIElement objects.
    ui_elements = [
        representation_utils.UIElement(text=first, hint_text='First name'),
        representation_utils.UIElement(text=last, hint_text='Last name'),
        representation_utils.UIElement(text='1244-455-4333', hint_text='Phone'),
        representation_utils.UIElement(
            text='Work', content_description='Work Phone'
        ),
    ]

    env = mock.create_autospec(interface.AsyncEnv)
    mock_forest_to_ui_elements.return_value = ui_elements
    self.assertEqual(test_utils.perform_task(contact_draft, env), 1)


if __name__ == '__main__':
  absltest.main()
