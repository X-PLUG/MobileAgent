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
from android_world.agents import seeact
from android_world.agents import seeact_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import representation_utils


class TestSeeAct(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(interface.AsyncEnv)
    self.seeact = seeact.SeeAct(self.mock_env)
    self.mock_execute_openai_request = mock.patch.object(
        seeact_utils, 'execute_openai_request'
    ).start()
    mock.patch.object(seeact_utils, 'create_grounding_messages_payload').start()
    mock.patch.object(
        seeact_utils, 'create_action_generation_messages_payload'
    ).start()
    mock.patch.object(actuation, 'execute_adb_action').start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_step(self):
    mock_action_gen_response_1 = {
        'choices': [{'message': {'content': 'Generated action 1'}}]
    }
    mock_action_ground_response_1 = {
        'choices': [
            {'message': {'content': 'ELEMENT: A\nACTION: CLICK\nVALUE: None'}}
        ]
    }

    mock_action_gen_response_2 = {
        'choices': [{'message': {'content': 'Generated action 2'}}]
    }
    mock_action_ground_response_2 = {
        'choices': [{
            'message': {
                'content': 'ELEMENT: None\nACTION: TERMINATE\nVALUE: None'
            }
        }]
    }

    self.mock_execute_openai_request.side_effect = [
        mock_action_gen_response_1,
        mock_action_ground_response_1,
        mock_action_gen_response_2,
        mock_action_ground_response_2,
    ]

    mock_ui_elements = [
        representation_utils.UIElement(
            text=None,
            content_description='AnImage',
            class_name='android.widget.ImageView',
        ),
        representation_utils.UIElement(
            text='Unchecked',
            class_name='android.widget.CheckBox',
            is_checked=False,
        ),
    ]
    self.mock_env.get_state.return_value.ui_elements = mock_ui_elements

    goal = 'Test goal'
    result = self.seeact.step(goal)
    done = result.done
    result_1 = result.data

    self.assertFalse(done)
    self.assertEqual(result_1['ui_elements'], mock_ui_elements)
    self.assertEqual(
        result_1['actionable_elements'][0].description,
        '"AnImage" image',
    )
    self.assertEqual(
        result_1['actionable_elements'][1].description,
        'a checkbox with the text "Unchecked" that is not checked',
    )
    self.assertEqual(result_1['action_gen_response'], 'Generated action 1')
    self.assertEqual(
        result_1['action_ground_response'],
        'ELEMENT: A\nACTION: CLICK\nVALUE: None',
    )
    self.assertIsInstance(result_1['seeact_action'], seeact_utils.SeeActAction)
    self.assertEqual(result_1['seeact_action'].action, 'CLICK')
    self.assertEqual(result_1['seeact_action'].element, 'A')
    self.assertEqual(result_1['seeact_action'].value, 'None')
    self.assertEqual(result_1['action_description'], '"AnImage" image -> CLICK')
    self.assertEqual(result_1['action'].action_type, 'click')
    self.assertEqual(result_1['action'].index, 0)
    self.assertEqual(self.seeact._actions, ['"AnImage" image -> CLICK'])

    result = self.seeact.step(goal)
    done = result.done
    result_2 = result.data

    self.assertTrue(done)
    self.assertEqual(result_2['ui_elements'], mock_ui_elements)
    self.assertEqual(
        result_2['actionable_elements'][0].description, '"AnImage" image'
    )
    self.assertEqual(
        result_2['actionable_elements'][1].description,
        'a checkbox with the text "Unchecked" that is not checked',
    )
    self.assertEqual(result_2['action_gen_response'], 'Generated action 2')
    self.assertEqual(
        result_2['action_ground_response'],
        'ELEMENT: None\nACTION: TERMINATE\nVALUE: None',
    )
    self.assertIsInstance(result_2['seeact_action'], seeact_utils.SeeActAction)
    self.assertEqual(result_2['seeact_action'].action, 'TERMINATE')
    self.assertIsNone(result_2['seeact_action'].element)
    self.assertEqual(result_2['seeact_action'].value, 'None')
    self.assertEqual(result_2['action_description'], 'TERMINATE')
    self.assertEqual(result_2['action'].action_type, 'status')
    self.assertEqual(result_2['action'].goal_status, 'task_complete')
    self.assertEqual(
        self.seeact._actions, ['"AnImage" image -> CLICK', 'TERMINATE']
    )


if __name__ == '__main__':
  absltest.main()
