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

from absl.testing import absltest
from android_world.agents import seeact_utils
from android_world.env import json_action
from android_world.env import representation_utils


class TestGenerateActionGenerationPrompt(absltest.TestCase):

  def test_generate_action_generation_prompt(self):
    task = "Test task"
    question_description = "Test question"
    previous_actions = ["Element A -> CLICK", "TERMINATE"]
    expected_prompt = (
        "You are asked to complete the following task: Test task\n\n"
        "Previous Actions:\n"
        "Element A -> CLICK\n"
        "TERMINATE\n\n"
        "Test question"
    )
    actual_prompt = seeact_utils.generate_action_generation_prompt(
        task, question_description, previous_actions
    )
    self.assertEqual(actual_prompt, expected_prompt)


class TestExtractText(absltest.TestCase):

  def test_extract_text_with_valid_value(self):
    self.assertEqual(
        seeact_utils._extract_text("  Hello, World!  "), "Hello, World!"
    )

  def test_extract_text_with_none_value(self):
    self.assertIsNone(seeact_utils._extract_text(None))

  def test_extract_text_with_none_string(self):
    self.assertIsNone(seeact_utils._extract_text("None"))


class TestExtractElementActionValue(absltest.TestCase):

  def test_extract_element_action_value_with_valid_lines(self):
    lines = ["ELEMENT: A", "ACTION: INPUT TEXT", "VALUE: Hello, World!"]
    expected_result = seeact_utils.SeeActAction(
        element="A", action="INPUT TEXT", value="Hello, World!"
    )
    self.assertEqual(
        seeact_utils.extract_element_action_value(lines), expected_result
    )

  def test_extract_element_action_value_with_missing_element(self):
    lines = ["ACTION: INPUT TEXT", "VALUE: Hello, World!"]
    with self.assertRaisesRegex(
        ValueError, "ELEMENT is required for INPUT TEXT action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_value(self):
    lines = ["ELEMENT: A", "ACTION: INPUT TEXT"]
    with self.assertRaisesRegex(
        ValueError, "VALUE is required for INPUT TEXT action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_invalid_action(self):
    lines = ["ELEMENT: A", "ACTION: INVALID_ACTION", "VALUE: Hello, World!"]
    with self.assertRaisesRegex(ValueError, "Invalid action: INVALID_ACTION"):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_input_text_value(self):
    lines = ["ELEMENT: A", "ACTION: INPUT TEXT"]
    with self.assertRaisesRegex(
        ValueError, "VALUE is required for INPUT TEXT action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_input_text_element(self):
    lines = ["ACTION: INPUT TEXT", "VALUE: Hello, World!"]
    with self.assertRaisesRegex(
        ValueError, "ELEMENT is required for INPUT TEXT action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_invalid_scroll_value(self):
    lines = ["ACTION: SWIPE", "VALUE: invalid_direction"]
    with self.assertRaisesRegex(
        seeact_utils.ParseActionError,
        'Invalid VALUE "invalid_direction" for SWIPE action; must be up, down,'
        " left, or right.",
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_open_app_value(self):
    lines = ["ACTION: OPEN APP"]
    with self.assertRaisesRegex(
        ValueError, "VALUE is required for OPEN APP action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_answer_value(self):
    lines = ["ACTION: ANSWER"]
    with self.assertRaisesRegex(
        ValueError, "VALUE is required for ANSWER action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_click_element(self):
    lines = ["ACTION: CLICK", "VALUE: None"]
    with self.assertRaisesRegex(
        ValueError, "ELEMENT is required for CLICK action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_missing_long_press_element(self):
    lines = ["ACTION: LONG PRESS", "VALUE: None"]
    with self.assertRaisesRegex(
        ValueError, "ELEMENT is required for LONG PRESS action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_invalid_click_value(self):
    lines = ["ELEMENT: A", "ACTION: CLICK", "VALUE: Invalid"]
    with self.assertRaisesRegex(
        ValueError, "VALUE should be 'None' for CLICK action"
    ):
      seeact_utils.extract_element_action_value(lines)

  def test_extract_element_action_value_with_invalid_long_press_value(self):
    lines = ["ELEMENT: A", "ACTION: LONG PRESS", "VALUE: Invalid"]
    with self.assertRaisesRegex(
        ValueError, "VALUE should be 'None' for LONG PRESS action"
    ):
      seeact_utils.extract_element_action_value(lines)


class TestFormatAndFilterElements(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ui_elements = [
        representation_utils.UIElement(
            text="Button 1", class_name="android.widget.Button"
        ),
        representation_utils.UIElement(
            text=None,
            content_description="AnImage",
            class_name="android.widget.ImageView",
        ),
        representation_utils.UIElement(
            text="Layout", class_name="android.widget.LinearLayout"
        ),
        representation_utils.UIElement(
            text="Unchecked",
            class_name="android.widget.CheckBox",
            is_checked=False,
        ),
        representation_utils.UIElement(
            text="Checked",
            class_name="android.widget.CheckBox",
            is_checked=True,
        ),
        representation_utils.UIElement(
            text="Selected",
            class_name="android.widget.RadioButton",
            is_checked=True,
        ),
        representation_utils.UIElement(
            text="abc", class_name="android.widget.Switch", is_checked=True
        ),
    ]

  def test_format_and_filter_elements(self):
    expected_output = [
        '"Button 1" button',
        '"AnImage" image',
        '"Layout" icon',
        'a checkbox with the text "Unchecked" that is not checked',
        'a checkbox with the text "Checked" that is checked',
        'a radio button with the text "Selected" that is selected',
        'a switch with the text "abc" that is checked',
    ]

    ui_elements = seeact_utils.format_and_filter_elements(self.ui_elements)
    descriptions = [e.description for e in ui_elements]

    self.assertEqual(
        descriptions,
        expected_output,
    )

  def test_empty_input(self):
    self.assertEqual(
        seeact_utils.format_and_filter_elements([]),
        [],
    )

  def test_no_class_name(self):
    no_class_element = representation_utils.UIElement(
        class_name=None, text="some text"
    )
    self.assertEqual(
        seeact_utils.format_and_filter_elements([no_class_element]),
        [],
    )


class TestGenerateActionDescription(absltest.TestCase):

  def test_click_action(self):
    action = seeact_utils.SeeActAction(action="CLICK", value="")
    element = seeact_utils.SeeActElement(description="button")
    expected = "button -> CLICK"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_input_text_action(self):
    action = seeact_utils.SeeActAction(action="INPUT TEXT", value="Hello World")
    element = seeact_utils.SeeActElement(description="search bar")
    expected = "search bar -> INPUT TEXT: Hello World"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_open_app_action(self):
    action = seeact_utils.SeeActAction(action="OPEN APP", value="Maps")
    element = seeact_utils.SeeActElement(description="app drawer")
    expected = "OPEN APP: Maps"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_long_press_action(self):
    action = seeact_utils.SeeActAction(action="LONG PRESS", value="")
    element = seeact_utils.SeeActElement(description="image")
    expected = "image -> LONG PRESS"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_navigate_home_action(self):
    action = seeact_utils.SeeActAction(action="NAVIGATE HOME", value="")
    element = seeact_utils.SeeActElement(description="None")
    expected = "NAVIGATE HOME"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_keyboard_enter_action(self):
    action = seeact_utils.SeeActAction(action="KEYBOARD ENTER", value="")
    element = seeact_utils.SeeActElement(description="None")
    expected = "KEYBOARD ENTER"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_navigate_back_action(self):
    action = seeact_utils.SeeActAction(action="NAVIGATE BACK", value="")
    element = seeact_utils.SeeActElement(description="None")
    expected = "NAVIGATE BACK"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_scroll_action(self):
    action = seeact_utils.SeeActAction(action="SWIPE", value="down")
    element = None
    expected = "SWIPE: down"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_scroll_action_with_target(self):
    action = seeact_utils.SeeActAction(action="SWIPE", value="down")
    element = seeact_utils.SeeActElement(description="app drawer")
    expected = "app drawer -> SWIPE: down"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_terminate_action(self):
    action = seeact_utils.SeeActAction(action="TERMINATE", value="")
    element = seeact_utils.SeeActElement(description="None")
    expected = "TERMINATE"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )

  def test_answer_action(self):
    action = seeact_utils.SeeActAction(action="ANSWER", value="42")
    element = seeact_utils.SeeActElement(description="None")
    expected = "ANSWER: 42"
    self.assertEqual(
        seeact_utils.generate_action_description(action, element), expected
    )


class TestConvertSeeActActionToJSONAction(absltest.TestCase):

  def test_click_action_with_valid_element(self):
    action = seeact_utils.SeeActAction(action="CLICK", element="B")
    elements = [
        seeact_utils.SeeActElement(
            description="Button 1", abc_index="A", index=0
        ),
        seeact_utils.SeeActElement(
            description="Button 2", abc_index="B", index=1
        ),
    ]
    expected_json_action = json_action.JSONAction(action_type="click", index=1)

    result = seeact_utils.convert_seeact_action_to_json_action(action, elements)

    self.assertEqual(result, expected_json_action)

  def test_click_action_with_invalid_element(self):
    action = seeact_utils.SeeActAction(action="CLICK", element="D")
    elements = [
        seeact_utils.SeeActElement(
            description="Button 1", abc_index="A", index=0
        ),
        seeact_utils.SeeActElement(
            description="Button 2", abc_index="B", index=1
        ),
    ]

    with self.assertRaisesRegex(
        seeact_utils.ParseActionError,
        "Action type is click, but received no target element or"
        " incorrect target element.",
    ):
      seeact_utils.convert_seeact_action_to_json_action(action, elements)

  def test_long_press_action_with_valid_element(self):
    action = seeact_utils.SeeActAction(action="LONG PRESS", element="A")
    elements = [
        seeact_utils.SeeActElement(
            description="Button 1", abc_index="A", index=0
        ),
        seeact_utils.SeeActElement(
            description="Button 2", abc_index="B", index=1
        ),
    ]
    expected_json_action = json_action.JSONAction(
        action_type="long_press", index=0
    )

    result = seeact_utils.convert_seeact_action_to_json_action(action, elements)

    self.assertEqual(result, expected_json_action)

  def test_input_text_action_with_valid_element(self):
    action = seeact_utils.SeeActAction(
        action="INPUT TEXT", element="B", value="Hello, world!"
    )
    elements = [
        seeact_utils.SeeActElement(
            description="Input 1", abc_index="A", index=0
        ),
        seeact_utils.SeeActElement(
            description="Input 2", abc_index="B", index=1
        ),
    ]
    expected_json_action = json_action.JSONAction(
        action_type="input_text", index=1, text="Hello, world!"
    )

    result = seeact_utils.convert_seeact_action_to_json_action(action, elements)

    self.assertEqual(result, expected_json_action)

  def test_terminate_action(self):
    action = seeact_utils.SeeActAction(action="TERMINATE")
    elements = []
    expected_json_action = json_action.JSONAction(
        action_type="status", goal_status="task_complete"
    )

    result = seeact_utils.convert_seeact_action_to_json_action(action, elements)

    self.assertEqual(result, expected_json_action)

  def test_answer_action(self):
    action = seeact_utils.SeeActAction(action="ANSWER", value="42")
    elements = []
    expected_json_action = json_action.JSONAction(
        action_type="answer", text="42"
    )

    result = seeact_utils.convert_seeact_action_to_json_action(action, elements)

    self.assertEqual(result, expected_json_action)

  def test_scroll_action(self):
    action = seeact_utils.SeeActAction(action="SWIPE", value="up")
    elements = []
    expected_json_action = json_action.JSONAction(
        action_type="scroll", direction="down"
    )

    result = seeact_utils.convert_seeact_action_to_json_action(action, elements)

    self.assertEqual(result, expected_json_action)


if __name__ == "__main__":
  absltest.main()
