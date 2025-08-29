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

"""Helper functions for the SeeAct experiment setup.

Sourced from https://github.com/OSU-NLP-Group/SeeAct.
"""

import base64
import dataclasses
import io
import os
import re
import string
from typing import Any
from absl import logging
from android_world.agents import infer
from android_world.env import json_action
from android_world.env import representation_utils
from IPython import display
from matplotlib.pylab import plt
import numpy as np
import PIL
import requests

# OpenAI model used for these experiments.
_GPT_TURBO = "gpt-4-turbo-2024-04-09"

VALID_ACTIONS = {
    "CLICK",
    "TERMINATE",
    "ANSWER",
    "LONG PRESS",
    "INPUT TEXT",
    "NAVIGATE HOME",
    "KEYBOARD ENTER",
    "NAVIGATE BACK",
    "SWIPE",
    "OPEN APP",
    "WAIT",
}
ACTIONS_WITHOUT_ELEMENT = {
    "KEYBOARD ENTER",
    "NAVIGATE HOME",
    "NAVIGATE BACK",
    "TERMINATE",
    "ANSWER",
    "OPEN APP",
    "WAIT",
}
ACTIONS_WITH_VALUE = {"INPUT TEXT", "SWIPE", "OPEN APP", "ANSWER"}


def generate_action_generation_prompt(
    task: str,
    question_description: str,
    previous_actions: list[str] | None = None,
) -> str:
  """Generate the first phase prompt for the SeeAct experiment setup.

  It focuses on the task description, previous actions and a question
  description without disruption from formatting or referring prompts.

  Args:
      task: The task description.
      question_description: A description of the question or task at hand.
      previous_actions: A list of previous actions taken.

  Returns:
      list: A list containing the system role and the generated query text.
  """
  query_text = "You are asked to complete the following task: " + task + "\n\n"
  previous_action_text = "Previous Actions:\n"
  if previous_actions is None:
    previous_actions = []
  for action_text in previous_actions:
    previous_action_text += action_text + "\n"
  query_text += previous_action_text + "\n" + question_description
  return query_text


def generate_grounding_prompt(
    referring_description: str = "",
    element_format: str = "",
    ui_element_choices: list[str] | None = None,
) -> str:
  """Generate a referring prompt that includes the element format, action format, and value format along with choices, if applicable, for the SeeAct experiment setup.

  Args:
      referring_description: Description on how to format the output.
      element_format: The format for specifying the element.
      ui_element_choices: A list of choices for the next action.

  Returns:
      The generated referring prompt.
  """
  referring_prompt = (
      referring_description + "\n\n" if referring_description else ""
  )

  if ui_element_choices:
    choice_text = format_action_options(ui_element_choices)
    referring_prompt += choice_text

  referring_prompt += f"{element_format}"

  return referring_prompt


def format_action_options(choices: list[str]) -> str:
  """Format the given choices into a structured option text for presentation in the prompt.

  Args:
    choices: A list of choices to be formatted.

  Returns:
    The formatted choices text.
  """
  option_text = ""
  for idx, choice in enumerate(choices):
    option_name = generate_multiple_choice(idx)
    option_text += f"{option_name}. {choice}\n"

  non_abcd = generate_multiple_choice(len(choices))
  option_text += (
      "If none of these elements match your target element, please select"
      f" {non_abcd}. None of the other options match the correct element.\n\n"
  )

  return option_text


def generate_multiple_choice(index: int) -> str:
  """Generate an option name based on the index.

  Args:
    index: The index of the option.

  Returns:
    The generated option name.
  """
  if index > 26 * 26:
    raise ValueError(f"Index {index} is greater than 26 * 26")

  if index < 26:
    return string.ascii_uppercase[index]
  else:
    first_letter_index = (index - 26) // 26
    second_letter_index = (index - 26) % 26
    first_letter = string.ascii_uppercase[first_letter_index]
    second_letter = string.ascii_uppercase[second_letter_index]
    return f"{first_letter}{second_letter}"


def create_action_generation_messages_payload(
    system_role_prompt: str,
    action_gen_prompt: str,
    image_array: np.ndarray,
) -> list[dict[str, Any]]:
  """Creates JSON input for action generation.

  Args:
    system_role_prompt: The general instructions to give to the agent.
    action_gen_prompt: Prompt for the specific task.
    image_array: Image of the current screen.

  Returns:
    JSON input for OpenAI API.
  """
  base64_image = infer.Gpt4Wrapper.encode_image(image_array)
  messages = [
      {
          "role": "system",
          "content": [{"type": "text", "text": system_role_prompt}],
      },
      {
          "role": "user",
          "content": [
              {"type": "text", "text": action_gen_prompt},
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}",
                      "detail": "high",
                  },
              },
          ],
      },
  ]

  return messages


def create_grounding_messages_payload(
    system_role_prompt: str,
    action_gen_prompt: str,
    image_array: np.ndarray,
    action_generation_output: str,
    action_grounding_prompt: str,
) -> list[dict[str, Any]]:
  """Creates JSON input for grounding.

  Args:
    system_role_prompt: The general instructions to give to the agent.
    action_gen_prompt: Prompt for generating the action.
    image_array: Image of the current screen.
    action_generation_output: Output from the action generation.
    action_grounding_prompt: Prompt for generating the grounding action.

  Returns:
    JSON input for OpenAI API.
  """
  base64_image = infer.Gpt4Wrapper.encode_image(image_array)
  messages = [
      {
          "role": "system",
          "content": [{"type": "text", "text": system_role_prompt}],
      },
      {
          "role": "user",
          "content": [
              {"type": "text", "text": action_gen_prompt},
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}",
                      "detail": "high",
                  },
              },
          ],
      },
      {
          "role": "assistant",
          "content": [
              {"type": "text", "text": f"\n\n{action_generation_output}"}
          ],
      },
      {
          "role": "user",
          "content": [{"type": "text", "text": action_grounding_prompt}],
      },
  ]

  return messages


def display_prompt(
    messages_payload: list[dict[str, Any]],
    ignore_system_prompt: bool = True,
    extra_text: str = "",
):
  """Displays content, image, and role for a web navigation task.

  For data viz purposes.

  Args:
    messages_payload: A dictionary containing the 'role', 'content', and image
      data.
    ignore_system_prompt: If True, skip printing out the system prompt.
    extra_text: Extra text to display.
  """

  for message in messages_payload:
    print(f"-------- Role: {message['role'].upper()} --------")  # Role Marker
    if ignore_system_prompt and message["role"] == "system":
      continue
    for content_item in message["content"]:
      if content_item["type"] == "text":
        print(content_item["text"])
      elif content_item["type"] == "image_url":
        base64_str = content_item["image_url"]["url"].replace(
            "data:image/jpeg;base64,", ""
        )
        image_data = base64.b64decode(base64_str)
        image = PIL.Image.open(io.BytesIO(image_data))

        image.thumbnail((512, 512))
        display.display(image)
      else:
        raise ValueError(
            f"Unknown content type: {content_item['type'].upper()}"
        )
  print(extra_text)


def execute_openai_request(
    messages_payload: list[dict[str, Any]],
    model: str = _GPT_TURBO,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
  """Executes a request to the OpenAI API with the given JSON input.

  Args:
    messages_payload: The JSON input created for action generation or grounding.
    model: The model to use for the request.
    temperature: Temperature setting for GPT's responses.
    max_tokens: Max number of output tokens.

  Returns:
    The response from the OpenAI API as a dictionary.
  """
  api_key = os.environ["OPENAI_API_KEY"]
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
  }
  payload = {
      "model": model,
      "messages": messages_payload,
      "temperature": temperature,
      "max_tokens": max_tokens,
  }

  response = requests.post(
      "https://api.openai.com/v1/chat/completions",
      headers=headers,
      json=payload,
  )

  return response.json()


@dataclasses.dataclass(frozen=True)
class SeeActAction:
  action: str
  element: str | None = None
  value: str | None = None


def _extract_text(value: str) -> str | None:
  if value is None or value == "None":
    return None
  return re.sub(r"[.]+|\s+", " ", value).strip()


class ParseActionError(ValueError):
  """Exception raised for errors in the parsing of an action."""

  pass


def _validate_action(element: str, action: str, value: str) -> None:
  """Validates the element, action, and value combination.

  Args:
    element: The element associated with the action.
    action: The action to validate.
    value: The value associated with the action.

  Raises:
    ValueError: If the action is invalid.
  """

  if action not in VALID_ACTIONS:
    raise ParseActionError(f"Invalid action: {action}")

  if action == "INPUT TEXT":
    if not value:
      raise ParseActionError("VALUE is required for INPUT TEXT action")
    if element is None:
      raise ParseActionError("ELEMENT is required for INPUT TEXT action")

  if action == "SWIPE" and value not in ["up", "down", "left", "right"]:
    raise ParseActionError(
        f'Invalid VALUE "{value}" for SWIPE action; must be up, down, left, or'
        " right."
    )

  if action == "OPEN APP" and not value:
    raise ParseActionError("VALUE is required for OPEN APP action")

  if action == "ANSWER" and not value:
    raise ParseActionError("VALUE is required for ANSWER action")

  if action in ["CLICK", "LONG PRESS"]:
    if element is None:
      raise ParseActionError(f"ELEMENT is required for {action} action")
    if value != "None":
      raise ParseActionError(f"VALUE should be 'None' for {action} action")

  if action in ACTIONS_WITHOUT_ELEMENT and value != "None":
    logging.error(
        "VALUE should be 'None' for %s action, but got %s", action, value
    )


def extract_element_action_value(lines: list[str]) -> SeeActAction:
  """Extracts the element, action, and value from the given lines.

  # Example.
  lines = ["ELEMENT: A", "ACTION: ENTER_TEXT", "VALUE: Hello, World!"]
  element, action, value = extract_element_action_value(lines)
  print(element)  # Output: "A"
  print(action)   # Output: "ENTER_TEXT"
  print(value)    # Output: "Hello, World!"

  Args:
    lines: A list of strings representing the lines to extract from. Each line
      should start with "ELEMENT:", "ACTION:", or "VALUE:" followed by the
      corresponding value.

  Returns:
    A tuple containing the extracted element, action, and value as strings.

  Raises:
    ValueError: If the element, action, or value is missing or None.

  Example:
  """
  element, action, value = None, None, None
  for line in lines:
    if line.startswith("ELEMENT:"):
      element = _extract_text(line.split(":")[1])
      if element == "None":
        element = None
    elif line.startswith("ACTION:"):
      action = _extract_text(line.split(":")[1])
    elif line.startswith("VALUE:"):
      value = line.split(":")[1].strip().strip(".")

  _validate_action(element, action, value)
  return SeeActAction(action=action, element=element, value=value)


@dataclasses.dataclass
class SeeActElement:
  description: str
  ui_element: representation_utils.UIElement | None = None
  index: int = -1
  abc_index: str = ""


def format_and_filter_elements(
    ui_elements: list[representation_utils.UIElement],
) -> list[SeeActElement]:
  """Formats and filters UI elements."""
  filtered = [
      SeeActElement(
          description=_get_element_description(ui_element),
          ui_element=ui_element,
      )
      for ui_element in ui_elements
      if _valid_element(ui_element)
  ]
  for i, element in enumerate(filtered):
    element.index = i
    element.abc_index = generate_multiple_choice(i)
  return filtered


def _valid_element(ui_element: representation_utils.UIElement) -> bool:
  """Returns true if element is valid."""

  if not ui_element.class_name:
    return False

  if (
      ui_element.text
      or ui_element.content_description
      or ui_element.hint_text
      or ui_element.resource_name
  ):
    return True
  else:
    return False


def _get_element_description(
    element: representation_utils.UIElement,
) -> str:
  """Produces a concise description of UI element given its properties.

  If it is not possible to describe a UI element at all, None will be returned.

  Args:
    element: The element to generate a description for.

  Returns:
    description: The description.  If it was not possible to generate a
      description for the element, None will be returned.
  """
  if not element.class_name:
    return "Unknown"
  class_string = element.class_name.split(".")[-1].lower()

  if "radiobutton" in class_string:
    return _describe_checkable_element(element, "radio button", "selected")
  elif "button" in class_string:
    return _describe_typed_element(element, "button")
  elif "image" in class_string:
    return _describe_typed_element(element, "image")
  elif "text" in class_string:
    if element.is_editable:
      return _describe_text_box(element)
    else:
      return _describe_typed_element(element, "icon")
  elif "switch" in class_string:
    return _describe_switch(element)
  elif "checkbox" in class_string:
    return _describe_checkable_element(element, "checkbox", "checked")
  else:
    return _describe_typed_element(element, "icon")


def _describe_text_box(element: representation_utils.UIElement) -> str:
  """Describes a text box."""
  if element.hint_text:
    description = 'a "{hint_text}" text box'.format(hint_text=element.hint_text)
  elif element.content_description:
    description = 'a "{content_description}" text box'.format(
        content_description=element.content_description
    )
  else:
    description = "a text box"

  if element.text:
    description += ' with the text "{text}"'.format(text=element.text)
  else:
    description += " that is empty"

  return description


def _describe_switch(element: representation_utils.UIElement) -> str:
  """Describes a switch."""
  if element.text:
    description = 'a switch with the text "{text}"'.format(text=element.text)
  elif element.hint_text:
    description = 'a "{hint_text}" switch'.format(hint_text=element.hint_text)
  elif element.content_description:
    description = 'a "{content_description}" switch'.format(
        content_description=element.content_description
    )
  else:
    description = "a switch"

  if element.is_checked:
    description += " that is checked"
  else:
    description += " that is not checked"

  return description


def _describe_checkable_element(
    element: representation_utils.UIElement,
    element_type: str,
    check_description: str,
) -> str:
  """Describes a checkable element.

  Examples of checkable elements are check boxes and radio buttons.

  Args:
    element: The element to describe.
    element_type: A generic description for the type of element (e.g.,
      checkbox).
    check_description: The word to use to describe when the element is checked
      (e.g., checked for a check box or selected for a radio button).

  Returns:
    The generated description.
  """
  if element.text:
    description = 'a {element_type} with the text "{text}"'.format(
        element_type=element_type, text=element.text
    )
  elif element.hint_text:
    description = 'a "{hint_text}" {element_type}'.format(
        element_type=element_type, hint_text=element.hint_text
    )
  elif element.content_description:
    description = 'a "{content_description}" {element_type}'.format(
        element_type=element_type,
        content_description=element.content_description,
    )
  else:
    description = "a {element_type}".format(element_type=element_type)

  if element.is_checked:
    description += " that is {:s}".format(check_description)
  else:
    description += " that is not {:s}".format(check_description)

  return description


def _describe_typed_element(
    element: representation_utils.UIElement, element_type: str
) -> str:
  """Provides a description of the form <details><type> for an element.

  Details can be pulled from the text, content_description or hint_text of
  the element.

  Args:
    element: The UI element to describe.
    element_type: How the type of the element should be described (e.g., button)

  Returns:
    The description.
  """
  if element.text:
    return '"{text}" {element_type}'.format(
        text=element.text, element_type=element_type
    )
  elif element.content_description:
    return '"{content_description}" {element_type}'.format(
        content_description=element.content_description,
        element_type=element_type,
    )
  elif element.hint_text:
    return '"{hint_text}" {element_type}'.format(
        hint_text=element.hint_text, element_type=element_type
    )
  elif element.resource_name:
    return '"{resource_name}" {element_type}'.format(
        resource_name=element.resource_name, element_type=element_type
    )
  else:
    return element_type


def get_referred_element(
    action: SeeActAction, elements: list[SeeActElement]
) -> SeeActElement | None:
  """Gets the referred element from the action.

  Args:
    action: The action to get the referred element from.
    elements: The list of elements to search through.

  Returns:
    The referred element, or None if it could not be found.

  Raises:
    ValueError: If the referred element does not exist.
  """
  if action.element is None:
    return None
  if action.action not in ("CLICK", "LONG PRESS", "INPUT TEXT", "SWIPE"):
    return None

  for element in elements:
    if element.abc_index == action.element:
      return element

  return None


def convert_seeact_action_to_json_action(
    action: SeeActAction, elements: list[SeeActElement]
) -> json_action.JSONAction:
  """Converts a SeeActAction object to a JSONAction object.

  Args:
      action: The SeeActAction object to convert.
      elements: UI elements.

  Returns:
      The corresponding JSONAction object.

  Raises:
    ParseActionError: If cannot convert action.
  """
  action_type_mapping = {
      "CLICK": json_action.CLICK,
      "TERMINATE": json_action.STATUS,
      "ANSWER": json_action.ANSWER,
      "LONG PRESS": json_action.LONG_PRESS,
      "INPUT TEXT": json_action.INPUT_TEXT,
      "NAVIGATE HOME": json_action.NAVIGATE_HOME,
      "KEYBOARD ENTER": json_action.KEYBOARD_ENTER,
      "NAVIGATE BACK": json_action.NAVIGATE_BACK,
      "SWIPE": json_action.SCROLL,
      "WAIT": json_action.WAIT,
      "OPEN APP": json_action.OPEN_APP,
  }
  action_type = action_type_mapping[action.action.upper()]

  index = None
  text = None
  direction = None
  goal_status = None
  app_name = None

  if action_type == json_action.INPUT_TEXT:
    text = action.value
  elif action_type == json_action.SCROLL:
    direction = _swipe_to_scroll(action.value)
  elif action_type == json_action.OPEN_APP:
    app_name = action.value
  elif action_type == json_action.ANSWER:
    text = action.value
  elif action_type == json_action.STATUS:
    goal_status = "task_complete"

  target_element = get_referred_element(action, elements)
  if target_element is None and action_type in [
      json_action.CLICK,
      json_action.LONG_PRESS,
      json_action.INPUT_TEXT,
  ]:
    raise ParseActionError(
        f"Action type is {action_type}, but received no target element or"
        " incorrect target element."
    )
  if target_element is not None:
    index = target_element.index

  return json_action.JSONAction(
      action_type=action_type,
      index=index,
      text=text,
      direction=direction,
      goal_status=goal_status,
      app_name=app_name,
  )


def _swipe_to_scroll(seeact_direction: str) -> str:
  """Maps SeeAct scroll direction to JSONAction scroll direction."""
  mapping = {
      "up": "down",
      "down": "up",
      "left": "right",
      "right": "left",
  }
  return mapping.get(seeact_direction.lower(), "unknown")


def generate_action_description(
    seeact_action: SeeActAction, element: SeeActElement
) -> str:
  """Generates a natural language description of the action."""

  if seeact_action.action in ACTIONS_WITHOUT_ELEMENT or (
      seeact_action.action == "SWIPE" and element is None
  ):
    new_action = seeact_action.action
  else:
    new_action = f"{element.description} -> {seeact_action.action}"

  if seeact_action.action in ACTIONS_WITH_VALUE:
    new_action += f": {seeact_action.value}"

  return new_action


def plot_to_html_img(figure) -> str:
  """Convert a matplotlib figure to an HTML img tag."""
  # Save the plot to a BytesIO object
  buf = io.BytesIO()
  figure.savefig(buf, format="png", bbox_inches="tight", dpi=200)
  buf.seek(0)
  # Encode the image in base64 to embed in HTML
  image_base64 = base64.b64encode(buf.read()).decode("utf-8")
  buf.close()
  return f'<img src="data:image/png;base64,{image_base64}" width="800" />'


def _generate_episode_html(episode: dict[str, Any], num: int) -> str:
  """Generate an HTML report from screenshot data and action descriptions for a single episode."""
  data = episode["episode_data"]
  title = episode["task_template"] + " - " + episode["goal"]
  success_text = (
      "<span style='color:green;'>SUCCESS</span>"
      if episode["is_successful"]
      else "<span style='color:red;'>FAILURE</span>"
  )
  header_details = (
      f"Runtime: {int(episode['run_time'])} seconds, "
      f"Episode Length: {episode['episode_length']} steps"
  )
  html_str = f"<h1>{num} {success_text} {title}</h1><h2>{header_details}</h2>"

  for i, (screenshot, description) in enumerate(
      zip(data["screenshot"], data["action_description"])
  ):
    fig, ax = plt.subplots()
    ax.imshow(screenshot)
    ax.axis("off")  # Turn off axis
    ax.set_title(f"TASK {i}: {description}", fontsize=12, color="blue")
    html_str += "<div>" + plot_to_html_img(fig) + "</div>"
    plt.close(fig)  # Close the plot to free up memory

  return html_str


def generate_full_report(
    episodes: list[dict[str, Any]], run_id: str = ""
) -> str:
  """Generate a comprehensive HTML report for multiple episodes."""
  html_str = f"<html><head><title>SeeAct: {run_id}</title></head><body>"
  html_str += "<h1>SeeAct Report</h1>"

  for i, episode in enumerate(episodes):
    if episode.get("exception_info") is not None:
      print(f"Skipping {i}; run failed.")
    else:
      html_str += _generate_episode_html(episode, i) + "<hr>"
    if i % 10 == 0:
      print(i)

  html_str += "</body></html>"
  return html_str
