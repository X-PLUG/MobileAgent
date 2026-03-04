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

"""Task for contacts apps."""

import random
import re
from absl import logging
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import contacts_validators
from android_world.utils import fuzzy_match_lib


class ContactsAddContact(contacts_validators.AddContact):
  """Task for adding a new contact."""

  complexity = 1.2
  app_names = ("contacts",)
  template = "Create a new contact for {name}. Their number is {number}."


def _contact_info_is_entered(
    first: str,
    last: str,
    phone: str,
    phone_label: str,
    ui_elements: list[representation_utils.UIElement],
) -> bool:
  """Checks if UI elements contain requested contact info.

  Specifically, it checks if current screen is the new contact screen and if the
  screen shows the function arguments filled out for the contacts.

  Args:
    first: First name.
    last: Last name.
    phone: Phone number.
    phone_label: Label for phone number.
    ui_elements: UI elements on screen.

  Returns:
    True if contact form is filled out.
  """
  first_name_element = None
  last_name_element = None
  phone_element = None
  phone_label_element = None

  # Mobile can appear twice on the screen, making this is a difficult edge case.
  # Do not use this input.
  assert phone_label != "Mobile"

  for element in ui_elements:
    if (
        element.text
        and fuzzy_match_lib.fuzzy_match(element.text, first)
        and element.hint_text == "First name"
    ):
      first_name_element = element
    elif (
        element.text
        and fuzzy_match_lib.fuzzy_match(element.text, last)
        and element.hint_text == "Last name"
    ):
      last_name_element = element
    elif element.text and (
        re.sub(r"\D", "", element.text) == re.sub(r"\D", "", phone)
        and element.hint_text == "Phone"
    ):
      phone_element = element
    elif (
        element.text == phone_label
        and element.content_description == phone_label + " Phone"
    ):
      phone_label_element = element

  # Content description may not be set properly, so we try to find the phone
  # label element by its text and its position if it is not found above
  if phone_element and phone_label_element is None:
    phone_label_element = _find_phone_label_element(
        phone_element, phone_label, ui_elements
    )
  if (
      first_name_element is None
      or last_name_element is None
      or phone_element is None
      or phone_label_element is None
  ):
    if first_name_element is None:
      logging.info("Missing 'first' UI element")
    if last_name_element is None:
      logging.info("Missing 'last' UI element")
    if phone_element is None:
      logging.info("Missing 'phone' UI element")
    if phone_label_element is None:
      logging.info("Missing 'phone_label' UI element")
    return False

  return True


def _find_phone_label_element(
    phone_element: representation_utils.UIElement,
    phone_label: str,
    ui_elements: list[representation_utils.UIElement],
) -> None | representation_utils.UIElement:
  for element in ui_elements:
    if element.text == phone_label and _vertically_adjacent(
        phone_element, element
    ):
      return element
  return None


def _vertically_adjacent(
    element1: representation_utils.UIElement,
    element2: representation_utils.UIElement,
) -> bool:
  if not element1.bbox_pixels or not element2.bbox_pixels:
    return False
  return (
      element1.bbox_pixels.y_max
      <= element2.bbox_pixels.y_min
      <= element1.bbox_pixels.y_max + element1.bbox_pixels.height
  )


class ContactsNewContactDraft(task_eval.TaskEval):
  """Task for entering contact info, but *not* hitting save."""

  app_names = ("contacts",)
  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "phone": {"type": "string"},
          "phone_label": {"type": "string"},
      },
      "required": ["first", "last", "phone", "phone_label"],
  }
  template = (
      "Go to the new contact screen and enter the following details: First"
      " Name: {first}, Last Name: {last}, Phone: {phone}, Phone Label:"
      " {phone_label}. Do NOT hit save."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    first_names = [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eva",
        "Frank",
        "Grace",
        "Hannah",
        "Ivan",
        "Jack",
    ]
    last_names = [
        "Johnson",
        "Smith",
        "Brown",
        "Taylor",
        "Adams",
        "Wilson",
        "Lee",
        "White",
        "Harris",
        "Clark",
    ]
    phone_labels = ["Home", "Work"]

    first = random.choice(first_names)
    last = random.choice(last_names)
    phone = (
        f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    )
    phone_label = random.choice(phone_labels)

    params = {
        "first": first,
        "last": last,
        "phone": phone,
        "phone_label": phone_label,
    }

    return params

  def is_successful(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    super().is_successful(env)
    ui_elements = representation_utils.forest_to_ui_elements(
        env.get_state().forest,
        exclude_invisible_elements=False,
    )
    return (
        1.0
        if _contact_info_is_entered(
            ui_elements=ui_elements,
            first=self.params["first"],
            last=self.params["last"],
            phone=self.params["phone"],
            phone_label=self.params["phone_label"],
        )
        else 0.0
    )
