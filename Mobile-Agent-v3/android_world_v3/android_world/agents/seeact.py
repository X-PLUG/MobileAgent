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

"""SeeAct agent for Android."""

from typing import Any

from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action

SEEACT_ONLINE_SYS_PROMPT = """Imagine that you are imitating humans operating an Android device for a task step by step. At each stage, you can see the Android screen like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can tap on an element, long-press an element, swipe, input text, open an app, or use the keyboard enter, home, or back key. (For your understanding, they are like `adb shell input tap`, `adb shell input swipe`, `adb shell input text`, `adb shell am start -n`, and `adb shell input keyevent`). One next step means one operation within these actions. Unlike humans, for typing (e.g., in text areas, text boxes), you should try directly typing the input or selecting the choice, bypassing the need for an initial click. You should not attempt to create accounts, log in or do the final submission. Terminate when you deem the task complete or if it requires potentially harmful actions."""

SEEACT_ONLINE_QUESTION_DESCRIPTION_NEW_EXP4 = """The screenshot below shows the Android screen you see. Follow the following guidance to think step by step before outlining the next action step at the current stage:

(Current Screen Identification)
Firstly, think about what the current screen is.

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step. Specifically, if the last action involved a INPUT TEXT, always evaluate whether it necessitates a confirmation step, because typically a single INPUT TEXT action does not make effect. (often, simply pressing 'Enter', assuming the default element involved in the last action, unless other clear elements are present for operation).

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the screen to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the screen to understand what you have done.

(Next Action Based on Android screen and Analysis)
Then, based on your analysis, in conjunction with human phone operation habits and the logic of app design, decide on the following action. And clearly outline which element on the Android screen users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules:
1. You should only issue a valid action given the current observation.
2. You should only issue one action at a time
3. For handling the select dropdown elements on a screen, it's not necessary for you to provide completely accurate options right now. The full list of options for these elements will be supplied later."""


SEEACT_CHOICE_PROMPT_DICT = {
    "system_prompt": SEEACT_ONLINE_SYS_PROMPT,
    "question_description": SEEACT_ONLINE_QUESTION_DESCRIPTION_NEW_EXP4,
    "referring_description": """(Reiteration)
First, reiterate your next target element, its detailed location, and the corresponding operation.

(Multichoice Question)
Below is a multi-choice question, where the choices are elements on the screen. All elements are arranged in the order based on their height on the screen, from top to bottom (and from left to right). This arrangement can be used to locate them. From the screenshot, find out where and what each one is on the screen, taking into account both their text content and details. Then, determine whether one matches your target element. Please examine the choices one by one. Choose the matching one. If multiple options match your answer, choose the most likely one by re-examining the screenshot, the choices, and your further reasoning. If you would like to perform a swipe action, you can optionally select the choice where you will swipe.""",
    "element_format": """(Final Answer)
Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.

Format:

ELEMENT: The uppercase letter of your choice. (No need for **ACTIONS_WITHOUT_ELEMENT**; and optional for SWIPE.)

ACTION: Choose an action from {**VALID_ACTIONS**}.

VALUE: Provide additional input based on ACTION.

The VALUE means:
If ACTION == INPUT TEXT, specify the text to be typed.
If ACTION == SWIPE, specify the direction: up, down, left, right.
If ACTION == OPEN APP, provide the name of the app to be opened.
If ACTION == ANSWER, specify the text of your answer to respond directly to a question or request for information.
For CLICK, LONG PRESS, KEYBOARD ENTER, NAVIGATE HOME, NAVIGATE BACK, WAIT, and TERMINATE, write "None".""".replace(
        "**ACTIONS_WITHOUT_ELEMENT**",
        ", ".join(seeact_utils.ACTIONS_WITHOUT_ELEMENT),
    ).replace(
        "**VALID_ACTIONS**", ", ".join(seeact_utils.VALID_ACTIONS)
    ),
}


def generate_seeact_prompts(
    task: str,
    previous_actions: list[str] | None = None,
    ui_element_choices: list[Any] | None = None,
    additional_guidelines: list[str] | None = None,
) -> tuple[str, str, str]:
  """Generates prompts for the SeeAct setup.

  Args:
      task: Description of the task to be performed.
      previous_actions: A list of actions previously taken.
      ui_element_choices: A list of choices available for the next action,
        derived from the accessibility tree.
      additional_guidelines: Task specific guidelines.

  Returns:
      A list of strings forming the complete prompt for the SeeAct task.
  """
  system_prompt_input = SEEACT_CHOICE_PROMPT_DICT["system_prompt"]
  question_description_input = SEEACT_CHOICE_PROMPT_DICT["question_description"]
  referring_input = SEEACT_CHOICE_PROMPT_DICT["referring_description"]
  element_format_input = SEEACT_CHOICE_PROMPT_DICT["element_format"]

  if additional_guidelines is not None:
    for guideline in additional_guidelines:
      system_prompt_input += f" {guideline}"

  return (
      system_prompt_input,
      seeact_utils.generate_action_generation_prompt(
          task,
          question_description_input,
          previous_actions=previous_actions,
      ),
      seeact_utils.generate_grounding_prompt(
          referring_description=referring_input,
          element_format=element_format_input,
          ui_element_choices=ui_element_choices,
      ),
  )


class SeeAct(base_agent.EnvironmentInteractingAgent):
  """SeeAct agent for Android."""

  def __init__(self, env: interface.AsyncEnv, name: str = "SeeAct"):
    super().__init__(env, name)
    self._actions = []
    self.additional_guidelines = None

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self._actions.clear()

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def step(
      self, goal: str, verbose: bool = True
  ) -> base_agent.AgentInteractionResult:
    result = {
        "ui_elements": None,
        "screenshot": None,
        "actionable_elements": None,
        "action_gen_payload": None,
        "action_gen_response": None,
        "action_ground_payload": None,
        "action_ground_response": None,
        "seeact_action": None,
        "action": None,
        "action_description": None,
    }
    state = self.get_post_transition_state()
    result["ui_elements"] = state.ui_elements
    result["screenshot"] = state.pixels
    actionable_elements = seeact_utils.format_and_filter_elements(
        state.ui_elements
    )
    result["actionable_elements"] = actionable_elements
    descriptions = [e.description for e in actionable_elements]
    sys_prompt, action_gen_prompt, action_ground_prompt = (
        generate_seeact_prompts(
            task=goal,
            previous_actions=self._actions,
            ui_element_choices=descriptions,
            additional_guidelines=self.additional_guidelines,
        )
    )

    # Action generation.
    payload = seeact_utils.create_action_generation_messages_payload(
        sys_prompt, action_gen_prompt, state.pixels
    )
    result["action_gen_payload"] = payload
    response = seeact_utils.execute_openai_request(payload)
    action_gen_response = response["choices"][0]["message"]["content"]
    result["action_gen_response"] = action_gen_response
    if verbose:
      (
          seeact_utils.display_prompt(
              result["action_gen_payload"],
              extra_text="\n~~~ANSWER~~~:" + action_gen_response,
          )
      )

    # Grounding.
    payload = seeact_utils.create_grounding_messages_payload(
        sys_prompt,
        action_gen_prompt,
        state.pixels,
        action_gen_response,
        action_ground_prompt,
    )
    result["action_ground_payload"] = payload
    response = seeact_utils.execute_openai_request(payload)
    action_ground_response = response["choices"][0]["message"]["content"]
    result["action_ground_response"] = action_ground_response

    # Parse action and convert to JSONAction.
    try:
      action_ground_response = result["action_ground_response"]
      seeact_action = seeact_utils.extract_element_action_value(
          action_ground_response.split("\n")
      )
      action = seeact_utils.convert_seeact_action_to_json_action(
          seeact_action, actionable_elements
      )
      result["seeact_action"] = seeact_action
      result["action"] = action
    except seeact_utils.ParseActionError as e:
      action_description = f"No Operation with error: {e}"
      action = json_action.JSONAction(action_type=json_action.UNKNOWN)
      result["seeact_action"] = None
      result["action"] = action
    else:
      target_element = seeact_utils.get_referred_element(
          seeact_action, actionable_elements
      )
      action_description = seeact_utils.generate_action_description(
          seeact_action, target_element
      )
      actuation.execute_adb_action(
          action,
          [e.ui_element for e in actionable_elements],
          self.env.logical_screen_size,
          self.env.controller,
      )

    result["action_description"] = action_description
    self._actions.append(action_description)

    if verbose:
      print("=" * 80)
      (
          seeact_utils.display_prompt(
              result["action_ground_payload"],
              extra_text="\n\n~~~~~~~~~ANSWER~~~~~~~~~:"
              + action_description
              + "\n\n",
          )
      )
      print("=" * 80)

    return base_agent.AgentInteractionResult(
        done=action.action_type == json_action.STATUS,
        data=result,
    )
