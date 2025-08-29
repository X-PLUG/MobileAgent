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

"""T3A: Text-only Autonomous Agent for Android."""

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message), like'
    ' user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by performing'
    ' actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step. At'
    ' each step, a list of descriptions for most UI elements on the'
    ' current screen will be given to you (each element can be specified by an'
    ' index), together with a history of what you have done in previous steps.'
    ' Based on these pieces of information and the goal, you must choose to'
    ' perform one of the action in the following list (action description'
    ' followed by the JSON format) by outputing the action in the correct JSON'
    ' format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    '- If you think the task is not'
    " feasible (including cases like you don't have enough information or can"
    ' not perform some necessary actions), finish by using the `status` action'
    ' with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into an editable text field (specified by its index), this'
    ' action contains clicking the text field, typing in the text and pressing'
    ' the enter, so no need to click on the target field to start:'
    ' `{{"action_type": "input_text", "text": <text_input>, "index":'
    ' <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not installed):'
    ' `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)

GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history), try to"
    ' switch to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations\n'
    '- Normally to select some text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)

SUMMARIZATION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is:{goal}\n'
    'Now I want you to summerize the latest step based on the action you'
    ' pick with the reason and descriptions for the before and after (the'
    ' action) screenshots.\n'
    'Here is the description for the before'
    ' screenshot:\n{before_elements}\n'
    'Here is the description for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    '\nBy comparing the descriptions for the two screenshots and the action'
    ' performed, give a brief summary of this step.'
    ' This summary will be added to action history and used in future action'
    ' selection, so try to include essential information you think that will'
    ' be most useful for future action selection like'
    ' what you intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might not be'
    ' correct), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short and in one line.\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


def _generate_ui_elements_description_list_full(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate description for a list of UIElement using full information.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: Logical screen size.

  Returns:
    Information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info


def _action_selection_prompt(
    goal: str,
    history: list[str],
    ui_elements_description: str,
    additional_guidelines: list[str] | None = None,
) -> str:
  """Generate the prompt for the action selection.

  Args:
    goal: The current task goal.
    history: Summaries for previous steps.
    ui_elements_description: A list of descriptions for the UI elements.
    additional_guidelines: Task specific guidelines.

  Returns:
    The text prompt for action selection that will be sent to gpt4v.
  """
  if history:
    history = '\n'.join(history)
  else:
    history = 'You just started, no action has been performed yet.'

  extra_guidelines = ''
  if additional_guidelines:
    extra_guidelines = 'For The Current Task:\n'
    for guideline in additional_guidelines:
      extra_guidelines += f'- {guideline}\n'

  return ACTION_SELECTION_PROMPT_TEMPLATE.format(
      history=history,
      goal=goal,
      ui_elements_description=ui_elements_description
      if ui_elements_description
      else 'Not available',
      additional_guidelines=extra_guidelines,
  )


def _summarize_prompt(
    goal: str,
    action: str,
    reason: str,
    before_elements: str,
    after_elements: str,
) -> str:
  """Generate the prompt for the summarization step.

  Args:
    goal: The overall goal.
    action: The action picked for the step.
    reason: The reason why pick the action.
    before_elements: Information for UI elements on the before screenshot.
    after_elements: Information for UI elements on the after screenshot.

  Returns:
    The text prompt for summarization that will be sent to gpt4v.
  """
  return SUMMARIZATION_PROMPT_TEMPLATE.format(
      goal=goal,
      action=action,
      reason=reason,
      before_elements=before_elements if before_elements else 'Not available',
      after_elements=after_elements if after_elements else 'Not available',
  )


class T3A(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size

    ui_elements = state.ui_elements
    before_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements

    action_prompt = _action_selection_prompt(
        goal,
        [
            'Step ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ],
        before_element_list,
        self.additional_guidelines,
    )
    step_data['action_prompt'] = action_prompt
    action_output, is_safe, raw_response = self.llm.predict(
        action_prompt,
    )

    if is_safe == False:  # pylint: disable=singleton-comparison
      #  is_safe could be None
      action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

    if not raw_response:
      raise RuntimeError('Error calling LLM in action selection phase.')

    step_data['action_output'] = action_output
    step_data['action_raw_response'] = raw_response

    reason, action = m3a_utils.parse_reason_action_output(action_output)

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    if (not reason) or (not action):
      print('Action prompt output is not in the correct format.')
      step_data['summary'] = (
          'Output for action selection is not in the correct format, so no'
          ' action is performed.'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    print('Action: ' + action)
    print('Reason: ' + reason)

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    if converted_action.action_type in ['click', 'long-press', 'input-text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        # Add mark for the target ui element, just used for visualization.
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
        )

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    after_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        self.env.logical_screen_size,
    )

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements

    summary_prompt = _summarize_prompt(
        goal,
        action,
        reason,
        before_element_list,
        after_element_list,
    )

    summary, is_safe, raw_response = self.llm.predict(
        summary_prompt,
    )
    if is_safe == False:  # pylint: disable=singleton-comparison
      #  is_safe could be None
      summary = """Summary triggered LLM safety classifier."""

    step_data['summary_prompt'] = summary_prompt
    step_data['summary'] = (
        f'Action selected: {action}. {summary}'
        if raw_response
        else 'Error calling LLM in summerization phase.'
    )
    print('Summary: ' + summary)
    step_data['summary_raw_response'] = raw_response

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )
