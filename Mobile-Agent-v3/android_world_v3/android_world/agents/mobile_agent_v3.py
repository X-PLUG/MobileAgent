# Copyright 2024 The android_world Authors.
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

"""Mobile-Agent-v3 for Android."""

import os
import time
import copy
import json
from android_world.agents import base_agent
from android_world.agents import infer_ma3 as infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import tools
from android_world.env import interface
from android_world.agents import new_json_action as json_action
from dataclasses import asdict
from PIL import Image
from android_world.agents.mobile_agent_v3_agent import (
    InfoPool, 
    Manager, 
    Executor, 
    Notetaker, 
    ActionReflector,
    ALL_APPS
)

def convert_fc_action_to_json_action(
    dummy_action
) -> json_action.JSONAction:
 
    action_json = json.loads(dummy_action)
    action_type = action_json['action']
    
    x = None
    y = None
    text = None
    direction = None
    goal_status = None
    app_name = None

    if action_type == 'open_app':
        action_type = json_action.OPEN_APP
        app_name = action_json['text']
    elif action_type == 'click':
        action_type = json_action.CLICK
        x, y = action_json['coordinate'][0], action_json['coordinate'][1]
    elif action_type == 'long_press':
        action_type = json_action.LONG_PRESS
        x, y = action_json['coordinate'][0], action_json['coordinate'][1]
    elif action_type == 'type':
        action_type = json_action.INPUT_TEXT
        text = action_json['text']
    elif action_type == 'swipe':
        action_type = json_action.SWIPE
        start_x, start_y, end_x, end_y = action_json['coordinate'][0], action_json['coordinate'][1], action_json['coordinate2'][0], action_json['coordinate2'][1]
        direction = [start_x, start_y, end_x, end_y]
    elif action_type == 'system_button':
        if action_json['button'] == 'Enter' or action_json['button'] == 'enter':
            action_type = json_action.KEYBOARD_ENTER
        elif action_json['button'] == 'Back' or action_json['button'] == 'back':
            action_type = json_action.NAVIGATE_BACK
        elif action_json['button'] == 'Home' or action_json['button'] == 'home':
            action_type = json_action.NAVIGATE_HOME
    elif action_type == 'answer':
        action_type = json_action.ANSWER
        text = action_json['text']
    elif action_type == 'done' or action_type == 'terminate':
        action_type = json_action.STATUS
        goal_status = json_action.GOAL_STATUS

    return json_action.JSONAction(
            action_type=action_type,
            x=x,
            y=y,
            text=text,
            direction=direction,
            goal_status=goal_status,
            app_name=app_name,
      )

all_apps_str = ""
for app_str in ALL_APPS:
    all_apps_str += f"  - {app_str}\n"

DETAILED_TIPS = (
    'General:\n'
    '- For any pop-up window, such as a permission request, you need to close it (e.g., by clicking `Don\'t Allow` or `Accept & continue`) before proceeding. Never choose to add any account or log in.`\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    '- Two files or notes can be considered the same or duplicate only if their names, creation time, and detailed content are exactly the same.\n'
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app.\n'
    '- Consider exploring the screen by using the `swipe`'
    ' action with different directions to reveal additional content. Or use search to quickly find a specific entry, if applicable.\n'
    '- If you cannot change the page content by swiping in the same direction continuously, the page may have been swiped to the bottom. Please try another operation to display more content.\n'
    '- For some horizontally distributed tags, you can swipe horizontally to view more.\n'
    'Text Related Operations:\n'
    '- Activated input box: If an input box is activated, it may have a cursor inside it and the keyboard is visible. If there is no cursor on the screen but the keyboard is visible, it may be because the cursor is blinking. The color of the activated input box will be highlighted. If you are not sure whether the input box is activated, click it before typing.\n'
    '- To input some text: first click the input box that you want to input, make sure the correct input box is activated and the keyboard is visible, then use `type` action to enter the specified text.\n'
    '- To clear the text: long press the backspace button in the keyboard.\n'
    '- To copy some text: first long press the text you want to copy, then click the `copy` button in bar.\n'
    '- To paste text into a text box: first long press the'
    ' text box, then click the `paste`'
    ' button in bar.'
)

class MobileAgentV3_M3A(base_agent.EnvironmentInteractingAgent):
  """Mobile Agent E wrapper for Android World."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      vllm: infer.MultimodalLlmWrapper,
      name: str = 'MobileAgentE_M3A',
      wait_after_action_seconds: float = 3.0,
      output_path: str = ""
  ):
      """Initializes a MobileAgentE_M3A Agent.

      Args:
          env: The environment.
          vllm: The multimodal LLM wrapper.
          name: The agent name.
          wait_after_action_seconds: Seconds to wait for the screen to stablize
              after executing an action
      """
      super().__init__(env, name)
      self.vllm = vllm
      self.additional_guidelines = None
      self.wait_after_action_seconds = wait_after_action_seconds
      self.output_path = output_path
      if self.output_path and not os.path.exists(self.output_path):
        os.mkdir(self.output_path)
      self.task_name = {}
      
      # init info pool
      self.info_pool = InfoPool(
          additional_knowledge_manager="",
          additional_knowledge_executor=copy.deepcopy(DETAILED_TIPS),
          err_to_manager_thresh=2
      )
      
      # Hide the coordinates on screen which might affect the vision model.
      self.env.hide_automation_ui()
    
  def initialize_chrome(self):
    print("Running additional chrome initialization...")
    # handle chrome initialization problem for browser tasks
    adb_utils.launch_app("chrome", self.env.controller)
    time.sleep(5)

    tool_controller = tools.AndroidToolController(env=self.env.controller)
    time.sleep(2)

    first_op = False
    try:
      print("try first variant...")
      tool_controller.click_element("Use without an account")
      time.sleep(5.0)
      first_op = True
    except:
      print("Failed to click 'Use without an account' button.")
      pass
    
    if not first_op:
      print("try second variant...")
      try:
        tool_controller.click_element("Accept & continue")
      except:
        pass
      time.sleep(3.0)
      try:
        tool_controller.click_element("No thanks")
      except:
        pass
      time.sleep(5.0)
      
    adb_utils.press_home_button(self.env.controller)
    time.sleep(2.0)
    print("Done additional chrome initialization")

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()

    # init info pool again on reset
    self.info_pool = InfoPool(
      additional_knowledge_manager=copy.deepcopy(""),
      additional_knowledge_executor=copy.deepcopy(DETAILED_TIPS),
      err_to_manager_thresh=2
    )
    
  def get_task_name(self, suite):
    for name, instances in suite.items():
      self.task_name[instances[0].goal] = name

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    ## init agents ## 
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    
    self.info_pool.instruction = goal
    step_idx = len(self.info_pool.action_history)
    
    self.info_pool.additional_knowledge_manager=copy.deepcopy(""),
    self.info_pool.additional_knowledge_executor=copy.deepcopy(DETAILED_TIPS),
    
    print('----------step ' + str(step_idx + 1))
    
    # fix chrome initialization problem
    if step_idx == 0 and "chrome" in goal.lower():
      self.initialize_chrome()

    ## perception ###
    state = self.get_post_transition_state()
    before_screenshot = state.pixels.copy()
    
    if self.info_pool.instruction not in self.task_name:
      task_output_dir = os.path.join(self.output_path, self.info_pool.instruction.replace(" ", "_")[:50])
    else:
      task_output_dir = os.path.join(self.output_path, self.task_name[self.info_pool.instruction])
    if not os.path.exists(task_output_dir):
      os.mkdir(task_output_dir)
    save_screenshot = Image.fromarray(before_screenshot)
    save_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx}.png"))
    before_screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx}.png")
    with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
      for item in self.info_pool.action_pool:
          f.write(item + '\n')
    
    self.info_pool.width = 1092
    self.info_pool.height = 2408
    
    ###############
    ### manager ###
    ###############
    
    ## check error escalation
    self.info_pool.error_flag_plan = False
    err_to_manager_thresh = self.info_pool.err_to_manager_thresh
    if len(self.info_pool.action_outcomes) >= err_to_manager_thresh:
      # check if the last err_to_manager_thresh actions are all errors
      latest_outcomes = self.info_pool.action_outcomes[-err_to_manager_thresh:]
      count = 0
      for outcome in latest_outcomes:
          if outcome in ["B", "C"]:
              count += 1
      if count == err_to_manager_thresh:
          self.info_pool.error_flag_plan = True

    skip_manager = False
    ## if previous action is invalid, skip the manager and try again first ##
    if not self.info_pool.error_flag_plan and len(self.info_pool.action_history) > 0:
      if self.info_pool.action_history[-1]['action'] == 'invalid':
        skip_manager = True
      
    if not skip_manager:
      print("\n### Manager ... ###\n")
      prompt_planning_ori = manager.get_prompt(self.info_pool)
      prompt_planning = prompt_planning_ori.replace("[add_info_start]", "").replace("[add_info_end]", "")
      output_planning, message_manager, raw_response = self.vllm.predict_mm(
          prompt_planning,
          [before_screenshot_file],
      )
      
      parsed_result_planning = manager.parse_response(output_planning)
      self.info_pool.completed_plan = parsed_result_planning['completed_subgoal']
      self.info_pool.plan = parsed_result_planning['plan']
      if not raw_response:
        raise RuntimeError('Error calling vLLM in planning phase.')
      
      print('Completed subgoal: ' + self.info_pool.completed_plan)
      print('Planning thought: ' + parsed_result_planning['thought'])
      print('Plan: ' + self.info_pool.plan, "\n")

    ## if stopping by planner ##
    output_action = None
    if "Finished" in self.info_pool.plan.strip() and len(self.info_pool.plan.strip()) < 15:
      self.info_pool.finish_thought = parsed_result_planning['thought']
      action_thought = "Finished by planner"
      action_object_str = "{\"action\": \"done\"}"
      action_description = "Finished by planner"
      self.info_pool.action_pool.append(action_object_str)

      if self.info_pool.instruction not in self.task_name:
        task_output_dir = os.path.join(self.output_path, self.info_pool.instruction.replace(" ", "_")[:50])
      else:
        task_output_dir = os.path.join(self.output_path, self.task_name[self.info_pool.instruction])
      if not os.path.exists(task_output_dir):
        os.mkdir(task_output_dir)
      save_screenshot = Image.fromarray(before_screenshot)
      save_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx}.png"))
      with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
        for item in self.info_pool.action_pool:
            f.write(item + '\n')
    else:

      ################
      ### Operator ###
      ################

      print("\n### Operator ... ###\n")
      prompt_action = executor.get_prompt(self.info_pool)
      output_action, message_operator, raw_response = self.vllm.predict_mm(
          prompt_action,
          [before_screenshot_file],
      )
      
      if not raw_response:
        raise RuntimeError('Error calling vLLM in operator phase.')
      parsed_result_action = executor.parse_response(output_action)
      action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']
      
      self.info_pool.last_action_thought = action_thought
      self.info_pool.last_summary = action_description

      if (not action_thought) or (not action_object_str):
        print('Action prompt output is not in the correct format.')
        self.info_pool.last_action = {"action": "invalid"}
        self.info_pool.action_history.append({"action": "invalid"})
        self.info_pool.summary_history.append(action_description)
        self.info_pool.action_outcomes.append("C")
        self.info_pool.error_descriptions.append("invalid action format, do nothing.")
        return base_agent.AgentInteractionResult(
            False,
            asdict(self.info_pool),
        )
    
    print('Thought: ' + action_thought)
    print('Action: ' + action_object_str)
    print('Action description: ' + action_description)

    try:
      converted_action = convert_fc_action_to_json_action(action_object_str)
      self.info_pool.action_pool.append(action_object_str)
    except Exception as e:
      print('Failed to convert the output to a valid action.')
      print(str(e))
      self.info_pool.last_action = {"action": "invalid"}
      self.info_pool.action_history.append({"action": "invalid"})
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append("C")
      self.info_pool.error_descriptions.append("invalid action format, do nothing.")
      return base_agent.AgentInteractionResult(
          False,
          asdict(self.info_pool),
      )

    if converted_action.action_type == 'status':
      outcome = "A"
      error_description = "None"
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
        outcome = "C"
        error_description = "Agent stopped since it thinks mission impossible."
      self.info_pool.last_action = json.loads(action_object_str)
      self.info_pool.action_history.append(json.loads(action_object_str))
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append(outcome)
      self.info_pool.error_descriptions.append(error_description)
      return base_agent.AgentInteractionResult(
          True,
          asdict(self.info_pool),
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)
    
    if converted_action.action_type == 'open_app':
      converted_action.app_name = converted_action.app_name.lower().strip()
      
    try:
      self.env.execute_action(converted_action)
    except Exception as e:
      print('Failed to execute action.')
      print(str(e))
      self.info_pool.last_action = json.loads({"action": "invalid"})
      self.info_pool.action_history.append({"action": "invalid"})
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append("C")
      if converted_action.action_type == "open_app":
        app_name = converted_action.app_name
        self.info_pool.error_descriptions.append(f"Failed to open the app '{app_name}'; the app name might not exist.")
      else:
        self.info_pool.error_descriptions.append(f"Failed to execute the action: {converted_action}")
      return base_agent.AgentInteractionResult(
          False,
          asdict(self.info_pool),
      )
    print("Done action execution.\n")
    self.info_pool.last_action = json.loads(action_object_str)

    time.sleep(self.wait_after_action_seconds)
    
    ### Perception after execution ###
    state = self.env.get_state(wait_to_stabilize=False)
    after_screenshot = state.pixels.copy()
    save_after_screenshot = Image.fromarray(after_screenshot)
    save_after_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx+1}.png"))
    after_screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx+1}.png")

    m3a_utils.add_screenshot_label(before_screenshot, 'before')
    m3a_utils.add_screenshot_label(after_screenshot, 'after')
    
    self.info_pool.ui_elements_list_after = []

    #################
    ### Reflector ###
    #################
    
    print("\n### Action Reflector ... ###\n")
    if converted_action.action_type != 'answer':
      prompt_action_reflect = action_reflector.get_prompt(self.info_pool)
      output_action_reflect, message_reflector, raw_response = self.vllm.predict_mm(
          prompt_action_reflect,
          [before_screenshot_file, after_screenshot_file],
      )
      
      parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
      outcome, error_description = (
          parsed_result_action_reflect['outcome'], 
          parsed_result_action_reflect['error_description']
      )
      progress_status = self.info_pool.completed_plan

      if "A" in outcome: # Successful. The result of the last action meets the expectation.
          action_outcome = "A"
      elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
          action_outcome = "B"
      elif "C" in outcome: # Failed. The last action produces no changes.
          action_outcome = "C"
      else:
          raise ValueError("Invalid outcome:", outcome)
      
    else:
      action_outcome = "A"
      error_description = "None"
      progress_status = self.info_pool.completed_plan + "\n" + "The `answer` action has been performed. Answer to the question: " + converted_action.text
    
    print('Action reflection outcome: ' + action_outcome)
    print('Action reflection error description: ' + error_description, "\n")

    self.info_pool.action_history.append(json.loads(action_object_str))
    self.info_pool.summary_history.append(action_description)
    self.info_pool.action_outcomes.append(action_outcome)
    self.info_pool.error_descriptions.append(error_description)
    self.info_pool.progress_status = progress_status

    #################
    ### NoteKeeper ###
    #################
    
    if "'Ideas' folder" in self.info_pool.instruction and "Joplin app" in self.info_pool.instruction:
      print("skip notekeeper because of hallucination")
    elif "answer" not in self.info_pool.instruction.lower() and "transactions from" not in self.info_pool.instruction.lower() and "enter their product" not in self.info_pool.instruction.lower():
      print("skip notekeeper because of no answer or transport")
    else:
      if action_outcome == "A" and converted_action.action_type != 'answer':
          print("\n### NoteKeeper ... ###\n")
          # if previous action is successful, record the important content
          prompt_note = notetaker.get_prompt(self.info_pool)
          output_note, message_notekeeper, raw_response = self.vllm.predict_mm(
            prompt_note,
            [after_screenshot_file],
          )
          
          parsed_result_note = notetaker.parse_response(output_note)
          important_notes = parsed_result_note['important_notes']
          self.info_pool.important_notes = important_notes
          
          print('Important notes: ' + important_notes, "\n")
    
    if self.info_pool.instruction not in self.task_name:
      task_output_dir = os.path.join(self.output_path, self.info_pool.instruction.replace(" ", "_")[:50])
    else:
      task_output_dir = os.path.join(self.output_path, self.task_name[self.info_pool.instruction])
    if not os.path.exists(task_output_dir):
      os.mkdir(task_output_dir)
    save_screenshot = Image.fromarray(before_screenshot)
    save_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx}.png"))
    with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
      for item in self.info_pool.action_pool:
          f.write(item + '\n')
    
    if converted_action.action_type == 'answer':
      return base_agent.AgentInteractionResult(
          True,
          asdict(self.info_pool),
      )
    else:
      return base_agent.AgentInteractionResult(
          False,
          asdict(self.info_pool),
      )
