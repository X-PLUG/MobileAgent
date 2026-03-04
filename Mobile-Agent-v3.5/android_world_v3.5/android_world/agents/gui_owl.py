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

"""GUI-Owl-1.5 for Android."""

from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.agents import mobile_agent_utils_new as mobile_agent_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import tools
from android_world.agents import new_json_action as json_action
from PIL import Image
import base64
import json
import pprint
import os
import time
from io import BytesIO
import copy

from android_world.agents.coordinate_resize import update_image_size_
import traceback

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fetch_resized_image(screenshot_file):
    screenshot = Image.open(screenshot_file)
    width, height = screenshot.size
    current_image_ele = update_image_size_({'image': screenshot_file, 'width': width, 'height': height})
    resized_width = current_image_ele['resized_width']
    resized_height = current_image_ele['resized_height']
    screenshot = screenshot.resize((resized_width, resized_height))
    return screenshot, resized_width, resized_height, current_image_ele

class GUIOwl(base_agent.EnvironmentInteractingAgent):
  """mobile agent for Android."""

  def __init__(self, env: interface.AsyncEnv, vllm, src_format, api_key, url, name: str = "Mobile_Agent", output_path = "", last_image=5):
    super().__init__(env, name)
    self._actions = []
    self._screenshots = []
    self.cur_user_messages = []
    self.output_path = output_path
    if self.output_path and not os.path.exists(self.output_path):
      os.mkdir(self.output_path)
    self.vllm = vllm
    self.src_format = src_format
    self.url = url
    self.api_key = api_key
    self.task_name = {}
    self.last_image = last_image

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self._actions.clear()
    self._screenshots.clear()
    self.cur_user_messages.clear()
  
  def cut_current_messages(self, messages, last_image=2):
    non_empty_user_indices = []
    for i, msg in enumerate(messages):
      if msg.get('role') == 'user' and msg.get('content') and len(msg['content']) > 0:
        non_empty_user_indices.append(i)

    if len(non_empty_user_indices) > last_image:
      indices_to_clear = non_empty_user_indices[:-last_image]
    else:
      indices_to_clear = []

    for index in indices_to_clear:
      if index == 1:
        messages[index]['content'] = [messages[index]['content'][0]]
      else:
        messages[index]['content'] = []

    return messages
  
  def convert_format(self, goal, messages):
    new_messages = copy.deepcopy(messages[:1])
    history = []
    for i, msg in enumerate(messages):
      if msg.get('role') == 'user' and (msg["content"] == [] or (len(msg["content"]) == 1 and msg["content"][0]["type"] == "text")):
        history.append(messages[i+1]["content"][0]["text"].split("Action:")[-1].split("<tool_call>")[0].strip())
      if i != 1 and msg.get('role') == 'user' and msg["content"] != []:
        if len(history) == 0:
          new_messages = copy.deepcopy(messages)
          new_messages[1]["content"][0]["text"] = f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {goal}\n\nPrevious actions:\nNo previous action."
          return new_messages
        history_string = ""
        for j, h in enumerate(history):
          history_string += f"Step{j+1}: {h}\n"
        history_string = history_string[:-1]
        new_messages.append(
          {
            "role": "user", "content": [
              {"type": "text", "text": f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {goal}\n\nPrevious actions:\n{history_string}"},
              {"type": "image_url", "image_url": {"url": msg["content"][0]["image_url"]["url"]}}
            ]
          }
        )
        new_messages += copy.deepcopy(messages[i+1:])
        return new_messages
      
    return copy.deepcopy(messages)

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
    
  def get_task_name(self, suite):
    for name, instances in suite.items():
      self.task_name[instances[0].goal] = name
  
  def step(
      self, goal: str) -> base_agent.AgentInteractionResult:
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
    step_idx = len(self._screenshots)
    state = self.get_post_transition_state()
    result["ui_elements"] = state.ui_elements

    result["screenshot"] = state.pixels.copy()
    screenshot = Image.fromarray(result["screenshot"])
    screenshot_file = f"screenshot_{step_idx}.png"
    screenshot_url = pil_to_base64(screenshot)
    
    if self.output_path:
      if goal not in self.task_name:
        task_output_dir = os.path.join(self.output_path, goal.replace(" ", "_")[:50])
      else:
        task_output_dir = os.path.join(self.output_path, self.task_name[goal])
      screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx}.png")
      if not os.path.exists(task_output_dir):
        os.mkdir(task_output_dir)
      screenshot.save(screenshot_file)
      with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
        for item in self._actions:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    self._screenshots.append(screenshot)

    screenshot, _, _, current_image_ele = fetch_resized_image(screenshot_file)
    action_response = ''
    action = None
    
    system_prompt = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''
    
    if step_idx == 0:
      self.cur_user_messages = [
        {
          "role": "system",
          "content": [{
            "type": "text",
            "text": system_prompt
          }]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {goal}\n\nPrevious actions:\nNo previous action."
            },
            {
              "type": "image_url",
              "image_url": {"url": screenshot_url}
            }
          ]
        }
      ]
    else:
      self.cur_user_messages.append(
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {"url": screenshot_url}
            }
          ]
        }
      )
    
    self.cur_user_messages = self.cut_current_messages(self.cur_user_messages, self.last_image)
    input_messages = self.convert_format(goal, self.cur_user_messages)
    action_response, _, _ = self.vllm.predict_mm(
        None,
        None,
        messages=input_messages
      )
    
    self.cur_user_messages.append(
      {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": action_response
        }],
      }
    )
    input_messages.append(
      {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": action_response
        }],
      }
    )
    result["action_response"] = action_response
    print('========== action_response ==========')
    pprint.pprint(action_response)

    dummy_action = None
    try:
      dummy_action = action_response.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
      dummy_action = json.loads(dummy_action)
      dummy_action['arguments']['action'] = dummy_action['arguments']['action'].replace('tap', 'click')
      if len(self._actions) > 0 and self._actions[-1]['arguments']['action'] == 'answer':
          dummy_action = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
          self.env.interaction_cache =  self._actions[-1]['arguments']['text']
  
      action, dummy_action_translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
          dummy_action, current_image_ele, src_format=self.src_format, tgt_format='abs_origin'
      )
      result["dummy_action"] = dummy_action
      result["dummy_action_translated"] = dummy_action_translated
      result["action"] = action
    except seeact_utils.ParseActionError as e:
      action = json_action.JSONAction(action_type=json_action.UNKNOWN)
      result["seeact_action"] = None
      result["action"] = action
    except:
        traceback.print_exc()
        print(action_response)
        raise
    else:
      actuation.execute_adb_action(
          action,
          [],
          self.env.logical_screen_size,
          self.env.controller
      )
      
      self._actions.append(dummy_action)

    if self.output_path:
      if goal not in self.task_name:
        task_output_dir = os.path.join(self.output_path, goal.replace(" ", "_")[:50])
      else:
        task_output_dir = os.path.join(self.output_path, self.task_name[goal])
      if not os.path.exists(task_output_dir):
        os.mkdir(task_output_dir)
      screenshot.save(screenshot_file)
      with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
        for item in self._actions:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    return base_agent.AgentInteractionResult(
        done=action.action_type == json_action.STATUS,
        data=result,
    )
