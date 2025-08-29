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
from qwen_vl_utils import smart_resize
from io import BytesIO

from android_world.agents.coordinate_resize import update_image_size_
import traceback

m3a_prompt = '''Instruction
You are an agent who can operate an Android phone on behalf of a user. Based on user’s goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks ”What is my schedule for today?”.
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.
When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the action in the following list (action description followed by the JSON format) by outputing the action in the correct JSON format.
- If you think the task has been completed, finish the task by using the status action with complete as goal status: ‘{”action type”: ”status”, ”goal status”: ”complete”}
- If you think the task is not feasible (including cases like you don’t have enough informa- tion or can not perform some necessary actions), finish by using the ‘status‘ action with infeasible as goal status: ‘{”action type”: ”status”, ”goal status”: ”infeasible”}
- Answer user’s question: ‘{”action type”: ”answer”, ”text”: ”answer text”}
- Click/tap on an element on the screen. Please describe the element you want to click using natural language. ‘{”action type”: ”click”, ”target”: target element description}‘. - Long press on an element on the screen, similar with the click action above, use the semantic description to indicate the element you want to long press: ‘{”action type”: ”long press”, ”target”: target element description}.
- Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the semantic de- scription to indicate the target text field: ‘{”action type”: ”input text”, ”text”: text input, ”target”: target element description}
- Press the Enter key: ‘{”action type”: ”keyboard enter”}
- Navigate to the home screen: ‘{”action type”: ”navigate home”}
- Navigate back: ‘{”action type”: ”navigate back”}
- Scroll the screen or a scrollable UI element in one of the four directions, use the same semantic description as above if you want to scroll a specific UI element, leave it empty when scroll the whole screen: ‘{”action type”: ”scroll”, ”direction”: up, down, left, right, ”element”: optional target element description}
- Open an app (nothing will happen if the app is not installed): ‘{”action type”: ”open app”, ”app name”: name}
- Wait for the screen to update: ‘{”action type”: ”wait”}

Guidelines
Here are some useful guidelines you need to follow:
General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn’t (you can see that from the history), SWITCH to other solutions.
- Sometimes you may need to navigate the phone to gather information needed to com- plete the task, for example if user asks ”what is my schedule tomorrow”, then you may want to open the calendar app (using the ‘open app‘ action), look up information there, answer user’s question (using the ‘answer‘ action) and finish (using the ‘status‘ action with complete as goal status).
- For requests that are questions (or chat messages), remember to use the ‘answer‘ action to reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is something like ”show me ...”).
- If the desired state is already achieved (e.g., enabling Wi-Fi when it’s already on), you can just complete the task.
Action Related:
- Use the ‘open app‘ action whenever you want to open an app (nothing will happen if the app is not installed), do not use the app drawer to open an app unless all other ways have failed.
- Use the ‘input text‘ action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- For ‘click‘, ‘long press‘ and ‘input text‘, the target element description parameter you choose must based on a VISIBLE element in the screenshot.
- Consider exploring the screen by using the ‘scroll‘ action with different directions to reveal additional content.
- The direction parameter for the ‘scroll‘ action can be confusing sometimes as it’s op- posite to swipe, for example, to view content at the bottom, the ‘scroll‘ direction should be set to ”down”. It has been observed that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as well.
Text Related Operations:
- Normally to select certain text on the screen: (i) Enter text selection mode by long pressing the area where the text is, then some of the words near the long press point will be selected (highlighted with two pointers indicating the range) and usually a text selection bar will also appear with options like ‘copy‘, ‘paste‘, ‘select all‘, etc. (ii) Select the exact text you need. Usually the text selected from the previous step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to select all text in the text field, simply click the ‘select all‘ button in the bar.
- At this point, you don’t have the ability to drag something around the screen, so in general you can not select arbitrary text.
- To delete some text: the most traditional way is to place the cursor at the right place and use the backspace button in the keyboard to delete the characters one by one (can long press the backspace to accelerate if there are many to delete). Another approach is to first select the text you want to delete, then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the ‘copy‘ button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a ‘paste‘ button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.
'''

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_to_base64(image_path):
  dummy_image = Image.open(image_path)
  MIN_PIXELS=3136
  MAX_PIXELS=10035200
  resized_height, resized_width  = smart_resize(dummy_image.height,
      dummy_image.width,
      factor=28,
      min_pixels=MIN_PIXELS,
      max_pixels=MAX_PIXELS,)
  dummy_image = dummy_image.resize((resized_width, resized_height))
  return f"data:image/png;base64,{pil_to_base64(dummy_image)}"


all_apps_str = """- simple calendar pro: A calendar app.\n  - settings: The Android system settings app for managing device settings such as Bluetooth, Wi-Fi, and brightness.\n  - markor: A note-taking app for creating, editing, deleting, and managing notes and folders.\n  - broccoli: A recipe management app.\n  - pro expense: An expense tracking app.\n  - simple sms messenger: An SMS app for sending, replying to, and resending text messages.\n  - opentracks: A sport tracking app for recording and analyzing activities.\n  - tasks: A task management app for tracking tasks, due dates, and priorities.\n  - clock: An app with stopwatch and timer functionality.\n  - joplin: A note-taking app.\n  - retro music: A music player app.\n  - simple gallery pro: An app for viewing images.\n  - camera: An app for taking photos and videos.\n  - chrome: A web browser app.\n  - contacts: An app for managing contact information.\n  - osmand: A maps and navigation app with support for adding location markers, favorites, and saving tracks.\n  - vlc: A media player app for playing media files.\n  - audio recorder: An app for recording and saving audio clips.\n  - files: A file manager app for the Android filesystem, used for deleting and moving files.\n  - simple draw pro: A drawing app for creating and saving drawings."""

DETAILED_TIPS = (
    'General:\n'
    '- If a previous action fails and the screen does not change, simply try again first.\n'
    '- For any pop-up window, such as a permission request, you need to close it (e.g., by clicking `Don\'t Allow` or `Accept & continue`) before proceeding. Never choose to add any account or log in.`\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.'
    ' ALL avaliable apps are listed as follows, please use the exact names (in lowercase) as argument for the `open_app` action.\n'
    f'{all_apps_str}'
    '- Use the `type` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `type`, the index parameter you'
    ' pick must be VISIBLE in the screenshot\n'
    '- Consider exploring the screen by using the `swipe`'
    ' action with different directions to reveal additional content. Or use search to quickly find a specific entry, if applicable.\n\n'
    'Text Related Operations:\n'
    '- When asked to save a file with a specific name, you can usually edit the name in the final step. For example, you can first record an audio clip then save it with a specific name.\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
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
    ' in the list.\n\n'
)

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

  def __init__(self, env: interface.AsyncEnv, vllm, src_format, api_key, url, name: str = "Mobile_Agent", output_path = ""):
    super().__init__(env, name)
    self._actions = []
    self._screenshots = []
    self._summarys = []
    self._thoughts = []
    self.output_result = {}
    self.output_path = output_path
    if self.output_path and not os.path.exists(self.output_path):
      os.mkdir(self.output_path)
    self.vllm = vllm

    self.add_thought = True
    self._text_actions = []
    self.src_format = src_format

    self.url = url
    self.api_key = api_key

    self.output_list = []
    self._response = []
    self.task_name = {}

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self._actions.clear()
    self._text_actions.clear()
    self._screenshots.clear() # TODO
    self._summarys.clear()
    self._thoughts.clear()
    self._response.clear()
  
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

    result["screenshot"] = state.pixels
    screenshot = Image.fromarray(state.pixels)
    screenshot_file = f"screenshot_{step_idx}.png"
    
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
  
    stage2_history = ''
    for idx, his in enumerate(self._summarys):
        if his is not None:
            stage2_history += 'Step ' + str(idx + 1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '
    stage2_user_prompt = goal

    screenshot, resized_width, resized_height, current_image_ele = fetch_resized_image(screenshot_file)
    action_response = ''
    action = None
    
    system_prompt_part, user_prompt_part = mobile_agent_utils.build_system_messages(stage2_user_prompt,
                                                                                    resized_width, resized_height,
                                                                                    '', stage2_history)
    
    user_prompt_part['content'].append({'image': image_to_base64(screenshot_file)})
    
    if os.environ.get('ADD_INFO', ''):
      user_prompt_part['content'].append({"type": "text", 'text': os.environ.get('ADD_INFO', '')})
    if os.environ.get('ADD_GENERAL_ADD_INFO', ''):
      user_prompt_part['content'].append({"type": "text", 'text': DETAILED_TIPS})
    
    messages = [system_prompt_part, user_prompt_part]

    action_response, _, _ = self.vllm.predict_mm(
          "",
          [],
          messages=messages
      )
    
    result["action_response"] = action_response
    print('========== action_response ==========')
    pprint.pprint(action_response)

    dummy_action = None
    thought = None
    summary = None
    try:
      if self.add_thought:
        if '</think>' in action_response:
          thought = action_response.split('</think>')[0].strip('<think>').strip('\n')
        else:
          thought = action_response.split('<thinking>\n')[1].split('\n</thinking>')[0]
        dummy_action = '{"name": "mobile_use"' + action_response.split('{"name": "mobile_use"')[1].split('}}\n')[0] + '}}'
        summary = action_response.split('<conclusion>\n')[1].split('\n</conclusion>')[0]
      else:
        dummy_action = '{"name": "mobile_use"' + action_response.split('{"name": "mobile_use"')[1].split('}}\n')[
            0] + '}}'
        thought = None
        summary = None

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
      
      self._text_actions.append(summary)
      self._actions.append(dummy_action)
      self._summarys.append(summary)
      self._thoughts.append(thought)
      self._response.append(action_response)

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
