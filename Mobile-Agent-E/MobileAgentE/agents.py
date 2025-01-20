from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from MobileAgentE.api import encode_image
from MobileAgentE.controller import tap, swipe, type, back, home, switch_app, enter, save_screenshot_to_file
from MobileAgentE.text_localization import ocr
import copy
import re
import json
import time
import os

### Helper Functions ###

def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
            "type": "text", 
            "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history


def add_response_two_image(role, prompt, chat_history, image):
    new_chat_history = copy.deepcopy(chat_history)

    base64_image1 = encode_image(image[0])
    base64_image2 = encode_image(image[1])
    content = [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}"
            }
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}"
            }
        },
    ]

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1) + "\n")
    print("*"*100)


def extract_json_object(text, json_type="dict"):
    """
    Extracts a JSON object from a text string.

    Parameters:
    - text (str): The text containing the JSON data.
    - json_type (str): The type of JSON structure to look for ("dict" or "list").

    Returns:
    - dict or list: The extracted JSON object, or None if parsing fails.
    """
    try:
        if "//" in text:
            # Remove comments starting with //
            text = re.sub(r'//.*', '', text)
        if "# " in text:
            # Remove comments starting with #
            text = re.sub(r'#.*', '', text)
        # Try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # Not a valid JSON, proceed to extract from text

    # Define patterns for extracting JSON objects or arrays
    json_pattern = r"({.*?})" if json_type == "dict" else r"(\[.*?\])"

    # Search for JSON enclosed in code blocks first
    code_block_pattern = r"```json\s*(.*?)\s*```"
    code_block_match = re.search(code_block_pattern, text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Failed to parse JSON inside code block

    # Fallback to searching the entire text
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue  # Try the next match

    # If all attempts fail, return None
    return None

########################


@dataclass
class InfoPool:
    """Keeping track of all information across the agents."""

    # User input / accumulated knowledge
    instruction: str = ""
    tips: str = ""
    shortcuts: dict = field(default_factory=dict)

    # Perception
    width: int = 1080
    height: int = 2340
    perception_infos_pre: list = field(default_factory=list) # List of clickable elements pre action
    keyboard_pre: bool = False # keyboard status pre action
    perception_infos_post: list = field(default_factory=list) # List of clickable elements post action
    keyboard_post: bool = False # keyboard status post action

    # Working memory
    summary_history: list = field(default_factory=list)  # List of action descriptions
    action_history: list = field(default_factory=list)  # List of actions
    action_outcomes: list = field(default_factory=list)  # List of action outcomes
    error_descriptions: list = field(default_factory=list)

    last_summary: str = ""  # Last action description
    last_action: str = ""  # Last action
    last_action_thought: str = ""  # Last action thought
    important_notes: str = ""
    
    error_flag_plan: bool = False # if an error is not solved for multiple attempts with the executor
    error_description_plan: bool = False # explanation of the error for modifying the plan

    # Planning
    plan: str = ""
    progress_status: str = ""
    progress_status_history: list = field(default_factory=list)
    finish_thought: str = ""
    current_subgoal: str = ""
    prev_subgoal: str = ""
    err_to_manager_thresh: int = 2

    # future tasks
    future_tasks: list = field(default_factory=list)


class BaseAgent(ABC):
    @abstractmethod
    def init_chat(self) -> list:
        pass
    @abstractmethod
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    @abstractmethod
    def parse_response(self, response: str) -> dict:
        pass


class Manager(BaseAgent):

    def init_chat(self):
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to track progress and devise high-level plans to achieve the user's requests. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        if info_pool.plan == "":
            # first time planning
            prompt += "---\n"
            prompt += "Think step by step and make an high-level plan to achieve the user's instruction. If the request is complex, break it down into subgoals. If the request involves exploration, include concrete subgoals to quantify the investigation steps. The screenshot displays the starting state of the phone.\n\n"
            
            if info_pool.shortcuts != {}:
                prompt += "### Available Shortcuts from Past Experience ###\n"
                prompt += "We additionally provide some shortcut functionalities based on past experience. These shortcuts are predefined sequences of operations that might make the plan more efficient. Each shortcut includes a precondition specifying when it is suitable for use. If your plan implies the use of certain shortcuts, ensure that the precondition is fulfilled before using them. Note that you don't necessarily need to include the names of these shortcuts in your high-level plan; they are provided as a reference.\n"
                for shortcut, value in info_pool.shortcuts.items():
                    prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
                prompt += "\n"
            prompt += "---\n"

            prompt += "Provide your output in the following format which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "A detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "1. first subgoal\n"
            prompt += "2. second subgoal\n"
            prompt += "...\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The first subgoal you should work on.\n\n"
        else:
            # continue planning
            prompt += "### Current Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"
            prompt += "### Previous Subgoal ###\n"
            prompt += f"{info_pool.current_subgoal}\n\n"
            prompt += f"### Progress Status ###\n"
            prompt += f"{info_pool.progress_status}\n\n"
            prompt += "### Important Notes ###\n"
            if info_pool.important_notes != "":
                prompt += f"{info_pool.important_notes}\n\n"
            else:
                prompt += "No important notes recorded.\n\n"
            if info_pool.error_flag_plan:
                prompt += "### Potentially Stuck! ###\n"
                prompt += "You have encountered several failed attempts. Here are some logs:\n"
                k = info_pool.err_to_manager_thresh
                recent_actions = info_pool.action_history[-k:]
                recent_summaries = info_pool.summary_history[-k:]
                recent_err_des = info_pool.error_descriptions[-k:]
                for i, (act, summ, err_des) in enumerate(zip(recent_actions, recent_summaries, recent_err_des)):
                    prompt += f"- Attempt: Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"

            prompt += "---\n"
            prompt += "The sections above provide an overview of the plan you are following, the current subgoal you are working on, the overall progress made, and any important notes you have recorded. The screenshot displays the current state of the phone.\n"
            prompt += "Carefully assess the current status to determine if the task has been fully completed. If the user's request involves exploration, ensure you have conducted sufficient investigation. If you are confident that no further actions are required, mark the task as \"Finished\" in your output. If the task is not finished, outline the next steps. If you are stuck with errors, think step by step about whether the overall plan needs to be revised to address the error.\n"
            prompt += "NOTE: If the current situation prevents proceeding with the original plan or requires clarification from the user, make reasonable assumptions and revise the plan accordingly. Act as though you are the user in such cases.\n\n"

            if info_pool.shortcuts != {}:
                prompt += "### Available Shortcuts from Past Experience ###\n"
                prompt += "We additionally provide some shortcut functionalities based on past experience. These shortcuts are predefined sequences of operations that might make the plan more efficient. Each shortcut includes a precondition specifying when it is suitable for use. If your plan implies the use of certain shortcuts, ensure that the precondition is fulfilled before using them. Note that you don't necessarily need to include the names of these shortcuts in your high-level plan; they are provided only as a reference.\n"
                for shortcut, value in info_pool.shortcuts.items():
                    prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
                prompt += "\n"
            
            prompt += "---\n"
            prompt += "Provide your output in the following format, which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "Provide a detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "If an update is required for the high-level plan, provide the updated plan here. Otherwise, keep the current plan and copy it here.\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The next subgoal to work on. If the previous subgoal is not yet complete, copy it here. If all subgoals are completed, write \"Finished\".\n"
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "plan": plan, "current_subgoal": current_subgoal}


# name: {arguments: [argument_keys], description: description}
ATOMIC_ACTION_SIGNITURES = {
    "Open_App": {
        "arguments": ["app_name"],
        "description": lambda info: "If the current screen is Home or App screen, you can use this action to open the app named \"app_name\" on the visible on the current screen."
    },
    "Tap": {
        "arguments": ["x", "y"],
        "description": lambda info: "Tap the position (x, y) in current screen."
    },
    "Swipe": {
        "arguments": ["x1", "y1", "x2", "y2"],
        "description": lambda info: f"Swipe from position (x1, y1) to position (x2, y2). To swipe up or down to review more content, you can adjust the y-coordinate offset based on the desired scroll distance. For example, setting x1 = x2 = {int(0.5 * info.width)}, y1 = {int(0.5 * info.height)}, and y2 = {int(0.1 * info.height)} will swipe upwards to review additional content below. To swipe left or right in the App switcher screen to choose between open apps, set the x-coordinate offset to at least {int(0.5 * info.width)}."
    },
    "Type": {
        "arguments": ["text"],
        "description": lambda info: "Type the \"text\" in an input box."
    },
    "Enter": {
        "arguments": [],
        "description": lambda info: "Press the Enter key after typing (useful for searching)."
    },
    "Switch_App": {
        "arguments": [],
        "description": lambda info: "Show the App switcher for switching between opened apps."
    },
    "Back": {
        "arguments": [],
        "description": lambda info: "Return to the previous state."
    },
    "Home": {
        "arguments": [],
        "description": lambda info: "Return to home page."
    },
    "Wait": {
        "arguments": [],
        "description": lambda info: "Wait for 10 seconds to give more time for a page loading."
    }
}

INIT_SHORTCUTS = {
    "Tap_Type_and_Enter": {
        "name": "Tap_Type_and_Enter",
        "arguments": ["x", "y", "text"],
        "description": "Tap an input box at position (x, y), Type the \"text\", and then perform the Enter operation. Very useful for searching and sending messages!",
        "precondition": "There is a text input box on the screen with no previously entered content.",
        "atomic_action_sequence":[
            {"name": "Tap", "arguments_map": {"x":"x", "y":"y"}},
            {"name": "Type", "arguments_map": {"text":"text"}},
            {"name": "Enter", "arguments_map": {}}
        ]
    }
}


class Operator(BaseAgent):
    def __init__(self, adb_path):
        self.adb = adb_path

    def init_chat(self):
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better understand the content in this screenshot, we have extracted positional information for the text elements and icons, including interactive elements such as search bars. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom)."
        )
        prompt += "The extracted information is as follows:\n"

        for clickable_info in info_pool.perception_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += (
            "Note that a search bar is often a long, rounded rectangle. If no search bar is presented and you want to perform a search, you may need to tap a search button, which is commonly represented by a magnifying glass.\n"
            "Also, the information above might not be entirely accurate. "
            "You should combine it with the screenshot to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Keyboard status ###\n"
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        if info_pool.tips != "":
            prompt += "### Tips ###\n"
            prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
            prompt += f"{info_pool.tips}\n\n"

        prompt += "### Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += "Here are some potentially important content relevant to the user's request you already recorded:\n"
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "---\n"
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions or the shortcuts. The shortcuts are predefined sequences of actions that can be used to speed up the process. Each shortcut has a precondition specifying when it is suitable to use. If you plan to use a shortcut, ensure the current phone state satisfies its precondition first.\n\n"
        
        prompt += "#### Atomic Actions ####\n"
        prompt += "The atomic action functions are listed in the format of `name(arguments): description` as follows:\n"

        if info_pool.keyboard_pre:
            for action, value in ATOMIC_ACTION_SIGNITURES.items():
                prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        else:
            for action, value in ATOMIC_ACTION_SIGNITURES.items():
                if "Type" not in action:
                    prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
            prompt += "NOTE: Unable to type. The keyboard has not been activated. To type, please activate the keyboard by tapping on an input box or using a shortcut, which includes tapping on an input box first.”\n"
        
        prompt += "\n"
        prompt += "#### Shortcuts ####\n"
        if info_pool.shortcuts != {}:
            prompt += "The shortcut functions are listed in the format of `name(arguments): description | Precondition: precondition` as follows:\n"
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"- {shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are available.\n"
        prompt += "\n"

        prompt += "### Latest Action History ###\n"
        if info_pool.action_history != []:
            prompt += "Recent actions you took previously and whether they were successful:\n"
            num_actions = min(5, len(info_pool.action_history))
            latest_actions = info_pool.action_history[-num_actions:]
            latest_summary = info_pool.summary_history[-num_actions:]
            latest_outcomes = info_pool.action_outcomes[-num_actions:]
            error_descriptions = info_pool.error_descriptions[-num_actions:]
            action_log_strs = []
            for act, summ, outcome, err_des in zip(latest_actions, latest_summary, latest_outcomes, error_descriptions):
                if outcome == "A":
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Successful\n"
                else:
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"
                prompt += action_log_str
                action_log_strs.append(action_log_str)
            if latest_outcomes[-1] == "C" and "Tap" in action_log_strs[-1] and "Tap" in action_log_strs[-2]:
                prompt += "\nHINT: If multiple Tap actions failed to make changes to the screen, consider using a \"Swipe\" action to view more content or use another way to achieve the current subgoal."
            
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "---\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        prompt += "Provide a detailed explanation of your rationale for the chosen action. IMPORTANT: If you decide to use a shortcut, first verify that its precondition is met in the current phone state. For example, if the shortcut requires the phone to be at the Home screen, check whether the current screenshot shows the Home screen. If not, perform the appropriate atomic actions instead.\n\n"

        prompt += "### Action ###\n"
        prompt += "Choose only one action or shortcut from the options provided. IMPORTANT: Do NOT return invalid actions like null or stop. Do NOT repeat previously failed actions.\n"
        prompt += "Use shortcuts whenever possible to expedite the process, but make sure that the precondition is met.\n"
        prompt += "You must provide your decision using a valid JSON format specifying the name and arguments of the action. For example, if you choose to tap at position (100, 200), you should write {\"name\":\"Tap\", \"arguments\":{\"x\":100, \"y\":100}}. If an action does not require arguments, such as Home, fill in null to the \"arguments\" field. Ensure that the argument keys match the action function's signature exactly.\n\n"
        
        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action and the expected outcome."
        return prompt

    def execute_atomic_action(self, action: str, arguments: dict, **kwargs) -> None:
        adb_path = self.adb
        
        if "Open_App".lower() == action.lower():
            screenshot_file = kwargs["screenshot_file"]
            ocr_detection = kwargs["ocr_detection"]
            ocr_recognition = kwargs["ocr_recognition"]
            app_name = arguments["app_name"].strip()
            text, coordinate = ocr(screenshot_file, ocr_detection, ocr_recognition)
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [int((coordinate[ti][0] + coordinate[ti][2])/2), int((coordinate[ti][1] + coordinate[ti][3])/2)]
                    tap(adb_path, name_coordinate[0], name_coordinate[1]- int(coordinate[ti][3] - coordinate[ti][1]))# 
                    break
            if app_name in ['Fandango', 'Walmart', 'Best Buy']:
                # additional wait time for app loading
                time.sleep(10)
            time.sleep(10)
        
        elif "Tap".lower() == action.lower():
            x, y = int(arguments["x"]), int(arguments["y"])
            tap(adb_path, x, y)
            time.sleep(5)
        
        elif "Swipe".lower() == action.lower():
            x1, y1, x2, y2 = int(arguments["x1"]), int(arguments["y1"]), int(arguments["x2"]), int(arguments["y2"])
            swipe(adb_path, x1, y1, x2, y2)
            time.sleep(5)
            
        elif "Type".lower() == action.lower():
            text = arguments["text"]
            type(adb_path, text)
            time.sleep(3)

        elif "Enter".lower() == action.lower():
            enter(adb_path)
            time.sleep(10)

        elif "Back".lower() == action.lower():
            back(adb_path)
            time.sleep(3)
        
        elif "Home".lower() == action.lower():
            home(adb_path)
            time.sleep(3)
        
        elif "Switch_App".lower() == action.lower():
            switch_app(adb_path)
            time.sleep(3)
        
        elif "Wait".lower() == action.lower():
            time.sleep(10)
        
    def execute(self, action_str: str, info_pool: InfoPool, screenshot_log_dir=None, iter="", **kwargs) -> None:
        action_object = extract_json_object(action_str)
        if action_object is None:
            print("Error! Invalid JSON for executing action: ", action_str)
            return None, 0, None
        action, arguments = action_object["name"], action_object["arguments"]
        action = action.strip()

        # execute atomic action
        if action in ATOMIC_ACTION_SIGNITURES:
            print("Executing atomic action: ", action, arguments)
            self.execute_atomic_action(action, arguments, info_pool=info_pool, **kwargs)
            if screenshot_log_dir is not None:
                time.sleep(1)
                screenshot_file = os.path.join(screenshot_log_dir, f"{iter}__{action.replace(' ', '')}.png")
                save_screenshot_to_file(self.adb, screenshot_file)
            return action_object, 1, None # number of atomic actions executed
        # execute shortcut
        elif action in info_pool.shortcuts:
            print("Executing shortcut: ", action)
            shortcut = info_pool.shortcuts[action]
            for i, atomic_action in enumerate(shortcut["atomic_action_sequence"]):
                try:
                    atomic_action_name = atomic_action["name"]
                    if atomic_action["arguments_map"] is None or len(atomic_action["arguments_map"]) == 0:
                        atomic_action_args = None
                    else:
                        atomic_action_args = {}
                        for atomic_arg_key, value in atomic_action["arguments_map"].items():
                            if value in arguments: # if the mapped key is in the shortcut arguments
                                atomic_action_args[atomic_arg_key] = arguments[value]
                            else: # if not: the values are directly passed
                                atomic_action_args[atomic_arg_key] = value
                    print(f"\t Executing sub-step {i}:", atomic_action_name, atomic_action_args, "...")
                    self.execute_atomic_action(atomic_action_name, atomic_action_args, info_pool=info_pool, **kwargs)
                    # log screenshot during shortcut execution
                    if screenshot_log_dir is not None:
                        time.sleep(1)
                        screenshot_file = os.path.join(screenshot_log_dir, f"{iter}__{action.replace(' ', '')}__{i}-{atomic_action_name.replace(' ', '')}.png")
                        save_screenshot_to_file(self.adb, screenshot_file)
                        
                except Exception as e:
                    e += f"\nError in executing step {i}: {atomic_action_name} {atomic_action_args}"
                    print("Error in executing shortcut: ", action, e)
                    return action_object, i, e
            return action_object, len(shortcut["atomic_action_sequence"]), None
        else:
            if action.lower() in ["null", "none", "finish", "exit", "stop"]:
                print("Agent choose to finish the task. Action: ", action)
            else:
                print("Error! Invalid action name: ", action)
            info_pool.finish_thought = info_pool.last_action_thought
            return None, 0, None

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ", " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}


class ActionReflector(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the last action produced the expected behavior and to keep track of the overall progress."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "---\n"
        prompt += f"The attached two images are two phone screenshots before and after your last action. " 
        prompt += f"The width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        prompt += (
            "To help you better perceive the content in these screenshots, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom).\n"
        )
        prompt += (
            "Note that these information might not be entirely accurate. "
            "You should combine them with the screenshots to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Screen Information Before the Action ###\n"
        for clickable_info in info_pool.perception_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status before the action: "
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"


        prompt += "### Screen Information After the Action ###\n"
        for clickable_info in info_pool.perception_infos_post:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status after the action: "
        if info_pool.keyboard_post:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        # assert info_pool.last_action != ""
        prompt += f"Action: {info_pool.last_action}\n"
        prompt += f"Expectation: {info_pool.last_summary}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action was successful, update the progress status accordingly. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure. Note that for the “Swipe” action, it may take multiple attempts to display the expected content. Thus, for a \"Swipe\" action, if the screen shows new content, it usually meets the expectation.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\", \"B\" or \"C\":\n"
        prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

        prompt += "### Progress Status ###\n"
        prompt += "If the action was successful or partially successful, update the progress status. If the action failed, copy the previous progress status.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Progress Status ###")[0].replace("\n", " ").replace("  ", " ").strip()
        progress_status = response.split("### Progress Status ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"outcome": outcome, "error_description": error_description, "progress_status": progress_status}


class Notetaker(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to take notes of important content relevant to the user's request."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"

        prompt += "### Existing Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "### Current Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better perceive the content in this screenshot, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom)."
        )
        prompt += "The extracted information is as follows:\n"

        for clickable_info in info_pool.perception_infos_post:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += (
            "Note that this information might not be entirely accurate. "
            "You should combine it with the screenshot to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information above to identify any important content that needs to be recorded. IMPORTANT: Do not take notes on low-level actions; only keep track of significant textual or visual information relevant to the user's request.\n\n"

        prompt += "Provide your output in the following format:\n"
        prompt += "### Important Notes ###\n"
        prompt += "The updated important notes, combining the old and new ones. If nothing new to record, copy the existing important notes.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        important_notes = response.split("### Important Notes ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"important_notes": important_notes}


SHORTCUT_EXMPALE = """
{
    "name": "Tap_Type_and_Enter",
    "arguments": ["x", "y", "text"],
    "description": "Tap an input box at position (x, y), Type the \"text\", and then perform the Enter operation (useful for searching or sending messages).",
    "precondition": "There is a text input box on the screen.",
    "atomic_action_sequence":[
        {"name": "Tap", "arguments_map": {"x":"x", "y":"y"}},
        {"name": "Type", "arguments_map": {"text":"text"}},
        {"name": "Enter", "arguments_map": {}}
    ]
}
"""


class ExperienceReflectorShortCut(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant specializing in mobile phone operations. Your goal is to reflect on past experiences and provide insights to improve future interactions."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### Current Task ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"

        prompt += "### Atomic Actions ###\n"
        prompt += "Here are the atomic actions in the format of `name(arguments): description` as follows:\n"
        for action, value in ATOMIC_ACTION_SIGNITURES.items():
            prompt += f"{action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        prompt += "\n"

        prompt += "### Existing Shortcuts from Past Experience ###\n"
        if info_pool.shortcuts != {}:
            prompt += "Here are some existing shortcuts you have created:\n"
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"- {shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are provided.\n"
        prompt += "\n"

        prompt += "### Full Action History ###\n"
        if info_pool.action_history != []:
            latest_actions = info_pool.action_history
            latest_summary = info_pool.summary_history
            action_outcomes = info_pool.action_outcomes
            error_descriptions = info_pool.error_descriptions
            progress_status_history = info_pool.progress_status_history
            for act, summ, outcome, err_des, progress in zip(latest_actions, latest_summary, action_outcomes, error_descriptions, progress_status_history):
                if outcome == "A":
                    prompt += f"- Action: {act} | Description: {summ} | Outcome: Successful | Progress: {progress}\n"
                else:
                    prompt += f"- Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        if len(info_pool.future_tasks) > 0:
            prompt += "---\n"
            prompt += "### Future Tasks ###\n"
            prompt += "Here are some tasks that you might be asked to do in the future:\n"
            for task in info_pool.future_tasks:
                prompt += f"- {task}\n"
            prompt += "\n"

        prompt += "---\n"
        prompt += "Carefully reflect on the interaction history of the current task. Check if there are any subgoals that are accomplished by a sequence of successful actions and can be consolidated into new \"Shortcuts\" to improve efficiency for future tasks? These shortcuts are subroutines consisting of a series of atomic actions that can be executed under specific preconditions. For example, tap, type and enter text in a search bar or creating a new note in Notes."

        prompt += "Provide your output in the following format:\n\n"

        prompt += "### New Shortcut ###\n"
        prompt += "If you decide to create a new shortcut (not already in the existing shortcuts), provide your shortcut object in a valid JSON format which is detailed below. If not, put \"None\" here.\n"
        prompt += "A shortcut object contains the following fields: name, arguments, description, precondition, and atomic_action_sequence. The keys in the arguements need to be unique. The atomic_action_sequence is a list of dictionaries, each containing the name of an atomic action and a mapping of its atomic argument names to the shortcut's argument name. If an atomic action in the atomic_action_sequence does not take any arugments, set the `arguments_map` to an empty dict. \n"
        prompt += "IMPORTANT: The shortcut must ONLY include the Atomic Actions listed above. Create a new shortcut only if you are confident it will be useful in the future. Ensure that duplicated shortcuts with overly similar functionality are not included.\n"
        prompt += "PRO TIP: Avoid creating shortcuts with too many arguments, such as involving multiple taps at different positions. All coordinate arguments required for the shortcut should be visible on the current screen. Imagine that when you start executing the shortcut, you are essentially blind.\n"
        prompt += f"Follow the example below to format the shortcut. Avoid adding comments that could cause errors with json.loads().\n {SHORTCUT_EXMPALE}\n\n"
        return prompt

    def add_new_shortcut(self, short_cut_str: str, info_pool: InfoPool) -> str:
        if short_cut_str is None or short_cut_str == "None":
            return
        short_cut_object = extract_json_object(short_cut_str)
        if short_cut_object is None:
            print("Error! Invalid JSON for adding new shortcut: ", short_cut_str)
            return
        short_cut_name = short_cut_object["name"]
        if short_cut_name in info_pool.shortcuts:
            print("Error! The shortcut already exists: ", short_cut_name)
            return
        info_pool.shortcuts[short_cut_name] = short_cut_object
        print("Updated short_cuts:", info_pool.shortcuts)

    def parse_response(self, response: str) -> dict:
        new_shortcut = response.split("### New Shortcut ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"new_shortcut": new_shortcut}


class ExperienceReflectorTips(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant specializing in mobile phone operations. Your goal is to reflect on past experiences and provide insights to improve future interactions."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### Current Task ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"
    
        prompt += "### Existing Tips from Past Experience ###\n"
        if info_pool.tips != "":
            prompt += f"{info_pool.tips}\n\n"
        else:
            prompt += "No tips recorded.\n\n"

        prompt += "### Full Action History ###\n"
        if info_pool.action_history != []:
            latest_actions = info_pool.action_history
            latest_summary = info_pool.summary_history
            action_outcomes = info_pool.action_outcomes
            error_descriptions = info_pool.error_descriptions
            progress_status_history = info_pool.progress_status_history
            for act, summ, outcome, err_des, progress in zip(latest_actions, latest_summary, action_outcomes, error_descriptions, progress_status_history):
                if outcome == "A":
                    prompt += f"- Action: {act} | Description: {summ} | Outcome: Successful | Progress: {progress}\n"
                else:
                    prompt += f"- Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"
            
        if len(info_pool.future_tasks) > 0:
            prompt += "---\n"
            # if the setting provides future tasks explicitly
            prompt += "### Future Tasks ###\n"
            prompt += "Here are some tasks that you might be asked to do in the future:\n"
            for task in info_pool.future_tasks:
                prompt += f"- {task}\n"
            prompt += "\n"

        prompt += "---\n"
        prompt += "Carefully reflect on the interaction history of the current task. Check if there are any general tips that might be useful for handling future tasks, such as advice on preventing certain common errors?\n\n"

        prompt += "Provide your output in the following format:\n\n"

        prompt += "### Updated Tips ###\n"
        prompt += "If you have any important new tips to add (not already included in the existing tips), combine them with the current list. If there are no new tips, simply copy the existing tips here. Keep your tips concise and general.\n"
        return prompt

    def parse_response(self, response: str) -> dict:
        updated_tips = response.split("### Updated Tips ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"updated_tips": updated_tips}


class ExperienceRetrieverShortCut(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant specializing in mobile phone operations. Your goal is to select relevant shortcuts from previous experience to the current task."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, instruction, shortcuts) -> str:
        
        prompt = "### Existing Shortcuts from Past Experience ###\n"
        for shortcut, value in shortcuts.items():
            prompt += f"- Name: {shortcut} | Description: {value['description']}\n"
        
        prompt += "\n"
        prompt += "### Current Task ###\n"
        prompt += f"{instruction}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to pick the shortcuts that can be helpful to the current task. Remove shortcuts that are irrelevant to the current task.\n"

        prompt += "Provide your output in the following format:\n\n"
        prompt += "### Selected Shortcuts ###\n"
        prompt += "Provide your answer as a list of selected shortcut names: [\"shortcut1\", \"shortcut2\", ...]. If there are no relevant shortcuts, put \"None\" here.\n"
        return prompt
    
    def parse_response(self, response: str) -> dict:
        selected_shortcuts_str = response.split("### Selected Shortcuts ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        try:
            selected_shortcut_names = extract_json_object(selected_shortcuts_str, json_type="list")
            selected_shortcut_names = [s.strip() for s in selected_shortcut_names]
        except:
            selected_shortcut_names = []
            
        return {"selected_shortcut_names": selected_shortcut_names}


class ExperienceRetrieverTips(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        sysetm_prompt = "You are a helpful AI assistant specializing in mobile phone operations. Your goal is to select relevant tips from previous experience to the current task."
        operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
        return operation_history

    def get_prompt(self, instruction, tips) -> str:
        prompt = "### Existing Tips from Past Experience ###\n"
        prompt += f"{tips}\n\n"
        
        prompt += "\n"
        prompt += "### Current Task ###\n"
        prompt += f"{instruction}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to pick the tips that can be helpful to the current task. Remove tips that are irrelevant to the current task.\n"

        prompt += "Provide your output in the following format:\n\n"
        prompt += "### Selected Tips ###\n"
        prompt += "Tips that are generally useful and relevant to the current task. Feel free to reorganize the bullets. If there are no relevant tips, put \"None\" here.\n"

        return prompt
    
    def parse_response(self, response: str) -> dict:
        selected_tips = response.split("### Selected Tips ###")[-1].replace("\n", " ").replace("  ", " ").strip()        
        return {"selected_tips": selected_tips}