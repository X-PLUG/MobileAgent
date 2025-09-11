from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re

@dataclass
class InfoPool:
    """Keeping track of all information across the agents."""
    
    # User input / accumulated knowledge
    instruction: str = ""
    task_name: str = ""
    additional_knowledge_manager: str = ""
    additional_knowledge_executor: str = ""
    add_info_token = "[add_info]"
    
    ui_elements_list_before: str = "" # List of UI elements with index
    ui_elements_list_after: str = "" # List of UI elements with index
    action_pool: list = field(default_factory=list)

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
    completed_plan: str = ""
    progress_status: str = ""
    progress_status_history: list = field(default_factory=list)
    finish_thought: str = ""
    current_subgoal: str = ""
    # prev_subgoal: str = ""
    err_to_manager_thresh: int = 2

    # future tasks
    future_tasks: list = field(default_factory=list)

class BaseAgent(ABC):
    @abstractmethod
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    @abstractmethod
    def parse_response(self, response: str) -> dict:
        pass

class Manager(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are an agent who can operate an Android phone on behalf of a user. Your goal is to track progress and devise high-level plans to achieve the user's requests.\n\n"
        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        task_specific_note = ""
        if ".html" in info_pool.instruction:
            task_specific_note = "NOTE: The .html file may contain additional interactable elements, such as a drawing canvas or a game. Do not open other apps without completing the task in the .html file."
        elif "Audio Recorder" in info_pool.instruction:
            task_specific_note = "NOTE: The stop recording icon is a white square, located fourth from the left at the bottom. Please do not click the circular pause icon in the middle."

        if info_pool.plan == "":
            # first time planning
            prompt += "---\n"
            prompt += "Make a high-level plan to achieve the user's request. If the request is complex, break it down into subgoals. The screenshot displays the starting state of the phone.\n"
            prompt += "IMPORTANT: For requests that explicitly require an answer, always add 'perform the `answer` action' as the last step to the plan!\n\n"
            if task_specific_note != "":
                prompt += f"{task_specific_note}\n\n"
            
            prompt += "### Guidelines ###\n"
            prompt += "The following guidelines will help you plan this request.\n"
            prompt += "General:\n"
            prompt += "Use search to quickly find a file or entry with a specific name, if search function is applicable.\n"
            prompt += "Task-specific:\n"
            if info_pool.additional_knowledge_manager != "":
                prompt += f"{info_pool.additional_knowledge_manager}\n\n"
            else:
                prompt += f"{info_pool.add_info_token}\n\n"
            
            prompt += "Provide your output in the following format which contains two parts:\n"
            prompt += "### Thought ###\n"
            prompt += "A detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "1. first subgoal\n"
            prompt += "2. second subgoal\n"
            prompt += "...\n"
        else:
            if info_pool.completed_plan != "No completed subgoal.":
                prompt += "### Historical Operations ###\n"
                prompt += "Operations that have been completed before:\n"
                prompt += f"{info_pool.completed_plan}\n\n"
            prompt += "### Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"
            prompt += f"### Last Action ###\n"
            prompt += f"{info_pool.last_action}\n\n"
            prompt += f"### Last Action Description ###\n"
            prompt += f"{info_pool.last_summary}\n\n"
            prompt += "### Important Notes ###\n"
            if info_pool.important_notes != "":
                prompt += f"{info_pool.important_notes}\n\n"
            else:
                prompt += "No important notes recorded.\n\n"
            prompt += "### Guidelines ###\n"
            prompt += "The following guidelines will help you plan this request.\n"
            prompt += "General:\n"
            prompt += "Use search to quickly find a file or entry with a specific name, if search function is applicable.\n"
            prompt += "Task-specific:\n"
            if info_pool.additional_knowledge_manager != "":
                prompt += f"{info_pool.additional_knowledge_manager}\n\n"
            else:
                prompt += f"{info_pool.add_info_token}\n\n"
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
            prompt += "Carefully assess the current status and the provided screenshot. Check if the current plan needs to be revised.\n Determine if the user request has been fully completed. If you are confident that no further actions are required, mark the plan as \"Finished\" in your output. If the user request is not finished, update the plan. If you are stuck with errors, think step by step about whether the overall plan needs to be revised to address the error.\n"
            prompt += "NOTE: 1. If the current situation prevents proceeding with the original plan or requires clarification from the user, make reasonable assumptions and revise the plan accordingly. Act as though you are the user in such cases. 2. Please refer to the helpful information and steps in the Guidelines first for planning. 3. If the first subgoal in plan has been completed, please update the plan in time according to the screenshot and progress to ensure that the next subgoal is always the first item in the plan. 4. If the first subgoal is not completed, please copy the previous round's plan or update the plan based on the completion of the subgoal.\n"
            prompt += "IMPORTANT: If the next steps require an `answer` action, make sure that there is a plan to perform the `answer` action. In this case, you should not mark the plan as \"Finished\" unless the last action is `answer`.\n"
            if task_specific_note != "":
              prompt += f"{task_specific_note}\n\n"

            prompt += "Provide your output in the following format, which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "An explanation of your rationale for the updated plan and current subgoal.\n\n"
            prompt += "### Historical Operations ###\n"
            prompt += "Try to add the most recently completed subgoal on top of the existing historical operations. Please do not delete any existing historical operation. If there is no newly completed subgoal, just copy the existing historical operations.\n\n"
            prompt += "### Plan ###\n"
            prompt += "Please update or copy the existing plan according to the current page and progress. Please pay close attention to the historical operations. Please do not repeat the plan of completed content unless you can judge from the screen status that a subgoal is indeed not completed.\n"
            
        return prompt

    def parse_response(self, response: str) -> dict:
        if "### Historical Operations" in response:
            thought = response.split("### Thought")[-1].split("### Historical Operations")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
            completed_subgoal = response.split("### Historical Operations")[-1].split("### Plan")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        else:
            thought = response.split("### Thought")[-1].split("### Plan")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
            completed_subgoal = "No completed subgoal."
        plan = response.split("### Plan")[-1].replace("\n", " ").replace("  ", " ").replace("###", "").strip()#.split("### Current Subgoal")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        return {"thought": thought, "completed_subgoal": completed_subgoal,  "plan": plan}#, "current_subgoal": current_subgoal

from utils.new_json_action import *

ATOMIC_ACTION_SIGNITURES_noxml = {
    ANSWER: {
        "arguments": ["text"],
        "description": lambda info: "Answer user's question. Usage example: {\"action\": \"answer\", \"text\": \"the content of your answer\"}"
    },
    CLICK: {
        "arguments": ["coordinate"],
        "description": lambda info: "Click the point on the screen with specified (x, y) coordinates. Usage Example: {\"action\": \"click\", \"coordinate\": [x, y]}"
    },
    LONG_PRESS: {
        "arguments": ["coordinate"],
        "description": lambda info: "Long press on the position (x, y) on the screen. Usage Example: {\"action\": \"long_press\", \"coordinate\": [x, y]}"
    },
    TYPE: {
        "arguments": ["text"],
        "description": lambda info: "Type text into current activated input box or text field. If you have activated the input box, you can see the words \"ADB Keyboard {on}\" at the bottom of the screen. If not, click the input box to confirm again. Please make sure the correct input box has been activated before typing. Usage Example: {\"action\": \"type\", \"text\": \"the text you want to type\"}"
    },
    SYSTEM_BUTTON: {
        "arguments": ["button"],
        "description": lambda info: "Press a system button, including back, home, and enter. Usage example: {\"action\": \"system_button\", \"button\": \"Home\"}"
    },
    SWIPE: {
        "arguments": ["coordinate", "coordinate2"],
        "description": lambda info: "Scroll from the position with coordinate to the position with coordinate2. Please make sure the start and end points of your swipe are within the swipeable area and away from the keyboard (y1 < 1400). Usage Example: {\"action\": \"swipe\", \"coordinate\": [x1, y1], \"coordinate2\": [x2, y2]}"
    }
}

INPUT_KNOW = "If you've activated an input field, you'll see \"ADB Keyboard {on}\" at the bottom of the screen. This phone doesn't display a soft keyboard. So, if you see \"ADB Keyboard {on}\" at the bottom of the screen, it means you can type. Otherwise, you'll need to tap the correct input field to activate it."

class Executor(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are an agent who can operate an Android phone on behalf of a user. Your goal is to decide the next action to perform based on the current state of the phone and the user's request.\n\n"

        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"
        
        prompt += "### Current Subgoal ###\n"
        current_goal = info_pool.plan
        current_goal = re.split(r'(?<=\d)\. ', current_goal)
        truncated_current_goal = ". ".join(current_goal[:4]) + '.'
        truncated_current_goal = truncated_current_goal[:-2].strip()
        prompt += f"{truncated_current_goal}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        if info_pool.additional_knowledge_executor != "":
            prompt += "### Guidelines ###\n"
            prompt += f"{info_pool.additional_knowledge_executor}\n"

        if "exact duplicates" in info_pool.instruction:
            prompt += "Task-specific:\nOnly two items with the same name, date, and details can be considered duplicates.\n\n"
        elif "Audio Recorder" in info_pool.instruction:
            prompt += "Task-specific:\nThe stop recording icon is a white square, located fourth from the left at the bottom. Please do not click the circular pause icon in the middle.\n\n"
        else:
            prompt += "\n"
        
        prompt += "---\n"        
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions.\n\n"
        
        prompt += "#### Atomic Actions ####\n"
        prompt += "The atomic action functions are listed in the format of `action(arguments): description` as follows:\n"

        for action, value in ATOMIC_ACTION_SIGNITURES_noxml.items():
            prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"

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
            
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "---\n"
        prompt += "IMPORTANT:\n1. Do NOT repeat previously failed actions multiple times. Try changing to another action.\n"
        prompt += "2. Please prioritize the current subgoal.\n\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        prompt += "Provide a detailed explanation of your rationale for the chosen action.\n\n"

        prompt += "### Action ###\n"
        prompt += "Choose only one action or shortcut from the options provided.\n"
        prompt += "You must provide your decision using a valid JSON format specifying the `action` and the arguments of the action. For example, if you want to type some text, you should write {\"action\":\"type\", \"text\": \"the text you want to type\"}.\n\n"
        
        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action. Do not describe expected outcome.\n"
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought")[-1].split("### Action")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        action = response.split("### Action")[-1].split("### Description")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        description = response.split("### Description")[-1].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        return {"thought": thought, "action": action, "description": description}

class ActionReflector(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are an agent who can operate an Android phone on behalf of a user. Your goal is to verify whether the last action produced the expected behavior and to keep track of the overall progress.\n\n"

        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"
        
        prompt += "### Progress Status ###\n"
        if info_pool.completed_plan != "":
            prompt += f"{info_pool.completed_plan}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "---\n"
        prompt += "The two attached images are phone screenshots taken before and after your last action. \n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        prompt += f"Action: {info_pool.last_action}\n"
        prompt += f"Expectation: {info_pool.last_summary}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action was successful, update the progress status accordingly. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure.\n\n"
        prompt += "Note: For swiping to scroll the screen to view more content, if the content displayed before and after the swipe is exactly the same, the swipe is considered to be C: Failed. The last action produces no changes. This may be because the content has been scrolled to the bottom.\n\n"

        prompt += "Provide your output in the following format containing two parts:\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your response as \"A\", \"B\" or \"C\":\n"
        prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome")[-1].split("### Error Description")[0].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        error_description = response.split("### Error Description")[-1].replace("\n", " ").replace("###", "").replace("  ", " ").strip()
        return {"outcome": outcome, "error_description": error_description}

class Notetaker(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to take notes of important content relevant to the user's request.\n\n"

        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"

        prompt += "### Existing Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        if "transactions" in info_pool.instruction and "Simple Gallery" in info_pool.instruction:
            prompt += "### Guideline ###\nYou can only record the transaction information in DCIM, because the other transactions are irrelevant to the task.\n"
        elif "enter their product" in info_pool.instruction:
            prompt += "### Guideline ###\nPlease record the number that appears each time so that you can calculate their product at the end.\n"
        
        prompt += "---\n"
        prompt += "Carefully examine the information above to identify any important content on the current screen that needs to be recorded.\n"
        prompt += "IMPORTANT:\nDo not take notes on low-level actions; only keep track of significant textual or visual information relevant to the user's request. Do not repeat user request or progress status. Do not make up content that you are not sure about.\n\n"

        prompt += "Provide your output in the following format:\n"
        prompt += "### Important Notes ###\n"
        prompt += "The updated important notes, combining the old and new ones. If nothing new to record, copy the existing important notes.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        important_notes = response.split("### Important Notes")[-1].replace("\n", " ").replace("  ", " ").replace("###", "").strip()
        return {"important_notes": important_notes}