import base64
import io
import json
import traceback
import time

import requests
from PIL import Image

END_POINT = "http://localhost:8000/v1/"  # Replace with actual endpoint
# system prompt
ACTION_SCHEMA = json.load(open('/evaluation/agentcpm_schema.json', encoding="utf-8"))
items = list(ACTION_SCHEMA.items())
insert_index = 3
items.insert(insert_index, ("required", ["thought"])) # enable/disable thought by setting it to "required"/"optional"
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f"""\nYou are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.\n\nYour expertise covers two types of digital tasks:\n    - Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.\n    - Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.\n\n\nYou are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:\n\n1. Basic Actions\nBasic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. \nBasic Action 1: CLICK \n    - purpose: Click at the specified position.\n    - format: CLICK <point>[[x-axis, y-axis]]</point>\n    - example usage: CLICK <point>[[101, 872]]</point>\n       \nBasic Action 2: TYPE\n    - purpose: Enter specified text at the designated location.\n    - format: TYPE [input text]\n    - example usage: TYPE [Shanghai shopping mall]\n\nBasic Action 3: SCROLL\n    - purpose: Scroll in the specified direction.\n    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n    - example usage: SCROLL [UP]\n    \n2.Custom Actions\nCustom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n\n\nCustom Action 1: LONG_PRESS \n    - purpose: Long press at the specified position.\n    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\nCustom Action 2: PRESS_BACK\n    - purpose: Press a back button to navigate to the previous screen.\n    - format: PRESS_BACK\n    - example usage: PRESS_BACK\n\nCustom Action 3: PRESS_HOME\n    - purpose: Press a home button to navigate to the home page.\n    - format: PRESS_HOME\n    - example usage: PRESS_HOME\n\nCustom Action 4: PRESS_RECENT\n    - purpose: Press the recent button to view or switch between recently used applications.\n    - format: PRESS_RECENT\n    - example usage: PRESS_RECENT\n\nCustom Action 5: IMPOSSIBLE\n    - purpose: Wait for the screen to load.\n    - format: WAIT\n    - example usage: WAIT\n\nCustom Action 6: COMPLETE\n    - purpose: Indicate the task is finished.\n    - format: COMPLETE\n    - example usage: COMPLETE\n\n\nIn most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.\nThoughts: Clearly outline your reasoning process for current step.\nActions: Specify the actual actions you will take based on your reasoning.\n\nYour current task instruction, action history, and associated screenshot are as follows:\nScreenshot: """

import base64
from io import BytesIO
def image_to_data_url(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"
def build_history_actions_str(history_list):
    history = ""
    for i, step_history in enumerate(history_list):
        history += f"Step {i+1}: {step_history}\n"
    return history
def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64-encoded string."""
    with io.BytesIO() as in_mem_file:
        image.save(in_mem_file, format="JPEG")
        in_mem_file.seek(0)
        return base64.b64encode(in_mem_file.read()).decode("utf-8")

def __resize__(origin_img):
    resolution = origin_img.size
    w,h = resolution
    max_line_res = 1120
    if max_line_res is not None:
        max_line = max_line_res
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
    img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
    return img

def predict(model_name, instruction, low_instruction, history,image):

    url = image_to_data_url(image)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": f"""\nTask: {instruction} You need to: {low_instruction}\nHistory: \n{history}\n"""}
        ]}
    ]

    # payload = {
    #     "model": "AgentCPM-GUI",  # Your model name
    #     "temperature": 0.1,
    #     "messages": messages,
    #     "max_tokens": 2048,
    # }

    # headers = {
    #     "Content-Type": "application/json",
    # }

    # response = requests.post('http://47.239.63.127:8000/v1/chat/completions', headers=headers, json=payload)
    # print(response)
    for i in range(32768):
        try:
            from openai import OpenAI
            bot = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key="EMPTY",
                base_url=END_POINT, 
                timeout=30
            )
            # # if os.environ.get('SEARCH_MODE','') or retry_flag:
            # if os.environ.get('SEARCH_MODE',''):

            #     # kwargs = {'extra_body': {"top_k": 50}, 'top_p': 0.9}
            #     kwargs = {'extra_body': {"top_k": 500}, 'temperature': 10.0}
            # else:
            kwargs = {'extra_body': {"top_k": 1}} # TODO
            # print(kwargs)
            chat_completion_from_url = bot.chat.completions.create(model=model_name, messages=messages, **kwargs)
            # logging.error(chat_completion_from_url)
            output = chat_completion_from_url.choices[0].message.content
            return output
        except:
            traceback.print_exc()
            print("Network Error:")
            try:
                print(output)
            except:
                print("Request Failed")
            time.sleep(2)
        else:
            break

import math

import json

def os_atlas_2minicpm(action_str):
    """
    Convert a string containing low-level thinking and action information to minicpm schema format
    
    Args:
        action_str (str): String containing low-level thinking and action information
        
    Returns:
        dict: Action dictionary in new format
    """
    result = {"STATUS": "continue"}
    
    try:
        # 提取动作部分
        action_start = action_str.find("Actions:")
        action_content = action_str[action_start + len("Actions:"):].strip()
        if action_start == -1:
            action_start = action_str.find("actions:")
            action_content = action_str[action_start + len("actions:"):].strip()
        if action_start == -1:
            raise ValueError("Cannot find action information")
        
        action_content = action_str[action_start + len("Actions:"):].strip()
        
        if "CLICK" in action_content:
            # Extract coordinates
            start = action_content.find("[[") + 2
            end = action_content.find("]]")
            coords_str = action_content[start:end]
            x, y = map(int, coords_str.split(","))
            result["POINT"] = [x, y]
        
        elif "TYPE" in action_content:
            # Extract input text
            start = action_content.find("[") + 1
            end = action_content.find("]")
            text = action_content[start:end]
            result["TYPE"] = text
        
        elif "SCROLL" in action_content:
            # Extract scroll direction
            start = action_content.find("[") + 1
            end = action_content.find("]")
            direction = action_content[start:end]
            direction = direction.strip().lower()
             # If has low instruction, need to reverse direction
            # if use_low_instruction:
            #     if direction == "UP":
            #         direction = "DOWN"
            #     elif direction == "DOWN":
            #         direction = "UP"
            #     elif direction == "LEFT":
            #         direction = "RIGHT"
            #     elif direction == "RIGHT":
            #         direction = "LEFT"
            result["to"] = direction
            result["POINT"] = [500, 500]  # 屏幕中心点
        
        elif "LONG_PRESS" in action_content:
            # Extract coordinates
            start = action_content.find("[[") + 2
            end = action_content.find("]]")
            coords_str = action_content[start:end]
            x, y = map(int, coords_str.split(","))
            result["POINT"] = [x, y]
            result["duration"] = 1000  # Default long press duration
        
        elif "PRESS_BACK" in action_content:
            result["PRESS"] = "BACK"
        
        elif "PRESS_HOME" in action_content:
            result["PRESS"] = "HOME"
        
        elif "PRESS_RECENT" in action_content:
            result["PRESS"] = "RECENT"
        
        elif "WAIT" in action_content:
            result["duration"] = 200
        
        elif "COMPLETE" in action_content:
            result["STATUS"] = "finish"
        
        else:
            print(f"Error, invalid action: {action_content}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    return result,action_content


if __name__ == "__main__":
    agent_action = {"thought":"click on the Add to Cart button","POINT":[501,952]}
    print(map_action_space2qwenvl(agent_action))
