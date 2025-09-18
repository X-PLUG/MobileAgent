import base64
import io
import json
import time
import traceback
import demjson3

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

def predict(model_name,instruction, low_instruction, history_list, image):
    """
    通过远程API调用UI-TARS模型，图片传image_url
    返回 Thought + Action 原始字符串
    """
    try:
        # 将 PIL.Image 上传到 OSS，获得 URL

        url = image_to_data_url(image)  # 返回图片可访问的临时url

        # 构造 history 文本
        history_str_list = []
        for h in history_list[-4:]:  # 只取最近4步
            # 用户截图
            image_url = image_to_data_url(Image.open(h["image_path"]))  # 返回图片可访问的临时url
            history_str_list.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image_url}
                ]
            })
            # 助手动作
            action = h.get("action", "")
            thought = h.get("low_instruction", "")
            history_str_list.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thought: {thought}\nAction: {action}"}
                ]
            })

        # 构造 UI-TARS 专用 prompt
        text = (
            "You are a GUI agent. You are given a task and your action history, with screenshots. "
            "You need to perform the next action to complete the task. \n\n"
            "## Output Format\n\n"
            "Thought: ...\n"
            "Action: ...\n\n\n"
            "## Action Space\n"
            "click(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
            "long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')\n"
            "type(content='')\n"
            "scroll(direction='down or up or right or left')\n"
            "press_back()\n"
            "press_home()\n"
            "wait()\n"
            "## Note\n"
            "- Use English in Thought part.\n\n"
            "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
            "## User Instruction\n" + instruction
        )

        # 构造 API messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful computer vision and GUI agent."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
        messages.extend(history_str_list)
        # 当前步骤的图片
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}}
            ]
        })

        # === 远程 API 请求 ===
        from openai import OpenAI
        bot = OpenAI(
            api_key="EMPTY",
            base_url=END_POINT,  # 换成你的UI-TARS部署地址
            timeout=60
        )
        kwargs = {'extra_body': {"top_k": 1}}
        chat_completion = bot.chat.completions.create(
            model=model_name,  # 你部署的UI-TARS模型名
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            **kwargs
        )

        output = chat_completion.choices[0].message.content
        return output.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[UI-TARS-predict Error]: {e}")
        # 返回一个兜底动作
        return "Thought: Unable to process\nAction: finished()"


import json
import math
import re


def uitars2minicpm(action_str):
    """
    Convert the ui-tars action string to the minicpm schema format
    
    Args:
        action_str (str): like "click(start_box='<|box_start|>(558,925)<|box_end|>')"
        
    Returns:
        dict: new format action dictionary
    """
    result = {"STATUS": "continue"}
    
    # auxiliary function to extract coordinates
    def extract_coords(s):
        # directly find and extract the coordinates in the parentheses
        first_bracket = s.find("(")
        start = s.find("(", first_bracket + 1)
        end = s.find(")")
        if start != -1 and end != -1:
            coords_str = s[start+1:end].strip()  # extract the content in (x,y)
            x, y = coords_str.split(",")
            return [int(x), int(y)]
        raise ValueError(f"Cannot find coordinates in the string: {s}")
    
    if "click(" in action_str:
        result["POINT"] = extract_coords(action_str)
        
    elif "long_press(" in action_str:
        result["POINT"] = extract_coords(action_str)
        if "time='" in action_str:
            time = action_str.split("time='")[1].split("'")[0]
            result["duration"] = int(time) if time else 1000
            
    elif "type(" in action_str:
        content = action_str.split("content='")[1].split("'")[0]
        result["TYPE"] = content
        
    elif "scroll(" in action_str:
        direction = action_str.split("direction='")[1].split("'")[0]
        result["POINT"] = [500, 500]  # screen center point
        #need reverse direction
        if direction == "down":
            direction = "up"
        elif direction == "up":
            direction = "down"
        elif direction == "right":
            direction = "left"
        elif direction == "left":
            direction = "right"
        result["to"] = direction
    elif "press_back()" in action_str:
        result["PRESS"] = "BACK"
        
    elif "press_home()" in action_str:
        result["PRESS"] = "HOME"
        
    elif "wait()" in action_str:
        result["duration"] = 200
        
    elif "finished()" in action_str:
        result["STATUS"] = "finish"
    elif "open_app(app_name=" in action_str:
        result["OPEN_APP"] = action_str.split("app_name='")[1].split("'")[0]
    else:
        print(f"Error, invalid action: {action_str}")
        
    return result,action_str

def extract_thought_action(output_str):
    """
    从模型输出的 Thought / Action 文本中解析出 thought_text 和 action_text

    Args:
        output_str (str): 模型返回的完整字符串，例如：
            Thought: I think
            Action: Press_back()

    Returns:
        tuple: (thought, action) - 都是 str，找不到时返回空字符串
    """
    thought = ""
    action = ""

    # 用正则匹配 Thought: 后面的内容
    thought_match = re.search(r"Thought\s*:\s*(.+)", output_str, flags=re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 用正则匹配 Action: 后面的内容
    action_match = re.search(r"Action\s*:\s*(.+)", output_str, flags=re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()

    return thought, action

if __name__ == "__main__":
    agent_action = {"thought":"click on the Add to Cart button","POINT":[501,952]}
    print(map_action_space2qwenvl(agent_action))
