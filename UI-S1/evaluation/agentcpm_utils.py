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
SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

import base64
from io import BytesIO

def image_to_data_url(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"



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

def predict(model_name: str,text_prompt: str, image: Image.Image):
    url = image_to_data_url(image)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"<Question>{text_prompt}</Question>\n当前屏幕截图：(<image>./</image>)"},
            {"type": "image_url", "image_url": {"url": url}}
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
                base_url=END_POINT, '
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

def map_action_space2qwenvl(agent_action, screen_size=(1080, 1920)) -> dict:
    """
    将 agent 输出的 JSON 字符串动作映射到 RAW_SPACE 定义的动作空间。
    
    Args:
        agent_action_str: str, 来自智能体的动作 JSON 字符串
        screen_size: (width, height), 屏幕分辨率，默认 1080x1920
    
    Returns:
       
       if dict: 格式为 {"action": ..., "arg": ...} 的动作，出错时返回 wait 或 terminate failure
    """
    if not isinstance(agent_action, dict):
        try:
            agent_action = json.loads(agent_action)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Invalid JSON received: {e}")
            return {"action": "terminate", "status": "failure"}
    width, height = screen_size

    def to_pixels(loc):
        """将 [0,1000] 范围的相对坐标转为像素坐标"""
        x = int(loc[0] * width / 1000)
        y = int(loc[1] * height / 1000)
        return [x, y]

    # 1. STATUS 优先级最高
    status = agent_action.get("STATUS")
    if status == "finish" or status == "satisfied":
        return {"action": "terminate", "status": "success"}
    elif status in ["impossible", "interrupt", "need_feedback"]:
        return {"action": "terminate", "status": "failure"}

    # 2. PRESS: 系统按钮
    press = agent_action.get("PRESS")
    if press:
        button_map = {
            "HOME": "Home",
            "BACK": "Back",
            "ENTER": "Enter"
        }
        return {"action": "system_button", "button": button_map.get(press, "Menu")}

    # 3. TYPE: 输入文本
    type_text = agent_action.get("TYPE")
    if type_text is not None:
        return {"action": "type", "text": type_text}

    # 4. to: 移动或滑动
    to = agent_action.get("to")
    point = agent_action.get("POINT")
    if to and point:
        start = to_pixels(point)
        if isinstance(to, str):
            # 方向滑动
            swipe_distance = 200  # 滑动距离（像素）
            direction = to.lower()
            x, y = start
            if direction == "up":
                end = [x, max(y - swipe_distance, 0)]
            elif direction == "down":
                end = [x, min(y + swipe_distance, height)]
            elif direction == "left":
                end = [max(x - swipe_distance, 0), y]
            elif direction == "right":
                end = [min(x + swipe_distance, width), y]
            else:
                end = [x, y]
            return {"action": "swipe", "coordinate": start, "coordinate2": end}
        elif isinstance(to, list) and len(to) == 2:
            # 移动到某个位置
            end = to_pixels(to)
            return {"action": "swipe", "coordinate": start, "coordinate2": end}

    # 5. POINT: 点击
    if point:
        coord = to_pixels(point)
        return {"action": "click", "coordinate": coord}

    # 6. duration: 等待
    duration = agent_action.get("duration", 0)
    if duration > 0:
        seconds = duration / 1000.0
        return {"action": "wait", "time": seconds}

    # 默认动作：继续（无操作）
    return {"action": "wait", "time": 0.1}


if __name__ == "__main__":
    agent_action = {"thought":"click on the Add to Cart button","POINT":[501,952]}
    print(map_action_space2qwenvl(agent_action))
