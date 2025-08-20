import base64
import requests
import time

import pdb
import dashscope
from dashscope import MultiModalConversation

from PIL import Image
import io
from openai import OpenAI
import json

def resize_encode_image(image_path, screen_scale_ratio=0.5):
    with Image.open(image_path) as img:
        new_width = int(img.width * screen_scale_ratio)
        new_height = int(img.height * screen_scale_ratio)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        buffered = io.BytesIO()
        resized_img.save(buffered, format="PNG")

        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode('utf-8')





def inference_chat(chat, model, api_url, token):    

    messages = []
    for role, content in chat:
        messages.append({"role": role, "content": content})

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=token, 
        base_url=api_url,
    )


    num_try = 5
    for _ in range(num_try):
        try:
            completion = client.chat.completions.create(
                model=model, # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=messages
            )
        except:
            print("Network Error:")
            try:
                print(completion.model_dump_json())
            except:
                print("Request Failed")
            time.sleep(2)
        else:
            break

    
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']

    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {token}"
    # }

    # data = {
    #     "model": model,
    #     "messages": [],
    #     "max_tokens": 2048,
    #     'temperature': 0.0,
    #     "seed": 1234
    # }

