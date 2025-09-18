

import base64
import copy
import json
import time
import traceback
import numpy as np
import requests
import torch
from PIL import Image
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from x.qwen.image import smart_resize
END_POINT = "http://localhost:8000/v1"  # Replace with actual endpoint
BBOX_ENLARGE_FACTOR = 1.2
POINT_DISTANCE_THRESHOLD = 0.04
import base64
from io import BytesIO
def image_to_data_url(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"
def check_text(text_pred, text_gt):
    text_pred = text_pred.lower().strip()
    text_gt = text_gt.lower().strip()
    return (text_pred in text_gt) or (text_gt in text_pred)

def check_click(click, candidate_bbox, gt_point):
    if len(candidate_bbox):
        candidate_bbox = enlarge_bbox(candidate_bbox, scale_factor=BBOX_ENLARGE_FACTOR)
        for bbox in candidate_bbox:
            if (bbox[0] <= click[0] <= bbox[2]) and (bbox[1] <= click[1] <= bbox[3]):
                return True
    if gt_point is not None:
        return np.linalg.norm([gt_point[0]-click[0], gt_point[1]-click[1]]) <= POINT_DISTANCE_THRESHOLD
    return False

def predict_direction(start, end):
    x1, y1 = start
    x2, y2 = end
    
    # 计算坐标变化
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    # 判断方向
    if abs(delta_x) > abs(delta_y):
        if delta_x > 0:
            return 'right'
        else:
            return 'left'
    else:
        if delta_y > 0:
            return 'down'
        else:
            return 'up'

def enlarge_bbox(bbox_list, scale_factor=1.2)->np.ndarray:
    """
    将每个 bounding box 放大一定倍数。

    :param bbox_list: bounding box 列表, 每个 bbox 是一个包含四个值的元组或列表, 表示 (xmin, ymin, xmax, ymax)
    :param scale_factor: 放大倍数
    :return: 放大后的 bounding box 列表
    """
    bbox_array = np.array(bbox_list)
    try:
        x_min, y_min, x_max, y_max = \
            bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    except:
        print(bbox_array)
        raise
    
    # 计算每个 bounding box 的中心点
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # 计算每个 bounding box 的宽度和高度
    width = (x_max - x_min) * scale_factor
    height = (y_max - y_min) * scale_factor
    
    # 计算放大后的 bounding box 的新的坐标
    new_x_min = x_center - width / 2
    new_y_min = y_center - height / 2
    new_x_max = x_center + width / 2
    new_y_max = y_center + height / 2
    
    # 将新的坐标组合成 bounding box 列表
    enlarged_bbox_list = np.vstack((new_x_min, new_y_min, new_x_max, new_y_max)).T
    
    return enlarged_bbox_list

def norm_coordinate(action, width, height):
    if 'candidate_bbox' in action:
        action['candidate_bbox'] = [[_[0]/width, _[1]/height, _[2]/width, _[3]/height] for _ in action['candidate_bbox']]
    if 'coordinate' in action:
        action['coordinate'] = [action['coordinate'][0]/width, action['coordinate'][1]/height]
    if 'coordinate2' in action:
        action['coordinate2'] = [action['coordinate2'][0]/width, action['coordinate2'][1]/height]
    return action

def evaluate_android_control_action(pred_action, current_check_pam, width, height, resized_width, resized_height,ignore_actions=[]):
    pred_action = norm_coordinate(copy.deepcopy(pred_action), resized_width, resized_height) # todo use resized width
    current_check_pam = norm_coordinate(copy.deepcopy(current_check_pam), width, height)
    print(current_check_pam, pred_action)
    if current_check_pam['action'] in ignore_actions:
        return True, True

    # type correct is ok
    if current_check_pam['action'] == 'wait':
        if pred_action['action'] == 'wait':
            return True, True
        return False, False
    elif current_check_pam['action'] == 'system_button':
        if pred_action['action'] == 'system_button':
            return True, current_check_pam['button'].lower().strip() == pred_action['button'].lower().strip()
        else:
            return False, False
    elif current_check_pam['action'] == 'type':
        if pred_action['action'] == 'type':
            return True, check_text(pred_action['text'], current_check_pam['text'])
        else:
            return False, False
    elif current_check_pam['action'] == 'open':
        if pred_action['action'] == 'open':
            return True, check_text(pred_action['text'], current_check_pam['text'])
        elif pred_action['action'] == 'click':
            if len(current_check_pam.get('candidate_bbox', []))>0:
                # 图中存在候选图标 检查是否命中
                return True, check_click(pred_action['coordinate'], current_check_pam['candidate_bbox'], gt_point=[(current_check_pam['candidate_bbox'][0][0]+current_check_pam['candidate_bbox'][0][2])/2, (current_check_pam['candidate_bbox'][0][1]+current_check_pam['candidate_bbox'][0][3])/2])
            else:
                # 图中没有图标 点击行为必然错误
                return False, False
        else:
            return False, False
    elif current_check_pam['action'] == 'swipe':
        if pred_action['action'] == 'swipe':
            # direction = predict_direction(pred_action['coordinate'], pred_action['coordinate2'])
            # gt_direction = current_check_pam['direction'] if 'direction' in current_check_pam else 
            # # the down up is wrong in gt, left right is ok
            # if gt_direction == 'down':
            #     gt_direction = 'up'
            # elif gt_direction == 'up':
            #     gt_direction = 'down'
            return True, True
        else:
            return False, False
    elif current_check_pam['action'] in ['long_press', 'click']:
        if pred_action['action'] == current_check_pam['action']:
            return True, check_click(pred_action['coordinate'], current_check_pam['candidate_bbox'], gt_point=current_check_pam['coordinate'])
        else:
            return False, False
    raise NotImplementedError


def message_translate(messages, to_format='dashscope'):
    screenshot_list = []
    if to_format == 'dashscope':
        return messages,screenshot_list
    
    if to_format == 'openai':
        messages = copy.deepcopy(messages)
        for msg in messages:
            if isinstance(msg['content'], str):
                msg['content'] = [msg['content']]
            new_contents = []
            for content in msg['content']:
                if  isinstance(content, str):
                    new_contents.append({"type": "text", 'text': content})
                elif 'text' in content:
                    new_contents.append({"type": "text", 'text': content['text']})
                elif 'image' in content:
                    screenshot_list.append(content['image'])
                    new_contents.append({"type": "image_url", "image_url": {"url": content['image']}})
                else:
                    raise NotImplementedError
            msg['content'] = new_contents
        return messages,screenshot_list
    if to_format == 'qwen':
        messages = copy.deepcopy(messages)
        for msg in messages:
            if isinstance(msg['content'], str):
                msg['content'] = [msg['content']]
            new_contents = []
            for content in msg['content']:
                if  isinstance(content, str):
                    new_contents.append({"type": "text", 'text': content})
                elif 'text' in content:
                    new_contents.append({"type": "text", 'text': content['text']})
                elif 'image' in content:
                    screenshot_list.append(content['image'])
                    new_contents.append({"type": "image", "image": content['image']})
                else:
                    raise NotImplementedError
            msg['content'] = new_contents
        return messages,screenshot_list

def call_mobile_agent_vllm(messages, model_name='qwen25vl_7b_1im_high_nlp_v9.7.1.1_mix_single', screenshot_list=[], retry_flag=False):

    messages ,screenshot_list = message_translate(messages, to_format='openai')
    # print(screenshot_list)
    screenshot_ptr = 0
    for msg in messages:
        # print(msg)
        for content in msg['content']:
            if 'image_url' in content:

                url = image_to_data_url(Image.open(screenshot_list[screenshot_ptr]))
                content['image_url']['url'] = url
                screenshot_ptr += 1
    # print(messages)
    assert screenshot_ptr == len(screenshot_list)
    output = ''
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
            print(output)
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

    return output

def find_last_image_ele(messages):
    image_ele = None
    for msg in messages:
        for content in msg['content']:
            if 'image' in content:
                image_ele = content

    last_image_path = image_ele['image']
    width, height = Image.open(last_image_path).size
    # width, height = line['width'], line['height']
    resized_height, resized_width = smart_resize(height, width, max_pixels=12800*28*28)
    return last_image_path,width, height, resized_width, resized_height
