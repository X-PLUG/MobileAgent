import base64
import copy
import json
import re
import time
import traceback

import numpy as np
import requests
# from x.data.agent.fake_uitars import PlainCallFormat
# from x.data.agent.space.std_space import RAW_SPACE
# from x.data.text import parse_tags
import torch

BBOX_ENLARGE_FACTOR = 1.2
POINT_DISTANCE_THRESHOLD = 0.04


def norm_coordinate(action, width, height):
    if 'candidate_bbox' in action and len(action['candidate_bbox']) == 4: # fix bug
        x, y, w, h = action['candidate_bbox']
        action['candidate_bbox'] = [[x / width, y / height, w / width, h / height]]
    if 'coordinate' in action:
        action['coordinate'] = [action['coordinate'][0]/width, action['coordinate'][1]/height]
    if 'coordinate2' in action:
        action['coordinate2'] = [action['coordinate2'][0]/width, action['coordinate2'][1]/height]
    return action


def check_text(text_pred, text_gt, text_retrict=False):
    text_pred = text_pred.lower().strip()
    text_gt = text_gt.lower().strip()
    if text_retrict:
        return text_pred == text_gt
    return (text_pred in text_gt) or (text_gt in text_pred)

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
        
def check_click(click, candidate_bbox, gt_point):
    if len(candidate_bbox):
        candidate_bbox = enlarge_bbox(candidate_bbox, scale_factor=BBOX_ENLARGE_FACTOR)
        for bbox in candidate_bbox:
            if (bbox[0] <= click[0] <= bbox[2]) and (bbox[1] <= click[1] <= bbox[3]):
                return True
    if gt_point is not None:
        return np.linalg.norm([gt_point[0]-click[0], gt_point[1]-click[1]]) <= POINT_DISTANCE_THRESHOLD
    return False

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
def check_response_match(pred_action, current_check_pam, width, height, resized_width, resized_height, text_retrict=False):
    pred_action = norm_coordinate(copy.deepcopy(pred_action), resized_width, resized_height) # todo use resized width
    current_check_pam = norm_coordinate(copy.deepcopy(current_check_pam), width, height)
    # print(current_check_pam, pred_action)
    if current_check_pam['action'] in ['wait', 'terminate']:
        if pred_action['action'] == current_check_pam['action']:
            return True, True
        return False, False
    elif current_check_pam['action'] == 'system_button':
        if pred_action['action'] == 'system_button':
            return True, current_check_pam['button'].lower().strip() == pred_action['button'].lower().strip()
        else:
            return False, False
    elif current_check_pam['action'] in ['type', 'answer', 'key']:
        if pred_action['action'] == 'type':
            return True, check_text(pred_action['text'], current_check_pam['text'], text_retrict=text_retrict)
        else:
            return False, False
    elif current_check_pam['action'] == 'open':
        if pred_action['action'] == 'open':
            return True, check_text(pred_action['text'], current_check_pam['text'], text_retrict=text_retrict)
        else:
            return False, False
    elif current_check_pam['action'] == 'swipe':
        if pred_action['action'] == 'swipe':
            if 'direction' in current_check_pam:
                gt_direction = current_check_pam['direction']
            else:
                gt_direction = predict_direction(current_check_pam['coordinate'], current_check_pam['coordinate2'])
            direction = predict_direction(pred_action['coordinate'], pred_action['coordinate2'])
            if gt_direction == 'down':
                gt_direction = 'up'
            elif gt_direction == 'up':
                gt_direction = 'down'
            return True, direction == gt_direction
        else:
            return False, False
    elif current_check_pam['action'] in ['long_press', 'click']:
        if pred_action['action'] == current_check_pam['action']:
            return True, check_click(pred_action['coordinate'], current_check_pam.get('candidate_bbox', []), gt_point=current_check_pam['coordinate'])
        else:
            return False, False
    return False, False