import base64
import copy
import json
import re
import time
import traceback

import numpy as np
import requests
import torch
# from x.data.agent.pyfunction import PythonCallFormat
from x.data.agent.mobile_use import MobileUseMultiTurnFormat
from x.data.agent.space.std_space import RAW_SPACE
from x.data.text import parse_tags

from .gui_utils.utils import check_response_match

# fm = PythonCallFormat(RAW_SPACE, add_thought=True)
fm = MobileUseMultiTurnFormat(RAW_SPACE, add_thought=True)






def gui_action_match_compute_score(solution_str, ground_truth, extra_info=None):
    if extra_info is None:
        extra_info = {}
    width, height, resized_width, resized_height = extra_info['width'], extra_info['height'], extra_info['resized_width'], extra_info['resized_height']

    format_score = 0.0
    action_score = 0.0
    type_match = False
    extract_match = False
    try:
        result = fm.parse_response(solution_str)
        think_str = result['thinking']
        if think_str is not None and think_str.strip():
            format_score = 1.0
        else:
            format_score = 0.0
        
        if 'action_content' in result:
            pred_action = result['action_content']
            type_match, extract_match = check_response_match(pred_action, ground_truth, width, height, resized_width, resized_height)
            if type_match:
                action_score = 0.5
                if extract_match:
                    action_score = 1.0
            else:
                action_score = 0.1 # 解析正确
        else:
            action_score = 0.0
    except:
        traceback.print_exc()
        print("Error Response:")
        print(solution_str)
        
    return {
        "score": format_score * 0.1 + action_score * 0.9,
        "format_score": format_score,
        "type_match": type_match,
        "extract_match": extract_match,
    }
