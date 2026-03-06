from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os, json
from tqdm import tqdm

import cv2
import numpy as np

import time
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import re
from typing import Dict, Any, Optional, Union

import numpy as np
import copy
import traceback
import random
import os
from datasets import load_dataset
from dataclasses import dataclass, field

from typing import Callable,Optional, Any, Dict, List, Tuple, Union
import torch

import math
from PIL import Image
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from qwen_vl_utils import process_vision_info
import argparse

DISTRIBUTED_INFO = {

}

def initialize_distributed():
    DISTRIBUTED_INFO['world_size'] = int(os.environ.get('WORLD_SIZE', 1))
    DISTRIBUTED_INFO['rank'] = int(os.environ.get('RANK', 0))

only_two_action_system_prompt = '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen\'s resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.\n* don\'t use any other computer use tool like type, key, scroll, left_click_drag and so on.\n* you can only use the left_click and mouse_move action to interact with the computer. if you can\'t find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'

infesible_prefix = 'Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return <tool_call>\n{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}\n</tool_call>'

MIN_PIXELS = 196*32*32
MAX_PIXELS = 2048*32*32

class JsonHelper:
    @staticmethod
    def load_json(data_path):
        with open(data_path, 'r') as f:
            return json.load(f)
    @staticmethod
    def save_json(data_path, data):
        file_dir = os.path.dirname(data_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # with open(data_path, "w") as f:
        #     json.dump(data, f, indent=2, ensure_ascii=False)
        with open(data_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

class JsonlHelper:
    """
    用于处理 .jsonl 文件的工具类。
    提供读取 .jsonl 文件为列表和保存列表为 .jsonl 文件的功能。
    """

    @staticmethod
    def load_jsonl_to_list(file_path):
        """
        读取 .jsonl 文件并将其转换为列表格式。
        
        参数:
            file_path (str): .jsonl 文件路径。
        
        返回:
            list: 包含所有 JSON 对象的列表，每个元素是一个字典。
        """
        data_list = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 将每一行解析为 JSON 对象
                    json_obj = json.loads(line.strip())
                    data_list.append(json_obj)
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
        except json.JSONDecodeError:
            print(f"文件内容不是有效的 JSON 格式: {file_path}")
        
        return data_list

    @staticmethod
    def save_list_to_jsonl(data_list, file_path):
        """
        将列表保存为 .jsonl 格式文件。
        
        参数:
            data_list (list): 要保存的列表，每个元素是一个字典。
            file_path (str): 输出文件路径。
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data_list:
                    # 确保每个元素是字典类型
                    if not isinstance(item, dict):
                        raise ValueError(f"列表中的元素必须是字典类型，但发现: {type(item)}")
                    
                    # 写入每一行 JSON 对象
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except IOError:
            print(f"无法写入文件: {file_path}")
        except ValueError as e:
            print(f"错误: {e}")
# 1. processor
def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor
def smart_resize(height, width, factor=32, min_pixels=56 * 56, max_pixels=32 * 32 * 2048, max_long_side=8192):
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def update_image_size_(image_ele, min_tokens=1, max_tokens=2048, merge_base=2, patch_size=16):
    height, width = image_ele["height"], image_ele["width"]
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=merge_base * patch_size,
        min_pixels=pixels_per_token * min_tokens,
        max_pixels=pixels_per_token * max_tokens,
        max_long_side=50000,
    )
    image_ele.update(
        {
            "resized_height": resized_height,
            "resized_width": resized_width,
            "seq_len": resized_height * resized_width // pixels_per_token + 2,
        }
    )
    return image_ele

def make_qwen_image_item(img_path: str, image=None, max_tokens=2048, patch_size=16):
    if image is not None:
        img = image
    else:
        from x.io.image_io import ImageIO, ImageIO2
        mio = ImageIO2()
        img = mio(str(img_path))
        # img = mio(img_path)
        # print("img: ", img)
    if isinstance(img_path, Path):
        img_path = str(img_path.absolute())
    if img_path.startswith("http") or img_path.startswith("oss://"):
        pass
    else:
        img_path = str(Path(img_path).absolute())
    # print("img_path: ", img_path)
    image_ele = {
        "image": img_path,
        "height": img.height,
        "width": img.width,
        "type": "image"
    }
    image_ele = update_image_size_(image_ele, max_tokens=max_tokens, patch_size=patch_size)
    return image_ele

def is_bbox_inside(inner, outer):
    x2_min, y2_min, x2_max, y2_max = inner
    x1_min, y1_min, x1_max, y1_max = outer
    return (x1_min <= x2_min and
            y1_min <= y2_min and
            x2_max <= x1_max and
            y2_max <= y1_max)

def read_json(path):
    def smart_json_loads(line):
        try:
            try:
                tmp = json.loads(line.strip())
                return tmp
            except KeyboardInterrupt:
                raise
            except:
                tmp = json5.loads(line.strip())
                return tmp
        except:
            print(f'Invalid Json Line [Json Start]{line}[Json End]')
            traceback.print_exc()
        return None
    path = Path(path)
    if path.suffix == '.json' or force_type == 'json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.suffix == '.jsonl' or force_type == 'jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                tmp = smart_json_loads(line)
                if tmp:
                    data.append(tmp)
                
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')
    return data

def crop_image_by_bbox(input_path, output_path, bbox, ratio=None):
    with Image.open(input_path) as img:
        cropped_img = img.crop(bbox)

        if ratio is not None and ratio != 1.0:
            # 计算新的尺寸
            new_size = (
                int(cropped_img.width * ratio),
                int(cropped_img.height * ratio)
            )
            # 使用高质量重采样方法
            cropped_img = cropped_img.resize(new_size, Image.Resampling.LANCZOS)
        
        cropped_img.save(output_path)

# =========for knowledge_bench=========
def normalize_coordinates_in_messages(data, original_width, original_height):
    """
    递归遍历 data['messages'] 中所有字符串，将形如 "int int" 的坐标转换为归一化格式 "x_norm y_norm"
    """
    def traverse(obj):
        if isinstance(obj, str):
            def replace_match(match):
                x, y = map(int, match.groups())
                x_norm = int(x / original_width * 1000)
                y_norm = int(y / original_height * 1000)
                return f"{x_norm} {y_norm}"

            # 避免匹配到更长数字/单词中的子串
            pattern = r'(?<![\d\w])(\d+)\s+(\d+)(?![\d\w])'
            return re.sub(pattern, replace_match, obj)

        if isinstance(obj, list):
            return [traverse(item) for item in obj]

        if isinstance(obj, dict):
            return {k: traverse(v) for k, v in obj.items()}

        return obj

    if "messages" in data:
        data["messages"] = traverse(data["messages"])
    return data

def unescape_string(s: Optional[str]) -> Optional[str]:
    if not s: return s
    result = s
    for _ in range(3):
        prev = result
        try:
            result = json.loads('"' + result + '"')
        except (json.JSONDecodeError, Exception):
            result = result.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
        if result == prev:
            break
    return result

def normalize_answer_for_choice_and_yn(answer: Optional[str]) -> Optional[str]:
    if answer is None: return None
    answer = answer.strip().strip('"').strip("'").strip('\\').strip()
    low = answer.lower()
    if low in ['yes', 'no', 'unknown']:
        return low
    if len(answer) >= 1 and answer[0].upper() in 'ABCDEFGH':
        if len(answer) == 1 or answer[1] in '.。)）: ,':
            return answer[0].upper()
    return answer.upper()

def robust_extract_answer_from_text(model_response: str) -> Optional[str]:
    if not model_response: return None
    raw = model_response.strip()
    candidates = set([raw])
    no_md = re.sub(r'```json\s*', '', raw)
    no_md = re.sub(r'```\s*', '', no_md).strip()
    no_md = re.sub(r'</?[a-zA-Z_]+>\s*$', '', no_md).strip()
    candidates.add(no_md)
    candidates.add(unescape_string(raw))
    candidates.add(unescape_string(no_md))

    for base in [raw, no_md]:
        if base:
            v = base.replace('\\\\\\"', '"').replace('\\"', '"').replace("\\\\'", "'").replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
            candidates.add(v)
            candidates.add(v.strip())

    for text in candidates:
        if not text: continue
        for match in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    for key in ['answer', 'Answer', 'ANSWER']:
                        if key in data:
                            ans = str(data[key]).strip().strip('"').strip("'").strip()
                            if ans: return ans
            except: pass
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ['answer', 'Answer', 'ANSWER']:
                    if key in data:
                        ans = str(data[key]).strip().strip('"').strip("'").strip()
                        if ans: return ans
        except: pass

    all_texts = list(candidates) + [raw, no_md]
    patterns_kv = [
        r'\\*["\']answer\\*["\']\s*:\s*\\*["\']([^"\'\\]+)\\*["\']',
        r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']answer["\']\s*:\s*["\']?([A-Ha-h])["\']?',
        r'["\']answer["\']\s*:\s*["\']?(yes|no|unknown)["\']?',
        r'\banswer\s*:\s*["\']([^"\']+)["\']',
        r'\banswer\s*:\s*([A-Ha-h])\b',
        r'\banswer\s*:\s*(yes|no|unknown)\b',
        r'["\']?\s*answer\s*["\']?\s*:\s*["\']?\s*([A-Ha-h])\s*["\']?',
        r'["\']?\s*answer\s*["\']?\s*:\s*["\']?\s*(yes|no|unknown)\s*["\']?',
    ]
    for text in all_texts:
        if not text: continue
        for pattern in patterns_kv:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().strip('"').strip("'").strip('\\').strip()
                if ans: return ans

    for text in all_texts:
        if not text: continue
        for pattern in [r'答案\s*[:：]\s*([A-Ha-h])\b', r'答案\s*[:：]\s*(yes|no|unknown)\b', r'答案是\s*[:：]?\s*([A-Ha-h])\b', r'答案是\s*[:：]?\s*(yes|no|unknown)\b']:
            m = re.search(pattern, text, re.IGNORECASE)
            if m: return m.group(1).strip()

    for text in all_texts:
        if not text: continue
        for pattern in [r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:：]?\s*[\"']?([A-Ha-h])[\"']?", r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:：]?\s*[\"']?(yes|no|unknown)[\"']?", r"(?:I\s+(?:would\s+)?(?:choose|select|pick))\s*[:：]?\s*[\"']?([A-Ha-h])[\"']?", r'\b(?:option|choice)\s+([A-Ha-h])\s+is\s+(?:the\s+)?(?:correct|right|best)', r'(?:so|therefore|thus|hence)[,.]?\s+(?:the\s+answer\s+is\s+)?["\']?([A-Ha-h])["\']?']:
            m = re.search(pattern, text, re.IGNORECASE)
            if m: return m.group(1).strip()

    m = re.search(r'answer[\\"\'\s:]+\s*([A-Ha-h]|yes|no|unknown)\b', model_response, re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def _extract_answer_from_plain_text_legacy(content: str, answer_type: Optional[str] = None) -> Optional[str]:
    if not content: return None
    content = str(content).strip()
    all_patterns = [
        r'(?:^|\n)\s*(?:final\s+)?answer\s*[：:]\s*(.+?)(?:\n|$)', r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[：:.\s]\s*(.+?)(?:\n|$)',
        r'(?:^|\n)\s*answer\s*=\s*(.+?)(?:\n|$)', r'(?:so|therefore|thus|hence)\s*,?\s*(?:the\s+)?answer\s+is\s*[：:.\s]\s*(.+?)(?:\n|$)',
        r'答案\s*(?:是|为|[：:])\s*(.+?)(?:\n|$|。)', r'(?:我)?选(?:择)?\s*(.+?)(?:\n|$|。)',
        r'I\s+(?:would\s+)?(?:choose|select|pick)\s+(.+?)(?:\n|$|\.(?:\s|$))', r'(?:option|choice)\s+([A-Ga-g])\b'
    ]
    for pat in all_patterns:
        m = re.search(pat, content, re.IGNORECASE | re.MULTILINE)
        if m:
            extracted = m.group(1).strip().strip('"').strip("'").strip('`').strip('.').strip()
            if extracted: return extracted
            
    if answer_type == "multiple_choice":
        for pat in [r'"{3}\s*([A-Ga-g])\s*"{3}', r'`{1,3}\s*([A-Ga-g])\s*`{1,3}', r'[\(\[]\s*([A-Ga-g])\s*[\)\]]', r'(?:^|[\s\.\,\;\:\!\?\"\'\`\n])([A-Ga-g])[\.\)\s]*$']:
            m = re.search(pat, content)
            if m: return m.group(1).strip()
        m = re.match(r'^\s*([A-Ga-g])\s*[\.\)\s:]*', content)
        if m and len(content.strip()) <= 5: return m.group(1).strip()

    elif answer_type == "yes_or_no":
        cleaned = content.lower().strip().strip('"').strip("'").strip('`').strip('.').strip()
        if cleaned in ['yes', 'no', 'unknown']: return cleaned
        for pat in [r'^(yes|no|unknown)\b', r'\b(yes|no|unknown)\s*[.!]?\s*$']:
            m = re.search(pat, cleaned, re.IGNORECASE)
            if m: return m.group(1).lower()
        if 'yes' in cleaned and 'no' not in cleaned: return 'yes'
        if 'no' in cleaned and 'yes' not in cleaned and 'unknown' not in cleaned: return 'no'
        if 'unknown' in cleaned: return 'unknown'

    cleaned_content = content.strip().strip('"').strip("'").strip('`').strip('.').strip()
    if len(cleaned_content) <= 10: return cleaned_content if cleaned_content else None
    return None
def extract_answer_from_response(model_response: Any, answer_type: Optional[str] = None) -> Optional[str]:
    def _from_content_text(content: str) -> Optional[str]:
        ans = robust_extract_answer_from_text(content)
        if ans is not None: return ans.strip()
        return _extract_answer_from_plain_text_legacy(content, answer_type)

    if hasattr(model_response, "choices"):
        try:
            choice = model_response.choices[0]
            content = getattr(choice.message, "content", "") or ""
            return _from_content_text(content)
        except: pass
    if isinstance(model_response, dict):
        try:
            if "choices" in model_response:
                content = model_response["choices"][0].get("message", {}).get("content", "") or ""
                return _from_content_text(content)
            if "answer" in model_response: return str(model_response["answer"]).strip()
        except: pass
    if isinstance(model_response, str):
        return _from_content_text(model_response)
    return None

def _compare_yes_no(pred: str, gt: str) -> bool:
    def normalize(text: str) -> Optional[str]:
        text = str(text).strip().lower()
        text = re.sub(r'^(the\s+answer\s+is\s*[:\.]?\s*)', '', text)
        text = re.sub(r'^(answer\s*[:\.]?\s*)', '', text, flags=re.IGNORECASE)
        text = text.strip().strip('"').strip("'").strip('.').strip()
        if text in ['yes', 'no', 'unknown']: return text
        if re.match(r'^yes\b', text): return 'yes'
        if re.match(r'^no\b', text) and 'unknown' not in text: return 'no'
        if re.match(r'^unknown\b', text): return 'unknown'
        if 'yes' in text and 'no' not in text: return 'yes'
        if 'no' in text and 'yes' not in text and 'unknown' not in text: return 'no'
        if 'unknown' in text: return 'unknown'
        return None
    return normalize(pred) is not None and normalize(pred) == normalize(gt)

def _compare_multiple_choice(pred: str, gt: str) -> bool:
    def extract_option(text: str) -> Optional[str]:
        text_upper = str(text).strip().upper()
        if text_upper in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']: return text_upper
        m = re.match(r'^\s*([A-Ha-h])[\.\):：\s]', text_upper)
        if m: return m.group(1).upper()
        for pat in [r'(?:THE\s+)?ANSWER\s*(?:IS|:)\s*([A-H])\b', r'(?:OPTION|CHOICE)\s+([A-H])\b', r'\b([A-H])\s*(?:IS\s+(?:THE\s+)?(?:CORRECT|RIGHT|ANSWER))']:
            m = re.search(pat, text_upper)
            if m: return m.group(1).upper()
        m = re.search(r'(?:^|[\s\.\,\;\:\!\?\"\'\`])([A-H])[\.\)\s]*$', text_upper)
        if m: return m.group(1).upper()
        return None
    pred_opt = extract_option(pred)
    gt_opt = extract_option(gt)
    return pred_opt is not None and gt_opt is not None and pred_opt == gt_opt

def validate_answer(answer_pred: Optional[str], answer_type: Optional[str], gt: Optional[str]) -> bool:
    if not answer_pred or not gt or not str(answer_pred).strip() or not str(gt).strip(): return False
    norm_pred = normalize_answer_for_choice_and_yn(answer_pred)
    norm_gt = normalize_answer_for_choice_and_yn(gt)
    
    if answer_type == "yes_or_no" or norm_gt in ['yes', 'no', 'unknown']:
        return _compare_yes_no(norm_pred or answer_pred, norm_gt or gt)
    elif answer_type == "multiple_choice" or norm_gt in list("ABCDEFGH"):
        return _compare_multiple_choice(norm_pred or answer_pred, norm_gt or gt)
    else:
        pred_str = str(answer_pred).strip().lower().strip('"').strip("'").strip('`').strip('.').strip()
        gt_str = str(gt).strip().lower().strip('"').strip("'").strip('`').strip('.').strip()
        pred_str = re.sub(r'\s+', ' ', pred_str)
        gt_str = re.sub(r'\s+', ' ', gt_str)
        return pred_str == gt_str or (len(gt_str) <= 20 and gt_str in pred_str)

def stdcoord(input_val, img_size):
    return int((input_val / img_size) * 1000)

def make_replacer(w: int, h: int):
    def replacer(match: re.Match) -> str:
        x = int(match.group(1))
        y = int(match.group(2))
        return f"{stdcoord(x, w)},{stdcoord(y, h)}"  # 保持空格分隔
    return replacer
#==========prompt for Knowledge_Bench================

def Actioneffectprompt(
    os_type: str,
    ds_path: str,
    annotate_image_path: str,      # 主截图（可与 ds_path 不同 root）
    question_image_dir_list: str,          # 用于读取原图尺寸（相对 ds_path）
    question_text: str,
    option_image_rel_paths: List[str],  # 相对 ds_path
    enable_thinking_prompt: bool = False,
) -> Dict[str, Any]:


    if os_type == "Android":
        system_prompt = (
            "You are a Graphical User Interface (GUI) agent. You will be given a screenshot, action descriptions, and multiple options, each containing an image. "
            "After performing one action on the screenshot, your goal is to select the option that correctly corresponds to the resulting screenshot after performing the action. "
            "Below is a short description of the action space:\n\n\n"
            "        ## Action Space\n\n"
            "        - click(point='x1 y1')\n"
            "        - long_press(point='x1 y1')\n"
            "        - type(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\n"
            "        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n\n"
            "        The size of the image is 1000x1000.\n"
        )
    else:
        system_prompt = (
            "You are a Graphical User Interface (GUI) agent. You will be given a screenshot, action descriptions, and multiple options, each containing an image. After performing one action on the screenshot, your goal is to select the option that correctly corresponds to the resulting screenshot after performing the action. Below is a short description of the action space:\n\n\n"
            "        Action Space\n\n"
            "        - click(point='x1 y1'): left click a position on the screen. \n"
            "        - left_double(point='x1 y1'): left double click a position on the screen. \n"
            "        - right_single(point='x1 y1'): right single click a position on the screen. \n"
            "        - drag(start_point='x1 y1', end_point='x2 y2'): drag the mouse from one position to another. \n"
            "        - hotkey(key='ctrl c'): keyboard shortcut, split keys with spaces\n"
            "        - type(content='xxx'): type an answer, use escape characters (', \", \\n) when needed. Add \\n at the end if it is the final submission.\n"
            "        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n\n"
            "        The size of the image is 1000x1000.\n"
        )

    # 读取原始 query 图像尺寸，用于把动作坐标映射到 1000 坐标系
    query_abs = os.path.join(ds_path, question_image_dir_list[0])
    w, h = Image.open(query_abs).size
    print(w, h)
    action_type = question_text.replace("ActionEffect: ", "").strip()
    indication_text = ""
    if any(action_type.startswith(p) for p in ["click", "left_double", "right_single", "scroll", "drag", "long_press"]):
        indication_text = "(as drawn in the initial screenshot.)"

    # 组 messages：严格用 {"text":...} 与 make_qwen_image_item 输出的 image item
    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": []},
    ]

    # 主截图（annotate_image_path 可能不在 ds_path 下，直接用传入路径）
    messages[1]["content"].append(make_qwen_image_item(annotate_image_path))
    messages[1]["content"].append({"text": "Above is the current screenshot.\n"})

    messages[1]["content"].append({
        "text": (
            f"After I perform the described action '{action_type}' {indication_text}, "
            f"which of the following options correctly corresponds to the resulting screenshot?\n"
        )
    })

    # options（按 A/B/C/D，且路径用 ds_path join）
    for i, rel in enumerate(option_image_rel_paths):
        label_chr = chr(65 + i)
        messages[1]["content"].append({"text": f"{label_chr}. \n"})
        opt_abs = os.path.join(ds_path, rel)
        messages[1]["content"].append(make_qwen_image_item(opt_abs))

    prompt_text = (
        "Please analyze the screenshot and the described action step by step:\n1. First, carefully examine the current screenshot (Image 1, the original screenshot) to understand the GUI state, including all visible elements, their positions, and the overall layout.\n2. Identify the exact element or area at the specified coordinates where the action will be performed.\n3. Understand what type of action is being performed (e.g., click, double-click, right-click, drag, scroll, type, hotkey) and what its expected effect would be on the GUI.\n4. Consider the typical behavior of this action on the identified element:\n   - What changes would occur on the screen?\n   - Would new windows, menus, dialogs, or panels open?\n   - Would any elements be selected, highlighted, expanded, or collapsed?\n   - Would the content or layout of the interface change?\n5. Now compare the original screenshot (Image 1) with each option image one by one, and analyze the specific differences:\n   - Compare Image 1 (original) vs Option A (Image 2): What are the differences? Does Option A reflect the expected result of the described action? List the specific changes you observe.\n   - Compare Image 1 (original) vs Option B (Image 3): What are the differences? Does Option B reflect the expected result of the described action? List the specific changes you observe.\n   - Compare Image 1 (original) vs Option C (Image 4): What are the differences? Does Option C reflect the expected result of the described action? List the specific changes you observe.\n   - Compare Image 1 (original) vs Option D (Image 5): What are the differences? Does Option D reflect the expected result of the described action? List the specific changes you observe.\n6. Based on your comparison above, determine which option's differences are consistent with the expected effect of the action:\n   - Eliminate options that show no meaningful change from the original screenshot.\n   - Eliminate options that show incorrect or unrelated changes.\n   - Eliminate options that show changes inconsistent with the described action.\n7. Select the option that best matches the expected resulting screenshot after the action is performed.\nYou must respond strictly in JSON format following this schema: {\"thought\": \"<your step-by-step reasoning, including the comparison between the original screenshot and each option>\", \"answer\": \"<A/B/C/D>\"}"
        if enable_thinking_prompt
        else "You must respond strictly in JSON format following this schema: Answer: {\"answer\": \"<A/B/C/D>\" } "
    )
    messages[1]["content"].append({"text": prompt_text})

    data = {"messages": messages}
    data = normalize_coordinates_in_messages(data, w, h)

    return data

def get_action_space_prompt(os_type: str) -> str:
    if os_type == "Android":
        return (
            "        ## Action Space\n\n"
            "        - click(point='x1 y1')\n"
            "        - long_press(point='x1 y1')\n"
            "        - type(content='') #If you want to submit your input, use \"\\n\" at the end of content.\n"
            "        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n\n"
            "        The size of the image is 1000x1000.\n"
        )
    else:
        return (
            "        Action Space\n\n"
            "        - click(point='x1 y1'): left click a position on the screen. \n"
            "        - left_double(point='x1 y1'): left double click a position on the screen. \n"
            "        - right_single(point='x1 y1'): right single click a position on the screen. \n"
            "        - drag(start_point='x1 y1', end_point='x2 y2'): drag the mouse from one position to another. \n"
            "        - hotkey(key='ctrl c'): keyboard shortcut, split keys with spaces\n"
            "        - type(content='xxx'): type an answer, use escape characters (', \", \\n) when needed. Add \\n at the end if it is the final submission.\n"
            "        - scroll(point='x1 y1', direction='down or up or right or left'): scroll to see more content\n\n"
            "        The size of the image is 1000x1000.\n"
        )

# ==========================================
# 1. Interface Perception (Layout, State, Widget, Icon etc.)
# ==========================================
def InterfacePerceptionPrompt(
    annotate_image_path: str,
    question_text: str,
    question_type: str,
    option_text_list: Optional[List[str]],
    knowledge: str,
    augmented_question: str,
    enable_thinking_prompt: bool = False,
) -> Dict[str, Any]:
    system_prompt = "You are a Graphical User Interface (GUI) agent. You will be given a screenshot, a question, and corresponding options. You need to choose one option as your answer.\n"
    
    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    
    messages[1]["content"].append(make_qwen_image_item(annotate_image_path))
    if augmented_question is not None:
        messages[1]["content"].append({"text": f"{augmented_question} \n"})
    else:
        messages[1]["content"].append({"text": f"{question_text} \n"})
    
    prompt_text = ""
    if question_type == 'multiple_choice':
        opt_str = ""
        if option_text_list:
            for i, opt in enumerate(option_text_list):
                label_chr = chr(65 + i)
                messages[1]["content"].append({"text": f"{label_chr}. {opt}\n"})
            opt_str = "/".join([chr(65 + i) for i in range(len(option_text_list))])
        else:
            opt_str = "A/B/C/D"
            
        prompt_text += "Which of the above options are correct according to the screenshot? "
        if enable_thinking_prompt:
            prompt_text += f'Think step by step. You must respond strictly in JSON format following this schema: {{\"thought\": \"<your reasoning>\", \"answer\": <{str(opt_str)}> }}' 
        else:
            prompt_text += f'You must respond strictly in JSON format following this schema: Answer: {{"answer": "<{opt_str}>" }} '
            
    elif question_type == 'yes_or_no':
        if enable_thinking_prompt:
            prompt_text += 'Think step by step. You must respond strictly in JSON format following this schema: {\"thought\": \"<your reasoning>\", \"answer\": \"<yes/no>\" } '
        else:
            prompt_text += 'You must respond strictly in JSON format following this schema: {"answer": "<yes/no>" } '

    if knowledge:
        if question_type == 'yes_or_no':
            prompt_text += f" Tips for answering this question: {knowledge}"
        else:
            prompt_text += f"Tips for answering this question: {knowledge}"
            
    messages[1]["content"].append({"text": prompt_text})
    return {"messages": messages}

# ==========================================
# 2. Action Prediction (Type)
# ==========================================

    
ALL_ACTION_DESCRIPTIONS = {
    "click": "click: typically selects an element, activates a button, or opens a single item.",
    "long_press": "long_press: typically triggers a context menu or special action by pressing and holding.",
    "left_double": "left_double: typically opens a file/folder or selects a word in text.",
    "right_single": "right_single: typically opens a context menu (right-click menu).",
    "drag": "drag: typically moves an element, selects a region, or resizes a window/panel.",
    "scroll": "scroll: typically shifts visible content up/down/left/right without changing selection.",
    "hotkey": "hotkey: typically triggers a shortcut action like copy, paste, undo, or switching views.",
    "type": "type: typically inserts or modifies text content in an input field or editor.",
}
def get_action_descriptions_for_options(option_text_list: List[str]) -> str:
    """
    根据选项文本列表，提取动作名称并构建对应的描述字符串 (参考 build_detailed_prompt 逻辑)
    """
    action_lines = []
    seen_actions = set()
    
    for opt in option_text_list:
        # 提取基本动作名，处理如 "click(point='...')" 的情况
        base_action = opt.split('(')[0].strip()
        
        if base_action in ALL_ACTION_DESCRIPTIONS and base_action not in seen_actions:
            line = f"   - {ALL_ACTION_DESCRIPTIONS[base_action]}"
            action_lines.append(line)
            seen_actions.add(base_action)
            
    return "\n".join(action_lines)

def ActionPredictionPrompt(
    ds_path: str,
    question_image_dir_list: List[str],
    option_text_list: List[str],
    os_type: str,
    enable_thinking_prompt: bool = False,
) -> Dict[str, Any]:
    system_prompt = "You are a Graphical User Interface (GUI) agent. You will be given two consecutive screenshots of the GUI, action descriptions, and multiple options. Your goal is to select which action was performed to transition from the first screenshot to the second. If the description specifies an action type, select the correct parameter value for the given action.\n\n\n"
    system_prompt += get_action_space_prompt(os_type)

    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    
    for img_rel in question_image_dir_list[:2]:
        img_abs = os.path.join(ds_path, img_rel)
        messages[1]["content"].append(make_qwen_image_item(img_abs))


        
    messages[1]["content"].append({"text": "Above are two consecutive screenshots, Your task is to select which action is performed in order to transition from the first screenshot to the second."})
    
    for i, opt in enumerate(option_text_list):
        label_chr = chr(65 + i)
        messages[1]["content"].append({"text": f"{label_chr}. {opt} \n"})
    opt_str = "/".join([chr(65 + i) for i in range(len(option_text_list))])

    answer_format = f"<{opt_str}>"
    action_descriptions = get_action_descriptions_for_options(option_text_list)

    if enable_thinking_prompt:
        # 4. 使用参考代码中的详细分析步骤 (1-6 步) 和 JSON 格式约束
        prompt_text = (
            "Please analyze the two consecutive screenshots and determine the action performed step by step:\n"
            "1. Carefully examine the first screenshot (before state) and the second screenshot (after state). "
            "Note the overall layout, visible elements, their positions, and any highlighted or selected items.\n"
            "2. Identify all differences between the two screenshots:\n"
            "   - Have any elements moved, appeared, or disappeared?\n"
            "   - Has any content been scrolled, resized, or rearranged?\n"
            "   - Are there any new menus, dialogs, tooltips, or context menus?\n"
            "   - Has any text been added, modified, or selected?\n"
            "   - Has any item been opened, expanded, or collapsed?\n"
            "3. Based on the observed differences, reason about which action type could produce such changes:\n"
            f"{action_descriptions}\n"
            "4. Eliminate options that cannot explain the observed differences.\n"
            "5. You should first choose the best matched action of transition from the first screenshot to the second.\n"
            "6. Select the option that matches the action.\n"
            f'You must respond strictly in JSON format following this schema: {{\"thought\": \"<your step-by-step reasoning>\", \"answer\": \"{answer_format}\"}}'
        )
    else:
        prompt_text = (f'Which of the above options are correct according to the screenshots? You must respond strictly in JSON format following this schema: {{\"thought\": \"<your reasoning>\", \"answer\": \"{answer_format}" }} '
        )   
    messages[1]["content"].append({"text": prompt_text})
    return {"messages": messages}

# ==========================================
# 3. Action Prediction Parameter
# ==========================================
def ActionPredictionParameterPrompt(
    ds_path: str,
    annotate_image_path: str,
    question_image_dir_list: List[str],
    question_text: str,
    option_text_list: List[str],
    os_type: str,
    enable_thinking_prompt: bool = False,
) -> Dict[str, Any]:
    system_prompt = "You are a Graphical User Interface (GUI) agent. You will be given two consecutive screenshots of the GUI, action descriptions, and multiple options. Your goal is to select which action was performed to transition from the first screenshot to the second. If the description specifies an action type, select the correct parameter value for the given action.\n\n\n"
    system_prompt += get_action_space_prompt(os_type)

    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    query_abs = os.path.join(ds_path, annotate_image_path)
    w, h = Image.open(query_abs).size
    print(w, h)
    messages[1]["content"].append(make_qwen_image_item(annotate_image_path))

    if len(question_image_dir_list) > 1:
        img_abs_2 = os.path.join(ds_path, question_image_dir_list[1])
        messages[1]["content"].append(make_qwen_image_item(img_abs_2))
        
    action_type = question_text.replace("ActionPrediction-Parameter: ", "").strip()
    messages[1]["content"].append({"text": f"Above are two consecutive screenshots, Your task is to select the option containing the right parameter value of the given action '{action_type}' to transition from the first to the second screenshot."})
    
    for i, opt in enumerate(option_text_list):
        label_chr = chr(65 + i)
        messages[1]["content"].append({"text": f"{label_chr}. {opt} \nAs is drawn in the first screenshot."})
    opt_str = "/".join([chr(65 + i) for i in range(len(option_text_list))])

    if enable_thinking_prompt:
        prompt_text = f"Which of the above options are correct according to the screenshots? Think step by step. You must respond strictly in JSON format following this schema: {{\"thought\": \"<your reasoning>\", \"answer\": \"<{opt_str}>\" }} "
    else:
        prompt_text = f'Which of the above options are correct according to the screenshots? You must respond strictly in JSON format following this schema: {{\"thought\": \"<your reasoning>\", \"answer\":\"<{opt_str}>\" }} '
        
    messages[1]["content"].append({"text": prompt_text})

    data = {"messages": messages}
    data = normalize_coordinates_in_messages(data, w, h)

    return data


# ==========================================
# 4. Goal Interpretation (Instruction Understanding Y/N)
# ==========================================
def GoalInterpretationPrompt(
    image_folder: str,
    question_image_dir_list: List[str],
    question_text: str,
    enable_thinking_prompt: bool = False,
) -> Dict[str, Any]:
    system_prompt = "You are a Graphical User Interface (GUI) agent.  You will be given a sequence of screenshots, a task instruction, and three possible answer options: yes, no, unknown. Your goal is to select the best option that indicates whether the task is completed: yes — The task is clearly completed.  no — The task is not completed. \n"
    
    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": []}
    ]
    
    for img_rel in question_image_dir_list:
        img_abs = os.path.join(image_folder, img_rel)
        messages[1]["content"].append(make_qwen_image_item(img_abs))
        
    messages[1]["content"].append({"text": f"According to the screenshots above, has the task \"{question_text}\" been completed?"})
    
    if enable_thinking_prompt:
        prompt_text = "Please analyze the screenshots and determine whether the task has been completed step by step:\n1. First, carefully read and understand the task instruction. Break down the task into sub-goals:\n   - What are the individual steps or requirements needed to complete this task?\n   - What would the final successful state look like?\n2. Examine each screenshot in sequence to understand the progression of actions:\n   - What application or interface is shown in each screenshot?\n   - What actions appear to have been taken between consecutive screenshots?\n   - What is the state of the interface in the final screenshot?\n3. Check each sub-goal against the evidence in the screenshots:\n   - Is there clear visual evidence that each sub-goal has been achieved?\n   - Are there any sub-goals that are clearly NOT achieved?\n   - Are there any sub-goals where the screenshots simply do not provide enough information to confirm or deny completion?\n4. Consider the three possible answers:\n   - Yes: ALL sub-goals of the task are clearly completed, with sufficient visual evidence in the screenshots.\n   - No: At least one sub-goal is clearly NOT completed, and there is evidence showing it was done incorrectly or not at all.\nKey information is missing, obscured, or the screenshots do not cover all necessary steps.\n5. Based on your analysis, select the most appropriate answer.\nYou must respond strictly in JSON format following this schema: {\"thought\": \"<your step-by-step reasoning>\", \"answer\": \"<yes/no>\"}"
    else:    
        prompt_text = " You must respond strictly in JSON format following this schema: Answer: Answer: {\"answer\": \"<yes/no>\" } "
        
    messages[1]["content"].append({"text": prompt_text})
    return {"messages": messages}

# ==========================================
# 5. Task Planning (Instruction Understanding MCQ)
# ==========================================
SYSTEM_PROMPT_TEMPLATE = (
    "You are a Graphical User Interface (GUI) agent. You will be given a task instruction, "
    "a screenshot, several GUI operations, and four options. Your goal is to select the best "
    "option that could solve the task.\n"
)
SYSTEM_PROMPT_YESNO_TEMPLATE = (
    "You are a Graphical User Interface (GUI) agent. You will be given a screenshot, a question, "
    "and corresponding options. You need to choose one option as your answer.\n"
)

YES_OR_NO_INSTRUCT_PROMPT = (
    " You must respond strictly in JSON format following this schema: Answer: Answer: {\"answer\": \"<yes/no/unknown>\" }"
)

MULTIPLE_CHOICE_DETAILED_PROMPT = (
    "Note: The given image is merely a reference for the interfacePlease analyze the given task and each option step by step:\n1. First, understand the overall goal of the task.\n2. For each option, examine the sequence of operations one by one:\n   - Is each operation relevant to the task?\n   - Is it performed in the correct order?\n   - Are there any missing steps or unnecessary steps?\n   - Does each operation logically follow from the previous one?\n3. Compare all options and identify which sequence would successfully accomplish the task.\nYou must respond strictly in JSON format following this schema: {\"thought\": \"<your step-by-step reasoning>\", \"answer\": \"<A/B/C/D>\" }"
)
YES_OR_NO_DETAILED_PROMPT = ("Note: The given image is merely a reference for the interfacePlease analyze the given task and the sequence of operations step by step:\n1. First, understand the overall goal of the task.\n2. Then, examine each operation in the given sequence one by one.\n3. For each operation, consider:\n   - Is this operation relevant to the task?\n   - Is it performed in the correct order relative to the previous and next steps?\n   - Are there any missing steps or unnecessary steps?\n   - Does this operation logically follow from the previous one?\n4. Check if the complete sequence, when executed in order, would successfully accomplish the task.\n5. Based on your analysis, determine whether the given sequence is correct (Yes) or incorrect (No).\nYou must respond strictly in JSON format following this schema: {\"thought\": \"<your step-by-step reasoning>\", \"answer\": \"<Yes/No>\"}"
)

def _make_text_item(text: str) -> Dict[str, Any]:
    return {"text": text}

# -------- 新增：把 "1. \"...\"" 解析成 {1: "..."} --------
_STEP_RE_LOOSE = re.compile(r'(?m)^\s*(\d+)\.\s*(.*)$')

def parse_task_steps(question_text: str) -> Dict[int, str]:
    """
    更宽松解析每行 `n. ...`，并尽量剥掉首尾引号。
    允许：
      1. "xxx"
      1. xxx
      1. "xxx".   （末尾多标点）
    """
    step_map: Dict[int, str] = {}
    for n, rest in _STEP_RE_LOOSE.findall(question_text or ""):
        s = rest.strip()

        # 去掉可能的首尾引号（中英文都支持）
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "“") or (s[0] == s[-1] == "”")):
            s = s[1:-1].strip()

        # 再次处理最常见形式：以 " 开头但不以 " 结尾（比如末尾带 .）
        if s.startswith('"'):
            s = s[1:].strip()
        if s.endswith('"'):
            s = s[:-1].strip()

        # 去掉末尾多余的句号/分号（可选）
        s = s.rstrip()

        step_map[int(n)] = s
    return step_map

# ========== 修复点：不要因为缺步骤直接抛 KeyError，避免 DataLoader 崩 ==========
def expand_option_numbers(opt_text: str, step_map: Dict[int, str], *, strict: bool = False) -> str:
    """
    strict=False：遇到缺失步骤时，不抛异常，保留原数字（或占位），避免训练/评测直接崩。
    strict=True：保持原行为，直接 raise KeyError。
    """
    opt_text = opt_text or ""

    # 兼容上游把 "A. 6, 5, 2" 也传进来的情况：先剥离 "A."
    opt_text = re.sub(r'^\s*[A-D]\.\s*', '', opt_text).strip()

    nums = [int(x) for x in re.findall(r"\d+", opt_text)]
    if not nums:
        return opt_text  # 没数字就认为已经是自然语言/已展开

    missing = [n for n in nums if n not in step_map]
    if missing and strict:
        raise KeyError(f"Option contains step id(s) not found in Task steps: {missing}")

    parts = []
    for n in nums:
        if n in step_map:
            parts.append(f"\"{step_map[n]}\"")
        else:
            # 缺失时保留数字，避免崩溃；你也可以改成 f"\"<missing step {n}>\""
            parts.append(str(n))
    return ", ".join(parts)

def TaskPlanningPrompt(
    ds_path: str,
    question_image_dir_list: List[str],
    question_text: str,
    option_text_list: List[str],
    q_type: str,
    enable_thinking_prompt: bool = True,
    # 可选：如果你想强制所有 option 必须能完全展开，把这个设 True
    strict_step_expand: bool = False,
) -> Dict[str, Any]:

    system_prompt = SYSTEM_PROMPT_YESNO_TEMPLATE if q_type == "yes_or_no" else SYSTEM_PROMPT_TEMPLATE

    messages = [
        {"role": "system", "content": [_make_text_item(system_prompt)]},
        {"role": "user", "content": []},
    ]

    for img_rel in question_image_dir_list:
        img_abs = os.path.join(ds_path, img_rel)
        messages[1]["content"].append(make_qwen_image_item(img_abs))

    messages[1]["content"].append(_make_text_item(question_text))

    if q_type == "multiple_choice":
        step_map = parse_task_steps(question_text)

        for i, opt in enumerate(option_text_list):
            label_chr = chr(65 + i)
            expanded_opt = expand_option_numbers(opt, step_map, strict=strict_step_expand)
            messages[1]["content"].append(_make_text_item(f"{label_chr}. {expanded_opt}"))

        if enable_thinking_prompt:
            messages[1]["content"].append(_make_text_item(MULTIPLE_CHOICE_DETAILED_PROMPT))
        else:
            messages[1]["content"].append(
                _make_text_item("Which of the above options are correct according to the screenshots? You must respond strictly in JSON format following this schema: Answer: {\"answer\": \"<A/B/C/D>\" } "
                )
            )
        return {"messages": messages}

    if q_type == "yes_or_no":
        if enable_thinking_prompt:
            messages[1]["content"].append(_make_text_item(YES_OR_NO_DETAILED_PROMPT))
        else:
            messages[1]["content"].append(_make_text_item(YES_OR_NO_INSTRUCT_PROMPT))
        return {"messages": messages}

def TaskPlanningPrompt_old(
    ds_path: str,
    question_image_dir_list: List[str],
    question_text: str,
    option_text_list: List[str],
    q_type: str,
    enable_thinking_prompt: bool = True,
    strict_step_expand: bool = False,
) -> Dict[str, Any]:

    system_prompt = SYSTEM_PROMPT_YESNO_TEMPLATE if q_type == "yes_or_no" else SYSTEM_PROMPT_TEMPLATE

    messages = [
        {"role": "system", "content": [_make_text_item(system_prompt)]},
        {"role": "user", "content": []},
    ]

    for img_rel in question_image_dir_list:
        img_abs = os.path.join(ds_path, img_rel)
        messages[1]["content"].append(make_qwen_image_item(img_abs))

    messages[1]["content"].append(_make_text_item(question_text))

    if q_type == "multiple_choice":
        for i, opt in enumerate(option_text_list):
            label_chr = chr(65 + i)
            messages[1]["content"].append(_make_text_item(f"{label_chr}. {opt}\n"))

        messages[1]["content"].append(
            _make_text_item(
                "Which of the above options are correct according to the screenshots? You must respond strictly in JSON format following this schema: Answer: {\"answer\": \"<A/B/C/D>\" } "
            )
        )
        return {"messages": messages}

    if q_type == "yes_or_no":
        messages[1]["content"].append(_make_text_item(YES_OR_NO_INSTRUCT_PROMPT))
        return {"messages": messages}


# 2. eval setting of each benchmarks
@dataclass()
class SingleRoundItem():
    messages: list = field(default_factory=list)
    gt_answer: str = ''
    index: int = 0
    extra: dict = field(default_factory=dict) # for evaluation
    model_response: Any = None
    gen_config: dict = field(default_factory=dict)
    # format_rule: Callable[[int, int], int] = lambda res: True
    def to_json(self,):
        messages = copy.deepcopy(self.messages)
        for msg in messages:
            for c in msg['content']:
                if 'image' in c and not isinstance(c['image'], str):
                    c['image'] = '-'
                if 'image' in c:
                    c['type'] = 'image'
                else:
                    c['type'] = 'text'
        return {
            'messages': messages,
            'gt_answer': self.gt_answer,
            'index': self.index,
            'extra': self.extra,
            'model_response': self.model_response,
            'gen_config': self.gen_config
        }   

    def to_dict(self):
        return {
            "index": getattr(self, "index", None),
            "messages": self.messages,
            "model_response": getattr(self, "model_response", None),
            "gt_answer": self.gt_answer,
            "extra": self.extra,
            "gen_config": self.gen_config
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        从字典还原为 SingleRoundItem 对象
        """
        return cls(
            index=data.get("index", 0),
            messages=data.get("messages", []),
            model_response=data.get("model_response", None),
            gt_answer=data.get("gt_answer", ''), 
            extra=data.get("extra", {}),
            gen_config=data.get("gen_config", {})
        )


class knowledge_bench(object):
    def __init__(self, ds_path, image_folder=None, gen_config=None, use_qwen3vl=True, max_tokens=2048, thinking= False):
        self.ds_path = ds_path
        self.image_folder = Path(ds_path,"Image")
        self.use_qwen3vl = use_qwen3vl         
        self.max_tokens = max_tokens
        self.gen_config = gen_config or {}
        
        self.suffix = "knowledge_bench"

        base = Path(ds_path) / "KnowledgeBench"
        lines = []
        for json_file in base.rglob("*.json"):
            tmp = read_json(json_file)
            if tmp:
                lines.append(tmp)
        self.dataset = lines
        self.enable_thinking_prompt = thinking
   
        self.tmp_folder = Path(ds_path, "AnnotateImage") 
    def find_image_path(self, image_path: str) -> str:
        candidate = os.path.join(self.image_folder, image_path)
        if os.path.exists(candidate): return candidate
        return os.path.join(self.ds_path, image_path) 
    def __getitem__(self, index):
        os.makedirs(self.tmp_folder, exist_ok=True)
        line = self.dataset[index]

        question_type = line.get("question_type", "multiple_choice") 
        if isinstance(question_type, list): question_type = question_type[0]
        
        groundtruth = str(line.get("groundtruth", ""))
        knowledge_info = line.get("knowledge", {})
        knowledge_type = knowledge_info.get("knowledge_type", "")
        knowledge_sub_type = knowledge_info.get("knowledge_sub_type", "")
        knowledge_text = line.get("needed_knowledge", "")
        
        question_text = line.get("question_text", "")
        question_image_dir_list = line.get("question_image_dir_list", [])
        if isinstance(question_image_dir_list, str): question_image_dir_list = [question_image_dir_list]
            
        option_text_list = line.get("option_text", [])
        option_image_dir_list = line.get("option_image_dir_list", [])
        os_type = line.get("os_type", "Windows")
        question_id = line.get("question_id", str(index))
        augmented_question = line.get("augmented_question", None)
        
        # --------------------------------------------------------
        # 动态图像生成与缓存逻辑
        # --------------------------------------------------------
        annotate_image_path = os.path.join(self.tmp_folder,f"{question_id}.png")
        base_img_path = self.find_image_path(question_image_dir_list[0]) if question_image_dir_list else ""
        if knowledge_type == "InterfacePerception" or knowledge_sub_type == "ActionEffect":
            if not os.path.exists(annotate_image_path) and base_img_path:
                print("Annotated image not found, gen_annotated image using official tools in KnowledgeBench")
            
            
        # fallback for empty annotate_image_path
        if os.path.exists(annotate_image_path) == False:
            annotate_image_path = base_img_path       
        # ========================================
        # 分发 Prompt 构建
        # ========================================
        messages_dict = {"messages": []}

        if knowledge_type == "InterfacePerception":
            messages_dict = InterfacePerceptionPrompt(
                annotate_image_path=annotate_image_path,
                question_text=question_text,
                question_type=question_type,
                option_text_list=option_text_list,
                knowledge=knowledge_text,
                augmented_question=augmented_question,
                enable_thinking_prompt=self.enable_thinking_prompt
            )

            
        elif knowledge_type == "InteractionPrediction":
            if knowledge_sub_type == "ActionEffect":
                messages_dict = Actioneffectprompt(
                    os_type=os_type,
                    ds_path=self.image_folder,
                    annotate_image_path=annotate_image_path,
                    question_image_dir_list=question_image_dir_list,
                    question_text=question_text,
                    option_image_rel_paths=option_image_dir_list,
                    enable_thinking_prompt=self.enable_thinking_prompt
                )
            elif knowledge_sub_type == "ActionPrediction":
                if "ActionPrediction-Parameter:" in question_text:
                    messages_dict = ActionPredictionParameterPrompt(
                        ds_path=self.image_folder,
                        annotate_image_path=annotate_image_path,
                        question_image_dir_list=question_image_dir_list,
                        question_text=question_text,
                        option_text_list=option_text_list,
                        os_type=os_type,
                        enable_thinking_prompt=self.enable_thinking_prompt
                    )
                    knowledge_sub_type = "ActionPredictionParameter"
                # 如果 question_text 带 Parameter 前缀，判定为 ActionPredictionParameter
                else:
                    messages_dict = ActionPredictionPrompt(
                        ds_path=self.image_folder,
                        question_image_dir_list=question_image_dir_list,
                        option_text_list=option_text_list,
                        os_type=os_type,
                        enable_thinking_prompt=self.enable_thinking_prompt
                )
                    
                  
        elif knowledge_type == "InstructionUnderstanding":
            if knowledge_sub_type == "GoalInterpretation":
                messages_dict = GoalInterpretationPrompt(
                    image_folder=self.image_folder,
                    question_image_dir_list=question_image_dir_list,
                    question_text=question_text,
                    enable_thinking_prompt=self.enable_thinking_prompt
                )
            elif knowledge_sub_type == "TaskPlanning":
                messages_dict = TaskPlanningPrompt(
                    ds_path=self.image_folder,
                    question_image_dir_list=question_image_dir_list,
                    question_text=question_text,
                    option_text_list=option_text_list,
                    q_type = question_type,
                    enable_thinking_prompt=self.enable_thinking_prompt
                )

        return SingleRoundItem(
            messages=messages_dict["messages"],
            gt_answer=groundtruth,
            index=index,
            gen_config=self.gen_config,
            extra={
                "question_id": question_id,
                "knowledge_type": knowledge_type,
                "knowledge_sub_type": knowledge_sub_type,
                "groundtruth": groundtruth,
                "answer_type": question_type,
                "app_type": line.get("app_type"),
                "os_type": os_type,
            }
        )

    def evaluate(self, singleitem_list):
        platform_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        subtype_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_correct = 0
        total = 0

        for item in singleitem_list:
            # 兼容字典和对象输入模式
            if isinstance(item, dict):
                extra = item.get('extra', {})
                response = item.get('model_response', '')
            else:
                extra = getattr(item, 'extra', {})
                response = getattr(item, 'model_response', '')

            platform = extra.get('knowledge_type', 'unknown')
            sub_type = extra.get('knowledge_sub_type', 'unknown')
            answer_type = extra.get('answer_type', None)
            gt = extra.get('groundtruth', None)

            # 鲁棒性答案提取
            answer_pred = extract_answer_from_response(response, answer_type=answer_type)
            correct = validate_answer(answer_pred, answer_type, gt)

            total += 1
            total_correct += int(correct)

        res = {}
        if total > 0:

            res['Overall Acc'] = total_correct / total * 100.0
        else:
            res['Overall Acc'] = 0.0

        print(f"total_correct: {total_correct}, total: {total}")
        print("Evaluation Results: ", res)
        return res


# # 3. run model
class InferenceSamplerV2(Sampler):
    def __init__(self, size):
        self._size = size
        assert size > 0
        # 使用 torch 原生分布式接口
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        # 简单的切片逻辑，确保每张卡处理不同的数据
        self._local_indices = [i for i in range(size) if i % self._world_size == self._rank]

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

class EvalGrounding(object):
    def __init__(self, ds_path, model_path, image_folder="", eval_benchmark_type="ssp"):
        if eval_benchmark_type == "kb":
            self.eval_benchmark_class = knowledge_bench(ds_path=ds_path,image_folder=image_folder,thinking=False)
        elif eval_benchmark_type == "kb-thinking":
            self.eval_benchmark_class = knowledge_bench(ds_path=ds_path,image_folder=image_folder,thinking=True)
        else:
            raise ValueError(f"Invalid eval_benchmark_type: {eval_benchmark_type}")
        self.eval_benchmark_type = eval_benchmark_type
        
        # 初始化分布式
        self.setup_distributed()

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map={'':torch.cuda.current_device()}
        )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    
    def setup_distributed(self):
        # 初始化分布式环境
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # 绑定当前进程到对应的 GPU
        torch.cuda.set_device(self.rank)
        print(f"Rank {self.rank} initialized on CUDA:{self.rank}")

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def call_model(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for retry in range(15):
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                break
            except Exception as e:
                time.sleep(1)
                continue
        try:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except:
            return "[Error Generation]"
        inputs = inputs.to(self.model.device)
        gen_config = {
            'top_p': 0.01,
            'top_k': 1,
            'temperature': 0.01,
            'repetition_penalty': 1.0,
            # 'out_seq_length': 2048,
        }
        inputs.update(gen_config)
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def main(self, save_path):
        if os.path.exists(save_path):
            if self.rank == 0:
                print(f"{save_path} already exists, skipping run model...")
                if ".jsonl" in save_path:
                    serializable = JsonlHelper.load_jsonl_to_list(save_path)
                elif ".json" in save_path:
                    serializable = JsonHelper.load_json(save_path)
                # final_outputs = [SingleRoundItem.from_dict(item) for item in serializable]

                final_outputs = serializable
                self.eval_benchmark_class.evaluate(final_outputs)
                print("Evaluation finished.")

            # 清理分布式环境
            self.cleanup_distributed()
            return
        
        loader = torch.utils.data.DataLoader(
            dataset=self.eval_benchmark_class, # 确保这里传的是 dataset 对象
            sampler=InferenceSamplerV2(size=len(self.eval_benchmark_class.dataset)),
            # sampler=InferenceSamplerV2(size=1),
            batch_size=1, 
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda m: m,
        )

        outputs = []
        # 仅在 Rank 0 显示进度条，避免刷屏
        if self.rank == 0:
            pbar = tqdm(loader, desc=f"{self.eval_benchmark_type}")
        else:
            pbar = loader

        for _, batch in enumerate(pbar):
            item = batch[0]
            for retry in range(20):
                try:
                    response = self.call_model(item.messages)
                    if response is None:
                        print(f"[Rank {self.rank}] call model error")
                        response = "[Error Generation]"
                    else:
                        break
                except Exception as e:
                    print(f"[Rank {self.rank}] Exception: {e}")
                    response = "[Error Generation]"
                    break
            print(f"[Rank {self.rank}] [model response] {response}")
            
            item.model_response = response
            outputs.append(item) # 直接 append item，方便后续处理

        # 3. 结果汇总 (Gather)
        # 将所有卡的 outputs 收集到 Rank 0
        all_outputs = [None for _ in range(self.world_size)] if self.rank == 0 else None
        dist.gather_object(outputs, all_outputs, dst=0)

        # 4. 仅在 Rank 0 进行评估
        if self.rank == 0:
            print("Gathering complete. Starting evaluation...")
            # 将列表展平
            final_outputs = [item for sublist in all_outputs for item in sublist]
            print("final_outputs: ", len(final_outputs))

            serializable = [it.to_dict() for it in final_outputs]
            JsonHelper.save_json(save_path, serializable)
            self.eval_benchmark_class.evaluate(serializable)
            print("Evaluation finished.")
        
        # 清理分布式环境
        self.cleanup_distributed()
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--ds_path", type=str, required=True, help="dataset path")
    parser.add_argument("--save_path", type=str, required=True, help="save/output path")
    parser.add_argument("--eval_benchmark_type", type=str, required=True, help="benchmark type")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_path
    ds_path = args.ds_path
    save_path = args.save_path
    eval_benchmark_type = args.eval_benchmark_type

    grounding_eval = EvalGrounding(ds_path, model_path=model_path, eval_benchmark_type=eval_benchmark_type)
    grounding_eval.main(save_path=save_path)
