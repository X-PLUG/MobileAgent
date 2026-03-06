from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os, json
from tqdm import tqdm

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
from typing import List
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Callable
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
MAX_PIXELS = 9800*32*32

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
def smart_resize(height, width, factor=32, min_pixels=56 * 56, max_pixels=32 * 32 * 9800, max_long_side=8192):
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

def update_image_size_(image_ele, min_tokens=1, max_tokens=9800, merge_base=2, patch_size=16):
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

def make_qwen_image_item(img_path: str, image=None, max_tokens=9800, patch_size=16):
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

class ScreenSpotv2(object):
    def __init__(self, ds_path, image_folder="", gen_config=None, use_qwen3vl=True, use_click_prefix=False, add_infesible_prefix=False, use_crop_tool=False, crop_image_save_dir="", zoomin_ratio=1):
        self.ds_path = ds_path
        self.img_folder = Path(ds_path, 'screenspotv2_image')
        self.use_qwen3vl = use_qwen3vl
        self.use_click_prefix = use_click_prefix
    
        self.suffix = "screenspot_v2"
        
        lines = []
        for platform in ['desktop', 'mobile', 'web']:
            tmp = read_json(Path(ds_path, f"screenspot_{platform}_v2.json"))
            for _ in tmp:
                _['platform'] = platform
                _['sub_task'] = platform
            lines.extend(tmp)
        self.dataset = lines
        self.add_infesible_prefix = add_infesible_prefix
        self.use_crop_tool = use_crop_tool
        self.crop_image_save_dir = crop_image_save_dir
        self.zoomin_ratio = zoomin_ratio
        assert not self.add_infesible_prefix or (self.add_infesible_prefix and self.use_qwen3vl)

    def __getitem__(self, index):
        line = self.dataset[index]

        if "http" in line['img_filename']:
            image_url = line['img_filename']
        else:
            if self.img_folder:
                if self.suffix == "osworld_g":
                    image_url = str(Path(self.img_folder, line['img_filename']).absolute())
                else:
                    image_url = str(Path(self.img_folder, line['img_filename'].strip("/")).absolute())
            else:
                image_url = str(Path(line['img_filename']).absolute())
        
        if self.use_qwen3vl:
            image_ele = make_qwen_image_item(image_url, max_tokens=9800, patch_size=16)
        else:
            image_ele = make_qwen_image_item(image_url)
        
        user_query = line['instruction']
        
        system_text = only_two_action_system_prompt
        if self.add_infesible_prefix:
            system_text = system_text + "\n" + infesible_prefix
        system_msg = {
                    "role": "system",
                    "content": [{"text": system_text, "type": "text"}]
                }

        messages = [
            system_msg,
            {
                "role": "user",
                "content": [
                    image_ele,
                    {
                        "type": "text",
                        "text": user_query
                    }
                ]
            }
        ]

        return SingleRoundItem(messages=messages, gt_answer=str(line['bbox']), index=index, extra={
            'bbox': line['bbox'],
            'image_ele': image_ele,
            'platform': line['platform'],
            'sub_task': line['sub_task'],
            'instruction': line['instruction'],
            'box_type': line.get('box_type', 'bbox')
        })
    
    def evaluate(self, singleitem_list):
        platform_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_correct = 0
        total = 0

        for item in singleitem_list:
            platform = item['extra']['platform']
            response = item['model_response']
            bbox = item['extra']['bbox']
            image_ele = item['extra']['image_ele']
            original_width = image_ele['width']
            original_height = image_ele['height']
            resized_width = image_ele['resized_width']
            resized_height = image_ele['resized_height']
            if self.use_qwen3vl:
                resized_width, resized_height = 1000, 1000

            box_type = item['extra']['box_type'] if 'box_type' in item['extra'] else "bbox"
            import re

            regex = r"\((\d+),\s*(\d+)\)"

            matches = re.findall(regex, response)
            if len(matches) == 0:
                regex = r"\[(\d+), (\d+)\]"
                matches = re.findall(regex, response)

            coordinates = []

            for match in matches:
                # 每个 match 是一个包含多个分组的元组
                if match[0] and match[1]:  # 匹配的是方括号的情况
                    x, y = int(match[0]), int(match[1])
                    coordinates.append((x, y))
            if coordinates:
                try:
                    x, y = coordinates[-1]
                    coord = [int(x), int(y)]
                except:
                    traceback.print_exc()
                    coord = [0, 0]
            else:
                coord = [0, 0]
            x_pred, y_pred = coord

            # 计算缩放比例
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height

            if box_type == "bbox":
                # 转换原始bbox到原图坐标 (假设原始格式为[x1,y1,w,h])
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h

                # 计算resized后的bbox坐标
                resized_x1 = math.floor(x1 * scale_x)
                resized_y1 = math.floor(y1 * scale_y)
                resized_x2 = math.ceil(x2 * scale_x)
                resized_y2 = math.ceil(y2 * scale_y)

                # 判断预测坐标是否在区域内
                correct = (resized_x1 <= x_pred <= resized_x2) and (resized_y1 <= y_pred <= resized_y2)
            elif box_type == "polygon":
                x, y = x_pred, y_pred
                polygon = bbox

                n = len(polygon) // 2
                correct = False
                j = n - 1

                for i in range(n):
                    # 缩放并读取当前点和前一点
                    xi = polygon[i * 2]     * scale_x
                    yi = polygon[i * 2 + 1] * scale_y
                    xj = polygon[j * 2]     * scale_x
                    yj = polygon[j * 2 + 1] * scale_y

                    # 射线法判断逻辑：跨过边则翻转 inside 状态
                    if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                        correct = not correct
                    j = i
            elif box_type == "refusal":
                center_point = [x_pred, y_pred]
                # print("center_point: ", center_point)
                # print("response: ", response)
                correct = all(center_point[i] == 0 for i in range(2))
            else:
                raise ValueError(f"Invalid box type: {box_type}")

            # print(f"coord: {coord}, gt: {[resized_x1, resized_y1, resized_x2, resized_y2]}, correct: {correct}")

            # 更新统计信息
            platform_stats[platform]['total'] += 1
            platform_stats[platform]['correct'] += correct
            total += 1
            total_correct += correct
            # print("total_correct: ", total_correct)
        res = {}
        for platform, stats in platform_stats.items():
            res[platform] = stats['correct']/stats['total']*100
        res['Overall Acc'] = total_correct/total*100
        print(f"total_correct: {total_correct}, total: {total}")
        print("Evaluation Results: ", res)
        return res
    
class ScreenSpotPro(ScreenSpotv2):
    def __init__(self, ds_path, image_folder="", gen_config=None, use_qwen3vl=True, use_click_prefix=False, add_infesible_prefix=False, use_function_cn=False, use_crop_tool=False, crop_image_save_dir="", zoomin_ratio=1):
        self.ds_path = ds_path
        if image_folder:
            self.img_folder = image_folder
        else:
            self.img_folder = Path(ds_path, 'images')
        # print("self.img_folder: ", self.img_folder)
    
        self.use_crop_tool = use_crop_tool
        self.suffix = "screenspot_pro"

        if self.use_crop_tool:
            self.suffix += f'_crop_tool_{str(self.use_crop_tool)}'

        lines = []
        for sub_task in Path(ds_path, 'annotations').iterdir():
            tmp = read_json(sub_task)
            for _ in tmp:
                try:
                    _['bbox'] = [_['bbox'][0], _['bbox'][1], _['bbox'][2]-_['bbox'][0], _['bbox'][3]-_['bbox'][1]] # xyxy -> xywh
                except:
                    print(f"{sub_task},\n{_}")
                _['platform'] = 'desktop'
                _['sub_task'] = sub_task.stem
            lines.extend(tmp)
        self.dataset = lines
        self.use_qwen3vl = use_qwen3vl
        self.use_click_prefix = use_click_prefix
        self.add_infesible_prefix = add_infesible_prefix
        
        self.crop_image_save_dir = crop_image_save_dir
        self.zoomin_ratio = zoomin_ratio

class MMBenchGUIL2(ScreenSpotv2):
    def __init__(self, ds_path, image_folder="", gen_config=None, use_qwen3vl=True, use_click_prefix=False, add_infesible_prefix=False, use_function_cn=False, use_crop_tool=False, crop_image_save_dir="", zoomin_ratio=1):
        self.ds_path = ds_path
        self.img_folder = Path(ds_path, 'offline_images')
        self.suffix = "mmbench_gui_l2"

        lines = read_json(Path(ds_path, 'L2_annotations.json'))
        for _ in lines:
            _['bbox'] = [_['bbox'][0], _['bbox'][1], _['bbox'][2]-_['bbox'][0], _['bbox'][3]-_['bbox'][1]] # xyxy -> xywh
            _['bbox'] = [round(_['bbox'][0]*_['image_size'][0]), round(_['bbox'][1]*_['image_size'][1]), round(_['bbox'][2]*_['image_size'][0]), round(_['bbox'][3]*_['image_size'][1])]
            _['sub_task'] = _['grounding_type']
            _['img_filename'] = f"{_['platform']}/{_['image_path']}"
        self.dataset = lines
        self.use_qwen3vl = use_qwen3vl
        self.use_click_prefix = use_click_prefix
        self.add_infesible_prefix = add_infesible_prefix
        self.use_crop_tool = use_crop_tool
        self.crop_image_save_dir = crop_image_save_dir
        self.zoomin_ratio = zoomin_ratio

class OSWorldG(ScreenSpotPro):
    def __init__(self, ds_path, image_folder="", gen_config=None, use_qwen3vl=True, use_click_prefix=False, add_infesible_prefix=True, use_function_cn=False, use_crop_tool=False, crop_image_save_dir="", zoomin_ratio=1):
        self.ds_path = ds_path
        self.img_folder = Path(ds_path, 'images')

        self.suffix = "osworld_g"
        lines = []
        for sub_task in Path(ds_path, 'annotations').iterdir():
            # if ".json" not in str(sub_task) and ".jsonl" not in str(sub_task):
            #     continue
            tmp = read_json(sub_task)
            # print("sub_task.stem: ", sub_task.stem)
            if "refined" in sub_task.stem:
                platform = "desktop_osworld_refined"
            else:
                platform = "desktop_osworld"
            for _ in tmp:
                _['platform'] = platform
                _['sub_task'] = "osworld_g"
            lines.extend(tmp)

        self.dataset = lines
        self.use_qwen3vl = use_qwen3vl
        self.add_infesible_prefix = add_infesible_prefix

        self.use_crop_tool = use_crop_tool
        self.crop_image_save_dir = crop_image_save_dir
        self.zoomin_ratio = zoomin_ratio

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
        if eval_benchmark_type == "spv2":
            self.eval_benchmark_class = ScreenSpotv2(ds_path=ds_path, image_folder=image_folder)
        elif eval_benchmark_type == "ssp":
            self.eval_benchmark_class = ScreenSpotPro(ds_path=ds_path, image_folder=image_folder)
        elif eval_benchmark_type == "mmbench_l2":
            self.eval_benchmark_class = MMBenchGUIL2(ds_path=ds_path, image_folder=image_folder)
        elif eval_benchmark_type == "osg":
            self.eval_benchmark_class = OSWorldG(ds_path=ds_path, image_folder=image_folder)
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
