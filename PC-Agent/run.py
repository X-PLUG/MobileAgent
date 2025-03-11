import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw

from PCAgent.api import inference_chat
from PCAgent.text_localization_old import ocr
from PCAgent.icon_localization import det
from PCAgent.prompt_qwen import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt, get_select_prompt
from PCAgent.prompt_qwen import get_subtask_prompt as get_subtask_prompt
from PCAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, init_subtask_chat

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent

from pynput.mouse import Button, Controller
import argparse
import pyautogui
import pyperclip
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes, merge_boxes_and_texts_new, merge_all_icon_boxes_new

import json
import pdb
import ast
import re

from pywin import WindowsACI, UIElement
from OpenOCR.tools.infer_e2e import OpenOCR

def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    match = chinese_pattern.search(text)
    return match is not None


import random
from PIL import ImageFont

def cmyk_to_rgb(c, m, y, k):
    r = 255 * (1.0 - c / 255) * (1.0 - k / 255)
    g = 255 * (1.0 - m / 255) * (1.0 - k / 255)
    b = 255 * (1.0 - y / 255) * (1.0 - k / 255)
    return int(r), int(g), int(b)

def draw_coordinates_boxes_on_image(image_path, coordinates, output_image_path, font_path, no_text=0):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    total_boxes = len(coordinates)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
              range(total_boxes)]
    # padding = height * 0.005

    for i, coord in enumerate(coordinates):
        c, m, y, k = colors[i]
        color = cmyk_to_rgb(c, m, y, k)

        draw.rectangle(coord, outline=color, width=int(height * 0.0025))
        
        if no_text != 1:
            font = ImageFont.truetype(font_path, int(height * 0.012))
            text_x = coord[0] + int(height * 0.0025)
            text_y = max(0, coord[1] - int(height * 0.013))
            draw.text((text_x, text_y), str(i + 1), fill=color, font=font)
    # image.show()
    image = image.convert('RGB')

    if os.path.exists(output_image_path):
        os.remove(output_image_path)
    image.save(output_image_path)


parser = argparse.ArgumentParser(description="PC Agent")
# parser.add_argument('--instruction', type=str, default="打开桌面上的memo文本文件，查看其中的第二个事件，并在闹钟中设定一个在这个事件前一小时的闹钟")
parser.add_argument('--instruction', type=str, default="在Chrome中搜索英伟达的股价，然后在股票信息excel文件中，将公司名写在A列，股票信息写在B列。")
# parser.add_argument('--instruction', type=str, default="在当前表格的A1位置写入数字100")


parser.add_argument('--use_som', type=int, default=1) # for action
parser.add_argument('--draw_text_box', type=int, default=0, help="whether to draw text boxes in som.")
parser.add_argument('--font_path', type=str, default="C:\Windows\Fonts\\times.ttf")
# parser.add_argument('--add_info', type=str, default="通过ctrl+t来新建标签页，从而进行新的搜索")
parser.add_argument('--add_info', type=str, default="")

parser.add_argument('--disable_reflection', type=int, default=1)
parser.add_argument('--clear_history_each_subtask', type=int, default=1)
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--use_a11y', type=int, default=1)
parser.add_argument('--text_len_thre', type=int, default=1000)
parser.add_argument('--num_step_limit', type=int, default=20)
parser.add_argument('--simple', type=int, default=0) # for simple instruction
parser.add_argument('--screenshot_root', type=str, default='task_')
parser.add_argument('--mute', type=int, default=0)

args = parser.parse_args()

exclude_words = ["系统"]
# TODO
# exclude_words = json.load(open('filter_icon.json', 'r'))

token_data = json.load(open('config.json', 'r'))
vl_model_version = token_data['vl_model_name']
llm_model_version = token_data['llm_model_name']
API_url = token_data['url']
token = token_data['token']

ctrl_key = "ctrl"
search_key = ["win", "s"]
ratio = args.ratio


def get_screenshot(screenshot_file):
    if os.path.exists(screenshot_file):
        os.remove(screenshot_file)
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_file)
    return

def home():
    key1 = 'win'
    key2 = 'd'
    pyautogui.keyDown(key1)
    pyautogui.keyDown(key2)
    pyautogui.keyUp(key2)
    pyautogui.keyUp(key1)

def open_app(name):
    if 'Outlook' in name:
        name = name.replace('Outlook', 'Outlook new')
    print('Action: open %s' % name)
    pyautogui.keyDown(search_key[0])
    pyautogui.keyDown(search_key[1])
    pyautogui.keyUp(search_key[1])
    pyautogui.keyUp(search_key[0])
    if contains_chinese(name):
        pyperclip.copy(name)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(name)
    time.sleep(1)
    pyautogui.press('enter')
    time.sleep(1)
    # time.sleep(0.1)
    # pyautogui.press('enter')

def tap(x, y, count=1):
    x, y = x//ratio, y//ratio
    print('Action: click (%d, %d) %d times' % (x, y, count))
    mouse = Controller()
    pyautogui.moveTo(x,y)
    mouse.click(Button.left, count=count)
    return

def shortcut(key1, key2):
    if key1 == 'command' or key1 == 'ctrl':
        key1 = ctrl_key
    print('Action: shortcut %s + %s' % (key1, key2))
    pyautogui.keyDown(key1)
    pyautogui.keyDown(key2)
    pyautogui.keyUp(key2)
    pyautogui.keyUp(key1)
    return

def presskey(key):
    print('Action: press %s' % key)
    pyautogui.press(key)

def tap_type_enter(x, y, text):
    x, y = x//ratio, y//ratio
    print('Action: click (%d, %d), enter %s and press Enter' % (x, y, text))
    pyautogui.click(x=x, y=y)
    if contains_chinese(text):
        pyperclip.copy(text)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(text)
    time.sleep(1)
    pyautogui.press('enter')
    return

def select(content, screenshot_file):
    prompt_select = get_select_prompt(content)
    chat_select = init_action_chat()
    chat_select = add_response("user", prompt_select, chat_select, [screenshot_file])
    output_select = inference_chat(chat_select, vl_model_version, API_url, token)
    print(output_select)
    first_line = output_select.split('<first>')[-1].split('</first>')[0][:30]
    last_line = output_select.split('<last>')[-1].split('</last>')[0][-30:]
    time.sleep(2)

    text_sys = OpenOCR(mode='mobile', drop_score=0.5,
                       det_box_type='quad')  # det_box_type: 'quad' or 'poly'
    res, _ = text_sys(img_path=screenshot_file, save_dir='e2e_results/', is_visualize=False)

    for item in res[0]:
        if first_line in item['transcription']:
            corr_first = item['points'][0]
            break
    for item in res[0]:
        if last_line in item['transcription']:
            corr_last = item['points'][2]
            break
    
    x1, y1 = corr_first
    x2, y2 = corr_last
    drag(x1, y1, x2, y2)
    return


def drag(x1, y1, x2, y2):
    x1, y1 = x1//ratio, y1//ratio
    x2, y2 = x2//ratio, y2//ratio
    pyautogui.moveTo(x1,y1)
    pyautogui.mouseDown()
    pyautogui.moveTo(x2,y2,duration=0.5)
    pyautogui.mouseUp()
    print('Action: drag from (%d, %d) to (%d, %d)' % (x1, y1, x2, y2))
    return

def replace(x, y, text):
    x, y = x//ratio, y//ratio
    print('Action: replace the content at (%d, %d) with %s and press Enter' % (x, y, text))
    mouse = Controller()
    pyautogui.moveTo(x,y)
    mouse.click(Button.left, count=2)
    shortcut('command', 'a')
    if contains_chinese(text):
        pyperclip.copy(text)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(text)
    time.sleep(1)
    pyautogui.press('enter')
    return


def append(x, y, text):
    x, y = x//ratio, y//ratio
    print('Action: append the content at (%d, %d) with %s and press Enter' % (x, y, text))
    mouse = Controller()
    pyautogui.moveTo(x,y)
    mouse.click(Button.left, count=1)
    shortcut('command', 'a')
    pyautogui.press('down')
    if contains_chinese(text):
        pyperclip.copy(text)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(text)
    time.sleep(1)
    pyautogui.press('enter')
    return


####################################### Edit your Setting #########################################

instruction = args.instruction

# You can add operational knowledge to help Agent operate more accurately.
add_info_basic = '''
When searching in the browser, click on the search bar at the top.
The input field in WeChat is near the send button.
When downloading files in the browser, it's preferred to use keyboard shortcuts.\n
'''

add_info = add_info_basic + args.add_info


# Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
reflection_switch = True if args.disable_reflection == 0 else False

# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = False # default: False
###################################################################################################


def get_perception_infos(screenshot_file, screenshot_som_file, font_path):
    get_screenshot(screenshot_file)
    
    total_width, total_height = Image.open(screenshot_file).size

    # no partition
    img_list = [screenshot_file]
    img_x_list = [0]
    img_y_list = [0]

    coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10

    for i, img in enumerate(img_list):
        width, height = Image.open(img).size

        sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition) # for old
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height,img_y_list[i] + coordinate[3] + padding))

        sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
        coordinates.extend(sub_coordinates_merge)
        texts.extend(sub_text_merge)
    merged_text, merged_text_coordinates = merge_boxes_and_texts(texts, coordinates)

    filtered_merged_text = []
    filtered_merged_text_coordinates = []
    for i in range(len(merged_text)):
        if len(merged_text[i]) <= args.text_len_thre:
            filtered_merged_text.append(merged_text[i])
            filtered_merged_text_coordinates.append(merged_text_coordinates[i])
    merged_text, merged_text_coordinates = filtered_merged_text, filtered_merged_text_coordinates


    if args.use_a11y == 0:
        coordinates = []
        for i, img in enumerate(img_list):
            width, height = Image.open(img).size
            sub_coordinates = det(img, "icon", groundingdino_model)
            for coordinate in sub_coordinates:
                coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
                coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
                coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
                coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))

            sub_coordinates = merge_all_icon_boxes(sub_coordinates)
            coordinates.extend(sub_coordinates)
        merged_icon_coordinates = merge_all_icon_boxes(coordinates)
    else:
        obs = {}
        obs["accessibility_tree"] = UIElement.systemWideElement()
        ACI = WindowsACI()
        elements = ACI.linearize_and_annotate_tree(obs)
        elements_filtered = [ele for ele in elements if len(ele['text'])<args.text_len_thre and ele['text'] not in exclude_words] # and "List Paragraph" not in ele['text']]
        elements = elements_filtered
        merged_icon_coordinates = [[ele['position'][0], ele['position'][1], ele['position'][0]+ele['size'][0], ele['position'][1]+ele['size'][1]] for ele in elements]

        ocr_bboxes = [(merged_text[i], merged_text_coordinates[i]) for i in range(len(merged_text))]
        filtered_ocr_bboxes = ACI.filter_ocr_elements(ocr_bboxes, elements)
        merged_text = [_[0] for _ in filtered_ocr_bboxes]
        merged_text_coordinates = [_[1] for _ in filtered_ocr_bboxes]

    if args.draw_text_box == 1:
        rec_list = merged_text_coordinates + merged_icon_coordinates
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(rec_list), screenshot_som_file, font_path)
    else:
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), screenshot_som_file, font_path)

    mark_number = 0
    perception_infos = []

    for i in range(len(merged_text_coordinates)):
        if args.use_som == 1 and args.draw_text_box == 1:
            mark_number += 1
            perception_info = {"text": "mark number: " + str(mark_number) + " text: " + merged_text[i], "coordinates": merged_text_coordinates[i]}
        else:
            perception_info = {"text": "text: " + merged_text[i], "coordinates": merged_text_coordinates[i]}
        perception_infos.append(perception_info)

    for i in range(len(merged_icon_coordinates)):
        if args.use_som == 1:
            mark_number += 1
            if args.use_a11y == 0:
                perception_info = {"text": "mark number: " + str(mark_number) + " icon", "coordinates": merged_icon_coordinates[i]}
            else:
                perception_info = {"text": "mark number: " + str(mark_number) + " icon: " + elements[i]['text'], "coordinates": merged_icon_coordinates[i]}
        else:
            if args.use_a11y == 0:
                perception_info = {"text": "icon", "coordinates": merged_icon_coordinates[i]}
            else:
                pass # TODO
        perception_infos.append(perception_info)

    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]

    return perception_infos, total_width, total_height


### Load ocr and icon detection model ###
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

if not args.use_a11y:
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)

def analyze_string(s):
    result = {
        'type': None,
        'format_keys': [],
        'dict_content': None
    }

    format_pattern = re.compile(r'\{(\w+)\}')

    #  {'key': 'value'}
    dict_pattern = re.compile(
        r'\{(?:\s*[\'\"]\w+[\'\"]\s*:\s*[\'\"][^{}\'\"]+[\'\"]\s*,?)*\}'
    )

    dict_matches = dict_pattern.findall(s)
    dicts = []
    for match in dict_matches:
        try:
            parsed_dict = ast.literal_eval(match)
            if isinstance(parsed_dict, dict):
                dicts.append(parsed_dict)
        except (ValueError, SyntaxError):
            continue

    has_dict = len(dicts) > 0

    s_without_dicts = dict_pattern.sub('', s)

    format_keys = format_pattern.findall(s_without_dicts)
    has_format = len(format_keys) > 0

    has_format_and_dict = has_format and has_dict

    if has_format_and_dict:
        result['type'] = 4
    elif has_format:
        result['type'] = 2
    elif has_dict:
        result['type'] = 3
    else:
        result['type'] = 1

    if has_format:
        result['format_keys'] = format_keys

    if has_dict:
        result['dict_content'] = dicts[0]

    return result

import re

def is_good_string(s):
    # Regex to match the dictionary-like part {'key1': 'value1', ...}
    dict_pattern = r"\{('[^']+' *: *'[^']+' *(, *'[^']+' *: *'[^']+')*)?\}"
    # Regex to match the item list part {item1, item2,...} with no single quotes in items
    item_pattern = r"\{([a-zA-Z0-9_]+( *, *[a-zA-Z0-9_]+)*)?\}"
    
    # Find all parts of the string contained within braces
    parts = re.findall(r'\{.*?\}', s)
    
    for part in parts:
        # Check if the part matches either the dictionary pattern or item pattern
        if not re.fullmatch(dict_pattern, part) and not re.fullmatch(item_pattern, part):
            return False
    return True

def check_subtask_dict(subtask_dict):
    num_subtask = len(list(subtask_dict.keys()))
    all_dict = {}
    for i in range(num_subtask):
        value = subtask_dict['subtask %d'%(i+1)]
        if is_good_string(value) == False:
            return False

        res = analyze_string(subtask_dict['subtask %d'%(i+1)])
        if res['type'] in [2, 4]:
            this_format_keys = res['format_keys']
            for format_key in this_format_keys:
                if format_key not in all_dict:
                    return False
        if res['type'] in [3, 4]:
            this_dict = res['dict_content']
            for k, v in this_dict.items():
                if k in all_dict:
                    return False
                all_dict[k] = v
    return True


output_for_save = []

thought_history = []
summary_history = []
action_history = []
reflection_history = []

reflection_thought = ""
summary = ""
action = ""
completed_requirements = ""
memory = ""
insight = ""
temp_file = "temp"
screenshot_root = args.screenshot_root + '%d/' % (1)

if os.path.exists(temp_file):
    shutil.rmtree(temp_file)
os.mkdir(temp_file)
if not os.path.exists(screenshot_root):
    os.mkdir(screenshot_root)
error_flag = False

# communication hub
answer_dict = {}

# call 4o gets the decomposed subtasks
prompt_subtask = get_subtask_prompt(instruction)
chat_subtask = init_subtask_chat()
chat_subtask = add_response("user", prompt_subtask, chat_subtask)
num_try_subtask = 5
flag_subtask = False

# Simple instruction, skips subtask decompose
if args.simple == 1:
    subtask_dict = {"subtask 1": instruction}
else:
    for i in range(num_try_subtask):
        try:
            output_subtask = inference_chat(chat_subtask, llm_model_version, API_url, token) # 2.2 modified

            if args.mute == 0:
                print(output_subtask)
            subtask_dict = json.loads(output_subtask.replace('```python', '```json').split('```json')[-1].split('```')[0])
            # check the format subtask 1, ...
            flag_check_subtask = True
            for j in range(len(list(subtask_dict.keys()))):
                if 'subtask %d'%(j+1) not in subtask_dict:
                    flag_check_subtask = False
                    break
            if not flag_check_subtask:
                continue

            # Check that the keys in the subtask have matching dict values in the previous subtasks
            if check_subtask_dict(subtask_dict):
                flag_subtask = True
                break
            else:
                continue

        except:
            continue
    if not flag_subtask:
        assert 0 == 1, "Fetch subtask_dict failed!"

task_dict = copy.deepcopy(subtask_dict)
task_dict['instruction'] = instruction
json.dump(task_dict, open(screenshot_root+'instruction.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

step_idx = 0

num_subtask = len(list(subtask_dict.keys()))
for i in range(num_subtask):
    sub_instruction = subtask_dict['subtask %d'%(i+1)]
    string_res = analyze_string(sub_instruction)
    if string_res['type'] in [2, 4]:
        this_format_keys = string_res['format_keys']
        try:
            value_dict = {}
            for key in this_format_keys:
                value_dict[key] = answer_dict[key]
            sub_instruction = sub_instruction.replace("{'", "{{'").replace("'}", "'}}").format_map(value_dict) # .format(**value_dict)
        except:
            assert 0 == 1, "loss some paramters"

    if args.mute == 0:
        print(sub_instruction)

    if args.clear_history_each_subtask == 1:
        thought_history = []
        summary_history = []
        action_history = []
        reflection_history = []

    iter = 0
    while True:
        start_time = time.perf_counter()
        iter += 1
        step_idx += 1
        if step_idx > args.num_step_limit:
            break
        if iter == 1:
            screenshot_file = screenshot_root+"screenshot_%d.png"%step_idx
            screenshot_som_file = screenshot_root+"screenshot_%d_som.png"%step_idx
            perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)

            shutil.rmtree(temp_file)
            os.mkdir(temp_file)
            json.dump(perception_infos, open(screenshot_root+'perception_infos_step%d.json'%step_idx, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

        output_for_save_this_step = {}

        prompt_action = get_action_prompt(sub_instruction, perception_infos, width, height, thought_history, summary_history, action_history, reflection_history, summary, action, reflection_thought, add_info, error_flag, completed_requirements, memory)
        chat_action = init_action_chat()
        if args.use_som == 1:
            chat_action = add_response("user", prompt_action, chat_action, [screenshot_som_file])
        else:
            chat_action = add_response("user", prompt_action, chat_action, [screenshot_file])

        output_action = inference_chat(chat_action, vl_model_version, API_url, token)

        output_for_save_this_step['action'] = output_action

        # thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
        # summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        # action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()

        action_json = json.loads(output_action.split('```json')[-1].split('```')[0])
        thought = action_json['Thought']
        summary = action_json['Summary']
        action = action_json['Action']

        chat_action = add_response("assistant", output_action, chat_action)
        status = "#" * 50 + " Decision " + "#" * 50
        if args.mute == 0:
            print(status)
            print(output_action)
            print('#' * len(status))

        if "Double TapIdx" in action:
            idx = action.split("(")[-1].split(")")[0]
            coordinate = perception_infos[i]['coordinates']
            x, y = int(coordinate[0]), int(coordinate[1])
            tap(x, y, 2)

        elif "Double Tap" in action:
            try:
                coordinate = action.split("(")[-1].split(")")[0].split(", ")
                x, y = int(coordinate[0]), int(coordinate[1])
                tap(x, y, 2)
            except:
                pass

        elif "Triple TapIdx" in action:
            coordinate = action.split("(")[-1].split(")")[0].split(", ")
            x, y = int(coordinate[0]), int(coordinate[1])
            tap(x, y, 3)

        elif "Triple Tap" in action:
            try:
                idx = action.split("(")[-1].split(")")[0]
                coordinate = perception_infos[i]['coordinates']
                x, y = int(coordinate[0]), int(coordinate[1])
                tap(x, y, 3)
            except:
                pass

        elif "TapIdx" in action:
            idx = action.split("(")[-1].split(")")[0]
            coordinate = perception_infos[i]['coordinates']
            x, y = int(coordinate[0]), int(coordinate[1])
            tap(x, y, 1)

        elif "Tap" in action:
            try:
                coordinate = action.split("(")[-1].split(")")[0].split(", ")
                x, y = int(coordinate[0]), int(coordinate[1])
                tap(x, y, 1)
            except:
                pass

        elif "Shortcut" in action:
            keys = action.split("(")[-1].split(")")[0].split(", ")
            key1, key2 = keys[0].lower(), keys[1].lower()
            shortcut(key1, key2)

        elif "Press" in action:
            key = action.split("(")[-1].split(")")[0]
            presskey(key)

        elif "Open App" in action:
            app = action.split("(")[-1].split(")")[0]
            open_app(app)

        elif "Type" in action:
            try:
                coordinate = action.split("(")[1].split(")")[0].split(", ")
                x, y = int(coordinate[0]), int(coordinate[1])
                if "[text]" not in action:
                    # for claude
                    if '[' not in action or ']' not in action:
                        # text = action.split('),')[-1].strip()
                        text = action.split('),')[-1].strip().split("(")[1].split(")")[0].replace("text: ", '').replace("'", "")
                    else:
                        text = action.split("[")[-1].split("]")[0]
                else:
                    text = action.split(" \"")[-1].split("\"")[0]
                actions = tap_type_enter(x, y, text)
                # actions = tap_type(x, y, text.strip("\"").strip("'"))
            except:
                pass

        elif "Select (" in action:
            content = action.split("(")[1].split(")")[0]
            select(content, screenshot_file)
        elif "Replace (" in action:
            try:
                coordinate = action.split("(")[1].split(")")[0].split(", ")
                x, y = int(coordinate[0]), int(coordinate[1])
                if "[text]" not in action:
                    # for claude
                    if '[' not in action or ']' not in action:
                        # text = action.split('),')[-1].strip()
                        text = action.split('),')[-1].strip().split("(")[1].split(")")[0].replace("text: ", '')
                    else:
                        if "] with " in action:
                            text = action.split("] with ")[-1]
                            text = text.replace("\"", '').replace("'", '').strip('.')
                        else:
                            text = action.split("[")[-1].split("]")[0]
                else:
                    text = action.split(" \"")[-1].split("\"")[0]
                replace(x, y, text.strip("\"").strip("'"))

            except:
                pass

        elif "Append (" in action:
            try:
                coordinate = action.split("(")[1].split(")")[0].split(", ")
                x, y = int(coordinate[0]), int(coordinate[1])
                if "[text]" not in action:
                    if '[' not in action or ']' not in action:
                        text = action.split('),')[-1].strip()
                    else:
                        text = action.split("[")[-1].split("]")[0]
                else:
                    text = action.split(" \"")[-1].split("\"")[0]
                append(x, y, text.strip("\""))
            except:
                pass

        elif "Tell (" in action:
            answer = '{' + action.split('({')[-1].split('})')[0]+'}'
            try:
                answer = ast.literal_eval(answer)
                for key, value in answer.items():
                    answer_dict[key] = value
                if args.mute == 0:
                    print(answer_dict)
            except:
                print(answer)
            output_for_save.append(output_for_save_this_step)
            json.dump(output_for_save, open(screenshot_root+'output_for_save.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
            break

        elif "Stop" in action:
            output_for_save.append(output_for_save_this_step)
            json.dump(output_for_save, open(screenshot_root+'output_for_save.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
            break

        time.sleep(2) # wait for the action to be excuted

        if memory_switch:
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            output_memory = inference_chat(chat_action, vl_model_version, API_url, token)
            chat_action = add_response("assistant", output_memory, chat_action)
            status = "#" * 50 + " Memory " + "#" * 50
            print(status)
            print(output_memory)
            print('#' * len(status))
            output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory

        last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = screenshot_root+"screenshot_%d.png"%(step_idx)

        if args.use_som == 1:
            last_screenshot_som_file = screenshot_root+"screenshot_%d_som.png"%(step_idx)

        screenshot_file = screenshot_root+"screenshot_%d.png"%(step_idx+1)
        screenshot_som_file = screenshot_root+"screenshot_%d_som.png"%(step_idx+1)

        perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)
        json.dump(perception_infos, open(screenshot_root+'perception_infos_step%d.json'%(step_idx+1), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

        if reflection_switch:
            prompt_reflect = get_reflect_prompt(sub_instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info)
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response("user", prompt_reflect, chat_reflect, [last_screenshot_file, screenshot_file])

            output_reflect = inference_chat(chat_reflect, vl_model_version, API_url, token)

            output_for_save_this_step['reflect'] = output_reflect

            reflection_thought = output_reflect.split("### Thought ###")[-1].split("### Answer ###")[0].replace("\n", " ").strip()
            reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
            chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            status = "#" * 50 + " Reflection " + "#" * 50
            if args.mute == 0:
                print(status)
                print(output_reflect)
                print('#' * len(status))

            if True:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)
                reflection_history.append(reflection_thought)

                prompt_planning = get_process_prompt(sub_instruction, thought_history, summary_history, action_history, completed_requirements, add_info, reflection_history)
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)

                output_planning = inference_chat(chat_planning, llm_model_version, API_url, token)

                output_for_save_this_step['planning'] = output_planning

                chat_planning = add_response("assistant", output_planning, chat_planning )
                status = "#" * 50 + " Planning " + "#" * 50
                if args.mute == 0:
                    print(status)
                    print(output_planning)
                    print('#' * len(status))
                completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
        else:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)

            prompt_planning = get_process_prompt(sub_instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning )
            output_planning = inference_chat(chat_planning, llm_model_version, API_url, token)
            output_for_save_this_step['planning'] = output_planning
            chat_planning = add_response("assistant", output_planning, chat_planning )
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()

        output_for_save.append(output_for_save_this_step)
        json.dump(output_for_save, open(screenshot_root+'output_for_save.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    if step_idx > args.num_step_limit:
        break

json.dump(output_for_save, open(screenshot_root+'output_for_save.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

