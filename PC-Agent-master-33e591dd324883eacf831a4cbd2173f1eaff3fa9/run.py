import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw

from PCAgent.api import inference_chat
from PCAgent.text_localization import ocr
from PCAgent.icon_localization import det
from PCAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from PCAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response

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
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes, merge_boxes_and_texts_new



import re

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

def draw_coordinates_boxes_on_image(image_path, coordinates, output_image_path, font_path):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    total_boxes = len(coordinates)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
              range(total_boxes)]
    # padding = height * 0.005

    for i, coord in enumerate(coordinates):
        # color = generate_color_from_hsv_pil(i, total_boxes)
        c, m, y, k = colors[i]
        color = cmyk_to_rgb(c, m, y, k)
        # print(color)

        # coord[0] = coord[0] - padding
        # coord[1] = coord[1] - padding
        # coord[2] = coord[2] + padding
        # coord[3] = coord[3] + padding

        draw.rectangle(coord, outline=color, width=int(height * 0.0025))

        font = ImageFont.truetype(font_path, int(height * 0.012))
        text_x = coord[0] + int(height * 0.0025)
        text_y = max(0, coord[1] - int(height * 0.013))
        draw.text((text_x, text_y), str(i + 1), fill=color, font=font)
    # image.show()
    image = image.convert('RGB')
    image.save(output_image_path)



parser = argparse.ArgumentParser(description="PC Agent")
parser.add_argument('--instruction', type=str, default='default')
parser.add_argument('--icon_caption', type=int, default=0) # 0: w/o icon_caption
parser.add_argument('--location_info', type=str, default='center') # center or bbox or icon_centor; icon_center: only icon center
parser.add_argument('--use_som', type=int, default=1) # for action
parser.add_argument('--draw_text_box', type=int, default=0, help="whether to draw text boxes in som.")
parser.add_argument('--font_path', type=str, default="/System/Library/Fonts/Times.ttc")
parser.add_argument('--pc_type', type=str, default="mac") # windows or mac
parser.add_argument('--api_url', type=str, default="https://api.openai.com/v1/chat/completions", help="GPT-4o api url.")
parser.add_argument('--api_token', type=str, help="Your GPT-4o api token.")
parser.add_argument('--qwen_api', type=str, default='', help="Input your Qwen-VL api if icon_caption=1.")
parser.add_argument('--add_info', type=str, default='')
parser.add_argument('--disable_reflection', action='store_true')

args = parser.parse_args()

if args.pc_type == "mac":
    ctrl_key = "command"
    search_key = ["command", "space"]
    ratio = 2
else:
    ctrl_key = "ctrl"
    search_key = ["win", "s"]
    ratio = 1
    args.font_path = "C:\Windows\Fonts\\times.ttf"

vl_model_version = 'gpt-4o'

def get_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save('screenshot/screenshot.png')
    return

def open_app(name):
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

def tap(x, y, count=1):
    x, y = x//ratio, y//ratio
    print('Action: click (%d, %d) %d times' % (x, y, count))
    mouse = Controller()
    pyautogui.moveTo(x,y)
    mouse.click(Button.left, count=count)
    return

def shortcut(key1, key2):
    if key1 == 'command' and args.pc_type != "mac":
        key1 = 'ctrl'
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


####################################### Edit your Setting #########################################

if args.instruction != 'default':
    instruction = args.instruction
else:
    # Your default instruction
    instruction = "Create a new doc on Word, write a brief introduction of Alibaba, and save the document."
    # instruction = "Help me download the pdf version of the 'Mobile Agent v2' paper on Chrome."

# Your GPT-4o API URL
API_url = args.api_url

# Your GPT-4o API Token
token = args.api_token

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
caption_call_method = "api"

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
caption_model = "qwen-vl-max"

# If you choose the api caption call method, input your Qwen api here
qwen_api = args.qwen_api

# You can add operational knowledge to help Agent operate more accurately.
if args.add_info == '':
    add_info = '''
    When searching in the browser, click on the search bar at the top.
    The input field in WeChat is near the send button.
    When downloading files in the browser, it's preferred to use keyboard shortcuts.
    '''
else:
    add_info = args.add_info

# Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
reflection_switch = True if not args.disable_reflection else False

# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = False # default: False
###################################################################################################


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size, coord[0] + point_size, coord[1] + point_size), fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def draw_rectangles_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for coord in coordinates:
        draw.rectangle([coord[0], coord[1]], outline="red", width=2)
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./temp/{i}.png")


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def process_image(image, query):
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': image
            },
            {
                'text': query
            },
        ]
    }]
    response = MultiModalConversation.call(model=caption_model, messages=messages)
    
    try:
        response = response['output']['choices'][0]['message']['content'][0]["text"]
    except:
        response = "An icon."
    
    return response


def generate_api(images, query):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    
    return icon_map


def split_image_into_4(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 2
    sub_height = height // 2

    # crop into 4 sub images
    quadrants = [
        (0, 0, sub_width, sub_height),
        (sub_width, 0, width, sub_height),
        (0, sub_height, sub_width, height),
        (sub_width, sub_height, width, height)
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

def ocr_parallel(img, ocr_detection, ocr_recognition, img_x_list, img_y_list, padding, i):
    width, height = Image.open(img).size
    sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
    for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(width*2, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(height*2,img_y_list[i] + coordinate[3] + padding))
    sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
    print('parallel end')
    return sub_text_merge, sub_coordinates_merge

def icon_parallel(img, det, img_x_list, img_y_list, padding, i):
    width, height = Image.open(img).size
    sub_coordinates = det(img, "icon", groundingdino_model)
    for coordinate in sub_coordinates:
        coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
        coordinate[2] = int(min(width*2, img_x_list[i] + coordinate[2] + padding))
        coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
        coordinate[3] = int(min(height*2, img_y_list[i] + coordinate[3] + padding))
    sub_coordinates = merge_all_icon_boxes(sub_coordinates)
    return sub_coordinates

def get_perception_infos(screenshot_file, screenshot_som_file, font_path):
    get_screenshot()
    
    total_width, total_height = Image.open(screenshot_file).size

    # Partition Image into 4 parts
    split_image_into_4(screenshot_file, './screenshot/screenshot')
    img_list = ['./screenshot/screenshot_part_1.png', './screenshot/screenshot_part_2.png',
                './screenshot/screenshot_part_3.png', './screenshot/screenshot_part_4.png']
    img_x_list = [0, total_width/2, 0, total_width/2]
    img_y_list = [0, 0, total_height/2, total_height/2]
    coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10

    for i, img in enumerate(img_list):
        width, height = Image.open(img).size

        sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height,img_y_list[i] + coordinate[3] + padding))

        sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
        coordinates.extend(sub_coordinates_merge)
        texts.extend(sub_text_merge)
    merged_text, merged_text_coordinates = merge_boxes_and_texts(texts, coordinates)

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
            perception_info = {"text": "mark number: " + str(mark_number) + " icon", "coordinates": merged_icon_coordinates[i]}
        else:
            perception_info = {"text": "icon", "coordinates": merged_icon_coordinates[i]}
        perception_infos.append(perception_info)
    
    if args.icon_caption == 1:
        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            # if perception_infos[i]['text'] == 'icon':
            if 'icon' in perception_infos[i]['text']: # TODO
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)

        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i])

        images = get_all_files_in_folder(temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a computer screen. Please briefly describe the shape and color of this icon in one sentence.'
            if caption_call_method == "local":
                for i in range(len(images)):
                    image_path = os.path.join(temp_file, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = generate_local(tokenizer, model, image_path, prompt)
                    icon_map[i+1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(temp_file, images[i])
                icon_map = generate_api(images, prompt)
            for i, j in zip(image_id, range(1, len(image_id)+1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] += ": " + icon_map[j]

    if args.location_info == 'center':
        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
    elif args.location_info == 'icon_center':
        for i in range(len(perception_infos)):
            if 'icon' in perception_infos[i]['text']:
                perception_infos[i]['coordinates'] = [
                    int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
                    int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)]

    return perception_infos, total_width, total_height

### Load caption model ###
device = "cuda"
torch.manual_seed(1234)
if caption_call_method == "local":
    if caption_model == "qwen-vl-chat":
        model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    elif caption_model == "qwen-vl-chat-int4":
        qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
        model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True,use_safetensors=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True, do_sample=False)
    else:
        print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
        exit(0)
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
elif caption_call_method == "api":
    pass
else:
    print("You must choose the caption model call function from \"local\" and \"api\"")
    exit(0)


### Load ocr and icon detection model ###
groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')


thought_history = []
summary_history = []
action_history = []
reflection_thought = ""
summary = ""
action = ""
completed_requirements = ""
memory = ""
insight = ""
temp_file = "temp"
screenshot = "screenshot"

if os.path.exists(temp_file):
    shutil.rmtree(temp_file)
os.mkdir(temp_file)
if not os.path.exists(screenshot):
    os.mkdir(screenshot)
error_flag = False


iter = 0
while True:
    iter += 1
    if iter == 1:
        screenshot_file = "./screenshot/screenshot.png"
        screenshot_som_file = "./screenshot/screenshot_som.png"
        perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)
        # print(perception_infos)

    prompt_action = get_action_prompt(instruction, perception_infos, width, height, thought_history, summary_history, action_history, summary, action, reflection_thought, add_info, error_flag, completed_requirements, memory, args.use_som, args.icon_caption, args.location_info)
    chat_action = init_action_chat()
    if args.use_som == 1:
        chat_action = add_response("user", prompt_action, chat_action, [screenshot_file, screenshot_som_file])
    else:
        chat_action = add_response("user", prompt_action, chat_action, [screenshot_file])

    output_action = inference_chat(chat_action, vl_model_version, API_url, token)
    thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
    summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
    action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
    chat_action = add_response("assistant", output_action, chat_action)
    status = "#" * 50 + " Decision " + "#" * 50
    print(status)
    print(output_action)
    print('#' * len(status))
    

    if "Double Tap" in action:
        coordinate = action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        tap(x, y, 2)

    elif "Triple Tap" in action:
        coordinate = action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        tap(x, y, 3)

    elif "Tap" in action:
        coordinate = action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        tap(x, y, 1)

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
        coordinate = action.split("(")[1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        if "[text]" not in action:
            text = action.split("[")[-1].split("]")[0]
        else:
            text = action.split(" \"")[-1].split("\"")[0]
        tap_type_enter(x, y, text)
        
    elif "Stop" in action:
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
    last_screenshot_file = "./screenshot/last_screenshot.png"
    if os.path.exists(last_screenshot_file):
        os.remove(last_screenshot_file)
    os.rename(screenshot_file, last_screenshot_file)
    if args.use_som == 1:
        last_screenshot_som_file = "./screenshot/last_screenshot_som.png"
        if os.path.exists(last_screenshot_som_file):
            os.remove(last_screenshot_som_file)
        os.rename(screenshot_som_file, last_screenshot_som_file)
    
    perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)
    
    if reflection_switch:
        prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info)
        chat_reflect = init_reflect_chat()
        chat_reflect = add_response("user", prompt_reflect, chat_reflect, [last_screenshot_file, screenshot_file])

        output_reflect = inference_chat(chat_reflect, vl_model_version, API_url, token)
        reflection_thought = output_reflect.split("### Thought ###")[-1].split("### Answer ###")[0].replace("\n", " ").strip()
        reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
        chat_reflect = add_response("assistant", output_reflect, chat_reflect)
        status = "#" * 50 + " Reflection " + "#" * 50
        print(status)
        print(output_reflect)
        print('#' * len(status))
    
        if 'A' in reflect:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)
            
            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(chat_planning, 'gpt-4o', API_url, token)
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
            
            error_flag = False
        
        elif 'B' in reflect:
            error_flag = True
            # presskey('esc')
            
        elif 'C' in reflect:
            error_flag = True
            # presskey('esc')
    
    else:
        thought_history.append(thought)
        summary_history.append(summary)
        action_history.append(action)
        
        prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
        chat_planning = init_memory_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning)
        output_planning = inference_chat(chat_planning, 'gpt-4o', API_url, token)
        chat_planning = add_response("assistant", output_planning, chat_planning)
        status = "#" * 50 + " Planning " + "#" * 50
        print(status)
        print(output_planning)
        print('#' * len(status))
        completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
         
    os.remove(last_screenshot_file)
    if args.use_som == 1:
        os.remove(last_screenshot_som_file)
