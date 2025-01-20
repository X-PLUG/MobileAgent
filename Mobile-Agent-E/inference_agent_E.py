import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw
from time import sleep

from MobileAgentE.api import inference_chat
from MobileAgentE.text_localization import ocr
from MobileAgentE.icon_localization import det
from MobileAgentE.controller import get_screenshot, start_recording, end_recording
from MobileAgentE.agents import (
    InfoPool, Manager, Operator, Notetaker, ActionReflector, ExperienceRetrieverShortCut, ExperienceRetrieverTips,
    INIT_SHORTCUTS, ExperienceReflectorShortCut, ExperienceReflectorTips
)
from MobileAgentE.agents import add_response, add_response_two_image
from MobileAgentE.agents import ATOMIC_ACTION_SIGNITURES

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent
import json
from dataclasses import dataclass, field, asdict

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

####################################### Edit your Setting #########################################
# Your ADB path
ADB_PATH = os.environ.get("ADB_PATH", default="adb")

## Reasoning model configs
BACKBONE_TYPE = os.environ.get("BACKBONE_TYPE", default="OpenAI") # "OpenAI" or "Gemini" or "Claude"
assert BACKBONE_TYPE in ["OpenAI", "Gemini", "Claude"], "Unknown BACKBONE_TYPE"
print("### Using BACKBONE_TYPE:", BACKBONE_TYPE)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default=None)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" # OpenAI compatible
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", default=None)

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", default=None)

if BACKBONE_TYPE == "OpenAI":
    REASONING_MODEL = "gpt-4o-2024-11-20"
    KNOWLEDGE_REFLECTION_MODEL = "gpt-4o-2024-11-20"
elif BACKBONE_TYPE == "Gemini":
    REASONING_MODEL = "gemini-1.5-pro-latest"
    KNOWLEDGE_REFLECTION_MODEL = "gemini-1.5-pro-latest"
elif BACKBONE_TYPE == "Claude":
    REASONING_MODEL = "claude-3-5-sonnet-20241022"
    KNOWLEDGE_REFLECTION_MODEL = "claude-3-5-sonnet-20241022"

## you can specify a jsonl file path for tracking API usage
USAGE_TRACKING_JSONL = None # e.g., usage_tracking.jsonl

## Perceptor configs
# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
CAPTION_CALL_METHOD = "api"
# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
CAPTION_MODEL = "qwen-vl-plus"

QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", default=None)


## Initial Tips provided by user; You can add additional custom tips ###
INIT_TIPS = """0. Do not add any payment information. If you are asked to sign in, ignore it or sign in as a guest if possible. Close any pop-up windows when opening an app.
1. By default, no APPs are opened in the background.
2. Screenshots may show partial text in text boxes from your previous input; this does not count as an error.
3. When creating new Notes, you do not need to enter a title unless the user specifically requests it.
"""

## other
TEMP_DIR = "temp"
SCREENSHOT_DIR = "screenshot"
SLEEP_BETWEEN_STEPS = 5

###################################################################################################
### Perception related functions ###

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


def crop(image, box, i, temp_file=TEMP_DIR):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    save_path = os.path.join(temp_file, f"{i}.jpg")
    cropped_image.save(save_path)


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def process_image(image, query, caption_model=CAPTION_MODEL):
    dashscope.api_key = QWEN_API_KEY
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
        response = "This is an icon."
    
    return response


def generate_api(images, query, caption_model=CAPTION_MODEL):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query, caption_model=caption_model): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    
    return icon_map


def merge_text_blocks(
    text_list,
    coordinates_list,
    x_distance_threshold=45,
    y_distance_min=-20,
    y_distance_max=30,
    height_difference_threshold=20,
):
    merged_text_blocks = []
    merged_coordinates = []

    # Sort the text blocks based on y and x coordinates
    sorted_indices = sorted(
        range(len(coordinates_list)),
        key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]),
    )
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue

        anchor = i
        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i + 1, num_blocks):
            if merge[j]:
                continue

            # Calculate differences and thresholds
            x_diff_left = abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0])
            x_diff_right = abs(sorted_coordinates_list[anchor][2] - sorted_coordinates_list[j][2])

            y_diff = sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3]
            height_anchor = sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1]
            height_j = sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1]
            height_diff = abs(height_anchor - height_j)

            if (
                (x_diff_left + x_diff_right) / 2 < x_distance_threshold
                and y_distance_min <= y_diff < y_distance_max
                and height_diff < height_difference_threshold
            ):
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = "\n".join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])
    return merged_text_blocks, merged_coordinates

###################################################################################################

def load_perception_models(
    device="cuda",
    caption_call_method=CAPTION_CALL_METHOD,
    caption_model=CAPTION_MODEL,
    groundingdino_model="AI-ModelScope/GroundingDINO",
    groundingdino_revision="v1.0.0",
    ocr_detection_model="iic/cv_resnet18_ocr-detection-db-line-level_damo",
    ocr_recognition_model="iic/cv_convnextTiny_ocr-recognition-document_damo",
    ):

    ### Load caption model ###
    if caption_call_method == "local":
        if caption_model == "qwen-vl-chat":
            model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
            vlm_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
            vlm_model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif caption_model == "qwen-vl-chat-int4":
            qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
            vlm_model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True,use_safetensors=True).eval()
            vlm_model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True, do_sample=False)
        else:
            print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
            exit(0)
        vlm_tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
    elif caption_call_method == "api":
        vlm_model = None
        vlm_tokenizer = None
        pass
    else:
        print("You must choose the caption model call function from \"local\" and \"api\"")
        exit(0)


    ### Load ocr and icon detection model ###
    groundingdino_dir = snapshot_download(groundingdino_model, revision=groundingdino_revision)
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
    ocr_detection = pipeline(Tasks.ocr_detection, model=ocr_detection_model) # dbnet (no tensorflow)
    ocr_recognition = pipeline(Tasks.ocr_recognition, model=ocr_recognition_model)

    print("INFO: Loaded perception models:")
    print("\t- Caption model method:", caption_call_method, "| caption vlm model:", caption_model)
    print("\t- Grounding DINO model:", groundingdino_model)
    print("\t- OCR detection model:", ocr_detection_model)
    print("\t- OCR recognition model:", ocr_recognition_model)
    return ocr_detection, ocr_recognition, groundingdino_model, vlm_model, vlm_tokenizer


DEFAULT_PERCEPTION_ARGS = {
    "device": "cuda",
    "caption_call_method": CAPTION_CALL_METHOD,
    "caption_model": CAPTION_MODEL,
    "groundingdino_model": "AI-ModelScope/GroundingDINO",
    "groundingdino_revision": "v1.0.0",
    "ocr_detection_model": "iic/cv_resnet18_ocr-detection-db-line-level_damo",
    "ocr_recognition_model": "iic/cv_convnextTiny_ocr-recognition-document_damo",
}

class Perceptor:
    def __init__(self, adb_path, perception_args = DEFAULT_PERCEPTION_ARGS):
        self.ocr_detection, self.ocr_recognition, self.groundingdino_model, \
            self.vlm_model, self.vlm_tokenizer = load_perception_models(**perception_args)
        self.adb_path = adb_path

    def get_perception_infos(self, screenshot_file, temp_file=TEMP_DIR):
        get_screenshot(self.adb_path)
        
        width, height = Image.open(screenshot_file).size
        
        text, coordinates = ocr(screenshot_file, self.ocr_detection, self.ocr_recognition)
        text, coordinates = merge_text_blocks(text, coordinates)
        
        center_list = [[(coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2] for coordinate in coordinates]
        draw_coordinates_on_image(screenshot_file, center_list)
        
        perception_infos = []
        for i in range(len(coordinates)):
            perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
            perception_infos.append(perception_info)
            
        coordinates = det(screenshot_file, "icon", self.groundingdino_model)
        
        for i in range(len(coordinates)):
            perception_info = {"text": "icon", "coordinates": coordinates[i]}
            perception_infos.append(perception_info)
            
        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            if perception_infos[i]['text'] == 'icon':
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)

        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i], temp_file=temp_file)

        images = get_all_files_in_folder(temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
            if CAPTION_CALL_METHOD == "local":
                for i in range(len(images)):
                    image_path = os.path.join(temp_file, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = generate_local(self.vlm_tokenizer, self.vlm_model, image_path, prompt)
                    icon_map[i+1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(temp_file, images[i])
                icon_map = generate_api(images, prompt, caption_model=CAPTION_MODEL)
            for i, j in zip(image_id, range(1, len(image_id)+1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] = "icon: " + icon_map[j]

        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
            
        return perception_infos, width, height

###################################################################################################

def finish(
        info_pool: InfoPool,
        persistent_tips_path=None,
        persistent_shortcuts_path=None
    ):
    
    print("Plan:", info_pool.plan)
    print("Progress Logs:")
    for i, p in enumerate(info_pool.progress_status_history):
        print(f"Step {i}:", p, "\n")
    print("Important Notes:", info_pool.important_notes)
    print("Finish Thought:", info_pool.finish_thought)
    if persistent_tips_path:
        print("Update persistent tips:", persistent_tips_path)
        with open(persistent_tips_path, "w") as f:
            f.write(info_pool.tips)
    if persistent_shortcuts_path:
        print("Update persistent shortcuts:", persistent_shortcuts_path)
        with open(persistent_shortcuts_path, "w") as f:
            json.dump(info_pool.shortcuts, f, indent=4)
    # exit(0)

import copy
import random
def get_reasoning_model_api_response(chat, model_type=BACKBONE_TYPE, model=None, temperature=0.0):
    
    # chat messages in openai format
    model = REASONING_MODEL if model is None else model
    if model_type == "OpenAI":
        return inference_chat(chat, model, OPENAI_API_URL, OPENAI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL, temperature=temperature)
    elif model_type == "Gemini":
        return inference_chat(chat, model, GEMINI_API_URL, GEMINI_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL, temperature=temperature)
    elif model_type == "Claude":
        return inference_chat(chat, model, CLAUDE_API_URL, CLAUDE_API_KEY, usage_tracking_jsonl=USAGE_TRACKING_JSONL, temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    


def run_single_task(
    instruction,
    future_tasks=[],
    run_name="test",
    log_root=f"logs/{REASONING_MODEL}/mobile_agent_E",
    task_id=None,
    tips_path=None,
    shortcuts_path=None,
    persistent_tips_path=None, # cross tasks
    persistent_shortcuts_path=None, # cross tasks
    perceptor: Perceptor = None,
    perception_args=DEFAULT_PERCEPTION_ARGS,
    max_itr=40,
    max_consecutive_failures=3,
    max_repetitive_actions=3,
    overwrite_log_dir=False,
    err_to_manager_thresh = 2, # 2 consecutive errors up-report to the manager
    enable_experience_retriever = False,
    temperature=0.0,
    screenrecord=False,
):

    ### set up log dir ###
    if task_id is None:
        task_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_root}/{run_name}/{task_id}"
    if os.path.exists(log_dir) and not overwrite_log_dir:
        print("The log dir already exists. And overwrite_log_dir is set to False. Skipping...")
        return
    os.makedirs(f"{log_dir}/screenshots", exist_ok=True)
    log_json_path = f"{log_dir}/steps.json"

    if screenrecord:
        # record one mp4 for each iteration
        screenrecord_dir = f"{log_dir}/screenrecords"
        os.makedirs(screenrecord_dir, exist_ok=True)
    
    # local experience save paths
    local_shortcuts_save_path = f"{log_dir}/shortcuts.json" # single-task setting
    local_tips_save_path = f"{log_dir}/tips.txt" # single-task setting

    ### Init Information Pool ###
    if shortcuts_path is not None and persistent_shortcuts_path is not None and shortcuts_path != persistent_shortcuts_path:
        raise ValueError("You cannot specify different shortcuts_path and persistent_shortcuts_path.")
    if tips_path is not None and persistent_tips_path is not None and tips_path != persistent_tips_path:
        raise ValueError("You cannot specify different tips_path and persistent_tips_path.")
    
    if shortcuts_path:
        initial_shortcuts = json.load(open(shortcuts_path, "r")) # load agent collected shortcuts
    elif persistent_shortcuts_path:
        initial_shortcuts = json.load(open(persistent_shortcuts_path, "r"))
    else:
        initial_shortcuts = copy.deepcopy(INIT_SHORTCUTS)
    print("INFO: Initial shortcuts:", initial_shortcuts)
    
    
    if tips_path:
        tips = open(tips_path, "r").read() # load agent updated tips
    elif persistent_tips_path:
        tips = open(persistent_tips_path, "r").read()
    else:
        tips = copy.deepcopy(INIT_TIPS) # user provided initial tips
    print("INFO: Initial tips:", tips)

    steps = []
    task_start_time = time.time()

    ## additional retrieval step before starting the task for selecting relevant tips and shortcuts ##
    if enable_experience_retriever:
        print("### Doing retrieval on provided Tips and Shortcuts ... ###")
        experience_retrieval_log = {
            "step": -1,
            "operation": "experience_retrieval",
            "original_tips": tips,
            "original_shortcuts": initial_shortcuts,
        }
        experience_retriever_start_time = time.time()

        # select shortcuts
        if len(initial_shortcuts) > 1:
            experience_retriever_shortcut = ExperienceRetrieverShortCut()
            experience_retriever_shortcut_prompt = experience_retriever_shortcut.get_prompt(instruction, initial_shortcuts)
            chat_experience_retrieval_shortcut = experience_retriever_shortcut.init_chat()
            chat_experience_retrieval_shortcut = add_response("user", experience_retriever_shortcut_prompt, chat_experience_retrieval_shortcut, image=None)
            output_experience_retrieval_shortcut = get_reasoning_model_api_response(chat_experience_retrieval_shortcut, model=KNOWLEDGE_REFLECTION_MODEL, temperature=temperature)
            parsed_experience_retrieval_shortcut = experience_retriever_shortcut.parse_response(output_experience_retrieval_shortcut)
            selected_shortcut_names = parsed_experience_retrieval_shortcut['selected_shortcut_names']
            if selected_shortcut_names is None or selected_shortcut_names == []:
                initial_shortcuts = copy.deepcopy(INIT_SHORTCUTS)
            else:
                selected_shortcuts = {}
                for key in selected_shortcut_names:
                    if key in initial_shortcuts:
                        selected_shortcuts[key] = initial_shortcuts[key]
                    else:
                        print(f"WARNING: {key} is not in initial_shortcuts.")
                if selected_shortcuts != {}:
                    initial_shortcuts = selected_shortcuts
        sleep(1)
        # select tips
        experience_retriever_tips = ExperienceRetrieverTips()
        experience_retrieval_tips_prompt = experience_retriever_tips.get_prompt(instruction, tips)
        chat_experience_retrieval_tips = experience_retriever_tips.init_chat()
        chat_experience_retrieval_tips = add_response("user", experience_retrieval_tips_prompt, chat_experience_retrieval_tips, image=None)
        output_experience_retrieval_tips = get_reasoning_model_api_response(chat_experience_retrieval_tips, model=KNOWLEDGE_REFLECTION_MODEL, temperature=temperature)
        parsed_experience_retrieval_tips = experience_retriever_tips.parse_response(output_experience_retrieval_tips)

        tips = parsed_experience_retrieval_tips['selected_tips']
        if tips.strip() == "None":
            tips = copy.deepcopy(INIT_TIPS)
        
        experience_retriever_end_time = time.time()
        experience_retrieval_log["experience_retrieval_shortcut_prompt"] = experience_retriever_shortcut_prompt
        experience_retrieval_log["experience_retrieval_tips_prompt"] = experience_retrieval_tips_prompt
        experience_retrieval_log["experience_retrieval_shortcut_response"] = output_experience_retrieval_shortcut
        experience_retrieval_log["experience_retrieval_tips_response"] = output_experience_retrieval_tips
        experience_retrieval_log["selected_tips"] = tips
        experience_retrieval_log["selected_shortcuts"] = initial_shortcuts
        experience_retrieval_log["duration"] = experience_retriever_end_time - experience_retriever_start_time
        
        print("selected_tips:", tips)
        print("selected_shortcuts:", initial_shortcuts)

        steps.append(experience_retrieval_log)
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)


    # init info pool
    info_pool = InfoPool(
        instruction = instruction,
        shortcuts = initial_shortcuts,
        tips = tips,
        future_tasks = future_tasks,
        err_to_manager_thresh=err_to_manager_thresh
    )

    ### temp dir ###
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    else:
        shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)
    if not os.path.exists(SCREENSHOT_DIR):
        os.mkdir(SCREENSHOT_DIR)

    ### Init Agents ###
    if perceptor is None:
        # if perceptor is not initialized, create the perceptor
        perceptor = Perceptor(ADB_PATH, perception_args=perception_args)
    manager = Manager()
    operator = Operator(adb_path=ADB_PATH)
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    exp_reflector_shortcuts = ExperienceReflectorShortCut()
    exp_reflector_tips = ExperienceReflectorTips()

    # save initial tips and shortcuts
    with open(local_tips_save_path, "w") as f:
        f.write(tips)
    with open(local_shortcuts_save_path, "w") as f:
        json.dump(initial_shortcuts, f, indent=4)

    ### Start the agent ###
    steps.append({
        "step": 0,
        "operation": "init",
        "instruction": instruction,
        "task_id": task_id,
        "run_name": run_name,
        "max_itr": max_itr,
        "max_consecutive_failures": max_consecutive_failures,
        "max_repetitive_actions": max_repetitive_actions,
        "future_tasks": future_tasks,
        "log_root": log_root,
        "tips_path": tips_path,
        "shortcuts_path": shortcuts_path,
        "persistent_tips_path": persistent_tips_path,
        "persistent_shortcuts_path": persistent_shortcuts_path,
        "perception_args": perception_args,
        "init_info_pool": asdict(info_pool)
    })
    with open(log_json_path, "w") as f:
        json.dump(steps, f, indent=4)

    iter = 0
    while True:
        iter += 1

        ## max iteration stop ##
        if max_itr is not None and iter >= max_itr:
            print("Max iteration reached. Stopping...")
            task_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "max_iteration",
                "max_itr": max_itr,
                "final_info_pool": asdict(info_pool),
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            return
        
        ## consecutive failures stop ##
        if len(info_pool.action_outcomes) >= max_consecutive_failures:
            last_k_aciton_outcomes = info_pool.action_outcomes[-max_consecutive_failures:]
            err_flags = [1 if outcome in ["B", "C"] else 0 for outcome in last_k_aciton_outcomes]
            if sum(err_flags) == max_consecutive_failures:
                print("Consecutive failures reaches the limit. Stopping...")
                task_end_time = time.time()
                steps.append({
                    "step": iter,
                    "operation": "finish",
                    "finish_flag": "max_consecutive_failures",
                    "max_consecutive_failures": max_consecutive_failures,
                    "final_info_pool": asdict(info_pool),
                    "task_duration": task_end_time - task_start_time,
                })
                with open(log_json_path, "w") as f:
                    json.dump(steps, f, indent=4)
                return
        
        ## max repetitive actions stop ##
        if len(info_pool.action_history) >= max_repetitive_actions:
            last_k_actions = info_pool.action_history[-max_repetitive_actions:]
            last_k_actions_set = set()
            try:
                for act_obj in last_k_actions:
                    if "name" in act_obj:
                        hash_key = act_obj['name']
                    else:
                        hash_key = json.dumps(act_obj)
                    if "arguments" in act_obj:
                        if act_obj['arguments'] is not None:
                            for arg, value in act_obj['arguments'].items():
                                hash_key += f"-{arg}-{value}"
                        else:
                            hash_key += "-None"
                    print("hashable action key:", hash_key)
                    last_k_actions_set.add(hash_key)
            except:
                last_k_actions_set = set() # not stopping if there is any error
                pass
            if len(last_k_actions_set) == 1:
                repeated_action_key = last_k_actions_set.pop()
                if "Swipe" not in repeated_action_key and "Back" not in repeated_action_key:
                    print("Repetitive actions reaches the limit. Stopping...")
                    task_end_time = time.time()
                    steps.append({
                        "step": iter,
                        "operation": "finish",
                        "finish_flag": "max_repetitive_actions",
                        "max_repetitive_actions": max_repetitive_actions,
                        "final_info_pool": asdict(info_pool),
                        "task_duration": task_end_time - task_start_time,
                    })
                    with open(log_json_path, "w") as f:
                        json.dump(steps, f, indent=4)
                    return


        # start recording for step iter #
        if screenrecord:
            cur_output_recording_path = f"{screenrecord_dir}/step_{iter}.mp4"
            recording_process = start_recording(ADB_PATH)

        if iter == 1: # first perception
            screenshot_file = "./screenshot/screenshot.jpg"
            print("\n### Perceptor ... ###\n")
            perception_start_time = time.time()
            perception_infos, width, height = perceptor.get_perception_infos(screenshot_file, temp_file=TEMP_DIR)
            shutil.rmtree(TEMP_DIR)
            os.mkdir(TEMP_DIR)
            
            keyboard = False
            keyboard_height_limit = 0.9 * height
            for perception_info in perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    keyboard = True
                    break
            
            info_pool.width = width
            info_pool.height = height

            ## log ##
            save_screen_shot_path = f"{log_dir}/screenshots/{iter}.jpg"
            Image.open(screenshot_file).save(save_screen_shot_path)

            perception_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "perception",
                "screenshot": save_screen_shot_path,
                "perception_infos": perception_infos,
                "duration": perception_end_time - perception_start_time,
            })
            print("Perception Infos:", perception_infos)
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
        
        ### get perception infos ###
        info_pool.perception_infos_pre = copy.deepcopy(perception_infos)
        info_pool.keyboard_pre = keyboard

        ### Manager: High-level Planning ###
        print("\n### Manager ... ###\n")
        ## check if stuck with errors for a long time ##
        # if so need to think about the high-level plan again
        info_pool.error_flag_plan = False
        if len(info_pool.action_outcomes) >= err_to_manager_thresh:
            # check if the last err_to_manager_thresh actions are all errors
            latest_outcomes = info_pool.action_outcomes[-err_to_manager_thresh:]
            count = 0
            for outcome in latest_outcomes:
                if outcome in ["B", "C"]:
                    count += 1
            if count == err_to_manager_thresh:
                info_pool.error_flag_plan = True
        ## 
        info_pool.prev_subgoal = info_pool.current_subgoal

        planning_start_time = time.time()
        prompt_planning = manager.get_prompt(info_pool)
        chat_planning = manager.init_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning, image=screenshot_file)
        output_planning = get_reasoning_model_api_response(chat_planning, temperature=temperature)
        parsed_result_planning = manager.parse_response(output_planning)
        
        info_pool.plan = parsed_result_planning['plan']
        info_pool.current_subgoal = parsed_result_planning['current_subgoal']

        ## log ##
        planning_end_time = time.time()
        steps.append({
            "step": iter,
            "operation": "planning",
            "prompt_planning": prompt_planning,
            "error_flag_plan": info_pool.error_flag_plan,
            "raw_response": output_planning,
            "thought": parsed_result_planning['thought'],
            "plan": parsed_result_planning['plan'],
            "current_subgoal": parsed_result_planning['current_subgoal'],
            "duration": planning_end_time - planning_start_time,
        })
        print("Thought:", parsed_result_planning['thought'])
        print("Overall Plan:", info_pool.plan)
        print("Current Subgoal:", info_pool.current_subgoal)
        
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        
        ###

        ### Experience Reflection: Update Tips & Shortcuts for Self-Evolving ###
        if len(info_pool.action_outcomes) > 0:
            # at the end of each task, update the tips and shortcuts
            if "Finished" in info_pool.current_subgoal.strip():
                print("\n### Experience Reflector ... ###\n")
                experience_reflection_start_time = time.time()
                # shortcuts
                prompt_knowledge_shortcuts = exp_reflector_shortcuts.get_prompt(info_pool)
                chat_knowledge_shortcuts = exp_reflector_shortcuts.init_chat()
                chat_knowledge_shortcuts = add_response("user", prompt_knowledge_shortcuts, chat_knowledge_shortcuts, image=None)
                output_knowledge_shortcuts = get_reasoning_model_api_response(chat_knowledge_shortcuts, model=KNOWLEDGE_REFLECTION_MODEL, temperature=temperature)
                parsed_result_knowledge_shortcuts = exp_reflector_shortcuts.parse_response(output_knowledge_shortcuts)
                new_shortcut_str = parsed_result_knowledge_shortcuts['new_shortcut']
                if new_shortcut_str != "None" and new_shortcut_str is not None:
                    exp_reflector_shortcuts.add_new_shortcut(new_shortcut_str, info_pool)
                print("New Shortcut:", new_shortcut_str)
                # tips
                prompt_knowledge_tips = exp_reflector_tips.get_prompt(info_pool)
                chat_knowledge_tips = exp_reflector_tips.init_chat()
                chat_knowledge_tips = add_response("user", prompt_knowledge_tips, chat_knowledge_tips, image=None)
                output_knowledge_tips = get_reasoning_model_api_response(chat_knowledge_tips, model=KNOWLEDGE_REFLECTION_MODEL, temperature=temperature)
                parsed_result_knowledge_tips = exp_reflector_tips.parse_response(output_knowledge_tips)
                updated_tips = parsed_result_knowledge_tips['updated_tips']
                info_pool.tips = updated_tips
                print("Updated Tips:", updated_tips)

                prompt_knowledge = [prompt_knowledge_shortcuts, prompt_knowledge_tips]
                output_knowledge = [output_knowledge_shortcuts, output_knowledge_tips]
                
                experience_reflection_end_time = time.time()
                steps.append({
                    "step": iter,
                    "operation": "experience_reflection",
                    "prompt_knowledge": prompt_knowledge,
                    "raw_response": output_knowledge,
                    "new_shortcut": new_shortcut_str,
                    "updated_tips": updated_tips,
                    "duration": experience_reflection_end_time - experience_reflection_start_time,
                })
                with open(log_json_path, "w") as f:
                    json.dump(steps, f, indent=4)
                ## save the updated tips and shortcuts ##
                with open(local_tips_save_path, "w") as f:
                    f.write(info_pool.tips)
                with open(local_shortcuts_save_path, "w") as f:
                    json.dump(info_pool.shortcuts, f, indent=4)
                
                    
        ### Stopping by planner ###
        if "Finished" in info_pool.current_subgoal.strip():
            info_pool.finish_thought = parsed_result_planning['thought']
            task_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "success",
                "final_info_pool": asdict(info_pool),
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            finish(
                info_pool,
                persistent_tips_path = persistent_tips_path,
                persistent_shortcuts_path = persistent_shortcuts_path
            )
            if screenrecord:
                end_recording(ADB_PATH, output_recording_path=cur_output_recording_path)
            return

        ### Executor: Action Decision ###
        print("\n### Operator ... ###\n")
        action_decision_start_time = time.time()
        prompt_action = operator.get_prompt(info_pool)
        chat_action = operator.init_chat()
        chat_action = add_response("user", prompt_action, chat_action, image=screenshot_file)
        output_action = get_reasoning_model_api_response(chat_action, temperature=temperature)
        parsed_result_action = operator.parse_response(output_action)
        action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']
        action_decision_end_time = time.time()

        info_pool.last_action_thought = action_thought
        ## execute the action ##
        action_execution_start_time = time.time()
        action_object, num_atomic_actions_executed, shortcut_error_message = operator.execute(action_object_str, info_pool, 
                        screenshot_file=screenshot_file, 
                        ocr_detection=perceptor.ocr_detection,
                        ocr_recognition=perceptor.ocr_recognition,
                        thought = action_thought,
                        screenshot_log_dir = os.path.join(log_dir, "screenshots"),
                        iter = str(iter)
                        )
        action_execution_end_time = time.time()
        if action_object is None:
            task_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "abnormal",
                "final_info_pool": asdict(info_pool),
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            finish(
                info_pool, 
                persistent_tips_path = persistent_tips_path,
                persistent_shortcuts_path = persistent_shortcuts_path
            ) # 
            print("WARNING!!: Abnormal finishing:", action_object_str)
            if screenrecord:
                end_recording(ADB_PATH, output_recording_path=cur_output_recording_path)
            return

        info_pool.last_action = action_object
        info_pool.last_summary = action_description
        
        
        ## log ##
        steps.append({
            "step": iter,
            "operation": "action",
            "prompt_action": prompt_action,
            "raw_response": output_action,
            "action_object": action_object,
            "action_object_str": action_object_str,
            "action_thought": action_thought,
            "action_description": action_description,
            "duration": action_decision_end_time - action_decision_start_time,
            "execution_duration": action_execution_end_time - action_execution_start_time,
        })
        print("Action Thought:", action_thought)
        print("Action Description:", action_description)
        print("Action:", action_object)
        
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        
        
        print("\n### Perceptor ... ###\n")
        ## perception on the next step ##
        perception_start_time = time.time()
        # last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = "./screenshot/last_screenshot.jpg"
        # last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)
        
        perception_infos, width, height = perceptor.get_perception_infos(screenshot_file, temp_file=TEMP_DIR)
        shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)
        
        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break
        
        info_pool.perception_infos_post = perception_infos
        info_pool.keyboard_post = keyboard
        assert width == info_pool.width and height == info_pool.height # assert the screen size not changed

        ## log ##
        Image.open(screenshot_file).save(f"{log_dir}/screenshots/{iter+1}.jpg")
        perception_end_time = time.time()
        steps.append({
            "step": iter+1,
            "operation": "perception",
            "screenshot": f"{log_dir}/screenshots/{iter+1}.jpg",
            "perception_infos": perception_infos,
            "duration": perception_end_time - perception_start_time
        })
        print("Perception Infos:", perception_infos)
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        
        ##

        print("\n### Action Reflector ... ###\n")
        ### Action Reflection: Check whether the action works as expected ###
        action_reflection_start_time = time.time()
        prompt_action_reflect = action_reflector.get_prompt(info_pool)
        chat_action_reflect = action_reflector.init_chat()
        chat_action_reflect = add_response_two_image("user", prompt_action_reflect, chat_action_reflect, [last_screenshot_file, screenshot_file])
        output_action_reflect = get_reasoning_model_api_response(chat_action_reflect, temperature=temperature)
        parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
        outcome, error_description, progress_status = (
            parsed_result_action_reflect['outcome'], 
            parsed_result_action_reflect['error_description'], 
            parsed_result_action_reflect['progress_status']
        )
        info_pool.progress_status_history.append(progress_status)
        action_reflection_end_time = time.time()

        if "A" in outcome: # Successful. The result of the last action meets the expectation.
            action_outcome = "A"
        elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
            action_outcome = "B"

            # NOTE: removing the automatic backing; always stopping at the failed state and then there will be a new perception step
            # no automatic backing
            # check how many backs to take
            action_name = action_object['name']
            if action_name in ATOMIC_ACTION_SIGNITURES:
                # back(ADB_PATH) # back one step for atomic actions
                pass
            elif action_name in info_pool.shortcuts:
                # shortcut_object = info_pool.shortcuts[action_name]
                # num_of_atomic_actions = len(shortcut_object['atomic_action_sequence'])
                if shortcut_error_message is not None:
                    error_description += f"; Error occured while executing the shortcut: {shortcut_error_message}"
                # for _ in range(num_atomic_actions_executed):
                #     back(ADB_PATH)   
            else:
                raise ValueError("Invalid action name:", action_name)

        elif "C" in outcome: # Failed. The last action produces no changes.
            action_outcome = "C"
        else:
            raise ValueError("Invalid outcome:", outcome)
        
        # update action history
        info_pool.action_history.append(action_object)
        info_pool.summary_history.append(action_description)
        info_pool.action_outcomes.append(action_outcome)
        info_pool.error_descriptions.append(error_description)
        info_pool.progress_status = progress_status

        ## log ##
        steps.append({
            "step": iter,
            "operation": "action_reflection",
            "prompt_action_reflect": prompt_action_reflect,
            "raw_response": output_action_reflect,
            "outcome": outcome,
            "error_description": error_description,
            "progress_status": progress_status,
            "duration": action_reflection_end_time - action_reflection_start_time,
        })
        print("Outcome:", action_outcome)
        print("Progress Status:", progress_status)
        print("Error Description:", error_description)
        
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)
        
        ##
        
        ### NoteTaker: Record Important Content ###
        if action_outcome == "A":
            print("\n### NoteKeeper ... ###\n")
            # if previous action is successful, record the important content
            notetaking_start_time = time.time()
            prompt_note = notetaker.get_prompt(info_pool)
            chat_note = notetaker.init_chat()
            chat_note = add_response("user", prompt_note, chat_note, image=screenshot_file) # new screenshot
            output_note = get_reasoning_model_api_response(chat_note, temperature=temperature)
            parsed_result_note = notetaker.parse_response(output_note)
            important_notes = parsed_result_note['important_notes']
            info_pool.important_notes = important_notes
            os.remove(last_screenshot_file)
            
            notetaking_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "notetaking",
                "prompt_note": prompt_note,
                "raw_response": output_note,
                "important_notes": important_notes,
                "duration": notetaking_end_time - notetaking_start_time,
            })
            print("Important Notes:", important_notes)
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

        elif action_outcome in ["B", "C"]:
            os.remove(last_screenshot_file)

        if screenrecord:
            end_recording(ADB_PATH, output_recording_path=cur_output_recording_path)
        print("\n=========================================================")
        print(f"sleeping for {SLEEP_BETWEEN_STEPS} before next iteration ...\n\n")
        sleep(SLEEP_BETWEEN_STEPS)