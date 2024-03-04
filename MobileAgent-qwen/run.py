import os
import clip
import shutil
import random
import argparse
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from MobileAgent.api import inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import load_model, det
from MobileAgent.crop import crop_for_clip, clip_for_icon
from MobileAgent.chat import init_chat, add_response, print_status
from MobileAgent.prompt import thought_prompt, action_prompt, format_prompt
from MobileAgent.controller import get_size, get_screenshot, tap, type, slide, back, back_to_desktop


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument("--grounding_ckpt", type=str)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--adb_path", type=str)
    parser.add_argument("--api", type=str)
    args = parser.parse_args()
    return args


def run(args):
    config_file = args.config_file
    ckpt_filenmae = args.grounding_ckpt
    device = 'cpu'
    groundingdino_model = load_model(config_file, ckpt_filenmae, device=device).eval()
    ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    api_token = args.api
    
    with open('./tutorial.txt', 'r', encoding='utf-8') as file:
        tutorial = file.read()
    
    instruction = args.instruction
    operation_history = init_chat(instruction)
    
    if not os.path.exists("screenshot"):
        os.mkdir("screenshot")
    if not os.path.exists("temp"):
        os.mkdir("temp")
    
    while True:
        x, y = get_size(args.adb_path)
        get_screenshot(args.adb_path)
        image = "./screenshot/screenshot.jpg"
        image_ori = "./screenshot/screenshot.png"
        temp_file = "./temp"
        iw, ih = Image.open(image).size

        if iw > ih:
            x, y = y, x
            iw, ih = ih, iw
        
        error_flag = 0
        stop_flag = 0
        while True:
            if error_flag == 0:
                operation_history = add_response("user", f"The user's instruction is {instruction}. " + tutorial + "\n" + thought_prompt, operation_history, image)
                thought, _ = inference_chat(operation_history, api_token)
                operation_history = add_response("assistant", thought, operation_history)
                operation_history = add_response("user", action_prompt, operation_history)
            
            while True:
                action, _ = inference_chat(operation_history, api_token)
                
                if "open" in action or "text" in action or "icon" in action or "type" in action:
                    try:
                        parameter = action.split("(")[1].split(")")[0]
                    except:
                        print("Not formated action:", action)
                        operation_history = add_response("assistant", action, operation_history)
                        operation_history = add_response("user", format_prompt, operation_history)
                    else:
                        break
                else:
                    break
            
            operation_history = add_response("assistant", action, operation_history)

            error_flag = 0

            if "stop" in action:
                stop_flag = 1
                break
            
            elif "open app" in action:
                parameter = action.split("(")[1].split(")")[0].replace("\"", "")
                in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
                
                if len(in_coordinate) >= 1:
                    tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"open app ({parameter})"
                    tap(args.adb_path, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)
                
                else:
                    error_prompt = f"Failed to execute action open App ({parameter}). The App {parameter} is not detected in the screenshot. Please change another action or parameter."
                    error_flag = 1
            
            elif "tap text" in action:
                parameter = action.split("(")[1].split(")")[0].replace("\"", "")
                in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
                    
                if len(out_coordinate) == 0:
                    error_prompt = f"Failed to execute action click text ({parameter}). The text {parameter} is not detected in the screenshot. Please change another action or parameter."
                    error_flag = 1
                
                elif len(out_coordinate) == 1:
                    tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"tap text ({parameter})"
                    tap(args.adb_path, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)
                
                else:
                    random_id = random.randint(0, len(out_coordinate)-1)
                    tap_coordinate = [(in_coordinate[random_id][0]+in_coordinate[random_id][2])/2, (in_coordinate[random_id][1]+in_coordinate[random_id][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"tap text ({parameter})"
                    tap(args.adb_path, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)
            
            elif "tap icon" in action:
                parameter = action.split("(")[1].split(")")[0].replace("\"", "")
                parameter1, parameter2 = parameter.split(",")[0].replace("\"", "").strip(), parameter.split(",")[1].replace("\"", "").strip()
                in_coordinate, out_coordinate = det(image, "icon", groundingdino_model)
                
                if len(out_coordinate) == 1:
                    tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"tap icon ({parameter})"
                    tap(args.adb_path, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)

                else:
                    temp_file = "./temp"
                    hash = []
                    clip_filter = []
                    for i, (td, box) in enumerate(zip(in_coordinate, out_coordinate)):
                        if crop_for_clip(image, td, i+1, parameter2):
                            hash.append(td)
                            crop_image = f"{i+1}.jpg"
                            clip_filter.append(os.path.join(temp_file, crop_image))
                        
                    clip_filter = clip_for_icon(clip_model, clip_preprocess, clip_filter, parameter1)
                    final_box = hash[clip_filter]
                    tap_coordinate = [(final_box[0]+final_box[2])/2, (final_box[1]+final_box[3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"tap icon ({parameter})"
                    tap(args.adb_path, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)
                        
            elif "scroll down" in action:
                action = "page down"
                slide(args.adb_path, 'page down', x, y)
            
            elif "scroll up" in action:
                action = "page up"
                slide(args.adb_path, 'page up', x, y)
            
            elif "type" in action:
                text = action.split("(")[1].split(")")[0].replace("\"", "")
                action = f"type ({text})"
                type(args.adb_path, text)
            
            elif "back" in action:
                action = "back"
                back(args.adb_path)

            elif "exit" in action:
                action = "exit"
                back_to_desktop(args.adb_path)

            else:
                error_prompt = format_prompt
                error_flag = 1

            if error_flag == 0:
                break
            else:
                operation_history = add_response("user", error_prompt, operation_history, image)
        
        if stop_flag == 1:
            break
        print_status(operation_history)
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

if __name__ == "__main__":
    args = get_args()
    run(args)