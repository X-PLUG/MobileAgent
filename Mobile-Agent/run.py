import shutil
import os
import clip
import copy
import argparse
from PIL import Image
from modelscope import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from MobileAgent.prompt import opreation_prompt, choose_opreation_prompt
from MobileAgent.icon_localization import det
from MobileAgent.text_localization import ocr
from MobileAgent.api import inference_chat
from MobileAgent.crop import crop, crop_for_clip, clip_for_icon
from MobileAgent.chat import init_chat, add_response, add_multiimage_response
from MobileAgent.controller import get_size, get_screenshot, tap, type, slide, back, back_to_desktop, get_xml, choose_clickable
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--adb_path", type=str)
    parser.add_argument("--api", type=str)
    args = parser.parse_args()
    return args


def run(args):
    device = 'cpu'
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
    ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    observation_list, thought_list, action_list = [], [], []
    instruction = args.instruction
    struct_operation_history = init_chat(instruction)
    
    if not os.path.exists("screenshot"):
        os.mkdir("screenshot")
    if not os.path.exists("temp"):
        os.mkdir("temp")
    
    while True:
        x, y = get_size(args.adb_path)
        get_screenshot(args.adb_path)
        image = "./screenshot/screenshot.jpg"
        image_ori = "./screenshot/screenshot.png"
        xml_path = "./screenshot/screenshot.xml"
        temp_file = "./temp"
        iw, ih = Image.open(image).size
        get_xml(args.adb_path, xml_path) 

        if iw > ih:
            x, y = y, x
            iw, ih = ih, iw

        choose_flag = 0
        error_flag = 0
        
        operation_history = copy.deepcopy(struct_operation_history)
        operation_history = add_response("user", opreation_prompt, operation_history, image)
        
        while True:
            response = inference_chat(operation_history, args.api)
            
            try:
                observation = re.search(r"Observation:(.*?)\n", response).group(1).strip()
                thought = re.search(r"Thought:(.*?)\n", response).group(1).strip()
                action = re.search(r"Action:(.*)", response).group(1).strip()
            except:
                print("Response not formatted, retry.")
            else:
                break
        
        observation_list.append(observation)
        thought_list.append(thought)
        action_list.append(action)
        
        if "stop" in action:
            break
        
        elif "open App" in action:
            parameter = re.search(r"\((.*?)\)", action).group(1)
            in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
            in_coordinate, out_coordinate = choose_clickable(in_coordinate, out_coordinate, xml_path, iw, ih, x, y)
            if len(in_coordinate) == 0:
                error_prompt = f"No App named {parameter}."
                error_flag = 1
            else:
                tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                tap(args.adb_path, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)
        
        elif "click text" in action:
            
            choose_chat = init_chat(instruction)
            choose_chat = add_response("user", choose_opreation_prompt, choose_chat, image)
            choose_chat = add_response("assistant", action, choose_chat)
            
            parameter = re.search(r"\((.*?)\)", action).group(1)
            in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
            in_coordinate, out_coordinate = choose_clickable(in_coordinate, out_coordinate, xml_path, iw, ih, x, y)   
            if len(out_coordinate) == 0:
                error_prompt = f"Failed to execute action click text ({parameter}). The text \"{parameter}\" is not detected in the screenshot."
                error_flag = 1
            elif len(out_coordinate) > 4:
                error_prompt = f"Failed to execute action click text ({parameter}). There are too many text \"{parameter}\" in the screenshot."
                error_flag = 1
            
            elif len(out_coordinate) == 1:
                tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                tap(args.adb_path, tap_coordinate[0], tap_coordinate[1], x, y)
            
            else:
                hash = {}
                for i, (td, box) in enumerate(zip(in_coordinate, out_coordinate)):
                    crop(image, box, i+1, td)
                    hash[i+1] = td

                images = []
                temp_file = "./temp"
                for i in range(len(hash.keys())):
                    crop_image = f"{i+1}.jpg"
                    images.append(os.path.join(temp_file, crop_image))
                
                ocr_prompt = f"The {str(len(out_coordinate))} red boxes are numbered 1 through {str(len(out_coordinate))}. Which red box with \"{parameter}\" do you want to click on? Please output just one number from 1 to {str(len(out_coordinate))}, such as 1, 2......"
                choose_chat = add_multiimage_response("user", ocr_prompt, choose_chat, images)
                choose_response = inference_chat(choose_chat, args.api)
                
                final_box = hash[int(choose_response)]
                tap_coordinate = [(final_box[0]+final_box[2])/2, (final_box[1]+final_box[3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                tap(args.adb_path, tap_coordinate[0], tap_coordinate[1], x, y)
                
                choose_flag = 1
                choose_user = ocr_prompt
                choose_images = images
                choose_response = choose_response
        
        elif "click icon" in action:
            
            choose_chat = init_chat(instruction)
            choose_chat = add_response("user", choose_opreation_prompt, choose_chat, image)
            choose_chat = add_response("assistant", action, choose_chat)
            
            parameter = re.search(r"\((.*?)\)", action).group(1)
            parameter1, parameter2 = parameter.split(',')[0].strip(), parameter.split(',')[1].strip()
            in_coordinate, out_coordinate = det(image, "icon", groundingdino_model)
            in_coordinate, out_coordinate = choose_clickable(in_coordinate, out_coordinate, xml_path, iw, ih, x, y)
            if len(out_coordinate) == 1:
                tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                tap(args.adb_path, tap_coordinate[0], tap_coordinate[1], x, y)
                
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
                tap(args.adb_path, tap_coordinate[0], tap_coordinate[1], x, y)
                    
        elif "page" in action:
            slide(args.adb_path, action, x, y)
        
        elif "type" in action:
            text = re.search(r"\((.*?)\)", response).group(1)
            type(args.adb_path, text)
        
        elif "back" in action:
            back(args.adb_path)

        elif "exit" in action:
            back_to_desktop(args.adb_path)

        else:
            error_prompt = "Please respond strictly to the output format!"
        
        struct_operation_history = add_response("user", "This is the current screenshot. Please give me your action.", struct_operation_history, image)
        struct_operation_history = add_response("assistant", action, struct_operation_history)
        
        if error_flag == 0:
            if choose_flag == 1:
                struct_operation_history = add_multiimage_response("user", choose_user, struct_operation_history, choose_images)
                struct_operation_history = add_response("assistant", choose_response, struct_operation_history)
        else:
            struct_operation_history = add_response("user", error_prompt, struct_operation_history, image)
            struct_operation_history = add_response("assistant", "I will try again with another action or parameter.", struct_operation_history)
        
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

if __name__ == "__main__":
    args = get_args()
    run(args)
