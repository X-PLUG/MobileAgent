import os
import clip
import uuid
import json
import shutil
import random
import argparse
from PIL import Image
from flask import Flask, request, jsonify
from modelscope import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from MobileAgent.api import inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.crop import crop_for_clip, clip_for_icon
from MobileAgent.chat import init_chat, add_response, print_status
from MobileAgent.prompt import thought_prompt, action_prompt, format_prompt

api_token = ''
device = 'cpu'

def set_params(param1, param2):
    global ckpt_filename, api_token
    ckpt_filename = param1
    api_token = param2

parser = argparse.ArgumentParser()
parser.add_argument("--grounding_ckpt", type=str)
parser.add_argument("--api", type=str)
args = parser.parse_args()
set_params(args.grounding_ckpt, args.api)

groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

print("*"*100)
print("Host start.")
print("*"*100)

if not os.path.exists("screenshot"):
    os.mkdir("screenshot")
if not os.path.exists("temp"):
    os.mkdir("temp")

now_history = {}

app = Flask(__name__)

@app.route('/a', methods=['POST'])
def run_host():

    if request.is_json:
        json_data = request.json
        session_id = json_data["session_id"]

        if session_id == '':
            session_id = uuid.uuid4().hex
            screenshot = json_data["screenshot"]
            instruction = json_data["query"]
            tutorial = json_data["tutorial"]
            if instruction[-1] == '.':
                instruction = instruction[:-1]
            
            ori_image = Image.open(screenshot)
            original_width, original_height = ori_image.size
            ori_image_path = "./screenshot/screenshot.png"
            ori_image.save(ori_image_path, "PNG")
            
            new_width = int(original_width * 0.5)
            new_height = int(original_height * 0.5)
            resize_image_path = "./screenshot/screenshot.jpg"
            resized_image = ori_image.resize((new_width, new_height))
            resized_image.convert("RGB").save(resize_image_path, "JPEG")
            
            image_ori = ori_image_path
            image = resize_image_path
            
            temp_file = "./temp"
            x, y = original_width, original_height
            iw, ih = new_width, new_height

            if iw > ih:
                x, y = y, x
                iw, ih = ih, iw
            
            operation_history = init_chat(instruction)
        
        else:
            screenshot = json_data["screenshot"]
            session_id = json_data["session_id"]
            tutorial = json_data["tutorial"]
            
            ori_image = Image.open(screenshot)
            original_width, original_height = ori_image.size
            ori_image_path = "./screenshot/screenshot.png"
            ori_image.save(ori_image_path, "PNG")
            
            new_width = int(original_width * 0.5)
            new_height = int(original_height * 0.5)
            resize_image_path = "./screenshot/screenshot.jpg"
            resized_image = ori_image.resize((new_width, new_height))
            resized_image.convert("RGB").save(resize_image_path, "JPEG")
            
            image_ori = ori_image_path
            image = resize_image_path
            
            temp_file = "./temp"
            x, y = original_width, original_height
            iw, ih = new_width, new_height

            if iw > ih:
                x, y = y, x
                iw, ih = ih, iw
            
            operation_history = now_history[session_id]["history"]
            instruction = now_history[session_id]["instruction"]
            if instruction[-1] == '.':
                instruction = instruction[:-1]

        error_flag = 0
        while True:
            if error_flag == 0:
                operation_history = add_response("user", f"The user's instruction is {instruction}. " + tutorial + thought_prompt, operation_history, image)
                thought, _ = inference_chat(operation_history, api_token)
                operation_history = add_response("assistant", thought, operation_history)
                operation_history = add_response("user", action_prompt, operation_history)
            
            while True:
                action, _ = inference_chat(operation_history, api_token)
                
                if "open" in action or "text" in action or "icon" in action or "type" in action:
                    try:
                        parameter = action.split("(")[1].split(")")[0]
                    except:
                        print_status(operation_history)
                        operation_history = add_response("assistant", action, operation_history)
                        operation_history = add_response("user", format_prompt, operation_history)
                    else:
                        break
                else:
                    break
            
            operation_history = add_response("assistant", action, operation_history)

            error_flag = 0

            if "stop" in action:
                data = {"operation": action, "action": "end", "parameter": "", "session_id": session_id}
            
            elif "open app" in action:
                parameter = action.split("(")[1].split(")")[0].replace("\"", "")
                in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
                
                if len(in_coordinate) >= 1:
                    tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"open app ({parameter})"
                    data = {"operation": action, "action": "tap", "parameter": json.dumps([tap_coordinate[0]*x, (tap_coordinate[1]-0.03)*y]), "session_id": session_id}
                
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
                    data = {"operation": action, "action": "tap", "parameter": json.dumps([tap_coordinate[0]*x, tap_coordinate[1]*y]), "session_id": session_id}
                
                else:
                    random_id = random.randint(0, len(out_coordinate)-1)
                    tap_coordinate = [(in_coordinate[random_id][0]+in_coordinate[random_id][2])/2, (in_coordinate[random_id][1]+in_coordinate[random_id][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"tap text ({parameter})"
                    data = {"operation": action, "action": "tap", "parameter": json.dumps([tap_coordinate[0]*x, tap_coordinate[1]*y]), "session_id": session_id}
            
            elif "tap icon" in action:
                parameter = action.split("(")[1].split(")")[0].replace("\"", "")
                parameter1, parameter2 = parameter.split(",")[0].replace("\"", "").strip(), parameter.split(",")[1].replace("\"", "").strip()
                in_coordinate, out_coordinate = det(image, "icon", groundingdino_model)
                
                if len(out_coordinate) == 1:
                    tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                    tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                    action = f"tap icon ({parameter})"
                    data = {"operation": action, "action": "tap", "parameter": json.dumps([tap_coordinate[0]*x, tap_coordinate[1]*y]), "session_id": session_id}

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
                    data = {"operation": action, "action": "tap", "parameter": json.dumps([tap_coordinate[0]*x, tap_coordinate[1]*y]), "session_id": session_id}
                        
            elif "scroll down" in action:
                action = "scroll down"
                data = {"operation": action, "action": "slide", "parameter": json.dumps([[int(x/2),int(y/2)], [int(x/2),int(y/4)]]), "session_id": session_id}
            
            elif "scroll up" in action:
                action = "scroll up"
                data = {"operation": action, "action": "slide", "parameter": json.dumps([[int(x/2),int(y/2)], [int(x/2),int(3*y/4)]]), "session_id": session_id}
            
            elif "type" in action:
                text = action.split("(")[1].split(")")[0].replace("\"", "")
                action = f"type ({text})"
                data = {"operation": action, "action": "type", "parameter": text, "session_id": session_id}
            
            elif "back" in action:
                action = "back"
                data = {"operation": action, "action": "back", "parameter": "", "session_id": session_id}

            elif "exit" in action:
                action = "exit"
                data = {"operation": action, "action": "back to desktop", "parameter": "", "session_id": session_id}

            else:
                error_prompt = format_prompt
                error_flag = 1

            if error_flag == 0:
                break
            else:
                operation_history = add_response("user", error_prompt, operation_history, image)

        now_history[session_id] = {"history": operation_history, "instruction": instruction}
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)
        print_status(operation_history)
        return jsonify(data)

if __name__ == '__main__':
    app.run()