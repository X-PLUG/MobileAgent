import requests
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from time import sleep
import dashscope
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
from PIL import Image
import torch
import json
import argparse
from statistic import get_result


def read_json(json_file):
    lines = []
    if not os.path.exists(json_file):
        return lines
    with open(json_file, 'r') as file:
        for line in file:
            lines.append(json.loads(line.strip()))
    return lines


def save_list_to_jsonl(data_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_str = json.dumps(item)
            f.write(json_str + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--data_dir', type=str, default='dataset/')
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--test_file', type=str, default='test.jsonl')
    parser.add_argument('--save_dir', type=str, default='output/')
    args = parser.parse_args()
    return args



def critic_inference(data, data_dir):
    def call_model(user_prompt, image):
        system_prompt = "You are a helpful critic agent for GUI operation."
        image = image if isinstance(image, list) else [image]
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],

            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        for img in image:
            messages[1]['content'].append(
                {
                    "type": "image",
                    "image": os.path.normpath(img),
                }, )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=2048, top_k=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    user_prompt = data['problem']
    image = [os.path.join(data_dir, data['images'][0])]

    i = 0
    while True:
        try:
            response = call_model(user_prompt, image)
            break
        except Exception as e:
            i += 1
            print(f"Attempt {i} failed: {e}")
            sleep(10)

            if i > 1:
                raise
    return response


if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    test_file = args.test_file
    save_dir = args.save_dir
    data_dir  = args.data_dir

    print("Test Args: \n", args)

    directory, filename = os.path.split(test_file)
    name, ext = os.path.splitext(filename)
    save_filename = f"{name}_output{ext}"
    save_file = os.path.join(save_dir, save_filename)


    data_list = read_json(test_file)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_dir, max_pixels=602112)
    model = model.to("cuda")
    output_list = []

    for index in tqdm(range(0, len(data_list))):
        data = data_list[index]
        ground_truth = data['solution']
        response = critic_inference(data, data_dir)
        print(response)
        try:
            answer = response.split("<score>")[-1].split("</score>")[0].strip()
            answer = answer.replace(' ', '').replace('\n', '')
        except Exception as e:
            print(e)
            answer = None
            accuracy_value = 0

        if answer == ground_truth:
            accuracy_value = 1
        else:
            accuracy_value = 0

        data.update({'critic_response': response})
        data.update({'critic_result': answer})
        data.update({'accuracy_value': accuracy_value})
        output_list.append(data)

    metric, output_list = get_result(data_list)
    save_list_to_jsonl(output_list, save_file)

