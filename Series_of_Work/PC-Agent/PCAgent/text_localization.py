import re
import os
import logging
import os
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client

class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> ocr_api20210707Client:
        config = open_api_models.Config(
            access_key_id=os.environ.get('OCR_ACCESS_KEY_ID'),
            access_key_secret=os.environ.get('OCR_ACCESS_KEY_SECRET'),
        )
        config.endpoint = f'ocr-api.cn-hangzhou.aliyuncs.com'
        return ocr_api20210707Client(config)

    @staticmethod
    def main(image) -> None:
        client = Sample.create_client()
        recognize_all_text_request = ocr_api_20210707_models.RecognizeAllTextRequest(
            body=image,
            type='Advanced',
            output_coordinate='points',
            output_oricoord=True,
        )
        runtime = util_models.RuntimeOptions()
        output = client.recognize_all_text_with_options(recognize_all_text_request, runtime)
        # logger.info(f'ocr response：{output}', extra={'request_id': ""})
        output = output.body.data.sub_images[0].block_info.block_details
        return output

def image_to_binary(image_path):
    with open(image_path, 'rb') as file:
        binary_data = file.read()
    return binary_data

def remove_punctuation(text):
    # 使用正则表达式删除标点符号、下划线和空格
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # 删除标点符号
    cleaned_text = re.sub(r'_', '', cleaned_text)  # 删除下划线
    cleaned_text = re.sub(r'\s', '', cleaned_text)  # 删除空格
    return cleaned_text.replace("v", "").replace("o", "").replace("O", "").replace("T", "").replace("Q", "").replace("丶", "")


class OCRError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def ocr(image_path):
    text = []
    coordinate = []
    image = image_to_binary(image_path)
    print(image_path)
    try:
        outputs = Sample.main(image)
    except Exception as e:
        raise OCRError(e.message)
    for output in outputs:
        text.append(remove_punctuation(output.block_content))
        bbox = [int(output.block_points[0].x), int(output.block_points[0].y), int(output.block_points[2].x), int(output.block_points[2].y)]
        coordinate.append(bbox)
    
    return text, coordinate