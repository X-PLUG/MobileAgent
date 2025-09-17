'''
Format all data line into dashscope chatml
'''

import copy
import re
from typing import Any

media_split_pattern_llava = re.compile(r'(<image>|<video>)')
media_split_pattern_owl = re.compile(r'(<|image|>|<|video|>)')

def line_llava2qwen(line, source=None):
    from x.qwen.image import update_image_size_
    from x.io import ImageIO
    mio = ImageIO()
    def build_message(raw_conv):
        messages = []
        image_ptr = 0
        video_ptr = 0
        for _ in raw_conv:
            
            if _['from'] == 'human':
                content = []
                for media in media_split_pattern_llava.split(_['value']):
                    if media == '<image>':
                        img = mio(line['image'][image_ptr])
                        media_item = {
                            "image": line['image'][image_ptr],
                            "width": img.width,
                            "height": img.height,
                        }
                        media_item = update_image_size_(media_item)
                        content.append(media_item)
                        image_ptr += 1
                    elif media == '<video>':
                        content.append({"video": line['video'][video_ptr]})
                        video_ptr += 1
                    else:
                        if len(media):
                            content.append({"text": media})
                messages.append({"role": "user", "content": content})
            elif _['from'] == 'gpt' or _['from'] == 'assistant':
                messages.append({"role": "assistant", "content":[{"text": _['value'].replace('<image>','').replace('<video>','')}]})
            else:
                assert 1==2
        return messages
    if 'image' in line and not isinstance(line['image'], list):
        line['image'] = [line['image']]
    if 'video' in line and not isinstance(line['video'], list):
        line['video'] = [line['video']]
        
    messages = build_message(line['conversations'])
    
    line = {
        'messages':  messages,
        'task_name': line.get('task_name', "mm_sft"),
        "type": "chatml", 
        "source": source if source else line.get('dataset_name', 'Undefined'), 
        'dataset_name': source if source else line.get('dataset_name', 'Undefined'),
    }
    return line
    

def line_owl2qwen(line):
    def build_message(raw_conv):
        messages = []
        image_ptr = 0
        video_ptr = 0
        for _ in raw_conv:
            if _['role'] == 'user':
                content = []
                for media in media_split_pattern_owl.split(_['content']):
                    if media == '<|image|>':
                        content.append({"image": line['image'][image_ptr]})
                        image_ptr += 1
                    elif media == '<|video|>':
                        video_item = line['video'][video_ptr]
                        if isinstance(video_item, str):
                            content.append({"video": video_item})
                        else:
                            video_path = video_item['video']
                            new_video_dict = {
                                'video': video_path
                            }
                            if 'bound' in video_item:
                                new_video_dict['video_start'] = video_item['bound'][0]
                                new_video_dict['video_end'] = video_item['bound'][1]
                            content.append({"video": new_video_dict})
                        video_ptr += 1
                    else:
                        content.append({"text": media})
                messages.append({"role": "user", "content":[_['content'].replace('<|image|>','').replace('<|video|>','')]})
            elif _['role'] == 'assistant':
                messages.append({"role": "assistant", "content":[_['content'].replace('<|image|>','').replace('<|video|>','')]})
            else:
                assert 1==2
        return messages

    messages = build_message(line['messages'])

    line = {
        'messages':  messages,
        'task_name': line.get('task_name', "mm_sft"),
        "type": "chatml", 
        "source": line.get('dataset_name', 'Undefined'), 
        'dataset_name': line.get('dataset_name', 'Undefined'),
    }
    return line

def line_qwen2qwen(line):
    line['task_name'] = line.get('task_name', "mm_sft")
    line["dataset_name"] = line['source']
    return line

class DataFormater():
    def __init__(self, source_format) -> None:
        self.source_format = source_format

    def __call__(self, line, source=None) -> Any:
        if self.source_format == 'llava':
            return line_llava2qwen(line, source=source)
        elif self.source_format == 'owl':
            return line_owl2qwen(line)
        elif self.source_format == 'qwen':
            return line


def slim_messages(messages, num_image_limit = 5):
    keep_image_index = []
    image_ptr = 0
    messages = copy.deepcopy(messages)
    for msg in messages:
        for content in msg['content']:
            if 'image' in content or 'image_url' in content:
                keep_image_index.append(image_ptr)
                image_ptr += 1
    keep_image_index = keep_image_index[-num_image_limit:]

    image_ptr = 0
    for msg in messages:
        new_content = []
        for content in msg['content']:
            if 'image' in content or 'image_url' in content:
                if image_ptr not in keep_image_index:
                    pass
                else:
                    new_content.append(content)
                image_ptr += 1
            else:
                new_content.append(content)
        msg['content'] = new_content
    return messages