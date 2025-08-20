import base64
import copy
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor

import re

def message_translate(messages, to_format='dashscope'):
    if to_format == 'dashscope':
        return messages
    
    if to_format == 'openai':
        messages = copy.deepcopy(messages)
        for msg in messages:
            if isinstance(msg['content'], str):
                msg['content'] = [msg['content']]
            new_contents = []
            for content in msg['content']:
                if  isinstance(content, str):
                    new_contents.append({"type": "text", 'text': content})
                elif 'text' in content:
                    new_contents.append({"type": "text", 'text': content['text']})
                elif 'image' in content:
                    if content['image'].startswith('/'):
                        content['image'] = 'file://' + content['image']
                    new_contents.append({"type": "image_url", "image_url": {"url": content['image']}})
                else:
                    raise NotImplementedError
            msg['content'] = new_contents
        return messages
    if to_format == 'qwen':
        messages = copy.deepcopy(messages)
        for msg in messages:
            if isinstance(msg['content'], str):
                msg['content'] = [msg['content']]
            new_contents = []
            for content in msg['content']:
                if  isinstance(content, str):
                    new_contents.append({"type": "text", 'text': content})
                elif 'text' in content:
                    new_contents.append({"type": "text", 'text': content['text']})
                elif 'image' in content:
                    new_contents.append({"type": "image", "image": content['image']})
                else:
                    raise NotImplementedError
            msg['content'] = new_contents
        return messages


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def draw_point(image: Image.Image, point: list, color=None, radius=None):
    from copy import deepcopy
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  # Set 50% opacity
        except ValueError:
            color = (255, 0, 0, 128)  # Fallback to red with 50% opacity
    else:
        color = (255, 0, 0, 128)  # Fallback to red with 50% opacity
    # Create a semi-transparent overlay
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    if radius is None:
        radius = min(image.size) * 0.05
    x, y = point

    # Draw a semi-transparent circle on the overlay
    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color  # Red with 50% opacity
    )

    # Composite the overlay onto the original image
    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

def extract_bboxes_from_brackets(input_string):
    extracted_lists = [[int(num1), int(num2), int(num3), int(num4)] for num1, num2, num3, num4 in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', input_string)]
    return extracted_lists


def parse_tags(xml_content, tag_names):
    result = {}
    
    for tag_name in tag_names:
        # Define a regex pattern to match content for the current tag
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        
        # Use re.search to find the first match of pattern in xml_content
        match = re.search(pattern, xml_content, re.DOTALL)
        
        if match:
            # Extract and return the captured content within the tags
            tag_content = match.group(1).strip()
            result[tag_name] = tag_content
        else:
            result[tag_name] = None
    
    return result


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