import base64
import io
import os
import json
from android_world.agents import new_json_action as json_action

from android_world.agents.function_call_mobile_answer import AndroidWorldMobileUse # TODO
print('------function_call_mobile_with_answer-------')

from android_world.agents.coordinate_resize import convert_point_format

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

nousprompt = NousFnCallPrompt()

import copy
import copy
import io
import base64

def pil_to_base64_url(image, format="JPEG"):
    """
    将 PIL 图像转换为 Base64 URL。

    :param image: PIL 图像对象
    :param format: 图像格式（如 "JPEG", "PNG"）
    :return: Base64 URL 字符串
    """
    # 将图像保存到字节流
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    
    # 将字节流编码为 Base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 构建 Base64 URL
    mime_type = f"image/{format.lower()}"
    base64_url = f"data:{mime_type};base64,{image_base64}"
    
    return base64_url

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
        
def generate_user_prompt_single_image(instruction, history, add_info='', add_thought=True, think_tag_begin='<thinking>', think_tag_end='</thinking>'):
    user_prompt = f'''The user query: {instruction}'''

    add_reflection = os.environ.get('ADD_REFLECTION', '')
    if add_thought:
        # if len(history) > 0:
        user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
        if len(add_info) > 0:
            # user_prompt += f'\n请根据以下提示操作: {add_info}.'
            user_prompt += f'\nThe following tips can help you complete user tasks: {add_info}.'
        if add_reflection and len(history):
            user_prompt += f'''
Before answering, you must:
1. Analyze if the previous action ({history[-1]}) was appropriate.
2. Verify if its effects match expectations. When you discover that your action was executed incorrectly, you need to try to correct it or attempt another method, rather than terminate.
3. provide reasoning step-by-step for your next action.

Fill the content in <thinking></thinking> tags following this structure:
<thinking>
[Action Analysis]
(1) Correctness assessment: ...
(2) Outcome alignment: ...
(3) Observation of the current screenshot: ...
(3) Next Step Planning: Step-by-step reasoning for subsequent actions...
(4) Action: Express the next steps in imperative form.

</thinking>
Finally provide the <tool_call></tool_call> XML tags.'''
        else:
            user_prompt += f'\nBefore answering, explain your reasoning step-by-step in {think_tag_begin}{think_tag_end} tags, and insert them before the <tool_call></tool_call> XML tags.'
        user_prompt += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'
    return user_prompt

def generate_user_prompt_multi_image(instruction, history, add_info='', add_thought=True):
    user_prompt = f'''The user query: {instruction}'''
    if add_thought:
        # if len(history) > 0:
        # user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
        if len(add_info) > 0:
            # user_prompt += f'\n请根据以下提示操作: {add_info}.'
            user_prompt += f'\nThe following tips can help you complete user tasks: {add_info}.'
        user_prompt += '\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'
        user_prompt += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'
    return user_prompt

def build_system_messages(instruction, resized_width, resized_height, add_info='', history='', infer_mode = 'N_image_infer', add_thought=True):
    mobile_use = AndroidWorldMobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        # TODO
    )
    think_tag_begin = '<thinking>'
    think_tag_end = '</thinking>'
    user_prompt = generate_user_prompt_single_image(instruction, history, add_info, add_thought=add_thought, think_tag_begin=think_tag_begin, think_tag_end=think_tag_end)

    query_messages = [
        Message(
            role="system", content=[ContentItem(text="You are a helpful assistant.")]
        ),
        Message(
            role="user",
            content=[ContentItem(text=user_prompt)],
        )
    ]

    messages = nousprompt.preprocess_fncall_messages(
        messages=query_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    messages = [m.model_dump() for m in messages]

    messages[0]['content'][0]['type'] = 'text'
    messages[0]['content'][1]['type'] = 'text'
    messages[1]['content'][0]['type'] = 'text'

    system_prompt_part = {'role': 'system', 'content': []} # TODO
    system_prompt_part['content'].append(
        {'text': messages[0]['content'][0]['text'] + messages[0]['content'][1]['text']})

    user_prompt_part = {'role': 'user', 'content': []}  # user
    user_prompt_part['content'].append({'text': messages[1]['content'][0]['text']})  # 46 * 1

    return system_prompt_part, user_prompt_part

def convert_mobile_agent_action_to_json_action(
    dummy_action, img_ele, src_format='abs_origin', tgt_format='abs_resized'
) -> json_action.JSONAction:
    """Converts a SeeActAction object to a JSONAction object.

    Args:
      action: the whole dymmay action
                  dummy_action = {
                    "name": ACTION_NAME,
                    "arguments": {
                        "action": "click",
                        "coordinate": [100, 200],
                    },
                }
      elements: UI elements.

    Returns:
      The corresponding JSONAction object.

    """
    action_type_mapping = {
      "click": json_action.CLICK,
      "terminate": json_action.STATUS,
      "answer": json_action.ANSWER, # TODO
      "long_press": json_action.LONG_PRESS,
      "type": json_action.INPUT_TEXT,
      "swipe": json_action.SWIPE,
      "wait": json_action.WAIT,
      "system_button": "system_button",
      "open": json_action.OPEN_APP, # TODO
      "open_app": json_action.OPEN_APP, # TODO
    }

    x = None
    y = None
    text = None
    direction = None
    goal_status = None
    app_name = None

    result_json = {}
    arguments = dummy_action['arguments']
    try:
        action_type_org = arguments['action']
    except:
        arguments = json.loads(arguments)
        action_type_org = arguments['action']
    action_type = action_type_mapping[action_type_org]

    dummy_action_translated = copy.deepcopy({'name': 'mobile_use', 'arguments': arguments}) # dummy_action

    if action_type == json_action.INPUT_TEXT:
        text = arguments['text']

    elif action_type == json_action.SWIPE:
        start_x, start_y = arguments['coordinate']
        end_x, end_y = arguments['coordinate2']
        start_x, start_y = convert_point_format([start_x, start_y], img_ele, src_format=src_format, tgt_format=tgt_format)
        end_x, end_y = convert_point_format([end_x, end_y], img_ele, src_format=src_format, tgt_format=tgt_format)

        dummy_action_translated['arguments']['coordinate'] = [start_x, start_y]
        dummy_action_translated['arguments']['coordinate2'] = [end_x, end_y]

        direction = [start_x, start_y, end_x, end_y]
        # direction = _swipe_to_scroll(arguments['coordinate'], arguments['coordinate2'])

    elif action_type == json_action.CLICK:
        x, y = arguments['coordinate']
        x, y = convert_point_format([x, y], img_ele, src_format=src_format, tgt_format=tgt_format)
        dummy_action_translated['arguments']['coordinate'] = [x, y]

    elif action_type == json_action.LONG_PRESS:
        x, y = arguments['coordinate']
        x, y = convert_point_format([x, y], img_ele, src_format=src_format, tgt_format=tgt_format)
        dummy_action_translated['arguments']['coordinate'] = [x, y]

    elif action_type == json_action.OPEN_APP:  # TODO
      app_name = dummy_action_translated['arguments']['text']

    elif action_type == json_action.ANSWER: # TODO
        text = arguments['text']

    elif action_type == json_action.STATUS:
        goal_status = "task_complete"

    elif action_type == 'system_button':
        if arguments['button'] == 'Back':
            action_type = json_action.NAVIGATE_BACK
        elif arguments['button'] == 'Home':
            action_type = json_action.NAVIGATE_HOME
        elif arguments['button'] == 'Enter':
            action_type = json_action.KEYBOARD_ENTER
        else:
            print("Unknown button: {}".format(arguments['button']))
            raise NotImplementedError
    return json_action.JSONAction(
          action_type=action_type,
          x=x,
          y=y,
          text=text,
          direction=direction,
          goal_status=goal_status,
          app_name=app_name,
      ), dummy_action_translated
