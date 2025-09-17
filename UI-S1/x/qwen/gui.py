import json


point_template = lambda point: f"<points x1=\"{point[0]}\" y1=\"{point[1]}\"></points>"

def mid2pretrain_qwen_data(line, index, bbox_style='qwen-vl'):
    from x.qwen.image import point_rep
    from x.qwen.tokenizer import get_text_seq_len

    train_item = []
    
    train_item.append(line['image'])
    train_item.append({'prompt': line['query'], 'seq_len': get_text_seq_len(line['query'])})
    point = line['point']
    point = point_rep(point, line['image'], bbox_style=bbox_style)
    point_str = point_template(point)
    train_item.append({'text': point_str,'seq_len': get_text_seq_len(point_str)})
    return True, train_item






tool_descs = '''{"type": "function", "function": {"name_for_human": "mobile device operation assistant", "name": "mobile_operation", "description": "The mobile device operation assistant service performs simulated touch operations on the mobile device according to the user's requirements and returns the type and parameters of the corresponding operation.", "parameters": ['''
tool_descs += '''{"name": "click_point", "type": "string", "description": "The format is: click_point: [x, y], which means click the point on the screen with coordinates [x, y]. The x increases from left to right and represents the horizontal coordinate, and the y increases from top to bottom and represents the vertical coordinate. Both x and y are relative coordinates, with values between 0 and 1000.", "required": true}, '''
tool_descs += '''{"name": "swipe_from_start_to_end", "type": "string", "description": "The format is: swipe_from_start_to_end: [x1, y1, x2, y2], which means sliding from the starting point with coordinates [x1, y1] to the end point with coordinates [x2, y2]. x1 and x2 increase from left to right, representing the horizontal coordinate; y1 and y2 increase from top to bottom, representing the vertical coordinate. The x1, y1, x2, y2 are all relative coordinates, with values between 0 and 1000.", "required": true}, '''
tool_descs += '''{"name": "scroll_with_direction", "type": "string", "description": "The format is: scroll_with_direction: [direction], which means scrolling the page in the specified direction to display more content. The direction options are [up, down, left, right], which indicates the direction in which the page scrolls.", "required": true}, '''
tool_descs += '''{"name": "type_text", "type": "string", "description": "The format is: type_text: [text], which means input the specified text into the activated input box. text is the text content to be input.", "required": true}, '''
tool_descs += '''{"name": "press_system_button", "type": "string", "description": "The format is: press_system_button: [button], which means pressing the system button. The button options are [Back, Home, Menu, Enter], where Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means entering the key.", "required": true}, '''
tool_descs += '''{"name": "complete_task", "type": "string", "description": "The format is: complete_task, which means the task has been completed.", "required": true}, '''
tool_descs += '''{"name": "wait", "type": "string", "description": "The format is: wait, which means waiting for a while without doing anything.", "required": true}], "args_format": "The input to this tool should be a JSON object."}}'''


def mid2sft_qwen_data(line, index, bbox_style='qwen-vl'):
    from x.qwen.image import point_rep
    from x.qwen.tokenizer import get_text_seq_len

    messages = [{
                'role':
                    'system',
                'content': [{
                        'text': f'''You are Qwen, created by Alibaba Cloud. You are a helpful mobile device operation assistant. You need to help me operate my mobile device to complete my instruction.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>'''
                }],
            }]
    messages.append({
        'role': 'user',
        'content': [
            {
                'image': line['image']
            },
            {
                'text': f'''The task that the user wants to complete on the current device: {line['query']}
Your historical operation track is shown in the above screenshot. 
Please combine the user task and historical operation track to generate the operation that needs to be performed on the current page.
Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'''
            },
        ],
    })
    messages.append({
        "role":"assistant",
        "content":[{"text":'''<thinking></thinking>

<tool_call>
{"name": "mobile device operation assistant", "arguments": "{\"type_text\": [\"味多美\"]}"}
</tool_call>'''}]
    })
    train_item.append(line['image'])
    train_item.append({'prompt': line['query'], 'seq_len': get_text_seq_len(line['query'])})
    point = line['point']
    point = point_rep(point, line['image'], bbox_style=bbox_style)
    point_str = point_template(point)
    train_item.append({'text': point_str,'seq_len': get_text_seq_len(point_str)})
    return True, messages