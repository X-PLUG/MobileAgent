import copy
from MobileAgent.api import encode_image


def init_chat(instruction):
    operation_history = []
    sysetm_prompt = "You are a helpful phone operating assistant. You need to help me operate the phone to complete my instruction.\n"
    sysetm_prompt += f"My instruction is: {instruction}"
    operation_history.append(["user", [{"type": "text", "text": sysetm_prompt}]])
    operation_history.append(["assistant", [{"type": "text", "text": "Sure. How can I help you?"}]])
    return operation_history


def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
            "type": "text", 
            "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history


def add_multiimage_response(role, prompt, chat_history, images):
    new_chat_history = copy.deepcopy(chat_history)
    content = [
        {
            "type": "text", 
            "text": prompt
        },
    ]
    for image in images:
        base64_image = encode_image(image)
        this_content = {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        content.append(this_content)

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1))
    print("*"*100)