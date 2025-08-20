import copy
from PCAgent.api import resize_encode_image


def init_subtask_chat():
    operation_history = []
    system_prompt = "You are a helpful AI assistant."
    operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
    return operation_history


def init_action_chat():
    operation_history = []
    system_prompt = "You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user\'s instruction."
    operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
    return operation_history


def init_reflect_chat():
    operation_history = []
    system_prompt = "You are a helpful AI PC operating assistant."
    operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
    return operation_history


def init_memory_chat():
    operation_history = []
    system_prompt = "You are a helpful AI PC operating assistant."
    operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
    return operation_history


def add_response_old(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = resize_encode_image(image)
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


def add_response(role, prompt, chat_history, image=[], use_qwen=False):
    new_chat_history = copy.deepcopy(chat_history)
    content = [
        {
        "type": "text", 
        "text": prompt
        },
    ]
    for i in range(len(image)):
        if not use_qwen:
            base64_image = resize_encode_image(image[i])
            content.append(
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            )
        else:
            content.append(
                {
                    "type": "image", 
                    "image": image[i]
                }
            )
    new_chat_history.append([role, content])
    return new_chat_history


def add_response_two_image(role, prompt, chat_history, image):
    new_chat_history = copy.deepcopy(chat_history)

    base64_image1 = resize_encode_image(image[0])
    base64_image2 = resize_encode_image(image[1])
    content = [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}"
            }
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}"
            }
        },
    ]

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1) + "\n")
    print("*"*100)