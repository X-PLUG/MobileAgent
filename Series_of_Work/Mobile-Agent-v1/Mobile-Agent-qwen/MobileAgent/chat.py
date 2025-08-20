import copy

def init_chat(instruction):
    operation_history = []
    sysetm_prompt = f"You are a helpful phone operating assistant. You need to help me operate the phone to complete user's instruction."
    operation_history.append(["system", [{"text": sysetm_prompt}]])
    return operation_history


def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image:
        content = [
            {
                "text": prompt
            },
            {
                "image": "file://" + image
            },
        ]
    else:
        content = [
            {
            "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history


def add_multiimage_response(role, prompt, chat_history, images):
    new_chat_history = copy.deepcopy(chat_history)
    content = [
        {
            "text": prompt
        },
    ]
    for image in images:
        this_content = {
            "image": "file://" + image
        }
        content.append(this_content)

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1))
        print()
    print("*"*100)