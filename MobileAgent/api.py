import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def inference_chat(chat, API_TOKEN):    
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    data = {
        "model": 'gpt-4-vision-preview',
        "messages": [],
        "max_tokens": 2048,
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    while 1:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res = res.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Network Error {e}")
        else:
            break
    
    return res