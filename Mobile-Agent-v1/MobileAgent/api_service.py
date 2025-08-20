import base64
import requests
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_action(image_base, query, session_id, url, token):    
    image_base = encode_image(image_base)

    headers = {
        'Authorization': token,
        'Content-Type': 'application/json'
    }

    data = {
        "model": "pre-Mobile_Agent_Server-1664",
        "input": {
            "screenshot": image_base,
            "query": query,
            "session_id": session_id
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response