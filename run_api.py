import os
import json
import argparse
from MobileAgent.api_service import get_action
from MobileAgent.controller_api import get_screenshot, tap, type, slide, back, back_to_desktop


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--adb_path", type=str)
    parser.add_argument("--url", type=str)
    parser.add_argument("--token", type=str)
    args = parser.parse_args()
    return args


def run(args):
    if not os.path.exists("screenshot"):
        os.mkdir("screenshot")

    first_request = 0
    session_id = ""
    
    while True:
        first_request += 1

        get_screenshot(args.adb_path)
        image = "./screenshot/screenshot.jpg"
        
        while True:
            if first_request == 1:
                response = get_action(image, args.instruction, '', args.url, args.token)
                print(response.json())
            else:
                response = get_action(image, '', session_id, args.url, args.token)
                print(response.json())
                
            try:
                action = response.json()['output']['action']
                parameter = response.json()['output']['parameter']
                session_id = response.json()['output']['session_id']
            except:
                print("Error:")
                print(response.json)
            else:
                break

        if action == 'end':
            parameter = str(parameter)
            if parameter != '':
                print(parameter)
            break
        
        elif action == 'tap':
            parameter = json.loads(parameter)
            tap(args.adb_path, parameter[0], parameter[1])
        
        elif action == 'slide':
            parameter = json.loads(parameter)
            slide(args.adb_path, parameter[0][0], parameter[0][1], parameter[1][0], parameter[1][1])
            
        elif "type" in action:
            parameter = str(parameter)
            type(args.adb_path, parameter)
        
        elif "back" in action:
            back(args.adb_path)

        elif "exit" in action:
            back_to_desktop(args.adb_path)

if __name__ == "__main__":
    args = get_args()
    run(args)
