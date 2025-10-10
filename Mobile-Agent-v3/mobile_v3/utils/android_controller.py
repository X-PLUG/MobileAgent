import os
import time
import subprocess
from .controller import Controller

class AndroidController(Controller):
    def __init__(self, adb_path):
        self.adb_path = adb_path

    def get_screenshot(self, save_path):
        command = self.adb_path + " shell rm /sdcard/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = self.adb_path + " shell screencap -p /sdcard/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = self.adb_path + f" pull /sdcard/screenshot.png {save_path}"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        
        if not os.path.exists(save_path):
            return False
        else:
            return True

    def tap(self, x, y):
        command = self.adb_path + f" shell input tap {x} {y}"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def type(self, text):
        text = text.replace("\\n", "_").replace("\n", "_")
        for char in text:
            if char == ' ':
                command = self.adb_path + f" shell input text %s"
                subprocess.run(command, capture_output=True, text=True, shell=True)
            elif char == '_':
                command = self.adb_path + f" shell input keyevent 66"
                subprocess.run(command, capture_output=True, text=True, shell=True)
            elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
                command = self.adb_path + f" shell input text {char}"
                subprocess.run(command, capture_output=True, text=True, shell=True)
            elif char in '-.,!?@\'°/:;()':
                command = self.adb_path + f" shell input text \"{char}\""
                subprocess.run(command, capture_output=True, text=True, shell=True)
            else:
                command = self.adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
                subprocess.run(command, capture_output=True, text=True, shell=True)

    def slide(self, x1, y1, x2, y2):
        command = self.adb_path + f" shell input swipe {x1} {y1} {x2} {y2} 500"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def back(self):
        command = self.adb_path + f" shell input keyevent 4"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def home(self):
        command = self.adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
        subprocess.run(command, capture_output=True, text=True, shell=True)
