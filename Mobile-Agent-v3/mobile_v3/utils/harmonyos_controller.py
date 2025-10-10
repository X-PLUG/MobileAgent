import os
import time
import subprocess
from .controller import Controller

class HarmonyOSController(Controller):
    def __init__(self, hdc_path):
        self.hdc_path = hdc_path

    def get_screenshot(self, save_path):
        command = self.hdc_path + " shell rm /data/local/tmp/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = self.hdc_path + " shell uitest screenCap -p /data/local/tmp/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = self.hdc_path + " file recv /data/local/tmp/screenshot.png " + save_path
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)

        if not os.path.exists(save_path):
            return False
        else:
            return True

    def tap(self, x, y):
        command = self.hdc_path + f" shell uitest uiInput click {x} {y}"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def type(self, text):
        text = text.replace("\\n", "_").replace("\n", "_")
        command = self.hdc_path + f" shell uitest uiInput inputText 1 1 {text}"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def slide(self, x1, y1, x2, y2):
        command = self.hdc_path + f" shell uitest uiInput swipe {x1} {y1} {x2} {y2} 500"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def back(self):
        command = self.hdc_path + " shell uitest uiInput keyEvent Back"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def home(self):
        command = self.hdc_path + " shell uitest uiInput keyEvent Home"
        subprocess.run(command, capture_output=True, text=True, shell=True)
