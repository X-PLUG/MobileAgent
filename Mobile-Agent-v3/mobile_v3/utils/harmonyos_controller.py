import os
import time
import subprocess

def get_screenshot(hdc_path, save_path):
    command = hdc_path + " shell rm /data/local/tmp/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = hdc_path + " shell uitest screenCap -p /data/local/tmp/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = hdc_path + " file recv /data/local/tmp/screenshot.png " + save_path
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    
    if not os.path.exists(save_path):
        return False
    else:
        return True

def tap(hdc_path, x, y):
    command = hdc_path + f" hdc shell uitest uiInput click {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)

def type(hdc_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    command = hdc_path + f" shell uitest uiInput inputText 1 1 {text}"
    subprocess.run(command, capture_output=True, text=True, shell=True)
 
def slide(hdc_path, x1, y1, x2, y2):
    command = hdc_path + f" shell uitest uiInput swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)

def back(hdc_path):
    command = hdc_path + " shell uitest uiInput keyEvent Back"
    subprocess.run(command, capture_output=True, text=True, shell=True)

def home(hdc_path):
    command = hdc_path + " shell uitest uiInput keyEvent Home"
    subprocess.run(command, capture_output=True, text=True, shell=True)
