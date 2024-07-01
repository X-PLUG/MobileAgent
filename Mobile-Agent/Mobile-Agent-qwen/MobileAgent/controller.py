import time
import subprocess
from PIL import Image
import xml.etree.ElementTree as ET

def get_size(adb_path):
    command = adb_path + " shell wm size"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    resolution_line = result.stdout.strip().split('\n')[-1]
    width, height = map(int, resolution_line.split(' ')[-1].split('x'))
    return width, height


def get_screenshot(adb_path):
    command = adb_path + " shell rm /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + " shell screencap -p /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + " pull /sdcard/screenshot.png ./screenshot"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    image_path = "./screenshot/screenshot.png"
    save_path = "./screenshot/screenshot.jpg"
    image = Image.open(image_path)
    original_width, original_height = image.size
    new_width = int(original_width * 0.5)
    new_height = int(original_height * 0.5)
    resized_image = image.resize((new_width, new_height))
    resized_image.convert("RGB").save(save_path, "JPEG")
    time.sleep(1)


def tap(adb_path, x, y, px, py):
    w = px
    h = py
    ax = int(x*w)
    ay = int(y*h)
    command = adb_path + f" shell input tap {ax} {ay}"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(1)


def type(adb_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            command = adb_path + f" shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == '_':
            command = adb_path + f" shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in '-.,!?@\'°/:;()':
            command = adb_path + f" shell input text \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(1)


def slide(adb_path, action, x, y):
    if "down" in action:
        command = adb_path + f" shell input swipe {int(x/2)} {int(y/2)} {int(x/2)} {int(y/4)} 500"
        subprocess.run(command, capture_output=True, text=True, shell=True)
    elif "up" in action:
        command = adb_path + f" shell input swipe {int(x/2)} {int(y/2)} {int(x/2)} {int(3*y/4)} 500"
        subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(1)


def back(adb_path):
    command = adb_path + f" shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(1)
    
    
def back_to_desktop(adb_path):
    command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(1)

def get_xml(adb_path, save_path):
    dump_command = adb_path + f" shell uiautomator dump /sdcard/dump.xml"
    pull_command = adb_path + f" pull /sdcard/dump.xml {save_path}"
    subprocess.run(dump_command, capture_output=True, text=True, shell=True)
    time.sleep(1)
    subprocess.run(pull_command, capture_output=True, text=True, shell=True)    
    time.sleep(1)

def is_clickable(xml_path,x, y):
    for event, element in ET.iterparse(xml_path, events=('start', 'end')):
        if event == 'start' and 'clickable' in element.attrib and element.attrib['clickable'] == 'true':
            bounds = element.attrib["bounds"][1:-1].split("][")
            x1, y1 = map(int, bounds[0].split(","))
            x2, y2 = map(int, bounds[1].split(","))
            
            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                return True
    else:
        return False
    
def choose_clickable(in_coordinate, out_coordinate, xml_path, iw, ih, x, y):
    res_in_coordinate, res_out_coordinate = [], []
    for i in range(len(out_coordinate)):
        tap_coordinate = [(in_coordinate[i][0] + in_coordinate[i][2]) / 2,
                                  (in_coordinate[i][1] + in_coordinate[i][3]) / 2]
        tap_coordinate = [round(tap_coordinate[0] / iw, 2), round(tap_coordinate[1] / ih, 2)]
        if is_clickable(xml_path, int(tap_coordinate[0]*x), int(tap_coordinate[1]*y)):
            res_in_coordinate.append(in_coordinate[i])
            res_out_coordinate.append(out_coordinate[i])
    return res_in_coordinate, res_out_coordinate