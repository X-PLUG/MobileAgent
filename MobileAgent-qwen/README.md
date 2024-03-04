# Mobile-Agent via Qwen-vl
We introduce Mobile-Agent based on the free model Qwen-vl. 

## ❗Note
 - The best performing model Qwen-vl-max of Qwen still has a gap in the ability to work as a mobile device assistant compared to GPT-4v.
 - To mitigate this gap, we refactored the framework and prompt. And we used additional manually generated tutorials to boost performance.
 - Nevertheless, Mobile-Agent via Qwen-vl still performs weaker than Mobile-Agent via GPT-4v. Even if a detailed tutorial is provided, there will be obstacles in performing some complex tasks.
 - We also tried other multi-modal large models such as mPLUG-Owl and Qwen-vl-plus and the same performance gap still exists.
 
## 📺Demo
https://github.com/X-PLUG/MobileAgent/assets/127390760/e697eeba-c2f9-478f-b1b8-140f493ef759

## 🔧Getting Started
### Open DashScope and create APPI_KEY
Please refer to [DashScope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) to get your API-KEY.
### Installation
```
git clone https://github.com/X-PLUG/MobileAgent.git
cd MobileAgent
pip install -r requirements.txt
pip install dashscope
pip install dashscope --upgrade
```

### Preparation for Visual Perception Tools
1. Download the icon detection model [Grounding DION](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
2. The text detection model will be automatically downloaded from modelscope after you run Mobile-Agent.

## Use via screenshot

### Create a host
```
cd MobileAgent-qwen
python host.py --grounding_ckpt /path/to/GroundingDION --api "your DashScopeAPI-KEY"
```

### Inference

First, create a post to send your instruction and the init screenshot. The ``query`` is your instruction, and the ``session_id`` should be vacant.
```
import requests

tutorial = '''The following content may help you complete the instruction:
1. Dark mode is in \"Display & brightness\" of Settings.
2. \"Display & brightness\" can be found by scrolling down the setting page.
3. Tap the \"Dark mode\" to turn on the dark mode.
'''

image = "../results/Settings/1/1.jpg"
query_data = {'screenshot': image, 'query': 'Turn on the dark mode', 'session_id':'', 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())
```
You will get a output like this:
```
{"operation": open app (Settings), "action": "tap", "parameter": "[x, y]", "session_id": an unique id}
```
Then, please operate your device according to ``action`` and ``parameter``, and then send a screenshot of the device after the operation.
```
image = "../results/Settings/1/2.jpg"
query_data = {'screenshot': image, 'query': '', 'session_id': response_query.json()['session_id'], 'tutorial': tutorial}
response_query = requests.post('http://127.0.0.1:5000/a', json=query_data)
print(response_query.json())
```
The ``query`` should be vacant and the ``session_id`` is the unique id you got the last time.

### Case
We provide four cases in ``run_darkmode.py``, ``run_tiotok.py``, ``run_westlake.py``, and ``run_深色模式.py`` respectively. After host starts, run it using the following command:
```
python run_xxx.py
```

## Use via your own device

### Note
❗Due to performance limitations, the framework can only handle simple instructions.

### Preparation for Connecting Mobile Device
1. Download the [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en).
2. Turn on the ADB debugging switch on your Android phone, it needs to be turned on in the developer options first.
3. Connect your phone to the computer with a data cable and select "Transfer files".
4. Test your ADB environment as follow: ```/path/to/adb devices```. If the connected devices are displayed, the preparation is complete.
5. If you are using a MAC or Linux system, make sure to turn on adb permissions as follow: ```sudo chmod +x /path/to/adb```
6. If you are using Windows system, the path will be ```xx/xx/adb.exe```

### Create a tutorial for your instruction
First, write a tutorial to guide Mobile-Agent to complete the instruction. For example, if your instruction is ``Turn on the dark mode``, the tutorial can be:
```
The following content may help you complete the instruction:
1. Dark mode is in \"Display & brightness\" of Settings.
2. \"Display & brightness\" can be found by scrolling down the setting page.
3. Tap the \"Dark mode\" to turn on the dark mode.
```
Then, write this tutorial into ``MobileAgent/MobileAgent-qwen/tutorial.txt``.

### Inference
```
python run.py --grounding_ckpt /path/to/GroundingDION --adb_path /path/to/adb --api "your DashScope API-KEY" --instruction "your instruction"
```
