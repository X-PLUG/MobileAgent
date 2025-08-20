## PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC

## 📢News
🔥[2025-02-21] We have released an updated version of PC-Agent. Check the [paper](https://arxiv.org/abs/2502.14282) for details. The code will be updated soon.

🔥[2024-08-23] We have released the code of PC-Agent, supporting both Mac and Windows platforms.

## 📺Demo
[https://github.com/X-PLUG/MobileAgent/blob/main/PC-Agent/PCAgent/demo/Download%20paper%20from%20Chorme.mp4](https://github.com/user-attachments/assets/5abb9dc8-d49b-438b-ac44-19b3e2da03cb)

[https://github.com/X-PLUG/MobileAgent/blob/main/PC-Agent/PCAgent/demo/Search%20NBA%20FMVP%20and%20send%20to%20friend.mp4](https://github.com/user-attachments/assets/b890a08f-8a2f-426d-9458-aa3699185030)

[https://github.com/X-PLUG/MobileAgent/blob/main/PC-Agent/PCAgent/demo/Write%20an%20introduction%20of%20Alibaba%20in%20Word.mp4](https://github.com/user-attachments/assets/37f0a0a5-3d21-4232-9d1d-0fe845d0f77d)

## 📋Introduction
* PC-Agent is a multi-agent collaboration system, which can achieve automated control of computer software (_e.g._ Chrome, Word, and WeChat) based on user instructions.
* The visual perception module designed for high-resolution screens is better suited for the PC platforms.
* The Planning-Decision-Reflection framework improves the success rate of operations.

<!-- * PC-Agent是一个多智能体协作的系统，基于视觉感知实现多种电脑端应用的自动控制，包括Chrome, Word, WeChat等。
* 针对高分辨率屏幕设计的视觉感知模块更好地适应PC平台。
* 规划-决策-反思框架提高了操作的成功率。
 -->

## 🔧Getting Started

### Installation
Both **MacOS** and **Windows** are supported.
```
# For MacOS
pip install -r requirements.txt
# For Windows
pip install -r requirements_win.txt
```

### Test on your computer

1. Run the *run.py* with your instruction and your GPT-4o api token. For example,
```
python run.py --instruction="Create a new doc on Word, write a brief introduction of Alibaba, and save the document." --api_token='Your GPT-4o API token.'
```

2. Optionally, you can add specific operational knowledge via the *--add_info* option to help PC-Agent operate more accurately.

3. To further improve the operation efficiency of PC-Agent, you can set *--disable_reflection* to skip the reflection process. Note that this may reduce the success rate of the operation.

