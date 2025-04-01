## PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC

## 📢News
🔥[2025-03-17] The PC-Eval instructions and related files have been uploaded at [HuggingFace](https://huggingface.co/datasets/StarBottle/PC-Eval).

🔥[2025-03-12] The code has been updated.

🔥[2025-02-21] We have released an updated version of PC-Agent. Check the [paper](https://arxiv.org/abs/2502.14282) for details. The code will be updated soon.

🔥[2024-08-23] We have released the code of PC-Agent, supporting both Mac and Windows platforms.

## 📺Demo
[https://github.com/X-PLUG/MobileAgent/blob/main/PC-Agent/PCAgent/demo/Download%20paper%20from%20Chorme.mp4](https://github.com/user-attachments/assets/5abb9dc8-d49b-438b-ac44-19b3e2da03cb)

[https://github.com/X-PLUG/MobileAgent/blob/main/PC-Agent/PCAgent/demo/Search%20NBA%20FMVP%20and%20send%20to%20friend.mp4](https://github.com/user-attachments/assets/b890a08f-8a2f-426d-9458-aa3699185030)

[https://github.com/X-PLUG/MobileAgent/blob/main/PC-Agent/PCAgent/demo/Write%20an%20introduction%20of%20Alibaba%20in%20Word.mp4](https://github.com/user-attachments/assets/37f0a0a5-3d21-4232-9d1d-0fe845d0f77d)

## 📋Introduction
* PC-Agent is a multi-agent collaboration system, which can achieve automated control of productivity scenarios (_e.g._ Chrome, Word, and WeChat) based on user instructions.
* Active perception module designed for dense and diverse interactive elements are better adapted to the PC platform.
* The hierarchical multi-agent cooperative structure improves the success rate of more complex task sequences.

<!-- * PC-Agent是一个面向复杂PC任务的多模态智能体框架，基于视觉感知实现多种生产力场景的自动控制，包括Chrome, Word, WeChat等。
* 针对密集多样的可交互元素设计的主动感知模块更好地适应PC平台。
* 层次化多智能体协作结构提高了更复杂任务序列的成功率。
 -->

## 🔧Getting Started

### Installation
Now both **Windows** and **Mac** are supported.
```
conda create --name pcagent python=3.10
source activate pcagent

# For Windows
pip install -r requirements.txt

# For Mac
pip install -r requirements_mac.txt

git clone https://github.com/Topdu/OpenOCR.git
pip install openocr-python
```

### Configuration
Edit config.json to add your API keys and customize settings:
```
# API configuration
{
  "vl_model_name": "gpt-4o",
  "llm_model_name": "gpt-4o",
  "token": "sk-...", # Replace with your actual API key
  "url": "https://api.openai.com/v1"
}
```

### Test on your computer

1. Run the *run.py* with your instruction and your GPT-4o api token. For example,
```
# For Windows
python run.py --instruction="Open Chrome and search the PC-Agent paper." --Mac 0

# For Mac
python run.py --instruction="Open Chrome and search the PC-Agent paper." --Mac 1
```

2. Optionally, you can add specific operational knowledge via the *--add_info* option to help PC-Agent operate more accurately.

3. To further improve the operation efficiency of PC-Agent, you can set *--disable_reflection* to skip the reflection process. Note that this may reduce the success rate of the operation.

4. If the task is not very complex, you can set *--simple 1* to skip the task decomposition.
