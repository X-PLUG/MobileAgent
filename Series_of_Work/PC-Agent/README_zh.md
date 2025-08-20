## PC-Agent: 用于PC上复杂任务自动化的分层多代理协作框架

<div align="center">
<a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>

## 📢新闻
🔥[2025-03-17] PC-Eval 说明及相关文件已上传至 [HuggingFace](https://huggingface.co/datasets/StarBottle/PC-Eval)。

🔥[2025-03-12] 代码已更新。

🔥[2025-02-21] 我们发布了 PC-Agent 的更新版本。详情请查看 [论文](https://arxiv.org/abs/2502.14282)。代码即将更新。

🔥[2024-08-23] 我们发布了 PC-Agent 的代码，支持 Mac 和 Windows 平台。

## 📺Demo
[https://github.com/X-PLUG/MobileAgent/blob/main/Series_of_Work/PC-Agent/PCAgent_v1/demo/Download%20paper%20from%20Chorme.mp4](https://github.com/user-attachments/assets/5abb9dc8-d49b-438b-ac44-19b3e2da03cb)

[https://github.com/X-PLUG/MobileAgent/blob/main/Series_of_Work/PC-Agent/PCAgent_v1/demo/Search%20NBA%20FMVP%20and%20send%20to%20friend.mp4](https://github.com/user-attachments/assets/b890a08f-8a2f-426d-9458-aa3699185030)

[https://github.com/X-PLUG/MobileAgent/blob/main/Series_of_Work/PC-Agent/PCAgent_v1/demo/Write%20an%20introduction%20of%20Alibaba%20in%20Word.mp4](https://github.com/user-attachments/assets/37f0a0a5-3d21-4232-9d1d-0fe845d0f77d)

## 📋介绍
* PC-Agent 是一个多智能体协作系统，能够根据用户指令实现对生产力场景（例如 Chrome、Word 和微信）的自动化控制。
* 针对密集且多样化的交互元素设计的主动感知模块，能够更好地适配 PC 平台。
* 分层多智能体协作结构能够提高更复杂任务序列的成功率。

<!-- * PC-Agent是一个面向复杂PC任务的多模态智能体框架，基于视觉感知实现多种生产力场景的自动控制，包括Chrome, Word, WeChat等。
* 针对密集多样的可交互元素设计的主动感知模块更好地适应PC平台。
* 层次化多智能体协作结构提高了更复杂任务序列的成功率。
 -->

## 🔧开始

### 安装
现在同时支持**Windows**和**Mac**。
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

### 配置
编辑 config.json 以添加您的 API 密钥并自定义设置：
```
# API configuration
{
  "vl_model_name": "gpt-4o",
  "llm_model_name": "gpt-4o",
  "token": "sk-...", # Replace with your actual API key
  "url": "https://api.openai.com/v1"
}
```

### 测试你的PC

1. 使用你的指令和 GPT-4o API 令牌运行 *run.py*。例如：
```
# For Windows
python run.py --instruction="Open Chrome and search the PC-Agent paper." --mac 0

# For Mac
python run.py --instruction="Open Chrome and search the PC-Agent paper." --mac 1
```

2. 您也可以选择通过 *--add_info* 选项添加具体的操作知识，以帮助 PC-Agent 更精准地运行。

3. 为了进一步提高 PC-Agent 的运行效率，您可以设置 *--disable_reflection* 来跳过反射过程。请注意，这可能会降低操作的成功率。

4. 如果任务不是很复杂，您可以设置 *--simple 1* 来跳过任务分解。
