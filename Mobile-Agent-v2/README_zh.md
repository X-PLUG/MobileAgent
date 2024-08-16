![](assets/logo.png?v=1&type=image)
## Mobile-Agent-v2: 通过多代理协作有效导航的移动设备操作助手
<div align="center">
	<a href="https://huggingface.co/spaces/junyangwang0410/Mobile-Agent"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"></a>
	<a href="https://modelscope.cn/studios/wangjunyang/Mobile-Agent-v2"><img src="assets/Demo-ModelScope-brightgreen.svg" alt="Demo ModelScope"></a>
  <a href="https://arxiv.org/abs/2406.01014 "><img src="https://img.shields.io/badge/Arxiv-2406.01014-b31b1b.svg?logo=arXiv" alt=""></a>
  <a href="https://huggingface.co/papers/2406.01014"><img src="https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg" alt=""></a>
</div>
<br>
<div align="center">
Junyang Wang<sup>1</sup>, Haiyang Xu<sup>2†</sup>,Haitao Jia<sup>1</sup>, Xi Zhang,<sup>2</sup>
</div>
<div align="center">
Ming Yan<sup>2†</sup>, Weizhou Shen<sup>2</sup>, Ji Zhang<sup>2</sup>, Fei Huang<sup>2</sup>, Jitao Sang<sup>1†</sup>
</div>
<div align="center">
{junyangwang, jtsang}@bjtu.edu.cn, {shuofeng.xhy, ym119608}@alibaba-inc.com
</div>
<br>
<div align="center">
<sup>1</sup>北京交通大学    <sup>2</sup>阿里巴巴集团
</div>
<div align="center">
<sup>†</sup>通讯作者
</div>

<div align="center">
<a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>
<!--
简体中文 | [English](README.md)
<hr>
-->

## 📢新闻
* 🔥[6.27] 我们在[Hugging Face](https://huggingface.co/spaces/junyangwang0410/Mobile-Agent)和[ModelScope](https://modelscope.cn/studios/wangjunyang/Mobile-Agent-v2)发布了可以上传手机截图体验Mobile-Agent-v2的Demo，无需配置模型和设备，即刻便可体验。
* [6. 4] 我们发布了新一代移动设备操作助手 [Mobile-Agent-v2](https://arxiv.org/abs/2406.01014), 通过多智能体协作实现有效导航。

## 📺演示
https://github.com/X-PLUG/MobileAgent/assets/127390760/d907795d-b5b9-48bf-b1db-70cf3f45d155

## 📋介绍

![](assets/role.jpg?v=1&type=image)
* 一个用于解决在长上下文图文交错输入中导航的多智能体架构。
* 增强的视觉感知模块，用于提升操作准确率。
* 凭借GPT-4o进一步提升操作性能和速度。

## 🔧开始

❗目前仅安卓和鸿蒙系统（版本号 <= 4）支持工具调试。其他系统如iOS暂时不支持使用Mobile-Agent。

### 安装依赖
```
pip install -r requirements.txt
```

### 准备通过ADB连接你的移动设备

1. 下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en)（ADB）。
2. 在你的移动设备上开启“USB调试”或“ADB调试”，它通常需要打开开发者选项并在其中开启。如果是HyperOS系统需要同时打开 "[USB调试(安全设置)](https://github.com/user-attachments/assets/05658b3b-4e00-43f0-87be-400f0ef47736)"。
3. 通过数据线连接移动设备和电脑，在手机的连接选项中选择“传输文件”。
4. 用下面的命令来测试你的连接是否成功: ```/path/to/adb devices```。如果输出的结果显示你的设备列表不为空，则说明连接成功。
5. 如果你是用的是MacOS或者Linux，请先为 ADB 开启权限: ```sudo chmod +x /path/to/adb```。
6.  ```/path/to/adb```在Windows电脑上将是```xx/xx/adb.exe```的文件格式，而在MacOS或者Linux则是```xx/xx/adb```的文件格式。

### 在你的移动设备上安装 ADB 键盘
1. 下载 ADB 键盘的 [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)  安装包。
2. 在设备上点击该 apk 来安装。
3. 在系统设置中将默认输入法切换为 “ADB Keyboard”。

### 选择适合的运行方式

1. 在 ```run.py``` 的22行起编辑你的设置， 并且输入你的 ADB 路径，指令，GPT-4 API URL 和 Token。

2.选择适合你的设备的图标描述模型的调用方法：
  - 如果您的设备配备了高性能GPU，我们建议使用“local”方法。它是指在本地设备中部署图标描述模型。如果您的设备足够强大，则通常具有更好的效率。
  - 如果您的设备不足以运行7B 大小的 LLM，请选择“api”方法。我们使用并行调用来确保效率。

3.选择图标描述模型：
  - 如果选择“local”方法，则需要在“qwen-vl-chat”和“qwen-vl-chat-int4”之间进行选择，其中“qwen-vl-chat”需要更多的GPU内存，但提供了更好的性能与“qwen-vl-chat-int4”相比。同时，“qwen_api”可以是空置的。
  - 如果您选择“api”方法，则需要在“qwen-vl-plus”和“qwen-vl-max”之间进行选择，其中“qwen-vl-max”需要更多的费用，但与“qwen-vl-plus”相比提供了更好的性能。此外，您还需要申请[Qwen-VL 的 API-KEY](https://help.aliyun.com/document_detail/2712195.html?spm=a2c4g.2712569.0.0.5d9e730aymB3jH)，并将其输入到“qwen_api”。

4.您可以在“add_info”中添加操作知识（例如，完成您需要的指令所需的特定步骤），以帮助更准确地运行移动设备。

5.如果您想进一步提高移动设备的效率，则可以将“ reflection_Switch”和“ memory_switch”设置为“ False”。
  - “ reflection_switch”用于确定是否在此过程中添加“反思智能体”。这可能会导致操作陷入死周期。但是您可以将操作知识添加到“ add_info”中以避免它。
  - “ memory_switch”用于决定是否将“内存单元”添加到该过程中。如果你的指令中不需要在后续操作中使用之前屏幕中的信息，则可以将其关闭。

### 运行
```
python run.py
```

## 📑引用

如果您发现Mobile-Agent-v2对研究和应用程序有用，请使用此Bibtex引用：
```
@article{wang2024mobile2,
  title={Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration},
  author={Wang, Junyang and Xu, Haiyang and Jia, Haitao and Zhang, Xi and Yan, Ming and Shen, Weizhou and Zhang, Ji and Huang, Fei and Sang, Jitao},
  journal={arXiv preprint arXiv:2406.01014},
  year={2024}
}
```
