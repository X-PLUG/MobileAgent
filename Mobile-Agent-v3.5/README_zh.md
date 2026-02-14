# GUI-Owl 1.5

<div align="center">
<img src=https://youke1.picui.cn/s1/2025/08/18/68a2f82fef3d4.png width="40%"/>
</div>

<div align="center">
<a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
<hr>
</div>

## 📢 最新动态
* 🔥[2026年2月] 我们在 [HuggingFace](https://huggingface.co/collections/mPLUG/gui-owl-15) 上发布了 GUI-Owl 1.5 模型系列。
<!-- * 🔥[2025年2月] 技术报告见[此处](https://arxiv.org/abs/xxxx.xxxxx)。-->

## 📍 待办事项
- [x] 开源 GUI-Owl 1.5 模型权重
- [x] 开源 GUI-Owl 1.5 模型使用指南（Cookbook）
- [ ] 开源各基准测试的评测代码
- [ ] 在您自己的设备上部署 Mobile-Agent-v3.5

## 简介

GUI-Owl 1.5 是基于 Qwen3-VL 构建的新一代原生 GUI 智能体模型系列，支持跨**桌面**、**移动设备**、**浏览器**等多平台的 GUI 自动化。依托可扩展的混合数据飞轮、统一的智能体能力增强方案以及多平台环境强化学习（MRPO），GUI-Owl 1.5 提供了完整的模型矩阵：

| 模型 | HuggingFace |
|------|-------------|
| GUI-Owl-1.5-2B-Instruct | [🤗 链接](https://huggingface.co/mPLUG/GUI-Owl-1.5-2B-Instruct) |
| GUI-Owl-1.5-4B-Instruct | [🤗 链接](https://huggingface.co/mPLUG/GUI-Owl-1.5-4B-Instruct) |
| GUI-Owl-1.5-8B-Instruct | [🤗 链接](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct) |
| GUI-Owl-1.5-8B-Thinking | [🤗 链接](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think) |
| GUI-Owl-1.5-32B-Instruct | [🤗 链接](https://huggingface.co/mPLUG/GUI-Owl-1.5-32B-Instruct) |
| GUI-Owl-1.5-32B-Thinking | [🤗 链接](https://huggingface.co/mPLUG/GUI-Owl-1.5-32B-Think) |

**核心亮点：**
- 🏆 **业界领先**：在 OSWorld-Verified、AndroidWorld、Mobile-World、WindowsAA、ScreenSpot-v2、ScreenSpot-Pro 等基准上，在多平台 GUI 模型中取得最优表现。
- 🔧 **工具与 MCP 调用**：原生支持外部工具调用和 MCP 服务器协调，在 OSWorld-MCP 和 Mobile-World 上取得顶尖性能。
- 🧠 **长程记忆**：内置记忆能力，无需外部工作流编排，在 MemGUI-Bench 上领先所有原生智能体模型。
- 🤝 **多智能体协作**：既可作为独立的端到端智能体，也可在 Mobile-Agent-v3.5 框架中充当规划器、执行器、验证器、记录员等专业角色。
- ⚡ **Instruct 与 Thinking 双模式**：小尺寸 Instruct 模型推理速度快，适合边缘部署；大尺寸 Thinking 模型具备更强的规划与反思能力，适合复杂任务。

<!-- 
## 在移动设备上部署 Mobile-Agent-v3.5

❗ 目前仅 **Android** 和 **鸿蒙 OS** 支持工具调试。**iOS** 等其他系统暂不支持使用 Mobile-Agent。

### 安装依赖
```bash
pip install qwen_agent
pip install qwen_vl_utils
pip install numpy
```

### 通过 ADB 连接移动设备
1. 下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=zh-cn)。
2. 在 Android 手机上开启 ADB 调试（需先开启开发者选项）。若为 HyperOS 系统，还需开启"USB 调试（安全设置）"。
3. 使用数据线连接手机与电脑，选择"传输文件"。
4. 测试 ADB 环境：`adb devices`。若显示已连接设备，则准备就绪。
5. Mac/Linux 系统请确保 ADB 权限：`sudo chmod +x /path/to/adb`
6. Windows 系统请使用：`xx\xx\adb.exe`

### 在移动设备上安装 ADB 键盘
1. 下载 [ADB Keyboard APK](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)。
2. 在移动设备上安装。
3. 在系统设置中将默认输入法切换为 "ADB Keyboard"。

### 运行

#### Android
```bash
cd Mobile-Agent-v3.5/mobile_v3_5
python run_mobileagentv3_5.py \
    --adb_path "您的 ADB 路径" \
    --api_key "您的 vllm 服务 API 密钥" \
    --base_url "您的 vllm 服务地址" \
    --model "您的 vllm 服务模型名称" \
    --instruction "您希望 Mobile-Agent-v3.5 完成的指令" \
    --add_info "补充信息，可为空"
```

#### 鸿蒙 OS
```bash
cd Mobile-Agent-v3.5/mobile_v3_5
python run_mobileagentv3_5.py \
    --hdc_path "您的 HDC 路径" \
    --api_key "您的 vllm 服务 API 密钥" \
    --base_url "您的 vllm 服务地址" \
    --model "您的 vllm 服务模型名称" \
    --instruction "您希望 Mobile-Agent-v3.5 完成的指令" \
    --add_info "补充信息，可为空"
```
 -->

### 注意事项
1. GUI-Owl 1.5 默认输出相对坐标（0–1000）。

<!-- 
## 在 OSWorld 上评测
1. 按照 [OSWorld 官方仓库](https://github.com/xlang-ai/OSWorld?tab=readme-ov-file#-installation) 安装 OSWorld 及相关依赖。

2. 在 `run_guiowl.sh` 或 `run_ma3_5.sh` 中填写 vLLM 服务信息（api_key、base_url、model）。

3. 运行：
```bash
cd Mobile-Agent-v3.5/os_world_v3_5
sh run_guiowl.sh
sh run_ma3_5.sh
```

## 在 AndroidWorld 上评测
1. 按照 [AndroidWorld 官方仓库](https://github.com/google-research/android_world?tab=readme-ov-file#installation) 安装 Android 模拟器及相关依赖。

2. 安装依赖：
```bash
pip install qwen_agent
pip install qwen_vl_utils
```

3. 在 `run_guiowl.sh` 或 `run_ma3_5.sh` 中填写 vLLM 服务信息。

4. 运行：
```bash
cd Mobile-Agent-v3.5/android_world_v3_5
sh run_guiowl.sh
sh run_ma3_5.sh
```
 -->

## 性能

### 端到端在线基准测试

| 模型 | OSWorld-Verified | AndroidWorld | OSWorld-MCP | Mobile-World | WindowsAA |
|------|:---:|:---:|:---:|:---:|:---:|
| GUI-Owl-1.5-2B-Instruct | 43.5 | 67.9 | 33.0 | 31.3 | 25.8 |
| GUI-Owl-1.5-4B-Instruct | 48.2 | 69.8 | 31.7 | 32.3 | 29.4 |
| GUI-Owl-1.5-8B-Instruct | 52.3 | 69.0 | 41.8 | 41.8 | 31.7 |
| GUI-Owl-1.5-8B-Thinking | 52.9 | **71.6** | 38.8 | 33.3 | 35.1 |
| GUI-Owl-1.5-32B-Instruct | - | 69.4 | **47.6** | **46.8** | **44.8** |
| GUI-Owl-1.5-32B-Thinking | - | 68.2 | 43.8 | 42.8 | 44.1 |

### Grounding 基准测试

详细结果请参阅技术报告，涵盖 ScreenSpot-v2、ScreenSpot-Pro、OSWorld-G、MMBench-GUI 等基准。

## 快速开始

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_name = "mPLUG/GUI-Owl-1.5-8B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "screenshot.png"},
            {"type": "text", "text": "Click on the search bar."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)
# 推理：生成输出
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## 引用

如果您觉得 GUI-Owl 1.5 对您的研究有帮助，请引用：

```bibtex
@article{guiowl15,
  title={Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents},
  author={...},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## 致谢

GUI-Owl 1.5 基于 [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) 构建。感谢 Qwen 团队提供的优秀开源基础模型。
