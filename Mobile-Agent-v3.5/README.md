# Mobile-Agent-v3.5

<div align="center">
<img src="../assets/gui_owl_15_logo.png" width="80%"/>
</div>

<div align="center">
<a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
<hr>
</div>

## 📢 News
* 🔥[Feb. 2026] We release the GUI-Owl 1.5 model family on [HuggingFace](https://huggingface.co/collections/mPLUG/gui-owl-15).
<!-- * 🔥[Feb. 2025] The technical report can be found [here](https://arxiv.org/abs/xxxx.xxxxx).-->

## 📍 TODO
- [x] Open source GUI-Owl 1.5 model weights
- [x] Open source GUI-Owl 1.5 model cookbook
- [x] Deploy Mobile-Agent-v3.5 on your own device
- [ ] Open source evaluation code on benchmarks


## Introduction

GUI-Owl 1.5 is the next-generation native GUI agent model family built on Qwen3-VL. It supports multi-platform GUI automation across **desktops**, **mobile devices**, **browsers**, and more. Powered by a scalable hybrid data flywheel, unified agent capability enhancement, and multi-platform environment RL (MRPO), GUI-Owl 1.5 offers a full spectrum of models:

| Model | HuggingFace |
|-------|-------------|
| GUI-Owl-1.5-2B-Instruct | [🤗 Link](https://huggingface.co/mPLUG/GUI-Owl-1.5-2B-Instruct) |
| GUI-Owl-1.5-4B-Instruct | [🤗 Link](https://huggingface.co/mPLUG/GUI-Owl-1.5-4B-Instruct) |
| GUI-Owl-1.5-8B-Instruct | [🤗 Link](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct) |
| GUI-Owl-1.5-8B-Thinking | [🤗 Link](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think) |
| GUI-Owl-1.5-32B-Instruct | [🤗 Link](https://huggingface.co/mPLUG/GUI-Owl-1.5-32B-Instruct) |
| GUI-Owl-1.5-32B-Thinking | [🤗 Link](https://huggingface.co/mPLUG/GUI-Owl-1.5-32B-Think) |

**Key highlights:**
- 🏆 **State-of-the-art** among multi-platform GUI models on OSWorld-Verified, AndroidWorld, Mobile-World, WindowsAA, ScreenSpot-v2, ScreenSpot-Pro, and more.
- 🔧 **Tool & MCP calling**: Native support for external tool invocation and MCP server coordination, achieving top performance on OSWorld-MCP and Mobile-World.
- 🧠 **Long-horizon memory**: Built-in memory capability without external workflow orchestration, leading all native agent models on MemGUI-Bench.
- 🤝 **Multi-agent ready**: Serves both as a standalone end-to-end agent and as specialized roles (planner, executor, verifier, notetaker) within the Mobile-Agent-v3.5 framework.
- ⚡ **Instruct & Thinking variants**: Smaller instruct models for fast inference and edge deployment; larger thinking models for complex tasks requiring planning and reflection.


## Deploy Mobile-Agent-v3.5 on Your Mobile Device

❗ At present, only **Android OS** support tool debugging. Other systems, such as **iOS**, do not support the use of Mobile-Agent for the time being.

### Install Dependencies
```bash
pip install qwen_agent
pip install qwen_vl_utils
pip install numpy
```

### Preparation for Connecting Mobile Device with ADB
1. Download the [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en).
2. Turn on ADB debugging on your Android phone (enable Developer Options first). For HyperOS, also enable "USB Debugging (Security Settings)".
3. Connect your phone to the computer with a data cable and select "Transfer files".
4. Test your ADB environment: `adb devices`. If connected devices are displayed, you're ready.
5. On Mac/Linux, ensure ADB permissions: `sudo chmod +x /path/to/adb`
6. On Windows, use: `xx\xx\adb.exe`

### Install ADB Keyboard on Your Mobile Device
1. Download the [ADB Keyboard APK](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk).
2. Install on your mobile device.
3. Switch the default input method to "ADB Keyboard" in system settings.

### Run
```bash
cd Mobile-Agent-v3.5/mobile_use
python run_gui_owl_1_5_for_mobile.py \
    --adb_path "Your ADB path" \
    --api_key "Your api key of vllm service" \
    --base_url "Your base url of vllm service" \
    --model "Your model name of vllm service" \
    --instruction "The instruction you want Mobile-Agent-v3.5 to complete" \
    --add_info "Some supplementary knowledge, can also be empty"
```

### Note
1. GUI-Owl 1.5 outputs relative coordinates (0–1000) by default.

## Deploy Mobile-Agent-v3.5 on Your Computer Device

### Install Dependencies
```bash
pip install pyautogui
pip install pyperclip
```

### Run
```bash
cd Mobile-Agent-v3.5/computer_use
python run_gui_owl_1_5_for_pc.py \
    --api_key "Your api key of vllm service" \
    --base_url "Your base url of vllm service" \
    --model "Your model name of vllm service" \
    --instruction "The instruction you want Mobile-Agent-v3.5 to complete" \
    --add_info "Some supplementary knowledge, can also be empty"
```

### Note
1. GUI-Owl 1.5 outputs relative coordinates (0–1000) by default.

## Deploy Mobile-Agent-v3.5 on Your Browser

### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Configure Environment Variables
```bash
# Model API keys (required)
export API_KEY="sk-xxx"          # Agent model API
export OMNI_API_KEY=""
export EVAL_API_KEY="sk-xxx"               # Evaluation model API
```

### Run
```bash
cd cd Mobile-Agent-v3.5/browser_use
python run_gui_owl_1_5_for_web.py \
  --task "Search for 'Tongyi Lab'" \
  --web "https://bing.com" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \
  --model "claude-sonnet-4-5-20250929" \
  --output_dir results/custom \
  --image_type base64 \
  --headless \
  --use_css_som
```

### Detailed configuration
Please refer to [**Link**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3.5/browser_use/README.md).

## Evaluation on AndroidWorld
1. Please follow the [official code repository](https://github.com/google-research/android_world?tab=readme-ov-file#installation) to install the Android emulator and necessary dependencies.

2. Install the dependencies.
```
cd Mobile-Agent-v3.5/android_world
pip install -r requirements.txt
```

3. Fill in your vllm service information in the `run_guiowl15.sh` or `run_ma35.sh` script, including api_key, base_url, and model.

4. Run the evaluation.
```
sh run_guiowl15.sh
sh run_ma35.sh
```

## Evaluation on Grounding Benchmarks
1. Please download the images and annotations for the grounding benchmarks from their official repository.

2. Install the dependencies required by the qwen model.
```
pip install qwen_agent
pip install qwen_vl_utils
```

3. Fill in your information in the `run_grounding.sh`, including MODEL_PATH, DS_PATH, SAVE_PATH and EVAL_TYPE. 

4. Run the evaluation.
```bash
cd Mobile-Agent-v3.5/grounding_and_kb
sh run_grounding.sh
```

## Evaluation on Tool Use Benchmark
1. OSWorld-MCP: Please follow the [official code repository](https://github.com/X-PLUG/OSWorld-MCP/blob/main/osworld/run_multienv_gui_owl_1_5.py) to run the evalutation
2. MobileWorld: Please follow the [official code repository](https://github.com/Tongyi-MAI/MobileWorld/blob/main/src/mobile_world/agents/implementations/gui_owl_1_5.py) to run the evalutation


## Evaluation on GUI Knowledge Benchmark
1. Download the images and annotations for the Knowledge Bench from the official repository. Follow the official instructions to draw the GUI actions, and save the annotated images to DS_PATH/AnnotateImage.

2. Install the dependencies required by the qwen model.
```
pip install qwen_agent
pip install qwen_vl_utils
```

3. Fill in your information in the `run_gui_kb.sh`, including MODEL_PATH, DS_PATH, SAVE_PATH and EVAL_TYPE. 

4. Run the evaluation.
```bash
cd Mobile-Agent-v3.5/grounding_and_kb
sh run_gui_kb.sh
```

## Evaluation on Web Benchmark
### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Configure Environment Variables
```bash
# Model API keys (required)
export API_KEY="sk-xxx"          # Agent model API
export OMNI_API_KEY=""
export EVAL_API_KEY="sk-xxx"               # Evaluation model API
```

### Benchmark task
+ **WebArena/VisualWebArena**: Start the corresponding environment services in advance (configure ports per the official docs).
+ **Port mapping**: Ports are preset in the code (see `visualwebarena_url_map` / `webarena_url_map`). To change them, modify `BASE_URL` and the port numbers.
+ **Task data**: Ensure `data/merged_test_raw.json` exists (includes task definitions, login requirements, initial screenshots, etc.).

```bash
# WebVoyager
cd Mobile-Agent-v3.5/web_benchmark
python main_for_eval.py \
  --task "Find out which four teams the NFC North contains in the NFL on ESPN." \
  --web https://www.espn.com/ \
  --output_dir results/WebVoyager \
  --image_type file \
  --task_id "validation_WebVoyager__ESPN--41" \
  --use_css_som \
  --headless
```

## Performance

### End-to-End Online Benchmarks

| Model | OSWorld-Verified | AndroidWorld | OSWorld-MCP | Mobile-World | WindowsAA | WebArena | VisualWebArena | WebVoyager | Online-Mind2Web
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GUI-Owl-1.5-2B-Instruct | 43.5 | 67.9 | 33.0 | 31.3 | 25.8 | - | - | - | - |
| GUI-Owl-1.5-4B-Instruct | 48.2 | 69.8 | 31.7 | 32.3 | 29.4 | - | - | - | - |
| GUI-Owl-1.5-8B-Instruct | 52.3 | 69.0 | 41.8 | 41.8 | 31.7 | 45.7 | 39.4 | 69.9 | 41.7 |
| GUI-Owl-1.5-8B-Thinking | 52.9 | **71.6** | 38.8 | 33.3 | 35.1 | 46.7 | 40.8 | 78.1 | **48.6** |
| GUI-Owl-1.5-32B-Instruct | **56.5** | 69.4 | **47.6** | **46.8** | **44.8** | - | - | - | - |
| GUI-Owl-1.5-32B-Thinking | 56.0 | 68.2 | 43.8 | 42.8 | 44.1 | **48.4** | **46.6** | **82.1** | - |


### Grounding Benchmarks

Please refer to the technical report for detailed results on ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, MMBench-GUI, and more.

## Quick Start

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
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

```

## Citation

If you find GUI-Owl 1.5 useful in your research, please cite:

```bibtex
@article{xu2026mobile,
  title={Mobile-Agent-v3. 5: Multi-platform Fundamental GUI Agents},
  author={Xu, Haiyang and Zhang, Xi and Liu, Haowei and Wang, Junyang and Zhu, Zhaozai and Zhou, Shengjie and Hu, Xuhao and Gao, Feiyu and Cao, Junjie and Wang, Zihua and others},
  journal={arXiv preprint arXiv:2602.16855},
  year={2026}
}
```

## Acknowledgments

GUI-Owl 1.5 is built upon [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL). We thank the Qwen team for their excellent open-source foundation models.
