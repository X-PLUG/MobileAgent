## Installation & Configuration

### 1. Install Dependencies
```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser (required for first run)
playwright install chromium
# For visual debugging, you can install the full bundle: playwright install
```

### 2. Configure Environment Variables
```bash
# Model API keys (required)
export API_KEY="sk-xxx"          # Agent model API
export OMNI_API_KEY=""
export EVAL_API_KEY="sk-xxx"               # Evaluation model API

# OSS config (optional; only needed when uploading screenshots to OSS)
export OSS_ACCESS_KEY_ID="xxx"
export OSS_ACCESS_KEY_SECRET="xxx"
```
---

## Run Examples

```bash
# Specify the task description and target URL directly (note: you must handle login logic yourself)
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

---

## Key Parameter Reference (main.py)

### 🔑 Task Settings
| **Parameter** | **Type** | **Default** | **Description** |
| --- | --- | --- | --- |
| --task_id | str | "" | Unique task identifier (auto-generates `WebAgent_task_0` if empty) |
| --task | str | "点击地图" | **Required**: task to execute in natural language |
| --web | str | "" | **Required**: target website URL |
| --login | flag | False | Enable login flow (deprecated for now) |
| --rollout_id | str | "0" | Identifier for multiple attempts of the same task (used to distinguish trajectories) |
| --max_iter | int | 100 | Max agent steps (prevents infinite loops) |
| --current_time | str | None | Manually specify timestamp |
| --seed | int | None | Random seed |
| --init_image_path | str | "" | Path to the initial task image |
| --download_dir | str | "downloads" | Directory to store downloaded files |

### 🤖 Agent Settings
| **Parameter** | **Type** | **Default** | **Description** |
| --- | --- | --- | --- |
| --model | str | pre-gui_owl_1_5_8b_... | **Core**: LLM model identifier (custom supported) |
| --base_url | str | "" | Model API endpoint |
| --max_tokens | int | 2048 | Max tokens per generation |
| --temperature | float | 0.6 | Temperature |
| --top_p | int | 0.95 | Top-p sampling threshold |
| --top_k | int | 20 | Top-k sampling |
| --repetition_penalty | float | 1 | Repetition penalty |
| --text_only | flag | False | **Disable vision**: text-only mode (deprecated for now) |
| --max_attached_imgs | int | 2 | Max screenshots attached per turn (deprecated for now) |
| --image_type | str | "base64" | Image transfer method: `file` (path), `base64`, or `oss` |

### 🌐 Browser Settings
| **Parameter** | **Type** | **Default** | **Description** |
| --- | --- | --- | --- |
| --headless | flag | False | **Headless mode**: hide browser window |
| --window_width | int | 1080 | Browser window width (px) |
| --window_height | int | 1440 | Browser window height (px) |
| --highlight_mouse | flag | False | **Visual debugging**: highlight mouse (generates GIF with red circle) |
| --use_css_som | flag | False | Use CSS-filtered SoM |
| --use_omni_som | flag | False | Use Omni SoM (multimodal element recognition) |
| --omni_url | str | "" | **OmniParser service endpoint** (OmniParser project: <https://github.com/mini-solution/OmniParser>); used to send requests to this service when --use_omni_som is enabled |
| --save_accessibility_tree | flag | False | Save accessibility tree per step (deprecated for now) |
| --force_device_scale | flag | False | Force device pixel ratio (deprecated for now) |
| --fix_box_color | flag | False | Fix annotation box colors |
| --keep_user_info | flag | False | Load and persist Cookies/LocalStorage (login across sessions) |

### 📊 Evaluation Settings
| **Parameter** | **Type** | **Default** | **Description** |
| --- | --- | --- | --- |
| --eval | flag | False | Run evaluation after the task finishes |
| --eval_only | flag | False | **Evaluation-only mode**: skip execution and evaluate existing results |
| --eval_mode | str | WebJudge_Online_... | Evaluation strategy |
| --eval_model | str | "o4-mini-2025-04-16" | Dedicated evaluation model (must match eval mode) |
| --eval_score_threshold | int | 3 | Threshold used in the Online WebJudge method |