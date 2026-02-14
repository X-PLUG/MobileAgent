"""
Utility functions for Mobile-Agent-v3.5:
  - ADB device interaction
  - Screenshot annotation
  - Image resizing
  - Message construction for the VLM
  - App name resolution via LLM
"""

import json
import math
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Optional
import abc
import base64
import numpy as np
from io import BytesIO
from openai import OpenAI
from typing import Any, Optional
from qwen_vl_utils import smart_resize

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# ADB Tools
# ---------------------------------------------------------------------------
# Setup instructions (for reference):
#
# 1. Download ADB (Android Debug Bridge) for your OS from:
#    https://developer.android.com/tools/releases/platform-tools
#
# 2. Enable "USB Debugging" (or "ADB Debugging") on your mobile device.
#    a. Developer Options is usually in System Settings.
#    b. If Developer Options is not visible, go to "About Phone" and
#       tap the build number 7 times.
#    c. On Xiaomi HyperOS, also enable "USB Debugging (Security Settings)".
#
# 3. Connect the device to your computer via USB; select "File Transfer" mode.
#
# 4. Install the ADB Keyboard APK on the device:
#    https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk
#
# 5. Set the default input method to "ADB Keyboard" in system settings.
#    Verify by tapping an input field — you should see "ADB Keyboard {No}"
#    at the bottom of the screen.
#
# 6. Test the connection from a terminal:
#      /path/to/adb devices
#    The device list should not be empty.
#    On Windows the binary is adb.exe; on macOS/Linux it is just adb.
#
# 7. On macOS / Linux, grant execute permission first:
#      sudo chmod +x /path/to/adb
#
# 8. Quick sanity check — open any app, then run:
#      /path/to/adb shell am start -a android.intent.action.MAIN \
#          -c android.intent.category.HOME
#    The device should return to the home screen.
#
# 9. Pass the adb_path when instantiating AdbTools. If multiple devices
#    are connected, obtain the device ID via `adb devices` and pass it
#    as the `device` argument.
# ---------------------------------------------------------------------------


class AdbTools:
    """Wrapper around ADB commands for device interaction."""

    def __init__(self, adb_path, device=None):
        self.adb_path = adb_path
        self.device = device
        self._device_flag = f" -s {device} " if device is not None else " "
        self.image_info = None

    # -- helpers ----------------------------------------------------------

    def _run(self, args):
        """Run an ADB command string."""
        cmd = self.adb_path + self._device_flag + args
        subprocess.run(cmd, capture_output=True, text=True, shell=True)

    def _load_image_info(self, path):
        """Cache the width and height of the screenshot."""
        width, height = Image.open(path).size
        self.image_info = (width, height)

    # -- screenshot -------------------------------------------------------

    def get_screenshot(self, image_path, retry_times=3):
        """
        Capture a screenshot from the device and save it to *image_path*.
        Returns True on success, False after exhausting retries.
        """
        device_flag = f" -s {self.device}" if self.device else ""
        cmd = f"{self.adb_path}{device_flag} exec-out screencap -p > {image_path}"

        for _ in range(retry_times):
            subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if os.path.exists(image_path):
                self._load_image_info(image_path)
                return True
            time.sleep(0.1)
        return False

    # -- input actions ----------------------------------------------------

    def click(self, x, y):
        """Tap at screen coordinate (x, y)."""
        self._run(f"shell input tap {x} {y}")

    def long_press(self, x, y, duration=800):
        """Long-press at (x, y) for *duration* milliseconds."""
        self._run(f"shell input swipe {x} {y} {x} {y} {duration}")

    def slide(self, x1, y1, x2, y2, slide_time=800):
        """Swipe from (x1, y1) to (x2, y2) over *slide_time* milliseconds."""
        self._run(f"shell input swipe {x1} {y1} {x2} {y2} {slide_time}")

    def back(self):
        """Press the Back button."""
        self._run("shell input keyevent 4")

    def home(self):
        """Press the Home button to return to the home screen."""
        self._run(
            "shell am start -a android.intent.action.MAIN "
            "-c android.intent.category.HOME"
        )

    def type(self, text):
        """
        Type text via ADB Keyboard (supports CJK and Latin characters).
        Requires ADB Keyboard to be installed on the device.
        """
        escaped_text = text.replace('"', '\\"').replace("'", "\\'")
        command_sequence = [
            "shell ime enable com.android.adbkeyboard/.AdbIME",
            "shell ime set com.android.adbkeyboard/.AdbIME",
            0.1,  # short delay for IME switch
            f'shell am broadcast -a ADB_INPUT_TEXT --es msg "{escaped_text}"',
            0.1,
            "shell ime disable com.android.adbkeyboard/.AdbIME",
        ]

        for item in command_sequence:
            if isinstance(item, (int, float)):
                time.sleep(item)
            else:
                self._run(item.strip())

    # -- package management -----------------------------------------------

    def get_package_name(self, all_packages=False):
        """
        Return a sorted list of installed package names.
        If *all_packages* is False, only third-party packages are listed.
        """
        try:
            flag = "" if all_packages else " -3"
            cmd = f"{self.adb_path}{self._device_flag}shell pm list packages{flag}"
            res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            pkgs = []
            for line in res.stdout.splitlines():
                s = line.strip()
                if not s:
                    continue
                # Strip the "package:" prefix
                if s.startswith("package:"):
                    s = s[len("package:"):]
                # If the line contains "=", the right side is the package name
                if "=" in s:
                    _, s = s.split("=", 1)
                if s:
                    pkgs.append(s)
            return sorted(set(pkgs))
        except Exception as e:
            print(f"[ERROR] Failed to list packages: {e}")
            return []

    def open_app(self, package_name):
        """Launch an app by its package name."""
        self._run(
            f"shell monkey -p {package_name} "
            "-c android.intent.category.LAUNCHER 1"
        )


# ---------------------------------------------------------------------------
# Screenshot annotation
# ---------------------------------------------------------------------------

def annotate_screenshot(image_path, action_parameter, save_path="screenshot_anno.png"):
    """
    Draw action annotations (click dot / swipe arrow) on a screenshot
    and save the result to *save_path*.

    Returns the save path on success, or None if the action type is
    not visualizable.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    action_type = action_parameter.get("action", "")

    if action_type == "click":
        radius = 15
        cx, cy = action_parameter["coordinate"]
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill="red",
            outline="red",
        )
    elif action_type in ("scroll", "swipe"):
        x1, y1 = action_parameter["coordinate"]
        x2, y2 = action_parameter["coordinate2"]
        color = "red"
        arrow_size = 10

        # Draw the line
        draw.line((x1, y1, x2, y2), fill=color, width=2)

        # Compute arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        ax1 = x2 - arrow_size * math.cos(angle - math.pi / 6)
        ay1 = y2 - arrow_size * math.sin(angle - math.pi / 6)
        ax2 = x2 - arrow_size * math.cos(angle + math.pi / 6)
        ay2 = y2 - arrow_size * math.sin(angle + math.pi / 6)
        draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill=color)
    else:
        return None

    image.save(save_path)
    return save_path


# ---------------------------------------------------------------------------
# Smart image resize (Qwen-VL style)
# ---------------------------------------------------------------------------

def smart_resize(height, width, factor=16, min_pixels=None, max_pixels=None):
    """
    Rescale dimensions so that:
      1. Both are divisible by *factor*.
      2. Total pixels is within [min_pixels, max_pixels].
      3. Aspect ratio is preserved as closely as possible.
    """
    IMAGE_MIN_TOKEN_NUM = 4
    IMAGE_MAX_TOKEN_NUM = 16384
    MAX_RATIO = 200

    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "max_pixels must be >= min_pixels."

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"Aspect ratio must be < {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    def _round(n):
        return round(n / factor) * factor

    def _floor(n):
        return math.floor(n / factor) * factor

    def _ceil(n):
        return math.ceil(n / factor) * factor

    h_bar = max(factor, _round(height))
    w_bar = max(factor, _round(width))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor(height / beta)
        w_bar = _floor(width / beta)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil(height * beta)
        w_bar = _ceil(width * beta)

    return h_bar, w_bar


# ---------------------------------------------------------------------------
# VLM message construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: \\"volume_up\\", \\"volume_down\\", \\"power\\", \\"camera\\", \\"clear\\".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `interact`: Resolve the blocking window by interacting with the user.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "interact", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`,and `action=interact`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''


def build_messages(image_path, instruction, history_output, model_name, history_n=4):
    """
    Construct the multi-turn message list for the VLM.

    Args:
        image_path:      Path to the current screenshot.
        instruction:     The user's task instruction.
        history_output:  List of dicts with keys 'output' and 'image'.
        model_name:      Model identifier (affects history summarization).
        history_n:       Number of recent history turns to include as images.

    Returns:
        A list of message dicts suitable for the DashScope API.
    """
    current_step = len(history_output)
    history_start_idx = max(0, current_step - history_n)

    # Summarize early actions (before the image-history window)
    previous_actions = []
    for i in range(history_start_idx):
        if i < len(history_output):
            text = history_output[i]["output"]
            if model_name.endswith(".mem"):
                if "<tool_call>" in text:
                    text = text.split("<tool_call>")[0].strip()
            else:
                if "Action:" in text and "<tool_call>" in text:
                    text = text.split("Action:")[1].split("<tool_call>")[0].strip()
            previous_actions.append(f"Step {i + 1}: {text}")

    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

    # Build date context
    today = datetime.today()
    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    formatted_date = today.strftime("%Y-%m-%d") + " " + weekday_names[today.weekday()]
    date_info = f"Today's date is: {formatted_date}."

    instruction_prompt = (
        f"Please generate the next move according to the UI screenshot, "
        f"instruction and previous actions.\n\n"
        f"Instruction: {date_info}{instruction}\n\n"
        f"Previous actions:\n{previous_actions_str}"
    )

    # Assemble messages
    messages = [
        {
            "role": "system",
            "content": [{"text": SYSTEM_PROMPT}],
        }
    ]

    history_len = min(history_n, len(history_output))
    if history_len > 0:
        for idx, item in enumerate(history_output[-history_n:]):
            if idx == 0:
                messages.append({
                    "role": "user",
                    "content": [
                        {"text": instruction_prompt},
                        {"image": "file://" + item["image"]},
                    ],
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [{"image": "file://" + item["image"]}],
                })
            messages.append({
                "role": "assistant",
                "content": [{"text": item["output"]}],
            })
        messages.append({
            "role": "user",
            "content": [{"image": "file://" + image_path}],
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"text": instruction_prompt},
                {"image": "file://" + image_path},
            ],
        })

    return messages


ERROR_CALLING_LLM = 'Error calling LLM'

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_to_base64(image_path):
    dummy_image = Image.open(image_path)
    MIN_PIXELS=3136
    MAX_PIXELS=10035200
    resized_height, resized_width  = smart_resize(dummy_image.height,
        dummy_image.width,
        factor=28,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,)
    dummy_image = dummy_image.resize((resized_width, resized_height))
    return f"data:image/png;base64,{pil_to_base64(dummy_image)}"

class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""
    @abc.abstractmethod
    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
        text_prompt: Text prompt.

        Returns:
        Text output, is_safe, and raw output.
        """

class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""
    @abc.abstractmethod
    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
        text_prompt: Text prompt.
        images: List of images as numpy ndarray.

        Returns:
        Text output and raw output.
        """

class GUIOwlWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str,
            max_retry: int = 10,
            temperature: float = 0.0,
    ):
        if max_retry <= 0:
            max_retry = 10
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 10)
        self.temperature = temperature
        self.model = model_name
        self.bot = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30
        )

    def convert_messages_format_to_openaiurl(self, messages):
      converted_messages = []
      for message in messages:
          new_content = []
          for item in message['content']:
              if list(item.keys())[0] == 'text':
                  new_content.append({'type': 'text', 'text': item['text']})
              elif list(item.keys())[0] == 'image':
                new_content.append({'type': 'image_url', 'image_url': {'url': image_to_base64(item['image'])}})
          converted_messages.append({'role': message['role'], 'content': new_content})

      return converted_messages
    
    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, messages = None
    ) -> tuple[str, Optional[bool], Any]:
        
        payload = messages
        payload = self.convert_messages_format_to_openaiurl(payload)

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
              chat_completion_from_url = self.bot.chat.completions.create(model=self.model, messages=payload, **{})
              return (chat_completion_from_url.choices[0].message.content, payload, chat_completion_from_url)
            except Exception as e:
                time.sleep(wait_seconds)
                wait_seconds *= 1
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None, None


# ---------------------------------------------------------------------------
# App name resolution via LLM
# ---------------------------------------------------------------------------

def resolve_app_name_via_llm(instruction, app_name_list_str, api_key, base_url, model="qwen-plus"):
    """
    Use an LLM to determine which installed app should be opened
    based on the user instruction.

    Args:
        instruction:        The user's natural-language instruction.
        app_name_list_str:  Comma-separated string of installed app names.
        api_key:            API key for the LLM service.
        base_url:           Base URL for the LLM service.
        model:              Model name to use.

    Returns:
        The resolved app name (str), or empty string if unresolvable.
    """
    from openai import OpenAI

    prompt = f'''Role and Task:
You are an app resolver. Given a natural language instruction and a list of
installed app names on a device, determine which app needs to be opened and
output the corresponding name.

Input:
User instruction: "{instruction}"
Installed apps: "{app_name_list_str}"

Rules:
- Only select from the given app name list; never fabricate names.
- If the instruction explicitly names an app (e.g., "open WeChat"):
  - If that app is in the list, return its name.
  - If not in the list, return an empty string.
  - Do NOT substitute with a similar app.
- If the instruction is generic (e.g., "open a browser / map / camera"):
  - Pick any matching candidate from the list.
  - If no candidate exists, return an empty string.

Output format (important):
Only output JSON, no extra text.
JSON fields:
{{
  "reason": "<brief decision reason (1-2 sentences)>",
  "app": "<string, empty string if unable to determine>"
}}

Examples (for style reference only):

User instruction: "open WeChat"
Installed apps: "WeChat, Taobao"
Output:
{{
  "reason": "WeChat is explicitly named and exists in the list.",
  "app": "WeChat"
}}

User instruction: "open a browser"
Installed apps: "Google Chrome, Firefox"
Output:
{{
  "reason": "No specific browser named; multiple exist; returning Google Chrome.",
  "app": "Google Chrome"
}}

User instruction: "open iQIYI"
Installed apps: "Tencent Video, Bilibili"
Output:
{{
  "reason": "iQIYI is explicitly named but not installed; returning empty.",
  "app": ""
}}

User instruction: "open a map"
Installed apps: "Taobao, WeChat"
Output:
{{
  "reason": "A map app is needed but none is installed; returning empty.",
  "app": ""
}}
'''

    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    res_text = completion.choices[0].message.content
    print(f"[APP RESOLVER] LLM response: {res_text}")

    parsed = _try_parse_json(res_text)
    if parsed and "app" in parsed:
        return parsed["app"]
    return ""


def _try_parse_json(text):
    """Attempt to parse a JSON object from text, handling markdown fences."""
    if not text:
        return None
    try:
        cleaned = text
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        return json.loads(cleaned)
    except Exception as e:
        print(f"[WARN] JSON parse failed: {e}")
        return None
