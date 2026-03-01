import ast
import json
import math
import os
import re
import sys
import textwrap
import time
import abc
import base64
import numpy as np
from io import BytesIO
from openai import OpenAI
from typing import Any, Optional

import pyautogui
import pyperclip
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Computer interaction tools
# ---------------------------------------------------------------------------

class ComputerTools:
    """Cross-platform wrapper for desktop GUI automation via pyautogui."""

    def __init__(self):
        self.image_info = None

    def _load_image_info(self, path):
        """Cache the width and height of the latest screenshot."""
        width, height = Image.open(path).size
        self.image_info = (width, height)

    # -- screenshot -------------------------------------------------------

    def get_screenshot(self, image_path, retry_times=3):
        """
        Capture a desktop screenshot and save to *image_path*.
        Returns True on success, False after exhausting retries.
        """
        if os.path.exists(image_path):
            os.remove(image_path)

        for _ in range(retry_times):
            screenshot = pyautogui.screenshot()
            screenshot.save(image_path)
            if os.path.exists(image_path):
                self._load_image_info(image_path)
                return True
            time.sleep(0.1)
        return False

    # -- window management ------------------------------------------------

    def reset(self):
        """Minimize all windows and show the desktop."""
        pyautogui.hotkey("win", "d")

    # -- keyboard actions -------------------------------------------------

    def press_key(self, keys):
        """
        Press one or more keys. If multiple keys are given, they are
        pressed as a hotkey combination.

        Args:
            keys: A single key string or a list of key strings.
        """
        if isinstance(keys, list):
            cleaned = []
            for key in keys:
                if isinstance(key, str):
                    # Strip any wrapper artifacts like "keys=[" or trailing "]"
                    key = key.strip()
                    for prefix in ("keys=[", "['", '["'):
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                    for suffix in ("]", "']", '"]'):
                        if key.endswith(suffix):
                            key = key[: -len(suffix)]
                    key = key.strip()

                    # Normalize arrow key names
                    arrow_map = {
                        "arrowleft": "left",
                        "arrowright": "right",
                        "arrowup": "up",
                        "arrowdown": "down",
                    }
                    key = arrow_map.get(key, key)
                    cleaned.append(key)
                else:
                    cleaned.append(key)
            keys = cleaned
        else:
            keys = [keys]

        if len(keys) > 1:
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(keys[0])

    def type(self, text):
        """
        Type text by copying to clipboard and pasting.
        This approach supports CJK and special characters.
        """
        pyperclip.copy(text)
        pyautogui.keyDown("ctrl")
        pyautogui.keyDown("v")
        pyautogui.keyUp("v")
        pyautogui.keyUp("ctrl")

    # -- app launching ----------------------------------------------------

    def open_app(self, app_name, wait=0.5):
        """
        Open an application by name using the OS search mechanism.
        Supports Windows, macOS, and Linux.
        """
        if app_name == "File Explorer":
            app_name = "文件资源管理器"

        if sys.platform == "win32":
            pyautogui.hotkey("winleft", "s")
            time.sleep(wait)
            pyperclip.copy(app_name)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.3)
            pyautogui.press("enter")
            time.sleep(0.5)

        elif sys.platform == "darwin":
            pyautogui.hotkey("command", "space")
            time.sleep(wait)
            pyperclip.copy(app_name)
            pyautogui.hotkey("command", "v")
            time.sleep(0.3)
            pyautogui.press("enter")

        else:
            # Linux — attempt Alt+F2 run dialog
            pyautogui.hotkey("alt", "f2")
            time.sleep(wait)
            pyperclip.copy(app_name)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.3)
            pyautogui.press("enter")

    # -- mouse actions ----------------------------------------------------

    def mouse_move(self, x, y):
        """Move the mouse cursor to absolute coordinate (x, y)."""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.moveTo(x, y)

    def left_click(self, x, y):
        """Left-click at coordinate (x, y)."""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.click()

    def left_click_drag(self, x, y):
        """Click and drag from the current position to (x, y)."""
        pyautogui.dragTo(x, y, duration=0.5)
        pyautogui.moveTo(x, y)

    def right_click(self, x, y):
        """Right-click at coordinate (x, y)."""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.rightClick()

    def middle_click(self, x, y):
        """Middle-click at coordinate (x, y)."""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.middleClick()

    def double_click(self, x, y):
        """Double-click at coordinate (x, y)."""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.doubleClick()

    def triple_click(self, x, y):
        """Triple-click at coordinate (x, y)."""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.tripleClick()

    def scroll(self, pixels):
        """
        Scroll the mouse wheel.
        Positive values scroll up, negative values scroll down.
        """
        pyautogui.scroll(pixels)


# ---------------------------------------------------------------------------
# Step popup (blocking, with countdown)
# ---------------------------------------------------------------------------

class StepPopup:
    """Topmost popup window for displaying step information."""

    @staticmethod
    def show_blocking(
        title,
        text,
        image_path=None,
        timeout_sec=5,
        width=960,
        height=540,
        pos=None,
        image_ratio=0.55,
    ):
        """
        Show a blocking, always-on-top popup with an image on top
        and scrollable text below.

        Args:
            title:       Window title.
            text:        Body text to display.
            image_path:  Optional path to an image to show.
            timeout_sec: Auto-close after this many seconds.
            width:       Window width in pixels.
            height:      Window height in pixels.
            pos:         (x, y) position tuple, or None for centered.
            image_ratio: Fraction of content area used for the image (0.4–0.75).
        """
        import tkinter as tk
        from PIL import ImageTk

        root = tk.Tk()
        root.title(title)
        root.attributes("-topmost", True)
        root.resizable(False, False)

        # Window positioning
        if pos is None:
            root.update_idletasks()
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            x = int((sw - width) / 2)
            y = int(sh * 0.12)
        else:
            x, y = pos
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Main container
        frm = tk.Frame(root, bg="#1f1f1f")
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Title label
        lbl_title = tk.Label(
            frm, text=title, bg="#1f1f1f", fg="#ffffff",
            font=("Segoe UI", 12, "bold"), anchor="w",
        )
        lbl_title.pack(fill="x", pady=(0, 6))

        # Compute available heights for image and text areas
        content_h = height - 90
        image_h = max(80, int(content_h * image_ratio))
        text_h = max(60, content_h - image_h)

        # Image area (fixed height, scaled to fit)
        image_frame = tk.Frame(frm, bg="#1f1f1f", height=image_h)
        image_frame.pack(fill="x")
        image_frame.pack_propagate(False)

        img_label = tk.Label(image_frame, bg="#1f1f1f")
        img_label.pack(fill="both", expand=True)

        photo_ref = {"img": None}  # prevent garbage collection

        def render_image():
            if not image_path:
                img_label.config(text="(No image)", fg="#bbbbbb")
                return
            try:
                with Image.open(image_path) as im_src:
                    img = im_src.convert("RGB")
                avail_w = width - 24
                avail_h = image_h - 10
                iw, ih = img.size
                ratio = min(avail_w / iw, avail_h / ih)
                new_w = max(1, int(iw * ratio))
                new_h = max(1, int(ih * ratio))
                img_resized = img.resize((new_w, new_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                img_label.config(image=photo)
                photo_ref["img"] = photo
            except Exception as e:
                img_label.config(text=f"Image load failed: {e}", fg="#ff6666")

        render_image()

        # Text area (scrollable)
        text_frame = tk.Frame(frm, bg="#1f1f1f", height=text_h)
        text_frame.pack(fill="both", expand=True, pady=(6, 0))
        text_frame.pack_propagate(False)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        txt = tk.Text(
            text_frame, wrap="word", bg="#262626", fg="#e8e8e8",
            insertbackground="#e8e8e8", relief="flat",
        )
        txt.pack(side="left", fill="both", expand=True)
        txt.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=txt.yview)

        txt.insert("1.0", text or "")
        txt.config(state="disabled")

        # Bottom bar: countdown + close button
        bottom = tk.Frame(frm, bg="#1f1f1f")
        bottom.pack(fill="x", pady=(6, 0))
        countdown_var = tk.StringVar()

        def close():
            try:
                root.destroy()
            except Exception:
                pass

        def on_key(event):
            if event.keysym in ("Escape", "Return"):
                close()

        root.bind("<Escape>", on_key)
        root.bind("<Return>", on_key)

        lbl_count = tk.Label(
            bottom, textvariable=countdown_var,
            bg="#1f1f1f", fg="#bbbbbb", font=("Segoe UI", 10),
        )
        lbl_count.pack(side="left")

        btn = tk.Button(bottom, text="Close", command=close)
        btn.pack(side="right")

        remaining = [timeout_sec]

        def tick():
            remaining[0] -= 1
            if remaining[0] <= 0:
                close()
            else:
                countdown_var.set(
                    f"Auto-close in {remaining[0]}s (Esc/Enter to dismiss)"
                )
                root.after(1000, tick)

        countdown_var.set(
            f"Auto-close in {timeout_sec}s (Esc/Enter to dismiss)"
        )
        root.after(1000, tick)

        root.mainloop()


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def format_step_text(thought, action_list, explanation, max_width=88):
    """Format step details (thought / actions / explanation) for display."""

    def wrap(s):
        if isinstance(s, str):
            return "\n".join(textwrap.wrap(s, width=max_width))
        return str(s)

    parts = [f"Thought:\n{wrap(thought or '')}"]
    parts.append("\nActions:")
    if isinstance(action_list, list):
        for i, a in enumerate(action_list, 1):
            parts.append(f"  {i}. {json.dumps(a, ensure_ascii=False)}")
    else:
        parts.append(f"  {wrap(str(action_list))}")
    parts.append(f"\nExplanation:\n{wrap(explanation or '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Smart image resize (Qwen-VL style)
# ---------------------------------------------------------------------------

def smart_resize(
    height, width,
    factor=28,
    min_pixels=56 * 56,
    max_pixels=14 * 14 * 4 * 1280,
    max_long_side=8192,
):
    """
    Rescale dimensions so that:
      1. Both are divisible by *factor*.
      2. Total pixels is within [min_pixels, max_pixels].
      3. Longest side does not exceed *max_long_side*.
      4. Aspect ratio is preserved as closely as possible.
    """

    def _round(n):
        return round(n / factor) * factor

    def _floor(n):
        return math.floor(n / factor) * factor

    def _ceil(n):
        return math.ceil(n / factor) * factor

    if height < 2 or width < 2:
        raise ValueError(
            f"height ({height}) and width ({width}) must be >= 2"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Aspect ratio must be < 200, "
            f"got {max(height, width) / min(height, width)}"
        )

    # Clamp longest side
    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height = int(height / beta)
        width = int(width / beta)

    h_bar = _round(height)
    w_bar = _round(width)

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
# Screenshot annotation
# ---------------------------------------------------------------------------

def annotate_screenshot(image_path, action_parameter, save_path="screenshot_anno.png"):
    """
    Draw action annotations (click dot / drag arrow) on a screenshot
    and save the result to *save_path*.

    Handles two cases:
      - 'coordinate' only: draws a red dot (click).
      - 'coordinate1' + 'coordinate2': draws a red arrow (drag/swipe).

    Returns the save path on success, or None if no coordinates found.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    if "coordinate" in action_parameter:
        # Single-point action (click)
        radius = 15
        cx, cy = action_parameter["coordinate"]
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill="red",
            outline="red",
        )
    elif "coordinate1" in action_parameter and "coordinate2" in action_parameter:
        # Two-point action (drag / swipe)
        x1, y1 = action_parameter["coordinate1"]
        x2, y2 = action_parameter["coordinate2"]
        color = "red"
        arrow_size = 10

        # Draw the line
        draw.line((x1, y1, x2, y2), fill=color, width=2)

        # Compute and draw arrowhead
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
# VLM message construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    '# Tools\n\n'
    'You may call one or more functions to assist with the user query.\n\n'
    'You are provided with function signatures within <tools></tools> XML tags:\n'
    '<tools>\n'
    '{"type": "function", "function": {"name": "computer_use", '
    '"description": "Use a mouse and keyboard to interact with a computer, '
    'and take screenshots.\\n'
    '* This is an interface to a desktop GUI. You do not have access to a '
    'terminal or applications menu. You must click on desktop icons to start '
    'applications.\\n'
    '* Some applications may take time to start or process actions, so you '
    'may need to wait and take successive screenshots to see the results of '
    'your actions. E.g. if you click on Firefox and a window doesn\'t open, '
    'try wait and taking another screenshot.\\n'
    '* The screen\'s resolution is 1000x1000.\\n'
    '* Make sure to click any buttons, links, icons, etc with the cursor tip '
    'in the center of the element. Don\'t click boxes on their edges unless '
    'asked.", '
    '"parameters": {"properties": {"action": {"description": '
    '"The action to perform. The available actions are:\\n'
    '* `key`: Performs key down presses on the arguments passed in order, '
    'then performs key releases in reverse order.\\n'
    '* `type`: Type a string of text on the keyboard.\\n'
    '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate '
    'on the screen.\\n'
    '* `left_click`: Click the left mouse button at a specified (x, y) pixel '
    'coordinate on the screen.\\n'
    '* `left_click_drag`: Click and drag the cursor to a specified (x, y) '
    'pixel coordinate on the screen.\\n'
    '* `right_click`: Click the right mouse button at a specified (x, y) '
    'pixel coordinate on the screen.\\n'
    '* `middle_click`: Click the middle mouse button at a specified (x, y) '
    'pixel coordinate on the screen.\\n'
    '* `double_click`: Double-click the left mouse button at a specified '
    '(x, y) pixel coordinate on the screen.\\n'
    '* `triple_click`: Triple-click the left mouse button at a specified '
    '(x, y) pixel coordinate on the screen.\\n'
    '* `scroll`: Performs a scroll of the mouse scroll wheel.\\n'
    '* `hscroll`: Performs a horizontal scroll.\\n'
    '* `wait`: Wait specified seconds for the change to happen.\\n'
    '* `terminate`: Terminate the current task and report its completion '
    'status.\\n'
    '* `answer`: Answer a question.\\n'
    '* `interact`: Resolve the blocking window by interacting with the user.", '
    '"enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", '
    '"right_click", "middle_click", "double_click", "triple_click", "scroll", '
    '"hscroll", "wait", "terminate", "answer", "interact"], "type": "string"}, '
    '"keys": {"description": "Required only by `action=key`.", '
    '"type": "array"}, '
    '"text": {"description": "Required only by `action=type`, `action=answer` '
    'and `action=interact`.", "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x (pixels from the left edge) '
    'and y (pixels from the top edge) coordinates to move the mouse to. '
    'Required only by `action=mouse_move` and `action=left_click_drag`.", '
    '"type": "array"}, '
    '"pixels": {"description": "The amount of scrolling to perform. Positive '
    'values scroll up, negative values scroll down. Required only by '
    '`action=scroll` and `action=hscroll`.", "type": "number"}, '
    '"time": {"description": "The seconds to wait. Required only by '
    '`action=wait`.", "type": "number"}, '
    '"status": {"description": "The status of the task. Required only by '
    '`action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, '
    '"required": ["action"], "type": "object"}}}\n'
    '</tools>\n\n'
    'For each function call, return a json object with function name and '
    'arguments within <tool_call></tool_call> XML tags:\n'
    '<tool_call>\n'
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    '</tool_call>\n\n'
    '# Response format\n\n'
    'Response format for every step:\n'
    '1) Action: a short imperative describing what to do in the UI.\n'
    '2) A single <tool_call>...</tool_call> block containing only the JSON: '
    '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
    'Rules:\n'
    '- Output exactly in the order: Action, <tool_call>.\n'
    '- Be brief: one for Action.\n'
    '- Do not output anything else outside those two parts.\n'
    '- If finishing, use action=terminate in the tool call.'
)


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
            if "Action:" in text and "<tool_call>" in text:
                text = text.split("Action:")[1].split("<tool_call>")[0].strip()
            previous_actions.append(f"Step {i + 1}: {text}")

    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

    instruction_prompt = (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
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


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------

def extract_tool_calls(text):
    """
    Extract all JSON objects from <tool_call>...</tool_call> blocks.

    Returns a list of parsed dicts. Blocks that fail to parse are skipped
    with a warning.
    """
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
    blocks = pattern.findall(text)

    actions = []
    for blk in blocks:
        blk = blk.strip()
        try:
            actions.append(ast.literal_eval(blk))
        except (ValueError, SyntaxError) as e:
            print(f"[WARN] Failed to parse tool_call block: {e} | snippet: {blk[:80]}...")
    return actions


# ---------------------------------------------------------------------------
# Output directory helper
# ---------------------------------------------------------------------------

def get_output_dir(subdir="anno"):
    """
    Create and return an output directory for annotated screenshots.
    Prefers ~/Desktop/<subdir>; falls back to ./<subdir>.
    """
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    base_dir = desktop if os.path.isdir(desktop) else os.getcwd()
    out_dir = os.path.join(base_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sanitize_filename(name):
    """Replace non-alphanumeric characters with underscores for safe filenames."""
    return "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name
    ).strip()


ERROR_CALLING_LLM = 'Error calling LLM'

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_to_base64(image_path):
    if image_path.startswith("file://"):
        image_path = image_path[7:]
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

