from typing import Union, Tuple, List

from qwen_agent.tools.base import BaseTool, register_tool

    
@register_tool("computer_use")
class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Input a string of text. Use the `clear` parameter to decide whether to overwrite the existing text, and use the `enter` parameter to decide whether the enter key should be pressed after typing the text.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `drag`: Click at a specified (x, y) pixel coordinate on the screen, and drag the cursor to another specified (x2, y2) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "click",
                    "drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "scroll",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "clear": {
                "description": "Assign it to 1 if the text should overwrite the existing text, otherwise assign it to 0. Using this argument clears all text in an element. Required only by `action=type`.",
                "type": "number",
            },
            "enter": {
                "description": "Assign it to 1 if the enter key should be pressed after typing the text, otherwise assign it to 0. Required only by `action=type`.",
                "type": "number",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x2, y2): The x2 (pixels from the left edge) and y2 (pixels from the top edge) coordinates to drag the cursor to. Required only by `action=drag`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. This value should be between -5 and 5. Required only by `action=scroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click"]:
            return self._mouse_click(action)
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str):
        raise NotImplementedError()

    def _key(self, keys: List[str]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _mouse_move(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _scroll(self, pixels: int):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()