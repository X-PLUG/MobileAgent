RAW_SPACE = {
  "action_space": [
    {
      "action": "key",
      "arguments": ["keys"],
      "action_desc": "Performs key down presses on the arguments passed in order, then performs key releases in reverse order."
    },
    {
      "action": "type",
      "arguments": ["text"],
      "action_desc": "Type a string of text on the keyboard."
    },
    {
      "action": "mouse_move",
      "arguments": ["coordinate"],
      "action_desc": "(x, y): Move the cursor to a specified (x, y) pixel coordinate on the screen."
    },
    {
      "action": "left_click",
      "arguments": ["coordinate"],
      "action_desc": "Click the left mouse button at the current cursor position."
    },
    {
      "action": "left_click_drag",
      "arguments": ["coordinate"],
      "action_desc": "Click and drag the cursor (with the left button) to a specified (x, y) pixel coordinate on the screen."
    },
    {
      "action": "right_click",
      "arguments": [],
      "action_desc": "Click the right mouse button at the current cursor position."
    },
    {
      "action": "middle_click",
      "arguments": [],
      "action_desc": "Click the middle mouse button at the current cursor position."
    },
    {
      "action": "double_click",
      "arguments": [],
      "action_desc": "Double-click the left mouse button at the current cursor position."
    },
    {
      "action": "scroll",
      "arguments": ["pixels"],
      "action_desc": "Performs a scroll of the mouse scroll wheel. Positive scrolls up, negative scrolls down."
    },
    {
      "action": "wait",
      "arguments": ["time"],
      "action_desc": "Wait specified seconds for the change to happen."
    },
    {
      "action": "terminate",
      "arguments": ["status"],
      "action_desc": "Terminate the current task and report its completion status."
    },
    {
      "action": "answer",
      "arguments": ["text"],
      "action_desc": "Output the specified answer."
    }
  ],
  "argument_space": [
    {
      "argument": "keys",
      "enum": None,
      "argument_desc": "A list of key names to press in order. Used by 'key' action."
    },
    {
      "argument": "text",
      "enum": None,
      "argument_desc": "Text input required by the 'type' action."
    },
    {
      "argument": "coordinate",
      "enum": None,
      "argument_desc": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move or drag the mouse to. Used by 'mouse_move', 'left_click' and 'left_click_drag' actions."
    },
    {
      "argument": "pixels",
      "enum": None,
      "argument_desc": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Used by 'scroll' action."
    },
    {
      "argument": "time",
      "enum": None,
      "argument_desc": "The seconds to wait. Used by the 'wait' action."
    },
    {
      "argument": "status",
      "enum": ["success", "failure"],
      "argument_desc": "The status of the terminated task. Used by 'terminate' action."
    }
  ]
}
ACTION_MAP = {
    _['action']: _ for _ in RAW_SPACE['action_space']
}
ARGS_MAP = {
    _['argument']: _ for _ in RAW_SPACE['argument_space']
}