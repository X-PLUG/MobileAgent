RAW_SPACE = {
  "action_space": [
    {
      "action": "click",
      "arguments": ["coordinate"],
      "action_desc": "Click the point on the screen with specified (x, y) coordinates."
    },
    {
      "action": "long_press",
      "arguments": ["coordinate"],
      "action_desc": "Long press on the position (x, y) on the screen."
    },
    {
      "action": "swipe",
      "arguments": ["coordinate", "coordinate2"],
      "action_desc": "Swipe from starting point with specified (x, y) coordinates to endpoint with specified (x2, y2) coordinates."
    },
    {
      "action": "type",
      "arguments": ["text"],
      "action_desc": "Input the specified text into the activated input box."
    },
    {
      "action": "answer",
      "arguments": ["text"],
      "action_desc": "Output the specified answer."
    },
    {
      "action": "system_button",
      "arguments": ["button"],
      "action_desc": "Press a system button, including back, home, and enter."
    },
    {
      "action": "open",
      "arguments": ["text"],
      "action_desc": "Open an application on the device specified by text."
    }
  ],
  "argument_space": [
    {
      "argument": "coordinate",
      "enum": None,
      "argument_desc": "(x, y): The x and y pixels coordinates from the left and top edges."
    },
    {
      "argument": "coordinate2",
      "enum": None,
      "argument_desc": "(x, y): The x and y pixels coordinates from the left and top edges for the endpoint of a swipe."
    },
    {
      "argument": "text",
      "enum": None,
      "argument_desc": "Text input required by actions like `type`, `answer`, and `open`."
    },
    {
      "argument": "button",
      "enum": ["Back", "Home", "Enter"],
      "argument_desc": "System buttons available for pressing: Back, Home, or Enter."
    }
  ]
}

ACTION_MAP = {
    _['action']: _ for _ in RAW_SPACE['action_space']
}
ARGS_MAP = {
    _['argument']: _ for _ in RAW_SPACE['argument_space']
}