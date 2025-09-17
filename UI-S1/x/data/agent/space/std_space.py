RAW_SPACE = {
  "action_space": [
    {
      "action": "key",
      "arguments": ["text"],
      "action_desc": "Perform a key event on the mobile device using adb's `keyevent` syntax."
    },
    {
      "action": "click",
      "arguments": ["coordinate"],
      "action_desc": "Click the point on the screen with specified (x, y) coordinates."
    },
    {
      "action": "long_press",
      "arguments": ["coordinate", "time"],
      "action_desc": "Press the point on the screen with specified (x, y) coordinates for a specified number of seconds."
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
      "action_desc": "Press the specified system button: Back, Home, Menu, or Enter."
    },
    {
      "action": "open",
      "arguments": ["text"],
      "action_desc": "Open an application on the device specified by text."
    },
    {
      "action": "wait",
      "arguments": ["time"],
      "action_desc": "Wait for a specified number of seconds for changes to occur."
    },
    {
      "action": "terminate",
      "arguments": ["status"],
      "action_desc": "Terminate the current task and report its completion status: success or failure."
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
      "argument_desc": "Text input required by actions like `key`, `type`, `answer`, and `open`."
    },
    {
      "argument": "time",
      "enum": None,
      "argument_desc": "The time in seconds required by actions like `long_press` and `wait`."
    },
    {
      "argument": "button",
      "enum": ["Back", "Home", "Menu", "Enter"],
      "argument_desc": "System buttons available for pressing: Back, Home, Menu, or Enter."
    },
    {
      "argument": "status",
      "enum": ["success", "failure"],
      "argument_desc": "The completion status of a terminated task."
    }
  ]
}

ACTION_MAP = {
    _['action']: _ for _ in RAW_SPACE['action_space']
}
ARGS_MAP = {
    _['argument']: _ for _ in RAW_SPACE['argument_space']
}

def make_new_space(lines):
    # 只能用于space裁剪
    NEW_SPACE = {
         "action_space": [],
         "argument_space": []
    }
    ac_names = []
    for line in lines:
        for step in line['steps']:
            action = step['action_content']
            ac_names.append(action['action'])
            
    ac_names = list(set(ac_names))
    ag_names = []
    for name in ac_names:
        NEW_SPACE['action_space'].append(ACTION_MAP[name])
        ag_names.extend(ACTION_MAP[name]['arguments'])
    ag_names = list(set(ag_names))
    for name in ag_names:
        NEW_SPACE['argument_space'].append(ARGS_MAP[name])
    return NEW_SPACE

