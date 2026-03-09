SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "browser_use", "name": "browser_use", "description": "Use a browser to interact with web pages and take labeled screenshots.
* This is an interface to a web browser. You can click elements, type into inputs, scroll, wait for loading, go back, etc.
* Each Observation screenshot contains Numerical Labels placed at the TOP LEFT of each Web Element. Use these labels to target elements.
* Some pages may take time to load; you may need to wait and take successive screenshots.
* Avoid clicking near element edges; target the center of the element.
* Execute exactly ONE interaction action per step; do not chain multiple interactions in one call.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `click`: Click a web element by numerical label.
* `type`: Clear existing content in a textbox/input and type content. The system will automatically press ENTER after typing.
* `scroll`: Scroll within WINDOW or within a specific scrollable element/area (by label).
* `select`: Selects a specific option from a menu or dropdown. Use the option text provided in the textual information.
* `wait`: Wait for page processes to finish (default 5 seconds unless specified).
* `go_back`: Go back to the previous page.
* `wikipedia`: Directly jump to the Wikipedia homepage to search for information.
* `answer`: Terminate the current task and output the final answer.", "enum": ["click", "type", "scroll", "select, "wait", "go_back", "wikipedia", "answer"], "type": "string"}, "label": {"description": "Numerical label of the target web element. Required only by `action=click`, `action=type`, `action=scroll`, and `action=select` when scrolling within a specific area. Use string value `WINDOW` to scroll the whole page.", "type": ["integer", "string"]}, "direction": {"description": "Scroll direction. Required only by `action=scroll`.", "enum": ["up", "down"], "type": "string"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "option": {"description": "The option to select. Required only by `action=select`", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=wait` when overriding the default.", "type": "integer"}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
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
- Be brief: one line for Action.
- Do not output anything else outside those two parts.
- Execute ONLY ONE interaction per iteration (one tool call).
- If finishing, use action=answer in the tool call."""

SYSTEM_PROMPT_FALLBACK = """# Tools

You may call one or more functions to assist with the user query.

IMPORTANT: You have reached the maximum allowed steps. You are NOT allowed to perform any further browsing interactions. The ONLY permitted action is `answer`. Do NOT use `click`, `type`, `scroll`, `select`, `wait`, `go_back`, or `wikipedia`.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "browser_use", "name": "browser_use", "description": "Use a browser to interact with web pages and take labeled screenshots.
* This is an interface to a web browser. You can click elements, type into inputs, scroll, wait for loading, go back, etc.
* Each Observation screenshot contains Numerical Labels placed at the TOP LEFT of each Web Element. Use these labels to target elements.
* Some pages may take time to load; you may need to wait and take successive screenshots.
* Avoid clicking near element edges; target the center of the element.
* Execute exactly ONE interaction action per step; do not chain multiple interactions in one call.

STEP LIMIT MODE: Browsing is disabled; you must only answer.", "parameters": {"properties": {"action": {"description": "The action to perform. ONLY `answer` is allowed in this mode.", "enum": ["answer"], "type": "string"}, "text": {"description": "Required only by `action=answer`.", "type": "string"}}, "required": ["action", "text"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
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
- Be brief: one line for Action.
- Do not output anything else outside those two parts.
- Execute ONLY ONE interaction per iteration (one tool call).
- You MUST finish now by using action=answer in the tool call."""