import json
import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# agents.py pulls in requests / PIL / cv2 through its sibling modules, none of
# which extract_json_object needs. Stub them so this test stays dependency free.
for _name, _attrs in (
    ("MobileAgentE.api", ("encode_image",)),
    ("MobileAgentE.controller", ("tap", "swipe", "type", "back", "home",
                                 "switch_app", "enter", "save_screenshot_to_file")),
    ("MobileAgentE.text_localization", ("ocr",)),
):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        for _attr in _attrs:
            setattr(_stub, _attr, lambda *a, **kw: None)
        sys.modules[_name] = _stub

from MobileAgentE.agents import extract_json_object


class ExtractJsonObjectTest(unittest.TestCase):
    def test_valid_json_is_never_mangled_by_comment_stripping(self):
        """A JSON string value may contain "//" or "# "; it must survive."""
        cases = (
            {"name": "Type", "arguments": {"text": "https://www.taobao.com"}},
            {"name": "Type", "arguments": {"text": "Order # 12345"}},
            {"name": "Tap_Type_and_Enter",
             "arguments": {"x": 500, "y": 120, "text": "https://www.google.com"}},
        )
        for action in cases:
            with self.subTest(action=action):
                self.assertEqual(extract_json_object(json.dumps(action)), action)

    def test_plain_action_still_parses(self):
        action = {"name": "Tap", "arguments": {"x": 100, "y": 200}}
        self.assertEqual(extract_json_object(json.dumps(action)), action)

    def test_trailing_line_comment_is_still_stripped(self):
        action = {"name": "Tap", "arguments": {"x": 100, "y": 200}}
        self.assertEqual(extract_json_object(json.dumps(action) + " // tap it"), action)
        self.assertEqual(extract_json_object(json.dumps(action) + " # tap it"), action)

    def test_json_code_block_still_parses(self):
        action = {"name": "Home", "arguments": None}
        self.assertEqual(
            extract_json_object("```json\n" + json.dumps(action) + "\n```"), action)

    def test_list_payload_still_parses(self):
        self.assertEqual(
            extract_json_object('["Tap_Type_and_Enter"]', json_type="list"),
            ["Tap_Type_and_Enter"])

    def test_unparsable_text_returns_none(self):
        self.assertIsNone(extract_json_object("I cannot do that."))


if __name__ == "__main__":
    unittest.main()
