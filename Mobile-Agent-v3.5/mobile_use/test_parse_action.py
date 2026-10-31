"""
Unit tests for parse_action() in run_gui_owl_1_5_for_mobile.py.

Run from this directory:
    python -m unittest test_parse_action -v

The tests only exercise the pure-Python parsing logic. Heavy runtime
dependencies of the run script (openai, PIL, numpy, ...) are stubbed when
they are not installed so the tests can run on a machine without them.
"""

import importlib
import sys
import types
import unittest
from unittest import mock


def _stub_missing(name, **attrs):
    """Insert a dummy module into sys.modules if `name` cannot be imported."""
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_missing("openai", OpenAI=mock.MagicMock())
_stub_missing("qwen_vl_utils", smart_resize=mock.MagicMock())
_stub_missing("numpy")
_stub_missing("PIL")
_stub_missing("PIL.Image", open=mock.MagicMock())
_stub_missing("PIL.ImageDraw")

from run_gui_owl_1_5_for_mobile import parse_action  # noqa: E402


VALID_BLOCK = (
    '{"name": "mobile_use", "arguments": {"action": "click", '
    '"coordinate": [500, 600]}}'
)


class ParseActionTest(unittest.TestCase):
    def test_normal_output(self):
        text = (
            "Action: Tap the search box.\n"
            "<tool_call>\n" + VALID_BLOCK + "\n</tool_call>"
        )
        action = parse_action(text)
        self.assertEqual(action["name"], "mobile_use")
        self.assertEqual(action["arguments"]["action"], "click")
        self.assertEqual(action["arguments"]["coordinate"], [500, 600])

    def test_tags_without_newlines_and_surrounding_text(self):
        text = "Action: tap <tool_call>" + VALID_BLOCK + "</tool_call> done"
        self.assertEqual(parse_action(text)["arguments"]["action"], "click")

    def test_windows_line_endings(self):
        text = "Action: tap\r\n<tool_call>\r\n" + VALID_BLOCK + "\r\n</tool_call>\r\n"
        self.assertEqual(parse_action(text)["arguments"]["action"], "click")

    def test_missing_closing_tag_is_tolerated(self):
        text = "Action: tap\n<tool_call>\n" + VALID_BLOCK
        self.assertEqual(parse_action(text)["arguments"]["action"], "click")

    def test_trailing_text_after_json_is_tolerated(self):
        text = "<tool_call>\n" + VALID_BLOCK + "\nsome trailing text\n</tool_call>"
        self.assertEqual(parse_action(text)["arguments"]["action"], "click")

    def test_nested_braces_in_arguments(self):
        block = (
            '{"name": "mobile_use", "arguments": {"action": "type", '
            '"text": "a {b} c", "extra": {"k": {"v": 1}}}}'
        )
        text = "<tool_call>\n" + block + "\n</tool_call>"
        self.assertEqual(parse_action(text)["arguments"]["text"], "a {b} c")

    def test_empty_output_raises_value_error(self):
        for text in ("", "   \n", None):
            with self.assertRaises(ValueError) as ctx:
                parse_action(text)
            self.assertIn("empty", str(ctx.exception))

    def test_missing_tool_call_tag_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            parse_action("Action: I will tap the button. " + VALID_BLOCK)
        self.assertIn("No <tool_call> block", str(ctx.exception))
        self.assertIn("I will tap", str(ctx.exception))

    def test_broken_json_raises_value_error(self):
        text = '<tool_call>\n{"name": "mobile_use", "arguments": {"action": }\n</tool_call>'
        with self.assertRaises(ValueError) as ctx:
            parse_action(text)
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_json_without_arguments_raises_value_error(self):
        text = '<tool_call>\n{"name": "mobile_use"}\n</tool_call>'
        with self.assertRaises(ValueError) as ctx:
            parse_action(text)
        self.assertIn("arguments", str(ctx.exception))

    def test_arguments_without_action_raises_value_error(self):
        text = '<tool_call>\n{"name": "mobile_use", "arguments": {"text": "x"}}\n</tool_call>'
        with self.assertRaises(ValueError) as ctx:
            parse_action(text)
        self.assertIn("'action'", str(ctx.exception))

    def test_never_raises_index_error(self):
        for text in ("", "<tool_call>", "</tool_call>", "<tool_call></tool_call>", "{}"):
            with self.assertRaises(ValueError):
                parse_action(text)


if __name__ == "__main__":
    unittest.main()
