import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image


class PCModelResponseTest(unittest.TestCase):
    def run_step(self, message, *, failed=False):
        utils = types.ModuleType('utils')
        for name in ('ComputerTools', 'StepPopup', 'annotate_screenshot',
                     'build_messages', 'extract_tool_calls', 'get_output_dir',
                     'sanitize_filename', 'smart_resize', 'GUIOwlWrapper'):
            setattr(utils, name, Mock())
        utils.get_output_dir.return_value = '/unused-fixture'
        utils.sanitize_filename.return_value = 'task'
        utils.smart_resize.return_value = (600, 800)
        utils.extract_tool_calls.return_value = []
        response = None if failed else types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
        utils.GUIOwlWrapper.return_value.predict_mm.return_value = (
            ('Error calling LLM', None, None) if failed else ('answer', [], response)
        )
        pil = types.ModuleType('PIL')
        pil.Image = Mock()
        pil.Image.open.return_value = types.SimpleNamespace(width=800, height=600)
        path = Path(__file__).with_name('run_gui_owl_1_5_for_pc.py')
        spec = importlib.util.spec_from_file_location('pc_runner_under_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'utils': utils, 'PIL': pil}):
            spec.loader.exec_module(module)
        args = types.SimpleNamespace(api_key='unused', base_url='http://unused.invalid',
                                     model='fixture', instruction='fixture', max_steps=1)
        with patch.object(module, 'parse_args', return_value=args), \
             patch.object(module.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        utils.GUIOwlWrapper.assert_called_once_with(args.api_key, args.base_url, args.model)
        utils.GUIOwlWrapper.return_value.predict_mm.assert_called_once()
        return utils.extract_tool_calls.call_args.args[0]

    def test_instruct_response_without_reasoning(self):
        self.assertEqual(self.run_step(types.SimpleNamespace(content='answer')), 'answer')

    def test_empty_reasoning(self):
        self.assertEqual(self.run_step(types.SimpleNamespace(content='answer', reasoning_content=None)), 'answer')

    def test_failed_wrapper_response_keeps_output(self):
        self.assertEqual(self.run_step(None, failed=True), 'Error calling LLM')

    def test_thinking_response(self):
        self.assertEqual(self.run_step(types.SimpleNamespace(content='answer', reasoning_content='reason')),
                         '<thinking>\nreason\n</thinking>answer')


class PCWrapperIntegrationTest(unittest.TestCase):
    def test_real_wrapper_and_message_conversion_use_one_response(self):
        output = '<tool_call>{"name": "computer_use", "arguments": {"action": "stop"}}</tool_call>'
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=output, reasoning_content="done"))])
        client = Mock()
        client.chat.completions.create.return_value = response
        sdk = types.ModuleType("openai")
        sdk.OpenAI = Mock(return_value=client)
        directory = Path(__file__).parent
        spec = importlib.util.spec_from_file_location("utils", directory / "utils.py")
        utils = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"openai": sdk,
                                      "pyautogui": types.ModuleType("pyautogui"),
                                      "pyperclip": types.ModuleType("pyperclip")}):
            spec.loader.exec_module(utils)
        spec = importlib.util.spec_from_file_location("pc_integrated", directory / "run_gui_owl_1_5_for_pc.py")
        runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"utils": utils}):
            spec.loader.exec_module(runner)
        tools = Mock()
        def screenshot(path):
            Image.new("RGB", (800, 600), "white").save(path)
            return True
        tools.get_screenshot.side_effect = screenshot
        with tempfile.TemporaryDirectory() as temp:
            args = types.SimpleNamespace(api_key="fixture-key", base_url="http://unused.invalid/v1",
                                         model="fixture-model", instruction="fixture", max_steps=3)
            with patch.object(runner, "parse_args", return_value=args), \
                 patch.object(runner, "ComputerTools", return_value=tools), \
                 patch.object(runner, "StepPopup"), \
                 patch.object(runner, "get_output_dir", return_value=temp), \
                 patch.object(runner.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
        sdk.OpenAI.assert_called_once_with(api_key=args.api_key, base_url=args.base_url, timeout=30)
        client.chat.completions.create.assert_called_once()
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], args.model)
        self.assertTrue(any(item.get("image_url", {}).get("url", "").startswith("data:image/")
                            for msg in request["messages"] for item in msg["content"]))
        tools.get_screenshot.assert_called_once()


if __name__ == '__main__':
    unittest.main()
