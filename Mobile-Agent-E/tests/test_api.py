import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MobileAgentE.api import build_api_url, inference_chat, track_usage


class ApiUrlTest(unittest.TestCase):
    def test_region_and_protocol_matrix(self):
        cases = (
            ("https://api.minimax.io/v1", "openai", "https://api.minimax.io/v1/chat/completions"),
            ("https://api.minimax.io/anthropic", "anthropic", "https://api.minimax.io/anthropic/v1/messages"),
            ("https://api.minimaxi.com/v1", "openai", "https://api.minimaxi.com/v1/chat/completions"),
            ("https://api.minimaxi.com/anthropic", "anthropic", "https://api.minimaxi.com/anthropic/v1/messages"),
        )
        for base_url, api_protocol, expected in cases:
            with self.subTest(base_url=base_url, api_protocol=api_protocol):
                self.assertEqual(build_api_url(base_url, api_protocol), expected)


class UsageTrackingTest(unittest.TestCase):
    def test_m3_usd_pricing_tiers(self):
        cases = (
            (512000, "standard", 0.3, 1.2),
            (512001, "standard", 0.6, 2.4),
            (512000, "priority", 0.45, 1.8),
            (512001, "priority", 0.9, 3.6),
        )
        for input_tokens, service_tier, input_rate, output_rate in cases:
            response = {
                "model": "MiniMax-M3",
                "service_tier": service_tier,
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": 100},
            }
            usage = track_usage(response, api_key="test-key")
            self.assertAlmostEqual(usage["prompt_token_price"], input_tokens * input_rate / 1000000)
            self.assertAlmostEqual(usage["completion_token_price"], 100 * output_rate / 1000000)

    def test_m27_cache_pricing(self):
        response = {
            "model": "MiniMax-M2.7",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 300,
            },
        }
        usage = track_usage(response, api_key="test-key")
        self.assertAlmostEqual(usage["prompt_token_price"], 0.0003)
        self.assertAlmostEqual(usage["completion_token_price"], 0.00012)
        self.assertAlmostEqual(usage["cache_read_token_price"], 0.000012)
        self.assertAlmostEqual(usage["cache_write_token_price"], 0.0001125)

    def test_m3_openai_cache_read_pricing(self):
        response = {
            "model": "MiniMax-M3",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 200},
            },
        }
        usage = track_usage(response, api_key="test-key")
        self.assertAlmostEqual(usage["prompt_token_price"], 0.00024)
        self.assertAlmostEqual(usage["cache_read_token_price"], 0.000012)
        self.assertIsNone(usage["cache_write_token_price"])

    def test_china_pricing_uses_cny(self):
        response = {
            "model": "MiniMax-M3",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        }
        usage = track_usage(response, api_key="test-key", price_currency="CNY")
        self.assertEqual(usage["price_currency"], "CNY")
        self.assertAlmostEqual(usage["prompt_token_price"], 0.0021)
        self.assertAlmostEqual(usage["completion_token_price"], 0.00084)


class RequestCaptureTest(unittest.TestCase):
    @patch("MobileAgentE.api.requests.post")
    def test_anthropic_request(self, post):
        post.return_value.json.return_value = {
            "id": "response-id",
            "model": "MiniMax-M3",
            "content": [
                {"type": "thinking", "thinking": "internal"},
                {"type": "text", "text": "done"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        chat = [
            ("system", [{"type": "text", "text": "system"}]),
            ("user", [
                {"type": "text", "text": "inspect"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4", "fps": 1}},
            ]),
        ]
        api_url = build_api_url("https://api.minimax.io/anthropic", "anthropic")

        result = inference_chat(
            chat,
            "MiniMax-M3",
            api_url,
            "test-key",
            api_protocol="anthropic",
            service_tier="standard",
            thinking="disabled",
        )

        self.assertEqual(result, "done")
        self.assertEqual(post.call_args.args[0], "https://api.minimax.io/anthropic/v1/messages")
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image")
        self.assertEqual(payload["messages"][0]["content"][2]["type"], "video")

    @patch("MobileAgentE.api.requests.post")
    def test_openai_request(self, post):
        post.return_value.json.return_value = {
            "id": "response-id",
            "model": "MiniMax-M2.7",
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        chat = [("user", [{"type": "text", "text": "inspect"}])]
        api_url = build_api_url("https://api.minimaxi.com/v1", "openai")

        result = inference_chat(chat, "MiniMax-M2.7", api_url, "test-key", api_protocol="openai")

        self.assertEqual(result, "done")
        self.assertEqual(post.call_args.args[0], "https://api.minimaxi.com/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()
