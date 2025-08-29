# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from unittest import mock
from absl.testing import absltest
from android_world.agents import infer
import google.ai.generativelanguage as glm
import google.generativeai as genai
from google.generativeai.types import answer_types
from google.generativeai.types import generation_types
import requests


class InferTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_post = mock.patch.object(requests, "post").start()
    self.mock_sleep = mock.patch.object(time, "sleep").start()
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    os.environ["GCP_API_KEY"] = "fake_api_key"

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse({
                "candidates": (
                    [{"content": {"parts": [{"text": "fake response"}]}}]
                )
            })
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "fake response")
    self.assertEqual(is_safe, True)

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp_error(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse(
                {"candidates": [{"content": {"parts": []}}]}
            )
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, output = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertIsNone(is_safe)
    self.assertIsNone(output)

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp_no_candidates(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse({"candidates": []})
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, output = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertIsNone(is_safe)
    self.assertIsNone(output)

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp_unsafe(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse({
                "candidates": (
                    [{
                        "content": {"parts": []},
                        "finish_reason": answer_types.FinishReason.SAFETY,
                    }]
                )
            })
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)

  def test_gpt4v(self):
    llm = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09")
    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "fake response"}}]}'
    )
    self.mock_post.return_value = mock_200_response

    text_output, _, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "fake response")

  def test_gpt4v_retry(self):
    gpt4v = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09")

    mock_429_response = requests.Response()
    mock_429_response.status_code = 429
    mock_429_response._content = (
        b'{"error": {"message": "Error 429: rate limit reached."}}'
    )

    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "ok."}}]}'
    )
    self.mock_post.side_effect = [mock_429_response, mock_200_response]

    gpt4v.predict_mm("fake prompt", [])
    self.mock_sleep.assert_called_once()


if __name__ == "__main__":
  absltest.main()
