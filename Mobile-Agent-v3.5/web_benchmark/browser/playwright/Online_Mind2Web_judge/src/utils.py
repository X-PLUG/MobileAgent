import base64
import io
import os

from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AsyncOpenAI,   # 注意这里用异步客户端
)
import backoff


def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_predication(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except Exception:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except Exception:
            return 0
    elif mode == "WebVoyager_eval":
        if "FAILURE" in response:
            return 0
        else:
            return 1
    elif mode == "WebJudge_Online_Mind2Web_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except Exception:
            return 0
    elif mode == "WebJudge_Online_Mind2Web_eval_env_error":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            elif "web error" in response.lower().split("status:")[1]:
                return -1
            else:
                return 0
        except Exception:
            return 0
    elif mode == "WebJudge_general_eval":
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            else:
                return 0
        except Exception:
            return 0
    else:
        raise ValueError(f"Unknown mode: {mode}")


class OpenaiEngine:
    def __init__(
        self,
        api_key=None,
        stop=None,
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri="",
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT engine (async version).

        Args:
            api_key (str | list, optional): Auth key from OpenAI. Defaults to env OPENAI_API_KEY.
            stop (list, optional): Stop tokens. Defaults to [].
            rate_limit (int, optional): Max req per minute. Defaults to -1 (no limit).
            model (str, optional): Model name.
        """
        if stop is None:
            stop = []

        assert (
            os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass api_key or set OPENAI_API_KEY"

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")

        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minimum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)

        # 使用 AsyncOpenAI 客户端；这里保持你之前的 dashscope 兼容 base_url
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @staticmethod
    def log_error(details):
        print(
            f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}"
        )

    # backoff 的异步版本：use async def + on_exception 一样可以用
    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=50,
        on_backoff=log_error,
    )
    async def generate(
        self,
        messages,
        max_new_tokens=2048*2,
        temperature=1,
        model=None,
        **kwargs,
    ):
        """异步调用 OpenAI / DashScope 兼容接口，返回 list[str]."""

        model = model if model else self.model

        resp = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        # 返回与你原来一样的结构：list[str]
        # print(resp)
        return [choice.message.content for choice in resp.choices]
