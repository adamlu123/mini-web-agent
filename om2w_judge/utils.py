import base64
import io
import os
from typing import Optional

import backoff
from openai import APIConnectionError, APIError, OpenAI, RateLimitError


def _uses_max_completion_tokens(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    lowered = model_name.lower()
    return lowered.startswith("o")


def encode_image(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_predication(response, mode):
    if mode in {
        "Autonomous_eval",
        "AgentTrek_eval",
        "WebJudge_Online_Mind2Web_eval",
        "WebJudge_Online_Mind2Web_Sandbox_eval",
        "WebJudge_Online_Mind2Web_Sandbox_eval_ctime",
        "WebJudge_Online_Mind2Web_Sandbox_ThoughtsOnly_eval",
        "WebJudge_general_eval",
    }:
        try:
            if "success" in response.lower().split("status:")[1]:
                return 1
            return 0
        except Exception:
            return 0
    if mode == "WebVoyager_eval":
        return 0 if "FAILURE" in response else 1
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
        del tokenizer, port, endpoint_target_uri, kwargs
        assert os.getenv("OPENAI_API_KEY", api_key) is not None, (
            "must pass on the api_key or set OPENAI_API_KEY in the environment"
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop or []
        self.temperature = temperature
        self.model = model
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.client = OpenAI(api_key=api_key)

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=log_error,
    )
    def generate(self, messages, max_new_tokens=1024, temperature=0, model=None, **kwargs):
        model = model if model else self.model
        create_kwargs = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        if _uses_max_completion_tokens(model):
            create_kwargs["max_completion_tokens"] = max_new_tokens
        else:
            create_kwargs["max_tokens"] = max_new_tokens
            create_kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**create_kwargs)
        return [choice.message.content for choice in response.choices]
