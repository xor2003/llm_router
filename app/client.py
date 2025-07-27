import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict

import google.api_core.exceptions
from openai import APIStatusError, AsyncOpenAI

from app.config import BackendModel
from app.router import LLMRouter


class BaseGenerativeClient(ABC):
    """Abstract base class for a generative AI client."""

    @abstractmethod
    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a non-streaming response."""

    @abstractmethod
    async def generate_stream(
        self,
        payload: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a streaming response."""


class GeminiClient(BaseGenerativeClient):
    def __init__(self, model_id: str, model_name: str, api_key: str):
        import google.generativeai as genai

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_id = model_id

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def _translate_payload_to_gemini(self, payload: Dict[str, Any]) -> list[Dict]:
        contents = []
        for msg in payload.get("messages", []):
            role = "user" if msg["role"] == "user" else "model"
            parts = []
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part["text"])
                    elif isinstance(part, str):
                        parts.append(part)
            elif content:
                parts.append(content)

            if parts:
                contents.append(
                    {"role": role, "parts": [{"text": text} for text in parts]},
                )
        return contents

    def _translate_gemini_chunk_to_openai(self, chunk) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-chunk-{time.time()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": chunk.text},
                    "finish_reason": None,
                },
            ],
        }

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        contents = self._translate_payload_to_gemini(payload)
        response = await self.client.generate_content_async(contents)
        return {
            "id": f"chatcmpl-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def generate_stream(
        self,
        payload: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        contents = self._translate_payload_to_gemini(payload)
        stream = await self.client.generate_content_async(contents, stream=True)

        async for chunk in stream:
            # НАДЕЖНАЯ ПРОВЕРКА: Игнорируем пустые чанки, которые вызывают ошибку
            if not chunk.parts:
                self.logger.debug(
                    f"Ignoring empty/metadata chunk from Gemini stream: {chunk.candidates[0].finish_reason}",
                )
                continue

            yield self._translate_gemini_chunk_to_openai(chunk)


class OpenAIClient(BaseGenerativeClient):
    def __init__(self, model_id: str, model_name: str, api_key: str, api_base: str):
        self.model_id = model_id
        self.logger = logging.getLogger(self.__class__.__name__)

        if model_name.startswith("openrouter/"):
            model_name = model_name[len("openrouter/") :]
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_copy = payload.copy()
        payload_copy["model"] = self.model_name
        response = await self.client.chat.completions.create(**payload_copy)
        return response.model_dump()

    async def generate_stream(
        self,
        payload: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload_copy = payload.copy()
        payload_copy["model"] = self.model_name
        payload_copy["stream"] = True
        stream = await self.client.chat.completions.create(**payload_copy)
        async for chunk in stream:
            yield chunk.model_dump()


class RateLimitException(Exception):
    def __init__(self, reset_time: float):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Reset at {reset_time}")


class LLMClient:
    def __init__(
        self,
        generative_client: BaseGenerativeClient,
        backend_model: BackendModel,
        router: LLMRouter,
    ):
        self.generative_client = generative_client
        self.backend_model = backend_model
        self.model_id = self.backend_model.id
        self.model_name = self.backend_model.model_name
        self.router = router
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.model_id}]")

    async def make_request(self, payload: dict[str, Any]) -> Any:
        self.logger.info(f"Making request to backend model: {self.model_name}")
        self.logger.debug(f"Request payload: {payload}")
        stream = payload.get("stream", False)

        try:
            if stream:
                return self.generative_client.generate_stream(payload)
            return await self.generative_client.generate(payload)
        except APIStatusError as e:
            self._handle_api_error(e)
        except Exception as e:
            self._handle_generic_error(e)

    def _handle_api_error(self, e: APIStatusError) -> None:
        if e.status_code == 429:
            self._handle_rate_limit_error(e)
        else:
            self.logger.error(f"Unhandled API error: {e.status_code}")
            raise e

    def _handle_rate_limit_error(self, e: APIStatusError) -> None:
        headers = e.response.headers
        reset_time = float(headers.get("X-RateLimit-Reset", time.time() + 60))
        if reset_time > 1e10:
            reset_time /= 1000.0
        self.logger.warning(f"Rate limit error: Reset at {time.ctime(reset_time)}")
        raise RateLimitException(reset_time)

    def _handle_generic_error(self, e: Exception) -> None:
        if isinstance(e, google.api_core.exceptions.ResourceExhausted):
            self.logger.warning(f"Gemini rate limit error: {e}")
            try:
                error_json = json.loads(e.message)
                retry_info = None
                for detail in error_json.get("error", {}).get("details", []):
                    if (
                        detail.get("@type")
                        == "type.googleapis.com/google.rpc.RetryInfo"
                    ):
                        retry_info = detail
                        break

                if retry_info and "retryDelay" in retry_info:
                    delay_str = retry_info["retryDelay"]
                    delay_seconds = int(re.sub(r"\D", "", delay_str))
                    reset_time = time.time() + delay_seconds
                    self.logger.info(
                        f"Gemini API requested a specific retry delay of {delay_seconds}s.",
                    )
                    raise RateLimitException(reset_time) from e
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as parse_error:
                self.logger.warning(
                    f"Could not parse Gemini retry delay, using default: {parse_error}",
                )

            raise RateLimitException(time.time() + 60) from e

        self.logger.exception(f"Unhandled generic error making request: {e}")
        raise e
