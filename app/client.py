import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List

from openai import APIStatusError, AsyncOpenAI

from app.config import BackendModel
from app.router import LLMRouter


class BaseGenerativeClient(ABC):
    """
    Abstract base class for a generative AI client.
    """

    @abstractmethod
    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a non-streaming response.
        Args:
            payload: The request payload, conforming to a standard OpenAI-like format.
        Returns:
            The response from the generative model, in a standard OpenAI-like format.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming response.
        Args:
            payload: The request payload, conforming to a standard OpenAI-like format.
        Yields:
            Chunks of the response from the generative model.
        """
        pass


class GeminiClient(BaseGenerativeClient):
    def __init__(self, model_name: str, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def _translate_payload_to_gemini(self, payload: Dict[str, Any]) -> List[Dict]:
        contents = []
        for msg in payload.get("messages", []):
            role = "user" if msg["role"] == "user" else "model"
            parts = []
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part["text"])
                    elif isinstance(part, str):
                        parts.append(part)
            else:
                content = msg.get("content", "")
                if content:
                    parts.append(content)

            if parts:
                contents.append(
                    {"role": role, "parts": [{"text": text} for text in parts]}
                )
        return contents

    def _translate_gemini_response_to_openai(self, response) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        contents = self._translate_payload_to_gemini(payload)
        response = self.client.generate_content(contents)
        return self._translate_gemini_response_to_openai(response)

    async def generate_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # For Gemini, we need to accumulate the full response when using XML tool workaround
        if payload.get("stream") is False:
            # Handle as non-streaming request
            response = await self.generate(payload)
            yield response
        else:
            # Normal streaming behavior
            contents = self._translate_payload_to_gemini(payload)
            stream = self.client.generate_content(contents, stream=True)
            for chunk in stream:
                yield self._translate_gemini_response_to_openai(chunk)


class OpenAIClient(BaseGenerativeClient):
    def __init__(self, model_name: str, api_key: str, api_base: str):
        self.model_name = model_name
        if self.model_name.startswith("openrouter/"):
            self.model_name = self.model_name[len("openrouter/") :]
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["model"] = self.model_name
        return await self.client.chat.completions.create(**payload)

    async def generate_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload["model"] = self.model_name
        payload["stream"] = True
        stream = await self.client.chat.completions.create(**payload)
        async for chunk in stream:
            yield chunk


class RateLimitException(Exception):
    def __init__(self, reset_time: float):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Reset at {reset_time}")


class MCPConnectionManager:
    """Manages connections to MCP servers"""

    def __init__(self):
        self.servers = {}

    def add_server(self, server_name: str, endpoint: str):
        self.servers[server_name] = endpoint

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        params: Dict[str, str],
    ) -> Any:
        """Calls an MCP tool and returns the result"""
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        return f"MCP tool {tool_name} executed on {server_name} with params {params}"


class LLMClient:
    def __init__(
        self,
        generative_client: BaseGenerativeClient,
        backend_model: BackendModel,
        router: LLMRouter,
    ):
        self.generative_client = generative_client
        self.backend_model = backend_model
        self.id = self.backend_model.id
        self.model_name = self.backend_model.model_name
        self.router = router
        self.mcp_manager = MCPConnectionManager()
        self.mcp_manager.add_server("telegram", "https://mcp.telegram.example.com")
        self.mcp_manager.add_server("weather", "https://mcp.weather.example.com")

    async def make_request(self, payload: dict[str, Any]) -> Any:
        """Send request to the LLM backend_model endpoint with error forwarding"""
        logging.info(f"Making request to {self.model_name}")
        logging.debug(f"Request payload: {payload}")
        stream = payload.get("stream", False)

        try:
            if stream:
                return self.generative_client.generate_stream(payload)
            else:
                return await self.generative_client.generate(payload)
        except APIStatusError as e:
            self._handle_api_error(e)
        except Exception as e:
            self._handle_generic_error(e)

    def _handle_api_error(self, e: APIStatusError) -> None:
        if e.status_code == 429:
            self._handle_rate_limit_error(e)
        else:
            logging.exception(f"API error for {self.model_name}: {e}")
            raise e

    def _handle_rate_limit_error(self, e: APIStatusError) -> None:
        headers = e.response.headers
        reset_time = float(headers.get("X-RateLimit-Reset", time.time() + 60))
        if reset_time > 1e10:
            reset_time /= 1000.0
        logging.warning(
            f"Rate limit error for {self.model_name}: Reset at {time.ctime(reset_time)}",
        )
        raise RateLimitException(reset_time)

    def _handle_generic_error(self, e: Exception) -> None:
        logging.exception(f"Error making request to {self.model_name}: {e}")
        raise e
