import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from openai import APIStatusError, AsyncOpenAI

from app.config import BackendModel
from app.router import LLMRouter


class BaseGenerativeClient(ABC):
    @abstractmethod
    async def generate_content_async(self, contents, stream=False):
        pass


class GeminiClient(BaseGenerativeClient):
    def __init__(self, model_name, api_key):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)

    async def generate_content_async(self, contents, stream=False):
        return self.client.generate_content(contents, stream=stream)


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

        # In a real implementation, we would make an HTTP request here
        # For now, simulate a successful response
        return f"MCP tool {tool_name} executed on {server_name} with params {params}"


class LLMClient:
    def __init__(self, backend_model: BackendModel, router: Optional[LLMRouter] = None):
        self.backend_model = backend_model
        self.id = self.backend_model.id
        self.model_name = self.backend_model.model_name
        self.router = router
        self.mcp_manager = MCPConnectionManager()

        # Add known MCP servers
        self.mcp_manager.add_server("telegram", "https://mcp.telegram.example.com")
        self.mcp_manager.add_server("weather", "https://mcp.weather.example.com")

        self.gemini = "gemini" in self.model_name.lower()
        self.openai_client = None
        self.generative_client = None

        if self.gemini:
            self.generative_client = GeminiClient(
                self.model_name,
                backend_model.api_key,
            )
        else:
            # For OpenRouter, remove the 'openrouter/' prefix
            if self.model_name.startswith("openrouter/"):
                self.model_name = self.model_name[len("openrouter/") :]

            self.openai_client = AsyncOpenAI(
                base_url=backend_model.api_base,
                api_key=backend_model.api_key,
            )

    async def make_request(self, payload: dict[str, Any]) -> Any:
        """Send request to the LLM backend_model endpoint with error forwarding"""
        logging.info(f"Making request to {self.model_name}")
        logging.debug(f"Request payload: {payload}")
        try:
            if self.gemini:
                # Определяем, был ли запрошен поток
                stream = payload.get("stream", False)

                # Преобразование сообщений в формат Gemini (общая часть)
                contents = []
                for msg in payload["messages"]:
                    role = "user" if msg["role"] == "user" else "model"
                    parts = []
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part["text"])
                            elif isinstance(part, str):
                                parts.append(part)
                    else:
                        # Добавляем проверку на None
                        content = msg.get("content", "")
                        if content:
                            parts.append(content)

                    if parts:
                        contents.append(
                            {"role": role, "parts": [{"text": text} for text in parts]},
                        )

                try:
                    if stream:
                        # ВЕТКА ДЛЯ ПОТОКА:
                        # Вызываем API с stream=True и ВОЗВРАЩАЕМ ПОТОК НАПРЯМУЮ
                        logging.info("Making Gemini request with streaming.")
                        response_stream = (
                            await self.generative_client.generate_content_async(
                                contents,
                                stream=True,
                            )
                        )
                        return response_stream
                    # ВЕТКА БЕЗ ПОТОКА (старая логика):
                    # Получаем полный ответ и форматируем его в словарь
                    logging.info("Making Gemini request without streaming.")
                    response = await self.generative_client.generate_content_async(
                        contents,
                    )
                    # Форматируем ответ под OpenAI
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
                            },
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                except Exception as e:
                    logging.exception(f"Gemini API error: {e}")
                    raise RuntimeError(f"Gemini API error: {e}") from e
            else:
                # Логика для OpenAI-совместимых моделей (остается без изменений)
                logging.info(f"Making request to {self.model_name} with payload")
                payload["model"] = self.model_name

                response = await self.openai_client.chat.completions.create(**payload)
                return response

        except APIStatusError as e:
            if e.status_code == 429:
                headers = e.response.headers
                reset_time = float(headers.get("X-RateLimit-Reset", time.time() + 60))
                if reset_time > 1e10:
                    reset_time /= 1000.0
                logging.warning(
                    f"Rate limit error for {self.model_name}: Reset at {time.ctime(reset_time)}",
                )
                raise RateLimitException(reset_time)
            logging.exception(f"API error for {self.model_name}: {e}")
            raise e
        except Exception as e:
            logging.exception(f"Error making request to {self.model_name}: {e}")
            raise e
