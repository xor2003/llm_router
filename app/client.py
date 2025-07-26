import logging
import time
import openai
from typing import Any

import google.generativeai as genai
from openai import APIStatusError, AsyncOpenAI
from app.config import BackendModel


class RateLimitException(Exception):
    def __init__(self, reset_time: float):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Reset at {reset_time}")


class LLMClient:
    def __init__(self, backend_model: BackendModel):
        self.backend_model = backend_model
        self.id = self.backend_model.id
        self.model_name = self.backend_model.model_name

        # Handle Gemini models specially
        if "gemini" in self.model_name.lower():
            genai.configure(api_key=backend_model.api_key)
            # Extract base model name without any prefixes
            base_model = self.model_name.split("/")[-1].split(":")[0]

            self.client = genai.GenerativeModel(base_model)
            self.gemini = True
        else:
            # For OpenRouter, remove the 'openrouter/' prefix
            if self.model_name.startswith("openrouter/"):
                self.model_name = self.model_name[len("openrouter/"):]
                
            self.client = AsyncOpenAI(
                base_url=backend_model.api_base,
                api_key=backend_model.api_key,
            )
            self.gemini = False

    async def make_request(self, payload: dict[str, Any]) -> Any:
        """Send request to the LLM backend_model endpoint with error forwarding"""
        logging.info(f"Making request to {self.model_name}")
        logging.debug(f"Request payload: {payload}")
        try:
            if self.gemini:
                # Gemini handling remains the same
                # Convert messages to Gemini format
                messages = []
                for msg in payload["messages"]:
                    role = "user" if msg["role"] == "user" else "model"
                    # The parts should be a list of strings or Part objects, not dicts
                    messages.append({"role": role, "parts": [msg["content"]]})

                # Create chat history and send message
                # The history needs to be correctly formatted for start_chat
                history = [
                    {"role": msg["role"], "parts": msg["parts"]}
                    for msg in messages[:-1]
                ]
                chat = self.client.start_chat(history=history)
                
                # The last message's content is sent
                last_message_content = messages[-1]["parts"]
                response = await chat.send_message_async(
                    last_message_content,
                )

                # Format response to match OpenAI format
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
            else:
                # Use existing OpenAI client for non-Gemini models
                logging.info(f"Making request to {self.model_name} with payload")
                payload["model"] = self.model_name
                
                # SIMPLIFIED LOGIC: Just make the request.
                # The 'tools' parameter has already been removed by the router if necessary.
                response = await self.client.chat.completions.create(**payload)

                # Log rate limit headers if available
                if hasattr(response, "headers"):
                    headers = response.headers
                    if "X-RateLimit-Limit" in headers:
                        logging.info(
                            f"Rate limit: {headers['X-RateLimit-Limit']} reqs, "
                            f"Remaining: {headers.get('X-RateLimit-Remaining', '?')}, "
                            f"Reset: {headers.get('X-RateLimit-Reset', '?')}"
                        )

                # Log response for debugging
                logging.debug(f"Response from {self.model_name}: {response}")
                return response
        except APIStatusError as e:
            if e.status_code == 429:
                # Extract rate limit info from headers
                headers = e.response.headers
                reset_time = float(headers.get("X-RateLimit-Reset", time.time() + 60))

                # Handle milliseconds format
                if reset_time > 1e10:  # Likely in milliseconds
                    reset_time = reset_time / 1000.0

                logging.warning(
                    f"Rate limit error for {self.model_name}: "
                    f"Limit={headers.get('X-RateLimit-Limit')} "
                    f"Remaining={headers.get('X-RateLimit-Remaining')} "
                    f"Reset={time.ctime(reset_time)}"
                )
                raise RateLimitException(reset_time)
            else:
                logging.error(f"API error for {self.model_name}: {e}")
                raise e
        except Exception as e:
            logging.error(f"Error making request to {self.model_name}: {e}")
            raise e
