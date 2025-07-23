import logging
import time
import google.generativeai as genai
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from app.config import ModelDeployment


class LLMClient:
    def __init__(self, deployment: ModelDeployment):
        self.model_name = deployment.litellm_params.model
        
        # Handle Gemini models specially
        if "gemini" in self.model_name.lower():
            genai.configure(api_key=deployment.litellm_params.api_key)
            # Extract base model name without any prefixes
            base_model = self.model_name.split("/")[-1].split(":")[0]

            self.client = genai.GenerativeModel(base_model)
            self.gemini = True
        else:
            self.client = AsyncOpenAI(
                base_url=deployment.litellm_params.api_base,
                api_key=deployment.litellm_params.api_key,
            )
            self.gemini = False

    async def make_request(self, payload: Dict[str, Any]) -> Any:
        """Send request to the LLM deployment endpoint"""
        logging.info(f"Making request to {self.model_name}")
        logging.debug(f"Making request to {self.model_name} with payload: {payload}")
        try:
            if self.gemini:
                # Convert messages to Gemini format
                messages = []
                for msg in payload["messages"]:
                    role = "user" if msg["role"] == "user" else "model"
                    messages.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    })
                
                # Create chat history and send message
                chat = self.client.start_chat(history=messages[:-1])
                response = await chat.send_message_async(messages[-1]["parts"][0]["text"])
                
                # Format response to match OpenAI format
                return {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.text,
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            else:
                # Use existing OpenAI client for non-Gemini models
                return await self.client.chat.completions.create(**payload)
        except Exception as e:
            logging.error(f"Error making request to {self.model_name}: {e}")
            raise
