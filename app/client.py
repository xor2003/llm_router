import logging
import time
import google.generativeai as genai
from typing import Any, Dict
from openai import AsyncOpenAI, APIStatusError
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
        """Send request to the LLM deployment endpoint with error forwarding"""
        logging.info(f"=== REQUEST START ===")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Gemini: {self.gemini}")
        logging.info(f"Payload keys: {list(payload.keys())}")
        logging.info(f"Messages count: {len(payload.get('messages', []))}")
        
        # Log first message content for debugging
        if payload.get('messages'):
            first_msg = payload['messages'][0]
            logging.info(f"First message role: {first_msg.get('role')}")
            logging.info(f"First message content preview: {str(first_msg.get('content', ''))[:100]}...")
        
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
                logging.info(f"Making request to {self.model_name} with payload")
                payload["model"] = self.model_name
                return await self.client.chat.completions.create(**payload)
        except Exception as e:
            # Forward all errors except 429 to the client
            if isinstance(e, APIStatusError) and e.status_code == 429:
                logging.warning(f"Rate limit error for {self.model_name}: {e}")
                raise  # We'll handle 429s in the state manager
            else:
                logging.error(f"Forwarding error for {self.model_name}: {e}")
                raise e  # Forward other errors directly to client
