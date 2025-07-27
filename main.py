#!/usr/bin/env python3
import logging
import os
import sys
import time

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from typing import Dict

import uvicorn
from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from httpx import HTTPStatusError

from app.client import GeminiClient, LLMClient, RateLimitException
from app.config import AppConfig
from app.dependencies import get_client_map, get_config, get_router, get_state_manager
from app.prompts import generate_xml_tool_definitions, parse_xml_tool_call
from app.router import LLMRouter
from app.state import ModelStateManager

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="LLM Proxy Server")

# Create a TTL (Time To Live) cache.
recent_prompts_cache = TTLCache(maxsize=1000, ttl=2)


@app.get("/v1/models")
@app.get("/models")
async def list_models(config: AppConfig = Depends(get_config)):
    """Return load balancing model names from configuration"""
    model_names = list({model.model_name for model in config.model_list})
    return {"models": model_names}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    router: LLMRouter = Depends(get_router),
    client_map: Dict[str, LLMClient] = Depends(get_client_map),
    state_manager: ModelStateManager = Depends(get_state_manager),
    config: AppConfig = Depends(get_config),
):
    """Proxy endpoint for LLM requests in OpenAI format"""
    try:
        payload = await request.json()
        model_group = payload.get("model")
        stream = payload.get("stream", False)

        logging.debug(f"Incoming request for model group '{model_group}': {payload}")

        # ====================================================================
        # START: GENERALIZED PROTOCOL ADAPTER & REQUEST FILTERING
        # ====================================================================
        user_prompt_content = ""
        if payload.get("messages"):
            for message in payload["messages"]:
                if message["role"] == "user":
                    if isinstance(message.get("content"), list):
                        user_prompt_content = message["content"][0].get("text", "")
                    else:
                        user_prompt_content = message.get("content", "")
                    break

        is_standard_tool_request = "tools" in payload and payload.get("tools")

        if user_prompt_content in recent_prompts_cache and not is_standard_tool_request:
            logging.warning(
                f"Duplicate prompt detected. Blocking likely title-generation request for: '{user_prompt_content}'",
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "A primary tool-use request for this prompt is already in progress.",
                },
            )

        # ====================================================================
        # END: GENERALIZED LOGIC
        # ====================================================================

        if not model_group:
            raise HTTPException(status_code=400, detail="Model parameter is required")

        retry_count = 0
        max_retries = 3
        last_error = None

        while retry_count < max_retries:
            backend_model = router.get_next_backend_model(model_group)
            if not backend_model:
                raise HTTPException(
                    status_code=503,
                    detail="All models in group are overloaded or unavailable",
                )

            logging.info(
                f"Attempt {retry_count+1}/{max_retries}: Using backend model id: {backend_model.id} "
                f"(backend model: {backend_model.model_name})",
            )

            try:
                payload_copy = payload.copy()
                payload_copy["model"] = backend_model.model_name
                payload_copy = payload.copy()
                payload_copy["model"] = backend_model.model_name
                client = client_map[backend_model.id]

                # Conditional MCP Workaround Logic
                if is_standard_tool_request and not backend_model.supports_tools:
                    logging.info(
                        "Applying XML tool workaround for model that does not support native tools."
                    )
                    recent_prompts_cache[user_prompt_content] = True
                    client_tools = payload_copy.pop("tools", [])
                    payload_copy.pop("tool_choice", None)

                    xml_tool_definitions = generate_xml_tool_definitions(client_tools)
                    final_system_prompt = config.mcp_tool_use_prompt_template.format(
                        TOOL_DEFINITIONS=xml_tool_definitions
                    )

                    system_message_found = False
                    for message in payload_copy["messages"]:
                        if message["role"] == "system":
                            message["content"] = final_system_prompt
                            system_message_found = True
                            break
                    if not system_message_found:
                        payload_copy["messages"].insert(
                            0, {"role": "system", "content": final_system_prompt}
                        )
                    logging.debug("Payload transformed with DYNAMIC tool definitions.")

                    # For workaround, we must handle the response as a non-streamed tool call
                    response = await client.make_request(payload_copy)
                    if isinstance(client.generative_client, GeminiClient):
                        response_json = response
                    else:
                        response_json = response.model_dump()

                    content = response_json["choices"][0]["message"]["content"]
                    tool_call = parse_xml_tool_call(content)

                    if tool_call:
                        logging.info(f"XML tool call detected: {tool_call}")
                        state_manager.record_success(backend_model.id)
                        # Return a compliant OpenAI tool call response
                        final_response = {
                            "id": f"chatcmpl-{time.time()}",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "id": f"call_{time.time()}",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call["tool_name"],
                                                    "arguments": str(
                                                        tool_call["parameters"]
                                                    ),
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                        }
                        return JSONResponse(content=final_response)
                    else:
                        # Fallback to standard response if no tool call detected
                        state_manager.record_success(backend_model.id)
                        return JSONResponse(content=response_json)

                # Native Tool-Calling Passthrough
                logging.info("Passing request to model with native tool support.")
                response = await client.make_request(payload_copy)
                state_manager.record_success(backend_model.id)

                if stream:
                    # For streaming, wrap the async generator in a StreamingResponse
                    return StreamingResponse(
                        response,
                        media_type="text/event-stream"
                    )
                else:
                    if isinstance(client.generative_client, GeminiClient):
                        return JSONResponse(content=response)
                    return JSONResponse(content=response.model_dump())

            except HTTPStatusError as e:
                # ... (логика обработки ошибок)
                status_code = e.response.status_code
                state_manager.record_failure(backend_model.id, status_code)
                last_error = e
                if status_code == 429:
                    headers = e.response.headers
                    reset_time = float(
                        headers.get("X-RateLimit-Reset", time.time() + 60),
                    )
                    if reset_time > 1e10:
                        reset_time /= 1000.0
                    state_manager.set_cooldown(backend_model.id, reset_time)
                    logging.warning(
                        f"Rate limit hit for {backend_model.id}, cooldown until {time.ctime(reset_time)}",
                    )
                else:
                    logging.warning(
                        f"Request failed with status {status_code} on backend model {backend_model.id}. Error: {e.response.text}",
                    )
                retry_count += 1
                continue
            except RateLimitException as e:
                # ... (логика обработки ошибок)
                state_manager.set_cooldown(backend_model.id, e.reset_time)
                logging.warning(
                    f"Rate limit error for {backend_model.id}, cooldown until {time.ctime(e.reset_time)}",
                )
                last_error = e
                retry_count += 1
                continue

        # If all retries failed
        if last_error:
            if isinstance(last_error, HTTPStatusError):
                raise HTTPException(
                    status_code=last_error.response.status_code,
                    detail=last_error.response.json(),
                )
            raise HTTPException(status_code=500, detail=str(last_error))
        raise HTTPException(
            status_code=503,
            detail="All models in group are overloaded or unavailable",
        )

    except Exception as e:
        logging.exception("Internal server error occurred")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config: AppConfig = get_config()
    uvicorn.run(
        "main:app",
        host=config.proxy_server_config.host,
        port=config.proxy_server_config.port,
        reload=True,
    )
