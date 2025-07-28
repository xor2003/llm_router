#!/usr/bin/env python3
import json
import logging
import os
import sys
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "app"))

from typing import Dict

import openai
import uvicorn
from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import HTTPStatusError

from app.client import CustomRateLimitException, LLMClient
from app.config import AppConfig
from app.dependencies import get_client_map, get_config, get_router, get_state_manager
from app.prompts import generate_xml_tool_definitions, parse_xml_tool_call
from app.router import LLMRouter
from app.state import ModelStateManager

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Proxy Server")
recent_prompts_cache = TTLCache(maxsize=1000, ttl=2)


@app.get("/v1/models")
@app.get("/models")
async def list_models(config: AppConfig = Depends(get_config)):
    model_names = list({model.model_name for model in config.model_list})
    return {"models": model_names}


@app.post("/v1/chat/completions")
@app.post("/chat_completions")
async def chat_completions(
    request: Request,
    router: LLMRouter = Depends(get_router),
    client_map: Dict[str, LLMClient] = Depends(get_client_map),
    state_manager: ModelStateManager = Depends(get_state_manager),
    config: AppConfig = Depends(get_config),
):
    try:
        payload = await request.json()
        model_group = payload.get("model")
        stream = payload.get("stream", False)
        logger.debug(f"Incoming request for model group '{model_group}': {payload}")

        user_prompt_content = ""
        if payload.get("messages"):
            for message in payload["messages"]:
                if message["role"] == "user":
                    content = message.get("content", "")
                    user_prompt_content = (
                        content[0]["text"] if isinstance(content, list) else content
                    )
                    break

        is_standard_tool_request = "tools" in payload and payload.get("tools")

        if user_prompt_content in recent_prompts_cache and not is_standard_tool_request:
            logger.warning(
                f"Duplicate prompt detected. Blocking likely title-generation request for: '{user_prompt_content}'",
            )
            return JSONResponse(
                status_code=409,
                content={"error": "A primary tool-use request is already in progress."},
            )

        if not model_group:
            raise HTTPException(status_code=400, detail="Model parameter is required")

        retry_count = 0
        # Maximum attempts equals the number of models in the group to guarantee trying all
        max_retries = len(router.model_groups.get(model_group, []))
        if max_retries == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Model group '{model_group}' not found or is empty.",
            )

        last_error = None

        while retry_count < max_retries:
            # Use new smart method to select "active" model or switch on failure
            backend_model = router.get_active_or_failover_model(
                model_group,
                state_manager,
            )

            if not backend_model:
                raise HTTPException(
                    status_code=503,
                    detail=f"All models in group '{model_group}' are currently overloaded or unavailable",
                )

            logger.info(
                f"Attempt {retry_count+1}/{max_retries}: Using backend model id: {backend_model.id} "
                f"(backend model: {backend_model.model_name})",
            )

            try:
                payload_copy = payload.copy()
                client = client_map[backend_model.id]

                # --- XML WORKAROUND LOGIC ---
                if is_standard_tool_request and not backend_model.supports_tools:
                    logger.info("Applying XML tool workaround...")
                    recent_prompts_cache[user_prompt_content] = True

                    client_tools = payload_copy.pop("tools", [])
                    payload_copy.pop("tool_choice", None)
                    xml_tool_definitions = generate_xml_tool_definitions(client_tools)
                    final_system_prompt = config.mcp_tool_use_prompt_template.format(
                        TOOL_DEFINITIONS=xml_tool_definitions,
                    )

                    system_message_found = False
                    for message in payload_copy["messages"]:
                        if message["role"] == "system":
                            message["content"] = final_system_prompt
                            system_message_found = True
                            break
                    if not system_message_found:
                        payload_copy["messages"].insert(
                            0,
                            {"role": "system", "content": final_system_prompt},
                        )

                    logger.debug("Payload transformed for XML workaround.")

                    response_stream = await client.make_request(payload_copy)

                    async def event_generator():
                        full_response_text = ""
                        async for chunk in response_stream:
                            text_delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                                or ""
                            )
                            full_response_text += text_delta

                        logger.debug(
                            f"Full streamed response for XML parsing: {full_response_text}",
                        )
                        tool_call = parse_xml_tool_call(full_response_text)

                        if tool_call:
                            logger.info(f"XML tool call detected: {tool_call}")
                            openai_tool_call = {
                                "id": f"call_{int(time.time())}",
                                "type": "function",
                                "function": {
                                    "name": tool_call["tool_name"],
                                    "arguments": json.dumps(tool_call["parameters"]),
                                },
                            }
                            final_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": backend_model.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [openai_tool_call],
                                        },
                                        "finish_reason": "tool_calls",
                                    },
                                ],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                        else:
                            logger.info(
                                "No XML tool call detected, sending plain text.",
                            )
                            final_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": backend_model.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": full_response_text,
                                        },
                                        "finish_reason": "stop",
                                    },
                                ],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"

                        yield "data: [DONE]\n\n"

                    state_manager.record_success(backend_model.id)
                    return StreamingResponse(
                        event_generator(),
                        media_type="text/event-stream",
                    )

                # --- NATIVE TOOL-CALLING / NO TOOLS LOGIC ---
                logger.info("Passing request with native tool support or no tools.")
                response = await client.make_request(payload_copy)
                state_manager.record_success(backend_model.id)

                if stream:

                    async def native_event_stream():
                        async for chunk in response:
                            yield f"data: {json.dumps(chunk)}\n\n"

                    return StreamingResponse(
                        native_event_stream(),
                        media_type="text/event-stream",
                    )
                return JSONResponse(content=response)

            except openai.RateLimitError as e:
                logger.warning(
                    f"OpenAI RateLimitError for {backend_model.id}. Error: {e.body}. Retrying...",
                )
                state_manager.record_failure(backend_model.id, 429)
                last_error = e
                reset_time = time.time() + 60
                state_manager.set_cooldown(backend_model.id, reset_time)
                retry_count += 1
                continue

            except HTTPStatusError as e:
                status_code = e.response.status_code
                state_manager.record_failure(backend_model.id, status_code)
                last_error = e

                if status_code == 400:
                    logger.error(
                        f"Fatal 400 Bad Request for {backend_model.id}: {e.response.text}. Forwarding to client.",
                    )
                    raise HTTPException(status_code=400, detail=e.response.json())

                if status_code == 429:
                    headers = e.response.headers
                    reset_time = float(
                        headers.get("X-RateLimit-Reset", time.time() + 60),
                    )
                    if reset_time > 1e10:
                        reset_time /= 1000.0
                    state_manager.set_cooldown(backend_model.id, reset_time)
                    logger.warning(
                        f"Rate limit hit for {backend_model.id}, cooldown until {time.ctime(reset_time)}. Retrying...",
                    )
                else:
                    logger.warning(
                        f"Request failed with status {status_code} on {backend_model.id}. Error: {e.response.text}. Retrying...",
                    )

                retry_count += 1
                continue

            except CustomRateLimitException as e:
                logger.warning(
                    f"Custom RateLimitException for {backend_model.id}, cooldown until {time.ctime(e.reset_time)}. Retrying...",
                )
                state_manager.record_failure(backend_model.id, 429)
                state_manager.set_cooldown(backend_model.id, e.reset_time)
                last_error = e
                retry_count += 1
                continue

        # If all attempts failed
        if last_error:
            if isinstance(last_error, (HTTPStatusError, openai.APIStatusError)):
                # Check if error has response body to avoid crashes
                try:
                    detail = last_error.response.json()
                except Exception:
                    detail = str(last_error)
                raise HTTPException(
                    status_code=last_error.response.status_code,
                    detail=detail,
                )
            raise HTTPException(status_code=500, detail=str(last_error))
        raise HTTPException(
            status_code=503,
            detail="All models in group are overloaded or unavailable",
        )

    except Exception as e:
        logger.exception("Internal server error occurred")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config: AppConfig = get_config()
    uvicorn.run(
        "main:app",
        host=config.proxy_server_config.host,
        port=config.proxy_server_config.port,
        reload=True,
    )
