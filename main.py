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

import uvicorn
from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import HTTPStatusError

from app.client import LLMClient, RateLimitException
from app.config import AppConfig
from app.dependencies import get_client_map, get_config, get_router, get_state_manager
from app.prompts import generate_xml_tool_definitions, parse_xml_tool_call
from app.router import LLMRouter
from app.state import ModelStateManager

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="LLM Proxy Server")
recent_prompts_cache = TTLCache(maxsize=1000, ttl=2)


@app.get("/v1/models")
@app.get("/models")
async def list_models(config: AppConfig = Depends(get_config)):
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
    try:
        payload = await request.json()
        model_group = payload.get("model")
        stream = payload.get("stream", False)
        logging.debug(f"Incoming request for model group '{model_group}': {payload}")

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
            logging.warning(
                f"Duplicate prompt detected. Blocking likely title-generation request for: '{user_prompt_content}'"
            )
            return JSONResponse(
                status_code=409,
                content={"error": "A primary tool-use request is already in progress."},
            )

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
                f"(backend model: {backend_model.model_name})"
            )

            try:
                payload_copy = payload.copy()
                client = client_map[backend_model.id]

                if is_standard_tool_request and not backend_model.supports_tools:
                    # --- XML WORKAROUND LOGIC ---
                    logging.info("Applying XML tool workaround...")
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
                        payload_copy["messages"].insert(0, {"role": "system", "content": final_system_prompt})
                    
                    logging.debug("Payload transformed for XML workaround.")

                    response_stream = await client.make_request(payload_copy)

                    async def event_generator():
                        full_response_text = ""
                        async for chunk in response_stream:
                            text_delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                            full_response_text += text_delta
                        
                        logging.debug(f"Full streamed response for XML parsing: {full_response_text}")
                        tool_call = parse_xml_tool_call(full_response_text)

                        if tool_call:
                            logging.info(f"XML tool call detected: {tool_call}")
                            openai_tool_call = {
                                "id": f"call_{int(time.time())}", "type": "function",
                                "function": {
                                    "name": tool_call["tool_name"],
                                    "arguments": json.dumps(tool_call["parameters"]),
                                },
                            }
                            final_chunk = {
                                "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
                                "created": int(time.time()), "model": backend_model.model_name,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": [openai_tool_call]}, "finish_reason": "tool_calls"}],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                        else:
                            logging.info("No XML tool call detected, sending plain text.")
                            final_chunk = {
                                "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
                                "created": int(time.time()), "model": backend_model.model_name,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": full_response_text}, "finish_reason": "stop"}],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                        
                        yield "data: [DONE]\n\n"

                    state_manager.record_success(backend_model.id)
                    return StreamingResponse(event_generator(), media_type="text/event-stream")

                else:
                    # --- NATIVE TOOL-CALLING / NO TOOLS LOGIC ---
                    logging.info("Passing request with native tool support or no tools.")
                    response = await client.make_request(payload_copy)
                    state_manager.record_success(backend_model.id)

                    if stream:
                        async def native_event_stream():
                            async for chunk in response:
                                yield f"data: {json.dumps(chunk)}\n\n"
                        return StreamingResponse(native_event_stream(), media_type="text/event-stream")
                    else:
                        return JSONResponse(content=response)

            # --- НАЧАЛО ИЗМЕНЕНИЙ В ОБРАБОТКЕ ОШИБОК ---
            except HTTPStatusError as e:
                status_code = e.response.status_code
                state_manager.record_failure(backend_model.id, status_code)
                last_error = e

                if status_code == 400:
                    logging.error(f"Fatal 400 Bad Request for {backend_model.id}: {e.response.text}. Forwarding to client.")
                    # Немедленно прерываем и сообщаем клиенту, возвращая тело ошибки от бэкенда
                    raise HTTPException(status_code=400, detail=e.response.json())

                if status_code == 429:
                    headers = e.response.headers
                    reset_time = float(headers.get("X-RateLimit-Reset", time.time() + 60))
                    if reset_time > 1e10: reset_time /= 1000.0
                    state_manager.set_cooldown(backend_model.id, reset_time)
                    logging.warning(f"Rate limit hit for {backend_model.id}, cooldown until {time.ctime(reset_time)}")
                else:
                    logging.warning(f"Request failed with status {status_code} on {backend_model.id}. Error: {e.response.text}")
                
                retry_count += 1
                continue
            # --- КОНЕЦ ИЗМЕНЕНИЙ В ОБРАБОТКЕ ОШИБОК ---
            
            except RateLimitException as e:
                state_manager.set_cooldown(backend_model.id, e.reset_time)
                logging.warning(f"Rate limit error for {backend_model.id}, cooldown until {time.ctime(e.reset_time)}")
                last_error = e
                retry_count += 1
                continue

        # Если все попытки провалились
        if last_error:
            if isinstance(last_error, HTTPStatusError):
                raise HTTPException(status_code=last_error.response.status_code, detail=last_error.response.json())
            raise HTTPException(status_code=500, detail=str(last_error))
        raise HTTPException(status_code=503, detail="All models in group are overloaded or unavailable")

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