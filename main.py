#!/usr/bin/env python3
import sys
import os
import logging
import time

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import HTTPStatusError
from typing import Dict
from cachetools import TTLCache
from app.client import LLMClient, RateLimitException
from app.config import AppConfig
from app.dependencies import get_config, get_client_map, get_router, get_state_manager
from app.prompts import generate_xml_tool_definitions
from app.router import LLMRouter
from app.state import ModelStateManager

import os
# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="LLM Proxy Server")

# Create a TTL (Time To Live) cache.
# It will store items for a maximum of 2 seconds.
# maxsize can be adjusted based on expected concurrent users.
recent_prompts_cache = TTLCache(maxsize=1000, ttl=2)

# Make endpoints available under both /v1 and / paths
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
    config: AppConfig = Depends(get_config),  # <-- Add this dependency
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

        # Extract the user prompt content for caching and comparison
        user_prompt_content = ""
        if payload.get("messages"):
            for message in payload["messages"]:
                if message["role"] == "user":
                    # Assuming content is a list of dicts with 'text' key
                    if isinstance(message.get("content"), list):
                        user_prompt_content = message["content"][0].get("text", "")
                    else:
                        user_prompt_content = message.get("content", "")
                    break
        
        is_standard_tool_request = "tools" in payload and payload.get("tools")

        # Check if this is a likely duplicate "title" request
        if user_prompt_content in recent_prompts_cache and not is_standard_tool_request:
            logging.warning(f"Duplicate prompt detected. Blocking likely title-generation request for: '{user_prompt_content}'")
            # Return an empty stream or a specific error to the client
            # This prevents the "title" request from ever reaching the LLM
            return JSONResponse(
                status_code=409, # 409 Conflict is a good code for this
                content={"error": "A primary tool-use request for this prompt is already in progress."}
            )

        if is_standard_tool_request:
            logging.info("Standard OpenAI tool request detected. Translating to XML workaround.")
            
            # Add the prompt to the cache to detect subsequent "fake" requests
            recent_prompts_cache[user_prompt_content] = True

            # 1. Generate XML tool definitions from client's tools payload
            client_tools = payload.get("tools", [])
            xml_tool_definitions = generate_xml_tool_definitions(client_tools)

            # 2. Create final system prompt by injecting tool definitions
            final_system_prompt = config.mcp_tool_use_prompt_template.format(
                TOOL_DEFINITIONS=xml_tool_definitions
            )

            # 3. Remove tool parameters
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

            # 4. Inject the dynamic system prompt
            system_message_found = False
            for message in payload["messages"]:
                if message["role"] == "system":
                    message["content"] = final_system_prompt
                    system_message_found = True
                    break
            
            if not system_message_found:
                payload["messages"].insert(0, {"role": "system", "content": final_system_prompt})
            
            logging.debug("Payload transformed with DYNAMIC tool definitions.")

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
                f"(backend model: {backend_model.model_name})"
            )

            try:
                # Make a copy of payload to avoid mutating original
                payload_copy = payload.copy()
                payload_copy["model"] = backend_model.model_name
                client = client_map[backend_model.id]
                response = await client.make_request(payload_copy)
                state_manager.record_success(backend_model.id)
        
                # Regular response
                if stream:
                    async def stream_generator():
                        try:
                            async for chunk in response:
                                yield f"data: {chunk.model_dump_json()}\n\n"
                        except Exception as e:
                            logging.error(f"Stream error for {backend_model.id}: {e}")
                        finally:
                            logging.info(f"Stream closed for {backend_model.id}")
    
                    return StreamingResponse(stream_generator(), media_type="text/event-stream")
                else:
                    response_json = response.model_dump()
                    logging.info(f"Response for {backend_model.model_name}: {response_json}")
                    return JSONResponse(content=response_json)

            except HTTPStatusError as e:
                status_code = e.response.status_code
                state_manager.record_failure(backend_model.id, status_code)
                last_error = e
                
                if status_code == 429:
                    # Extract reset time from headers
                    headers = e.response.headers
                    reset_time = float(headers.get("X-RateLimit-Reset", time.time() + 60))
                    
                    # Handle milliseconds format
                    if reset_time > 1e10:  # Likely in milliseconds
                        reset_time = reset_time / 1000.0
                    
                    # Set precise cooldown
                    state_manager.set_cooldown(backend_model.id, reset_time)
                    logging.warning(
                        f"Rate limit hit for {backend_model.id}, "
                        f"cooldown until {time.ctime(reset_time)}"
                    )
                else:
                    logging.warning(
                        f"Request failed with status {status_code} on backend model {backend_model.id}. "
                        f"Error: {e.response.text}"
                    )
                
                retry_count += 1
                continue
                
            except RateLimitException as e:
                # Set precise cooldown from exception
                state_manager.set_cooldown(backend_model.id, e.reset_time)
                logging.warning(
                    f"Rate limit error for {backend_model.id}, "
                    f"cooldown until {time.ctime(e.reset_time)}"
                )
                last_error = e
                retry_count += 1
                continue

        # If all retries failed
        logging.error(f"All {max_retries} attempts failed for model group {model_group}")
        if last_error is not None:
            if isinstance(last_error, HTTPStatusError):
                raise HTTPException(
                    status_code=last_error.response.status_code,
                    detail=last_error.response.json()
                )
            else:
                raise HTTPException(status_code=500, detail=str(last_error))
        else:
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
        reload=True,  # Enable for development
    )
