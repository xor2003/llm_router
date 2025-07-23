#!/usr/bin/env python3
import sys
import os
import logging

# Добавляем корневую директорию проекта в путь Python
# Это позволяет находить модуль 'app' при запуске как 'python main.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import HTTPStatusError
from typing import Dict
from app.client import LLMClient
from app.config import AppConfig
from app.dependencies import get_config, get_client_map, get_router, get_state_manager
from app.router import LLMRouter
from app.state import ModelStateManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="LLM Proxy Server")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    router: LLMRouter = Depends(get_router),
    client_map: Dict[str, LLMClient] = Depends(get_client_map),
    state_manager: ModelStateManager = Depends(get_state_manager),
):
    """Эндпоинт, проксирующий запросы к LLM в формате OpenAI."""
    try:
        payload = await request.json()
        model_group = payload.get("model")
        stream = payload.get("stream", False)

        logging.debug(f"Incoming request for model group '{model_group}': {payload}")

        if not model_group:
            raise HTTPException(status_code=400, detail="Параметр 'model' обязателен.")

        deployment = router.get_next_deployment(model_group)
        if not deployment:
            raise HTTPException(
                status_code=503,
                detail="Все модели в группе перегружены или недоступны.",
            )

        logging.info(
            f"Selected deployment: {deployment.deployment_id} "
            f"(model: {deployment.litellm_params.model})"
        )

        try:
            payload["model"] = deployment.litellm_params.model
            client = client_map[deployment.deployment_id]
            # The make_request function now correctly returns an awaitable object
            # for both streaming and non-streaming responses.
            async_response = await client.make_request(payload)
            state_manager.record_success(deployment.deployment_id)

            if stream:
                # The response is an AsyncStream, which can be iterated over directly.
                async def stream_generator():
                    try:
                        async for chunk in async_response:
                            yield f"data: {chunk.model_dump_json()}\n\n"
                    except Exception as e:
                        logging.error(f"Error during streaming for {deployment.deployment_id}: {e}")
                    finally:
                        logging.info(f"Stream closed for {deployment.deployment_id}")

                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                # The response is a ChatCompletion object.
                response_json = async_response.model_dump()
                logging.info(f"Outgoing response for {deployment.litellm_params.model}: {response_json}")
                return JSONResponse(content=response_json)

        except HTTPStatusError as e:
            state_manager.record_failure(
                deployment.deployment_id, e.response.status_code
            )
            logging.error(
                f"Ошибка от API {deployment.litellm_params.model}: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.json()
            )

    except Exception as e:
        logging.exception("Произошла внутренняя ошибка сервера.")
        raise HTTPException(status_code=500, detail=str(e))


# Этот блок теперь будет работать корректно
if __name__ == "__main__":
    config: AppConfig = get_config()
    uvicorn.run(
        "main:app",
        host=config.proxy_server_config.host,
        port=config.proxy_server_config.port,
        reload=True,  # Включите для разработки
    )
