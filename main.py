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
from fastapi.responses import JSONResponse
from httpx import HTTPStatusError

from app.client import LLMClient
from app.config import AppConfig
from app.dependencies import get_config, get_llm_client, get_router, get_state_manager
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
    client: LLMClient = Depends(get_llm_client),
    state_manager: ModelStateManager = Depends(get_state_manager),
):
    """Эндпоинт, проксирующий запросы к LLM в формате OpenAI."""
    response = ""
    try:
        payload = await request.json()
        model_group = payload.get("model")
        
        # Log full incoming request
        logging.info(f"Incoming request for model group '{model_group}': {payload}")

        if not model_group:
            raise HTTPException(status_code=400, detail="Параметр 'model' обязателен.")

        # 1. Выбрать доступную модель
        deployment = router.get_next_deployment(model_group)
        if not deployment:
            raise HTTPException(
                status_code=503,
                detail="Все модели в группе перегружены или недоступны.",
            )
        
        # Log selected deployment details
        logging.info(f"Selected deployment: {deployment.deployment_id} "
                     f"(model: {deployment.litellm_params.model}, "
                     f"endpoint: {deployment.endpoint_url})")

        # 2. Отправить запрос
        try:
            payload["model"] = deployment.model_name
            response = await client.make_request(deployment, payload)
            state_manager.record_success(deployment.deployment_id)

            # 3. Адаптировать ответ, если нужно
            response_json = response.json()
            if deployment.litellm_params.model.startswith("gemini/"):
                response_json = client.adapt_gemini_response(
                    response_json, deployment.litellm_params.model
                )
            
            # Log full outgoing response
            logging.info(f"Outgoing response for {deployment.deployment_id}: {response_json}")
            
            return JSONResponse(content=response_json, status_code=response.status_code)

        except HTTPStatusError as e:
            # Обработка ошибок от нижестоящих API
            state_manager.record_failure(
                deployment.deployment_id, e.response.status_code
            )
            logging.error(
                f"Ошибка от API {deployment.deployment_id}: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.json()
            )

    except Exception as e:
        logging.exception("Произошла внутренняя ошибка сервера. %s", str(response))
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
