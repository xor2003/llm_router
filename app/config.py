import logging
import os
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings


# Загрузка .env файла
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


# Модели Pydantic для структуры config.yaml
class LiteLLMParams(BaseModel):
    model: str
    api_key: str
    api_base: Optional[str] = None
    rpm: Optional[int] = None
    # Добавьте другие параметры litellm по необходимости


class ModelDeployment(BaseModel):
    model_name: str  # Имя группы/маршрута, например "my-router"
    litellm_params: LiteLLMParams
    # Уникальный ID для отслеживания состояния
    deployment_id: str = Field(..., alias="id")

class ProxyServerConfig(BaseModel):
    port: int = 4000
    host: str = "0.0.0.0"


class RouterSettings(BaseModel):
    routing_strategy: Literal["simple-shuffle"] = "simple-shuffle"
    num_retries: int = 1


class AppConfig(BaseModel):
    proxy_server_config: ProxyServerConfig
    model_list: List[ModelDeployment]
    router_settings: RouterSettings


def load_config(path: str) -> AppConfig:
    """Загружает, парсит и валидирует конфигурацию из YAML файла."""
    logging.info(f"Загрузка конфигурации из {path}...")
    try:
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Парсинг переменных окружения
        valid_models = []

        for model_def in raw_config.get("model_list", []):
            params = model_def.get("litellm_params", {})

            # Generate unique ID
            model_id = model_def["model_name"] # + "/" + params["api_key"][-4:]
            model_def["id"] = model_id

            if params.get("api_base") is None and params["model"].startswith("gemini/"):
                # Use the correct Gemini endpoint
                params["api_base"] = "https://generativelanguage.googleapis.com/v1beta/models/"
                params["model"] = params["model"].split("/")[1]
            else:
                params["model"] = "/".join(params["model"].split("/")[1:])
            key_str = params.get("api_key", "")
            if key_str.startswith("os.environ/"):
                env_var = key_str.split("/")[-1]
                params["api_key"] = os.getenv(env_var)
                if not params["api_key"]:
                    logging.error(
                        f"Переменная окружения {env_var} не найдена. Пропускаем модель."
                    )
                    continue

            if not params.get("api_key"):
                logging.warning(
                    f"API ключ для модели {params.get('model', 'unknown')} отсутствует. Пропускаем."
                )
                continue

            valid_models.append(model_def)

        raw_config["model_list"] = valid_models

        # Set default router settings if missing
        if "router_settings" not in raw_config:
            raw_config["router_settings"] = {
                "routing_strategy": "simple-shuffle",
                "num_retries": 1,
            }

        return AppConfig.parse_obj(raw_config)
    except (FileNotFoundError, ValidationError, TypeError) as e:
        logging.error(f"Ошибка загрузки или валидации конфигурации: {e}")
        raise
