import logging
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings


# Модели Pydantic для структуры config.yaml
class LiteLLMParams(BaseModel):
    model: str
    api_key: str
    api_base: str | None = None
    rpm: int | None = None
    # Добавьте другие параметры litellm по необходимости


class BackendModel(BaseModel):
    """Represents a single backend model provider"""

    model_name: str  # Router group name, e.g., "gateway-model"
    litellm_params: LiteLLMParams
    # Unique ID for state tracking
    backend_model_id: str = Field(..., alias="id")


class ProxyServerConfig(BaseModel):
    port: int = 4000
    host: str = "0.0.0.0"


class RouterSettings(BaseModel):
    routing_strategy: Literal["simple-shuffle"] = "simple-shuffle"
    num_retries: int = 1


class AppConfig(BaseModel):
    proxy_server_config: ProxyServerConfig
    model_list: list[BackendModel]
    router_settings: RouterSettings


# Загрузка .env файла
class EnvSettings(BaseSettings):
    GEMINI_API_KEY: str
    OPENROUTER_KEY1: str
    OPENROUTER_KEY2: str
    OPENROUTER_KEY3: str

    class Config:
        env_file = ".env"
        extra = "ignore"


env_settings = EnvSettings()


def load_config(path: str) -> AppConfig:
    """Загружает, парсит и валидирует конфигурацию из YAML файла."""
    logging.info(f"Загрузка конфигурации из {path}...")
    try:
        with open(path) as f:
            raw_config = yaml.safe_load(f)

        # Парсинг переменных окружения
        valid_models = []

        for model_def in raw_config.get("model_list", []):
            params = model_def.get("litellm_params", {})

            # Generate unique ID
            model_id = model_def["model_name"]  # + "/" + params["api_key"][-4:]
            model_def["id"] = model_id

            if params.get("api_base") is None and params["model"].startswith("gemini/"):
                # Use the correct Gemini endpoint
                params["api_base"] = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                )
                params["model"] = params["model"].split("/")[1]
            else:
                params["model"] = "/".join(params["model"].split("/")[1:])
            key_str = params.get("api_key", "")
            if key_str.startswith("os.environ/"):
                env_var = key_str.split("/")[-1]
                # Use the env_settings object instead of os.getenv
                params["api_key"] = getattr(env_settings, env_var, None)
                if not params["api_key"]:
                    logging.error(
                        f"Переменная окружения {env_var} не найдена. Пропускаем модель.",
                    )
                    continue

            if not params.get("api_key"):
                logging.warning(
                    f"API ключ для модели {params.get('model', 'unknown')} отсутствует. Пропускаем.",
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
        logging.exception(f"Ошибка загрузки или валидации конфигурации: {e}")
        raise
