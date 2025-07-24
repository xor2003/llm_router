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


class DeploymentModel(BaseModel):
    """Represents a single backend model provider"""

    model_name: str  # Router group name, e.g., "gateway-model"
    litellm_params: LiteLLMParams


class BackendModel(BaseModel):
    """Represents a single backend model provider"""

    id: str = Field(...) # Unique backend LLM ID for state tracking
    group_name: str # LLM group name, e.g., "my-router"
    model_name: str # Full model identifier
    api_key: str
    api_base: str | None = None
    rpm: int | None = None


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

        for model_definition in raw_config.get("model_list", []):
            # Use group name from configuration
            group_name = model_definition["model_name"]
            litellm_params = model_definition.get("litellm_params", {})

            api_key = litellm_params.get("api_key", "")

            model_struct = {
                "group_name": group_name,  # Set group name directly
                "model_name": litellm_params["model"]  # Full model identifier
            }

            if not api_key:
                logging.warning(
                    f"API ключ для модели {litellm_params.get('model', 'unknown')} отсутствует. Пропускаем.",
                )
                continue

            if api_key.startswith("os.environ/"):
                env_var = api_key.split("/")[-1]
                # Use the env_settings object instead of os.getenv
                model_struct["api_key"] = getattr(env_settings, env_var, None)
                if not model_struct["api_key"]:
                    logging.error(
                        f"Переменная окружения {env_var} не найдена. Пропускаем модель.",
                    )
                    continue

            # Generate unique ID
            model_struct["id"] = (
                litellm_params["model"] + "/" + model_struct["api_key"][-4:]
            )
            model_struct["rpm"] = litellm_params["rpm"]
            model_struct["rpm"] = litellm_params["rpm"]

            if litellm_params.get("api_base") is None and litellm_params["model"].startswith("gemini/"):
                # Use the correct Gemini endpoint
                model_struct["api_base"] = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                )
                model_struct["model_name"] = litellm_params["model"].split("/")[1]
            else:
                model_struct["api_base"] = litellm_params["api_base"]
                # Set model_name to the full model identifier
                model_struct["model_name"] = litellm_params["model"]

            valid_models.append(model_struct)

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
