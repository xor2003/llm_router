from functools import lru_cache
from typing import Dict
from app.client import LLMClient
from app.config import AppConfig, load_config, ModelDeployment
from app.router import LLMRouter
from app.state import ModelStateManager


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Зависимость для получения конфигурации."""
    return load_config("config.yaml")


#@lru_cache(maxsize=None)
def get_llm_client(deployment: ModelDeployment) -> LLMClient:
    """Зависимость для получения HTTP клиента."""
    return LLMClient(deployment)


@lru_cache(maxsize=1)
def get_state_manager() -> ModelStateManager:
    """Зависимость для получения менеджера состояний."""
    config = get_config()
    # Инициализируем с deployment_id из конфигурации
    return ModelStateManager(config.model_list)


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    """Зависимость для получения роутера."""
    config = get_config()
    state_manager = get_state_manager()
    return LLMRouter(config.model_list, state_manager, config.router_settings)


@lru_cache(maxsize=1)
def get_client_map() -> Dict[str, LLMClient]:
    config = get_config()
    return {
        dep.deployment_id: get_llm_client(dep)
        for dep in config.model_list
    }
