from functools import lru_cache

from app.client import LLMClient
from app.config import AppConfig, BackendModel, load_config
from app.router import LLMRouter
from app.state import ModelStateManager


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Зависимость для получения конфигурации."""
    return load_config("config.yaml")


# @lru_cache(maxsize=None)
def get_llm_client(backend_model: BackendModel) -> LLMClient:
    """Зависимость для получения HTTP клиента."""
    return LLMClient(backend_model)


@lru_cache(maxsize=1)
def get_state_manager() -> ModelStateManager:
    """Зависимость для получения менеджера состояний."""
    config = get_config()
    # Инициализируем с backend_model_id из конфигурации
    return ModelStateManager()


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    """Зависимость для получения роутера."""
    config = get_config()
    state_manager = get_state_manager()
    return LLMRouter(config.model_list, state_manager, config.router_settings)


@lru_cache(maxsize=1)
def get_client_map() -> dict[str, LLMClient]:
    config = get_config()
    return {model.backend_model_id: get_llm_client(model) for model in config.model_list}
