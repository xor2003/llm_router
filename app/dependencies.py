from functools import lru_cache

from .config import AppConfig, load_config
from .router import LLMRouter
from .state import ModelStateManager
from .client import LLMClient


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config("config.yaml")


@lru_cache(maxsize=1)
def get_state_manager() -> ModelStateManager:
    config = get_config()
    return ModelStateManager(config.model_list)


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    return LLMRouter(get_config().model_list, get_state_manager())


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient()
