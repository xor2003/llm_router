from functools import lru_cache

from app.client import GeminiClient, LLMClient, OpenAIClient
from app.config import AppConfig, BackendModel, load_config
from app.router import LLMRouter
from app.state import ModelStateManager


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Зависимость для получения конфигурации."""
    return load_config("config.yaml")


def get_llm_client(backend_model: BackendModel, router: LLMRouter) -> LLMClient:
    """Factory function to create the appropriate LLMClient."""
    if backend_model.provider == "gemini":
        generative_client = GeminiClient(
            model_name=backend_model.model_name,
            api_key=backend_model.api_key,
        )
    else:  # Default to openai
        generative_client = OpenAIClient(
            model_name=backend_model.model_name,
            api_key=backend_model.api_key,
            api_base=backend_model.api_base,
        )
    return LLMClient(generative_client, backend_model, router)


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
    return {
        model.id: get_llm_client(model, get_router()) for model in config.model_list
    }
