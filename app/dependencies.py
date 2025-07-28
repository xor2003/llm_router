from functools import lru_cache

from app.client import GeminiClient, LLMClient, OpenAIClient
from app.config import AppConfig, BackendModel, load_config
from app.router import LLMRouter
from app.state import ModelStateManager


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Dependency for getting configuration."""
    return load_config("config.yaml")


def get_llm_client(backend_model: BackendModel, router: LLMRouter) -> LLMClient:
    """Factory function to create the appropriate LLMClient."""
    if backend_model.provider == "gemini":
        generative_client: GeminiClient | OpenAIClient = GeminiClient(
            model_id=backend_model.id,
            model_name=backend_model.model_name,
            api_key=backend_model.api_key,
        )
    else:  # Default to openai
        if not backend_model.api_base:
            raise ValueError("api_base is required for OpenAI models")
        generative_client = OpenAIClient(
            model_id=backend_model.id,
            model_name=backend_model.model_name,
            api_key=backend_model.api_key,
            api_base=backend_model.api_base,
        )
    return LLMClient(generative_client, backend_model, router)


@lru_cache(maxsize=1)
def get_state_manager() -> ModelStateManager:
    """Dependency for getting state manager."""
    # Initialize with backend_model_id from configuration
    return ModelStateManager()


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    """Dependency for getting router."""
    config = get_config()

    return LLMRouter(config)


@lru_cache(maxsize=1)
def get_client_map() -> dict[str, LLMClient]:
    config = get_config()
    return {model.id: get_llm_client(model, get_router()) for model in config.model_list}
