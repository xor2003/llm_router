from functools import lru_cache

from app.client import GeminiClient, LLMClient, OpenAIClient
from app.config import AppConfig, BackendModel, load_config
from app.router import LLMRouter
from app.state import AvailabilityChecker, ModelStateManager, StateUpdater
from app.utils.model_grouper import ModelGrouper


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Dependency for getting configuration."""
    return load_config("config.yaml")


def get_llm_client(
    backend_model: BackendModel,
    router: LLMRouter,
    config: AppConfig,
) -> LLMClient:
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
    return LLMClient(
        generative_client=generative_client,
        backend_model=backend_model,
        router=router,
        pii_config=config.pii_config,
    )


@lru_cache(maxsize=1)
def get_state_manager() -> ModelStateManager:
    """Dependency for getting state manager."""
    availability_checker = AvailabilityChecker()
    state_updater = StateUpdater()
    return ModelStateManager(availability_checker, state_updater)


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    """Dependency for getting router."""
    config = get_config()
    grouper = ModelGrouper()
    return LLMRouter(config, grouper)


@lru_cache(maxsize=1)
def get_client_map() -> dict[str, LLMClient]:
    config = get_config()
    router = get_router()
    return {
        model.id: get_llm_client(model, router, config) for model in config.model_list
    }
