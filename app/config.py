import logging
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# Pydantic models for config.yaml structure
class LiteLLMParams(BaseModel):
    model: str
    api_key: str
    api_base: str | None = None
    rpm: int | None = None
    # Add other litellm parameters as needed


class DeploymentModel(BaseModel):
    """Represents a single backend model provider"""

    model_name: str  # Router group name, e.g., "gateway-model"
    litellm_params: LiteLLMParams


class BackendModel(BaseModel):
    """Represents a single backend model provider"""

    id: str = Field(...)  # Unique backend LLM ID for state tracking
    group_name: str  # LLM group name, e.g., "my-router"
    model_name: str  # Full model identifier
    api_key: str
    api_base: str | None = None
    rpm: int | None = None
    supports_tools: bool = False  # Whether this model supports tool calls
    supports_mcp: bool = Field(
        default=False,
        description="Whether this model supports MCP protocol",
    )
    provider: Literal["openai", "gemini"] = "openai"


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
    mcp_tool_use_prompt_template: str

    # For backwards compatibility with Pydantic v1
    @classmethod
    def parse_obj(cls, obj: Any) -> "AppConfig":
        return cls.model_validate(obj)


# Loading .env file
class EnvSettings(BaseSettings):
    GEMINI_API_KEY: str = ""
    OPENROUTER_KEY1: str = ""
    OPENROUTER_KEY2: str = ""
    OPENROUTER_KEY3: str = ""
    OPENROUTER_KEY4: str = ""
    OPENROUTER_KEY5: str = ""
    OPENROUTER_KEY6: str = ""
    OPENROUTER_KEY7: str = ""
    OPENROUTER_KEY8: str = ""
    OPENROUTER_KEY9: str = ""
    OPENROUTER_KEY10: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


env_settings = EnvSettings()


def load_config(path: str) -> AppConfig:
    """Loads, parses and validates configuration from YAML file."""
    logger.info(f"Loading configuration from {path}...")
    try:
        with open(path) as f:
            raw_config = yaml.safe_load(f)

        # Parsing environment variables
        valid_models = []

        for model_definition in raw_config.get("model_list", []):
            # Use group name from configuration
            group_name = model_definition["model_name"]
            litellm_params = model_definition.get("litellm_params", {})

            # Skip models missing required 'model' parameter
            if "model" not in litellm_params:
                logger.error(
                    f"`model` parameter missing for group {group_name}. Skipping.",
                )
                continue

            api_key = litellm_params.get("api_key", "")

            model_struct = {
                "group_name": group_name,  # Set group name directly
                "model_name": litellm_params["model"],  # Full model identifier
            }

            if not api_key:
                logger.warning(
                    f"API key for model {litellm_params.get('model', 'unknown')} is missing. Skipping.",
                )
                continue

            if api_key.startswith("os.environ/"):
                env_var = api_key.split("/")[-1]
                # Use the env_settings object instead of os.getenv
                model_struct["api_key"] = getattr(env_settings, env_var, None)
                if not model_struct["api_key"]:
                    logger.error(
                        f"Environment variable {env_var} not found. Skipping model.",
                    )
                    continue

            # Generate unique ID
            model_struct["id"] = litellm_params["model"] + "/" + model_struct["api_key"][-4:]
            model_struct["rpm"] = litellm_params["rpm"]

            # Detect tool support based on model name
            model_name_str = litellm_params["model"]
            model_struct["supports_tools"] = any(
                kw in model_name_str.lower() for kw in ("gpt-4", "claude", "command-r", "mixtral")
            )

            if model_name_str.startswith("gemini/"):
                model_struct["provider"] = "gemini"
                model_struct["model_name"] = model_name_str.split("/")[1]
                model_struct["api_base"] = None  # Not used by GeminiClient
            else:
                model_struct["provider"] = "openai"
                model_struct["model_name"] = model_name_str
                model_struct["api_base"] = litellm_params.get("api_base")
            valid_models.append(model_struct)

        raw_config["model_list"] = valid_models

        # Set default router settings if missing
        if "router_settings" not in raw_config:
            raw_config["router_settings"] = {
                "routing_strategy": "simple-shuffle",
                "num_retries": 1,
            }

        # Load the MCP tool use prompt template
        with open("prompts/mcp_tool_use_prompt.txt") as f:
            raw_config["mcp_tool_use_prompt_template"] = f.read()

        return AppConfig.model_validate(raw_config)
    except (FileNotFoundError, ValidationError, TypeError) as e:
        logger.exception(f"Error loading or validating configuration: {e}")
        raise
