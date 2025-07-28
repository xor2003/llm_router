from unittest.mock import MagicMock

import pytest

from app.config import BackendModel
from app.router import LLMRouter
from app.state import ModelStateManager


@pytest.fixture
def mock_models():
    return [
        BackendModel(
            id="model1",
            group_name="group1",
            model_name="gemini/gemini-pro",
            api_key="key1",
            api_base="https://gemini.example.com",
            provider="gemini",
        ),
        BackendModel(
            id="model2",
            group_name="group1",
            model_name="gpt-4",
            api_key="key2",
            api_base="https://openai.example.com",
            provider="openai",
        ),
        BackendModel(
            id="model3",
            group_name="group2",
            model_name="gpt-3.5",
            api_key="key3",
            api_base="https://openai.example.com",
            provider="openai",
        ),
    ]


@pytest.fixture
def router(mock_models):
    state_manager = MagicMock(spec=ModelStateManager)
    settings = MagicMock()
    return LLMRouter(settings)
