import pytest
from unittest.mock import MagicMock
from app.router import LLMRouter
from app.config import BackendModel
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
            provider="gemini"
        ),
        BackendModel(
            id="model2",
            group_name="group1",
            model_name="gpt-4",
            api_key="key2",
            api_base="https://openai.example.com",
            provider="openai"
        ),
        BackendModel(
            id="model3",
            group_name="group2",
            model_name="gpt-3.5",
            api_key="key3",
            api_base="https://openai.example.com",
            provider="openai"
        )
    ]

@pytest.fixture
def router(mock_models):
    state_manager = MagicMock(spec=ModelStateManager)
    settings = MagicMock()
    return LLMRouter(mock_models, state_manager, settings)

# Test model selection
def test_model_group_selection(router):
    """Verify router selects model from group."""
    router._state_manager.is_available.return_value = True
    model = router.get_next_backend_model("group1")
    assert model.id in ["model1", "model2"]

# Test fallback mechanism
def test_fallback_mechanism(router):
    """Verify router falls back to next model when unavailable."""
    # First model is unavailable
    router._state_manager.is_available.side_effect = [False, True]
    
    model = router.get_next_backend_model("group1")
    assert model.id == "model2"
    
# Test edge cases
def test_no_models_available(router):
    """Verify router handles no models available."""
    router._state_manager.is_available.return_value = False
    model = router.get_next_backend_model("group1")
    assert model is None

def test_invalid_group(router):
    """Verify router handles invalid group name."""
    model = router.get_next_backend_model("invalid_group")
    assert model is None