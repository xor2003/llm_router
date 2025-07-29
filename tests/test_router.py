import pytest

from app.config import AppConfig, BackendModel
from app.router import LLMRouter
from app.state import ModelStateManager
from app.utils.model_grouper import ModelGrouper


@pytest.fixture
def mock_app_config():
    models = [
        BackendModel(
            id="model1",
            group_name="group1",
            model_name="model1",
            api_key="key1",
            supports_tools=True,
            supports_mcp=False,
        ),
        BackendModel(
            id="model2",
            group_name="group1",
            model_name="model2",
            api_key="key2",
            supports_tools=False,
            supports_mcp=False,
        ),
    ]

    return AppConfig(
        proxy_server_config={"port": 8000, "host": "localhost"},
        model_list=models,
        router_settings={
            "routing_strategy": "simple-shuffle",
            "num_retries": 3,
        },
        mcp_tool_use_prompt_template="Test prompt template",
    )


def test_router_initialization(mock_app_config):
    model_grouper = ModelGrouper()
    router = LLMRouter(mock_app_config, model_grouper)
    assert "group1" in router.model_groups
    assert len(router.model_groups["group1"]) == 2


def test_get_active_model(mock_app_config):
    from app.state import AvailabilityChecker, StateUpdater

    availability_checker = AvailabilityChecker()
    state_updater = StateUpdater()
    state_manager = ModelStateManager(availability_checker, state_updater)
    model_grouper = ModelGrouper()
    router = LLMRouter(mock_app_config, model_grouper)

    # Should get first model initially
    model = router.get_active_or_failover_model("group1", state_manager)
    assert model.id == "model1"

    # After failure, should get next model
    state_manager.record_failure("model1", 429)
    model = router.get_active_or_failover_model("group1", state_manager)
    assert model.id == "model2"

    # After all models fail, should return None
    state_manager.record_failure("model2", 429)
    model = router.get_active_or_failover_model("group1", state_manager)
    assert model is None
