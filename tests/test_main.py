from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import BackendModel
from app.dependencies import get_client_map, get_router
from main import app


# Mock the dependencies
@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_unsupported_model():
    return BackendModel(
        id="unsupported-model/abcd",
        group_name="test-group",
        model_name="unsupported-model",
        api_key="fake-key",
        supports_tools=False,
    )


@pytest.fixture
def mock_supported_model():
    return BackendModel(
        id="supported-model/efgh",
        group_name="test-group",
        model_name="gpt-4-turbo",
        api_key="fake-key",
        supports_tools=True,
    )


def test_tool_call_with_workaround(client, mock_unsupported_model):
    """Verify that the XML tool call workaround is applied for models
    that do not support native tool calling.
    """
    # Mock the router to return our unsupported model
    mock_router = MagicMock()
    mock_router.get_next_backend_model.return_value = mock_unsupported_model

    # Mock the client to return a pre-canned XML tool call response
    mock_llm_client = MagicMock()
    mock_response_obj = MagicMock()
    mock_response_dict = {
        "choices": [{"message": {"content": "<echo><message>test</message></echo>"}}],
    }
    mock_response_obj.model_dump.return_value = mock_response_dict
    mock_llm_client.make_request = AsyncMock(return_value=mock_response_obj)

    client_map = {mock_unsupported_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    payload = {
        "model": "test-group",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "echo", "parameters": {}}}],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    response_json = response.json()
    tool_call = response_json["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "echo"
    assert "'message': 'test'" in tool_call["function"]["arguments"]


def test_tool_call_native_passthrough(client, mock_supported_model):
    """Verify that requests are passed through without modification for models
    that support native tool calling.
    """
    mock_router = MagicMock()
    mock_router.get_next_backend_model.return_value = mock_supported_model

    mock_llm_client = MagicMock()
    mock_response_obj = MagicMock()
    mock_response_dict = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"message": "test"}',
                            },
                        },
                    ],
                },
            },
        ],
    }
    mock_response_obj.model_dump.return_value = mock_response_dict
    mock_llm_client.make_request = AsyncMock(return_value=mock_response_obj)

    client_map = {mock_supported_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    payload = {
        "model": "test-group",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "echo", "parameters": {}}}],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    response_json = response.json()
    assert "tool_calls" in response_json["choices"][0]["message"]
    # Assert that the original payload was passed to the client
    mock_llm_client.make_request.assert_awaited_once()
    sent_payload = mock_llm_client.make_request.call_args[0][0]
    assert "tools" in sent_payload
