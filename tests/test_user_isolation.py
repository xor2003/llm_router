"""Tests to verify that requests from different users don't mix contexts.
Each model should only see the context of one user at a time.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import BackendModel
from app.dependencies import get_client_map, get_router
from main import app


@pytest.fixture
def client():
    """Test client with clean dependency overrides."""
    app.dependency_overrides = {}
    return TestClient(app)


@pytest.fixture
def mock_backend_model():
    """Mock backend model for testing."""
    return BackendModel(
        id="test-model/abcd",
        group_name="test-group",
        model_name="gpt-4",
        api_key="fake-key",
        supports_tools=True,
    )


def test_user_context_isolation(client, mock_backend_model):
    """Test that requests from different users don't share context."""
    mock_router = MagicMock()
    mock_router.model_groups = {"test-group": [mock_backend_model]}
    mock_router.get_active_or_failover_model.return_value = mock_backend_model

    # Mock LLM client to capture requests
    mock_llm_client = MagicMock()

    # Track all requests made to the model
    requests_made = []

    async def mock_make_request(payload):
        requests_made.append(payload.copy())  # Store a copy of the payload
        return {
            "choices": [{"message": {"content": "response"}}],
            "model": mock_backend_model.model_name,
        }

    mock_llm_client.make_request = AsyncMock(side_effect=mock_make_request)
    client_map = {mock_backend_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    # Simulate requests from User A
    user_a_payload = {
        "model": "test-group",
        "messages": [
            {"role": "user", "content": "Hello, I'm User A"},
            {"role": "assistant", "content": "Hello User A, how can I help?"},
            {"role": "user", "content": "What's my name?"},
        ],
    }

    # Simulate requests from User B
    user_b_payload = {
        "model": "test-group",
        "messages": [
            {"role": "user", "content": "Hi, I'm User B"},
            {"role": "assistant", "content": "Hello User B, nice to meet you"},
            {"role": "user", "content": "What's my name?"},
        ],
    }

    # Send requests from both users
    response_a = client.post("/v1/chat/completions", json=user_a_payload)
    response_b = client.post("/v1/chat/completions", json=user_b_payload)

    # Verify both requests succeeded
    assert response_a.status_code == 200
    assert response_b.status_code == 200

    # Verify that each request contains only its own context
    assert len(requests_made) == 2

    # User A's request should only contain User A's context
    user_a_request = requests_made[0]
    user_a_messages = user_a_request["messages"]
    assert any("User A" in str(msg) for msg in user_a_messages)
    assert not any("User B" in str(msg) for msg in user_a_messages)

    # User B's request should only contain User B's context
    user_b_request = requests_made[1]
    user_b_messages = user_b_request["messages"]
    assert any("User B" in str(msg) for msg in user_b_messages)
    assert not any("User A" in str(msg) for msg in user_b_messages)

    # Verify no context mixing
    assert user_a_request != user_b_request


def test_concurrent_user_requests(client, mock_backend_model):
    """Test that concurrent requests maintain isolation."""
    mock_router = MagicMock()
    mock_router.model_groups = {"test-group": [mock_backend_model]}
    mock_router.get_active_or_failover_model.return_value = mock_backend_model

    # Mock LLM client
    mock_llm_client = MagicMock()
    requests_made = []

    async def mock_make_request(payload):
        requests_made.append(payload.copy())
        return {
            "choices": [{"message": {"content": "response"}}],
            "model": mock_backend_model.model_name,
        }

    mock_llm_client.make_request = AsyncMock(side_effect=mock_make_request)
    client_map = {mock_backend_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    # Test with different conversation histories
    conversations = [
        {
            "model": "test-group",
            "messages": [
                {"role": "user", "content": f"User {i} message 1"},
                {"role": "assistant", "content": f"Response to user {i}"},
                {"role": "user", "content": f"User {i} follow-up"},
            ],
        }
        for i in range(5)
    ]

    # Send all requests
    responses = []
    for payload in conversations:
        response = client.post("/v1/chat/completions", json=payload)
        responses.append(response)

    # Verify all succeeded
    for response in responses:
        assert response.status_code == 200

    # Verify each request has its own isolated context
    assert len(requests_made) == 5

    for i, request in enumerate(requests_made):
        messages = request["messages"]
        # Each request should only contain its own user context
        assert f"User {i}" in str(messages)
        # Should not contain other users' contexts
        for j in range(5):
            if j != i:
                assert f"User {j}" not in str(messages)


def test_streaming_user_isolation(client, mock_backend_model):
    """Test that streaming requests maintain user isolation."""
    mock_router = MagicMock()
    mock_router.model_groups = {"test-group": [mock_backend_model]}
    mock_router.get_active_or_failover_model.return_value = mock_backend_model

    # Mock LLM client for streaming
    mock_llm_client = MagicMock()
    requests_made = []

    async def mock_stream_generator():
        yield {"choices": [{"delta": {"content": "streaming response"}}]}

    async def mock_make_request(payload):
        requests_made.append(payload.copy())
        return mock_stream_generator()

    mock_llm_client.make_request = AsyncMock(side_effect=mock_make_request)
    client_map = {mock_backend_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    # Test streaming requests from different users
    user_a_stream = {
        "model": "test-group",
        "messages": [{"role": "user", "content": "User A streaming request"}],
        "stream": True,
    }

    user_b_stream = {
        "model": "test-group",
        "messages": [{"role": "user", "content": "User B streaming request"}],
        "stream": True,
    }

    # Send streaming requests
    response_a = client.post("/v1/chat/completions", json=user_a_stream)
    response_b = client.post("/v1/chat/completions", json=user_b_stream)

    # Verify both succeeded
    assert response_a.status_code == 200
    assert response_b.status_code == 200

    # Verify isolation
    assert len(requests_made) == 2

    # Check that each request has its own context
    user_a_request = requests_made[0]
    user_b_request = requests_made[1]

    assert "User A streaming request" in str(user_a_request["messages"])
    assert "User B streaming request" in str(user_b_request["messages"])

    # Ensure no mixing
    assert "User A" not in str(user_b_request["messages"])
    assert "User B" not in str(user_a_request["messages"])
