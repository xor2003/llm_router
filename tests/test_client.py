import time
from unittest.mock import MagicMock

import pytest

from app.client import CustomRateLimitException, LLMClient, OpenAIClient
from app.config import BackendModel

# ... existing tests ...


@pytest.mark.asyncio
async def test_rate_limiting_exception():
    # Create a backend model with RPM limit
    backend_model = BackendModel(
        id="test-model",
        group_name="test-group",
        model_name="gpt-4",
        api_key="test-key",
        rpm=2,
    )

    # Create mock dependencies
    mock_generative_client = MagicMock()
    mock_router = MagicMock()

    # Create LLMClient
    client = LLMClient(
        generative_client=mock_generative_client,
        backend_model=backend_model,
        router=mock_router,
    )

    # Mock the generative client to raise CustomRateLimitException
    mock_generative_client.generate.side_effect = CustomRateLimitException(time.time() + 60)

    # The request should raise CustomRateLimitException
    with pytest.raises(CustomRateLimitException):
        await client.make_request({})


@pytest.mark.asyncio
async def test_streaming_generation():
    # Create a backend model
    backend_model = BackendModel(
        id="test-model",
        group_name="test-group",
        model_name="gpt-4",
        api_key="test-key",
    )

    # Create mock dependencies
    mock_generative_client = MagicMock()
    mock_router = MagicMock()

    # Create mock async stream
    mock_chunk1 = MagicMock()
    mock_chunk1.model_dump.return_value = {"content": "Hello"}
    mock_chunk2 = MagicMock()
    mock_chunk2.model_dump.return_value = {"content": " World"}

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    mock_generative_client.chat.completions.create.return_value = mock_stream()

    # Create OpenAIClient with required parameters
    client = OpenAIClient(
        model_id=backend_model.id,
        model_name=backend_model.model_name,
        api_key=backend_model.api_key,
        api_base="https://api.openai.com",
    )
    # Set the client property to our mock
    client.client = mock_generative_client

    # Test streaming
    chunks = []
    async for chunk in client.generate_stream({"messages": [{"role": "user", "content": "Hello"}]}):
        chunks.append(chunk)

    # Verify results
    assert chunks == [{"content": "Hello"}, {"content": " World"}]
    mock_generative_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )


@pytest.mark.asyncio
async def test_generate_stream_async_iteration():
    """Test that generate_stream properly handles both async generators and coroutines"""
    # Create a backend model
    backend_model = BackendModel(
        id="test-model",
        group_name="test-group",
        model_name="gpt-4",
        api_key="test-key",
    )

    # Create mock dependencies
    mock_generative_client = MagicMock()

    # Test case 1: Async generator
    mock_chunk1 = MagicMock()
    mock_chunk1.model_dump.return_value = {"content": "First"}
    mock_chunk2 = MagicMock()
    mock_chunk2.model_dump.return_value = {"content": "Second"}

    async def mock_async_generator():
        yield mock_chunk1
        yield mock_chunk2

    mock_generative_client.chat.completions.create.return_value = mock_async_generator()

    # Create OpenAIClient
    client = OpenAIClient(
        model_id=backend_model.id,
        model_name=backend_model.model_name,
        api_key=backend_model.api_key,
        api_base="https://api.openai.com",
    )
    client.client = mock_generative_client

    # Test streaming with async generator
    chunks = []
    async for chunk in client.generate_stream({"messages": [{"role": "user", "content": "Test"}]}):
        chunks.append(chunk)

    assert chunks == [{"content": "First"}, {"content": "Second"}]

    # Reset mock
    mock_generative_client.reset_mock()
    chunks.clear()

    # Test case 2: Coroutine returning async generator
    async def mock_coroutine():
        return mock_async_generator()

    mock_generative_client.chat.completions.create.return_value = mock_coroutine()

    # Test streaming with coroutine
    async for chunk in client.generate_stream({"messages": [{"role": "user", "content": "Test"}]}):
        chunks.append(chunk)

    assert chunks == [{"content": "First"}, {"content": "Second"}]
