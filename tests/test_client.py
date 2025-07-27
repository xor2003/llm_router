import pytest
from unittest.mock import AsyncMock, MagicMock
from app.client import LLMClient, RateLimitException, BaseGenerativeClient
from app.config import BackendModel
import time
from openai import APIStatusError
from httpx import Request, Response

@pytest.fixture
def mock_backend_model():
    return BackendModel(
        id="test-model",
        group_name="test-group",
        model_name="gemini/test-model",
        api_key="test-key",
        api_base="https://test.example.com",
        rpm=1000,
        supports_tools=False,
        supports_mcp=False
    )

@pytest.fixture
def mock_openai_backend_model():
    return BackendModel(
        id="openai-model",
        group_name="test-group",
        model_name="gpt-4",
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        rpm=1000,
        supports_tools=True,
        supports_mcp=False
    )


@pytest.mark.asyncio
async def test_gemini_non_streaming_request(mock_backend_model):
    client = LLMClient(mock_backend_model)
    client.generative_client = AsyncMock(spec=BaseGenerativeClient)
    client.generative_client.generate_content_async.return_value = MagicMock(text="Test response")

    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }

    response = await client.make_request(payload)

    assert response["model"] == "gemini/test-model"
    assert response["choices"][0]["message"]["content"] == "Test response"
    assert isinstance(response["created"], int)
    client.generative_client.generate_content_async.assert_awaited_once()

@pytest.mark.asyncio
async def test_gemini_streaming_request(mock_backend_model):
    client = LLMClient(mock_backend_model)
    client.generative_client = AsyncMock(spec=BaseGenerativeClient)
    mock_stream = MagicMock()
    mock_stream.__aiter__.return_value = [MagicMock(text="Chunk1"), MagicMock(text="Chunk2")]
    client.generative_client.generate_content_async.return_value = mock_stream

    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }

    response_stream = await client.make_request(payload)
    chunks = [chunk async for chunk in response_stream]

    assert len(chunks) == 2
    assert chunks[0].text == "Chunk1"
    assert chunks[1].text == "Chunk2"

@pytest.mark.asyncio
async def test_openai_compatible_request(mock_openai_backend_model):
    client = LLMClient(mock_openai_backend_model)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
    
    client.openai_client = AsyncMock()
    client.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7
    }
    
    response = await client.make_request(payload)
    
    assert response.choices[0].message.content == "OpenAI response"
    client.openai_client.chat.completions.create.assert_awaited_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7
    )

@pytest.mark.asyncio
async def test_rate_limit_handling(mock_openai_backend_model):
    client = LLMClient(mock_openai_backend_model)

    request = Request(method="POST", url="https://api.openai.com/v1/chat/completions")
    response = Response(
        status_code=429,
        request=request,
        headers={"X-RateLimit-Reset": str(time.time() + 5)}
    )
    mock_exception = APIStatusError(message="Rate limit exceeded", response=response, body=None)

    client.openai_client.chat.completions.create = AsyncMock(side_effect=mock_exception)

    with pytest.raises(RateLimitException) as exc_info:
        await client.make_request({"messages": [{"role": "user", "content": "Hello"}]})

    assert exc_info.value.reset_time > time.time()

def test_mcp_connection_management(mock_openai_backend_model):
    client = LLMClient(mock_openai_backend_model)
    
    assert "telegram" in client.mcp_manager.servers
    assert "weather" in client.mcp_manager.servers
    assert client.mcp_manager.servers["telegram"] == "https://mcp.telegram.example.com"

@pytest.mark.asyncio
async def test_gemini_message_formatting(mock_backend_model):
    client = LLMClient(mock_backend_model)
    client.generative_client = AsyncMock(spec=BaseGenerativeClient)
    client.generative_client.generate_content_async.return_value = MagicMock(text="Test response")

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"}
            ]
        }],
        "stream": False
    }

    await client.make_request(payload)

    call_args = client.generative_client.generate_content_async.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["role"] == "user"
    assert len(call_args[0]["parts"]) == 2
    assert call_args[0]["parts"][0]["text"] == "Hello"
    assert call_args[0]["parts"][1]["text"] == "World"