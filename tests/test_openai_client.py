from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import BackendModel
from app.dependencies import get_llm_client
from app.router import LLMRouter


@pytest.fixture
def mock_openai_backend_model():
    return BackendModel(
        id="openai-model",
        group_name="test-group",
        model_name="gpt-4",
        api_key="test-openai-key",
        api_base="https://api.openai.com/v1",
        provider="openai",
    )


@pytest.fixture
def mock_router():
    return MagicMock(spec=LLMRouter)


# Test streaming
@pytest.mark.asyncio
async def test_openai_streaming(mock_openai_backend_model, mock_router):
    """Verify streaming requests for OpenAI client."""
    from app.config import AppConfig, PIIConfig, ProxyServerConfig, RouterSettings
    config = AppConfig(
        proxy_server_config=ProxyServerConfig(),
        model_list=[],
        router_settings=RouterSettings(),
        mcp_tool_use_prompt_template="",
        pii_config=PIIConfig()
    )
    llm_client = get_llm_client(mock_openai_backend_model, mock_router, config)
    mock_chunks = [{"content": "chunk1"}, {"content": "chunk2"}]

    async def async_gen():
        for item in mock_chunks:
            yield item

    with patch.object(
        llm_client.generative_client,
        "generate_stream",
        return_value=async_gen(),
    ) as mock_generate_stream:
        payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}
        response_stream = await llm_client.make_request(payload)
        chunks = [chunk async for chunk in response_stream]

        assert chunks == mock_chunks
        mock_generate_stream.assert_called_once_with(payload)


# Test error propagation
@pytest.mark.asyncio
async def test_openai_error_propagation(mock_openai_backend_model, mock_router):
    """Verify OpenAI client propagates errors correctly."""
    from app.config import AppConfig, PIIConfig, ProxyServerConfig, RouterSettings
    config = AppConfig(
        proxy_server_config=ProxyServerConfig(),
        model_list=[],
        router_settings=RouterSettings(),
        mcp_tool_use_prompt_template="",
        pii_config=PIIConfig()
    )
    llm_client = get_llm_client(mock_openai_backend_model, mock_router, config)

    with patch.object(
        llm_client.generative_client,
        "generate",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.side_effect = Exception("API error")
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        with pytest.raises(Exception) as exc_info:
            await llm_client.make_request(payload)

        assert "API error" in str(exc_info.value)
        mock_generate.assert_awaited_once_with(payload)
