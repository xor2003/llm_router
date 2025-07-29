from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import BackendModel
from app.dependencies import get_llm_client
from app.router import LLMRouter


@pytest.fixture
def mock_gemini_backend_model():
    return BackendModel(
        id="gemini-model",
        group_name="test-group",
        model_name="gemini/gemini-pro",
        api_key="test-gemini-key",
        api_base="https://generativelanguage.googleapis.com/v1beta/models/",
        provider="gemini",
    )


@pytest.fixture
def mock_router():
    return MagicMock(spec=LLMRouter)


# Test XML translation
@pytest.mark.asyncio
async def test_gemini_xml_translation(mock_gemini_backend_model, mock_router):
    """Verify Gemini handles XML responses correctly."""
    from app.config import AppConfig, PIIConfig, ProxyServerConfig, RouterSettings

    config = AppConfig(
        proxy_server_config=ProxyServerConfig(),
        model_list=[],
        router_settings=RouterSettings(),
        mcp_tool_use_prompt_template="",
        pii_config=PIIConfig(),
    )
    llm_client = get_llm_client(mock_gemini_backend_model, mock_router, config)
    xml_response = "<response><message>Test XML</message></response>"
    mock_response = {"choices": [{"message": {"content": xml_response}}]}

    with patch.object(
        llm_client.generative_client,
        "generate",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = mock_response
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        response = await llm_client.make_request(payload)

        # Verify XML is properly handled in response
        assert "<" in response["choices"][0]["message"]["content"]
        mock_generate.assert_awaited_once_with(payload)


# Test streaming
@pytest.mark.asyncio
async def test_gemini_streaming(mock_gemini_backend_model, mock_router):
    """Verify streaming requests for Gemini client."""
    from app.config import AppConfig, PIIConfig, ProxyServerConfig, RouterSettings

    config = AppConfig(
        proxy_server_config=ProxyServerConfig(),
        model_list=[],
        router_settings=RouterSettings(),
        mcp_tool_use_prompt_template="",
        pii_config=PIIConfig(),
    )
    llm_client = get_llm_client(mock_gemini_backend_model, mock_router, config)
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


# Test error handling
@pytest.mark.asyncio
async def test_gemini_error_handling(mock_gemini_backend_model, mock_router):
    """Verify Gemini client handles API errors correctly."""
    from app.config import AppConfig, PIIConfig, ProxyServerConfig, RouterSettings

    config = AppConfig(
        proxy_server_config=ProxyServerConfig(),
        model_list=[],
        router_settings=RouterSettings(),
        mcp_tool_use_prompt_template="",
        pii_config=PIIConfig(),
    )
    llm_client = get_llm_client(mock_gemini_backend_model, mock_router, config)

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
