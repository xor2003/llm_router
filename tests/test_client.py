import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Request, Response
from openai import APIStatusError

from app.client import GeminiClient, LLMClient, OpenAIClient, RateLimitException
from app.config import BackendModel
from app.dependencies import get_llm_client
from app.router import LLMRouter


@pytest.fixture
def mock_gemini_backend_model():
    """Fixture for a Gemini backend model configuration."""
    return BackendModel(
        id="gemini-model",
        group_name="test-group",
        model_name="gemini/gemini-pro",
        api_key="test-gemini-key",
        api_base="https://generativelanguage.googleapis.com/v1beta/models/",
        provider="gemini",
    )


@pytest.fixture
def mock_openai_backend_model():
    """Fixture for an OpenAI backend model configuration."""
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
    """Fixture for a mock router."""
    return MagicMock(spec=LLMRouter)


@pytest.mark.asyncio
async def test_llm_client_with_gemini_streaming(mock_gemini_backend_model, mock_router):
    """Verify streaming requests for Gemini client."""
    llm_client = get_llm_client(mock_gemini_backend_model, mock_router)
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


@pytest.mark.asyncio
async def test_llm_client_with_openai_non_streaming(
    mock_openai_backend_model,
    mock_router,
):
    """Verify non-streaming requests for OpenAI client."""
    llm_client = get_llm_client(mock_openai_backend_model, mock_router)
    mock_response = {"choices": [{"message": {"content": "OpenAI response"}}]}

    with patch.object(
        llm_client.generative_client,
        "generate",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = mock_response
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        response = await llm_client.make_request(payload)

        assert response == mock_response
        mock_generate.assert_awaited_once_with(payload)


@pytest.mark.asyncio
async def test_llm_client_with_openai_streaming(mock_openai_backend_model, mock_router):
    """Verify streaming requests for OpenAI client."""
    llm_client = get_llm_client(mock_openai_backend_model, mock_router)
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


def test_get_llm_client_instantiates_gemini_client(
    mock_gemini_backend_model,
    mock_router,
):
    """Ensure the correct client (Gemini) is instantiated based on config."""
    with patch("app.dependencies.GeminiClient", spec=GeminiClient) as mock_gemini:
        llm_client = get_llm_client(mock_gemini_backend_model, mock_router)
        assert isinstance(llm_client, LLMClient)
        mock_gemini.assert_called_once_with(
            model_id=mock_gemini_backend_model.id,
            model_name=mock_gemini_backend_model.model_name,
            api_key=mock_gemini_backend_model.api_key,
        )
        assert llm_client.generative_client == mock_gemini.return_value


def test_get_llm_client_instantiates_openai_client(
    mock_openai_backend_model,
    mock_router,
):
    """Ensure the correct client (OpenAI) is instantiated based on config."""
    with patch("app.dependencies.OpenAIClient", spec=OpenAIClient) as mock_openai:
        llm_client = get_llm_client(mock_openai_backend_model, mock_router)
        assert isinstance(llm_client, LLMClient)
        mock_openai.assert_called_once_with(
            model_id=mock_openai_backend_model.id,
            model_name=mock_openai_backend_model.model_name,
            api_key=mock_openai_backend_model.api_key,
            api_base=mock_openai_backend_model.api_base,
        )
        assert llm_client.generative_client == mock_openai.return_value


@pytest.mark.asyncio
async def test_rate_limit_exception_handling(mock_openai_backend_model, mock_router):
    """Verify that rate limit errors are correctly handled."""
    llm_client = get_llm_client(mock_openai_backend_model, mock_router)

    request = Request(method="POST", url="https://api.openai.com/v1/chat/completions")
    response = Response(
        status_code=429,
        request=request,
        headers={"X-RateLimit-Reset": str(time.time() + 60)},
    )
    api_error = APIStatusError(
        message="Rate limit exceeded",
        response=response,
        body=None,
    )

    with patch.object(
        llm_client.generative_client,
        "generate",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.side_effect = api_error
        with pytest.raises(RateLimitException) as exc_info:
            await llm_client.make_request(
                {"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert exc_info.value.reset_time > time.time()
        mock_generate.assert_awaited_once()
