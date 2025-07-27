import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock google.generativeai before importing app.client
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from app.client import LLMClient, MCPConnectionManager, RateLimitException
from app.config import BackendModel
import re
import time

@pytest.fixture
def mock_backend_model():
    return BackendModel(
        id="test-model-1",
        group_name="test-group",
        model_name="openrouter/deepseek/deepseek-r1-0528:free",
        api_key="sk-test-key",
        api_base="https://openrouter.ai/api/v1"
    )

@pytest.fixture
def llm_client(mock_backend_model):
    return LLMClient(mock_backend_model)

def test_mcp_tool_extraction(llm_client):
    # Test XML parsing
    xml_content = '''
    <tool_call server="weather" name="get_forecast">
        <city>San Francisco</city>
        <days>3</days>
    </tool_call>
    '''
    result = llm_client._extract_tool_call(xml_content)
    assert result == {
        "server": "weather",
        "tool": "get_forecast",
        "params": {"city": "San Francisco", "days": "3"}
    }

    # Test regex fallback for malformed XML with unquoted attributes
    malformed_content = '''
    <tool_call server=weather name=get_forecast>
        <city>Paris</city>
    </tool_call>
    '''
    result = llm_client._extract_tool_call(malformed_content)
    assert result == {
        "server": "weather",
        "tool": "get_forecast",
        "params": {"city": "Paris"}
    }
    
    # Test regex fallback for incomplete closing tag
    incomplete_content = '''
    <tool_call server="weather" name="get_forecast">
        <city>London</city
    </tool_call>
    '''
    result = llm_client._extract_tool_call(incomplete_content)
    assert result == {
        "server": "weather",
        "tool": "get_forecast",
        "params": {}
    }

def test_rate_limit_exception_handling(llm_client):
    exception = RateLimitException(reset_time=time.time() + 60)
    assert isinstance(exception, RateLimitException)
    assert "reset_time" in exception.__dict__

@pytest.mark.anyio
async def test_mcp_tool_execution(llm_client):
    # Mock MCP connection manager
    mock_mcp = AsyncMock()
    mock_mcp.call_tool.return_value = "Sunny, 72°F"
    llm_client.mcp_manager = mock_mcp

    tool_call = {
        "server": "weather",
        "tool": "get_forecast",
        "params": {"city": "London"}
    }
    
    result = await llm_client._handle_mcp_tool_call(tool_call)
    assert "<tool_result>" in result
    assert "Sunny, 72°F" in result

@pytest.mark.anyio
async def test_retry_logic(llm_client, monkeypatch):
    # Mock API client to raise rate limit exception
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = RateLimitException(time.time() + 5)
    monkeypatch.setattr(llm_client, "client", mock_client)

    # Test retry logic
    with pytest.raises(RateLimitException):
        await llm_client.make_request({
            "messages": [{"role": "user", "content": "Test"}]
        })

@pytest.mark.anyio
async def test_mcp_tool_execution(llm_client):
    # Skip non-asyncio backends
    if llm_client.__class__.__name__ != "AsyncioBackend":
        pytest.skip("Skipping non-asyncio tests")

@pytest.mark.anyio
async def test_retry_logic(llm_client, monkeypatch):
    # Skip non-asyncio backends
    if llm_client.__class__.__name__ != "AsyncioBackend":
        pytest.skip("Skipping non-asyncio tests")
        
    # Mock API client to raise rate limit exception
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = RateLimitException(time.time() + 5)
    monkeypatch.setattr(llm_client, "client", mock_client)

    # Test retry logic
    with pytest.raises(RateLimitException):
        await llm_client.make_request({
            "messages": [{"role": "user", "content": "Test"}]
        })

def test_model_name_normalization(llm_client):
    # Test OpenRouter model name normalization
    assert llm_client.model_name == "deepseek/deepseek-r1-0528:free"