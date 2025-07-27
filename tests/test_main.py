import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import BackendModel
from app.dependencies import get_client_map, get_router
from main import app


# Mock the dependencies
@pytest.fixture
def client():
    # Сбрасываем переопределения зависимостей перед каждым тестом
    app.dependency_overrides = {}
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
    mock_router = MagicMock()
    mock_router.get_next_backend_model.return_value = mock_unsupported_model

    # --- ИЗМЕНЕНИЕ №1: Мок должен возвращать АСИНХРОННЫЙ ГЕНЕРАТОР ---
    # Наш код теперь ожидает поток (генератор), а не простой объект.
    async def mock_stream_generator():
        # Имитируем, что модель возвращает XML по частям
        yield {"choices": [{"delta": {"content": "<echo>"}}]}
        yield {"choices": [{"delta": {"content": "<message>test</message>"}}]}
        yield {"choices": [{"delta": {"content": "</echo>"}}]}

    mock_llm_client = MagicMock()
    # `make_request` теперь возвращает асинхронный генератор
    mock_llm_client.make_request = AsyncMock(return_value=mock_stream_generator())

    client_map = {mock_unsupported_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    payload = {
        "model": "test-group",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        "stream": True, # Указываем, что клиент запрашивает поток
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    
    # --- ИЗМЕНЕНИЕ №2: Правильно читаем поток SSE ---
    # Ответ теперь - это поток Server-Sent Events, его нужно парсить по-другому.
    lines = response.text.strip().split('\n')
    # Нас интересует только строка с данными, не пустые строки и не "[DONE]"
    data_line = [line for line in lines if line.startswith('data:') and "[DONE]" not in line][0]
    
    # Убираем "data: " и парсим JSON
    response_json = json.loads(data_line.replace("data: ", ""))

    # Проверяем, что в ответе есть правильно сформированный tool_call
    tool_call = response_json["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "echo"
    # Аргументы теперь - это строка JSON, поэтому парсим и ее
    arguments = json.loads(tool_call["function"]["arguments"])
    assert arguments["message"] == "test"


def test_tool_call_native_passthrough(client, mock_supported_model):
    """Verify that requests are passed through without modification for models
    that support native tool calling.
    """
    mock_router = MagicMock()
    mock_router.get_next_backend_model.return_value = mock_supported_model

    # --- ИЗМЕНЕНИЕ №3: Мок должен возвращать СЛОВАРЬ, а не MagicMock ---
    # Это исправит ошибку `TypeError: Object of type MagicMock is not JSON serializable`
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
                "finish_reason": "tool_calls"
            },
        ],
    }
    mock_llm_client = MagicMock()
    # `make_request` теперь возвращает словарь
    mock_llm_client.make_request = AsyncMock(return_value=mock_response_dict)

    client_map = {mock_supported_model.id: mock_llm_client}

    app.dependency_overrides[get_router] = lambda: mock_router
    app.dependency_overrides[get_client_map] = lambda: client_map

    payload = {
        "model": "test-group",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        "stream": False, # Указываем, что это не-потоковый запрос
    }

    response = client.post("/v1/chat/completions", json=payload)

    # Теперь тест должен проходить
    assert response.status_code == 200
    response_json = response.json()
    assert "tool_calls" in response_json["choices"][0]["message"]
    
    # Проверяем, что `make_request` был вызван с оригинальным payload
    mock_llm_client.make_request.assert_awaited_once()
    sent_payload = mock_llm_client.make_request.call_args[0][0]
    assert "tools" in sent_payload
    