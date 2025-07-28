import time
from unittest.mock import MagicMock

import pytest

from app.client import CustomRateLimitException, LLMClient
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
