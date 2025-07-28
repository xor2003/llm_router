import time
from unittest.mock import patch

from app.state import ModelState, ModelStateManager

def test_model_state_initialization():
    state = ModelState()
    assert state.last_used == 0.0
    assert state.is_on_cooldown is False
    assert state.cooldown_until == 0.0
    assert state.failure_count == 0

def test_record_success():
    manager = ModelStateManager()
    manager.record_success("model1")
    state = manager.states["model1"]
    assert state.failure_count == 0
    assert state.last_used > 0

def test_record_failure():
    manager = ModelStateManager()
    manager.record_failure("model1", 500)
    state = manager.states["model1"]
    assert state.failure_count == 1

def test_record_failure_rate_limit():
    manager = ModelStateManager()
    current_time = time.time()
    manager.record_failure("model1", 429)
    state = manager.states["model1"]
    assert state.failure_count == 1
    assert state.is_on_cooldown is True
    assert state.cooldown_until > current_time + 59  # Should be ~60s in future

@patch('time.time')
def test_is_available(mock_time):
    mock_time.return_value = 1000.0
    manager = ModelStateManager()
    
    # Initially, model should be available (no previous use)
    assert manager.is_available("model1") is True
    
    # After first success, model should be temporarily unavailable due to rate limiting
    manager.record_success("model1")
    assert manager.is_available("model1") is False
    
    # After 50ms, still unavailable
    mock_time.return_value = 1000.05
    assert manager.is_available("model1") is False
    
    # After 110ms, should be available again
    mock_time.return_value = 1000.11
    assert manager.is_available("model1") is True
    
    # Immediately after, it should be unavailable due to min time between requests
    mock_time.return_value = 1000.05  # 50ms later
    assert manager.is_available("model1") is False
    
    # After 110ms, it should be available again
    mock_time.return_value = 1000.11  # 110ms later
    assert manager.is_available("model1") is True
    
    # Record a rate limit failure (429) to trigger cooldown
    manager.record_failure("model1", 429)
    assert manager.is_available("model1") is False
    
    # After cooldown period, it should be available again
    mock_time.return_value = 1061.0  # 61 seconds later
    assert manager.is_available("model1") is True