import time
from unittest.mock import patch

from app.state import AvailabilityChecker, ModelState, ModelStateManager, StateUpdater


# Helper function to create a test manager
def create_test_manager():
    checker = AvailabilityChecker()
    updater = StateUpdater()
    return ModelStateManager(checker, updater)


def test_model_state_initialization():
    state = ModelState()
    assert state.last_used == 0.0
    assert state.is_on_cooldown is False
    assert state.cooldown_until == 0.0
    assert state.failure_count == 0


def test_record_success():
    manager = create_test_manager()
    manager.record_success("model1")
    state = manager.states["model1"]
    assert state.failure_count == 0
    assert state.last_used > 0


def test_record_failure():
    manager = create_test_manager()
    manager.record_failure("model1", 500)
    state = manager.states["model1"]
    assert state.failure_count == 1


def test_record_failure_rate_limit():
    manager = create_test_manager()
    current_time = time.time()
    manager.record_failure("model1", 429)
    state = manager.states["model1"]
    assert state.failure_count == 1
    assert state.is_on_cooldown is True
    assert state.cooldown_until > current_time + 59  # Should be ~60s in future


@patch("time.time")
def test_is_available(mock_time):
    mock_time.return_value = 1000.0
    manager = create_test_manager()

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


# Test AvailabilityChecker directly
def test_availability_checker():
    checker = AvailabilityChecker()
    state = ModelState()
    current_time = time.time()

    # Test cooldown logic
    state.is_on_cooldown = True
    state.cooldown_until = current_time - 1  # Cooldown expired
    assert checker.check_availability(state, current_time) is True
    assert state.is_on_cooldown is False

    state.is_on_cooldown = True
    state.cooldown_until = current_time + 10  # Cooldown not expired
    assert checker.check_availability(state, current_time) is False

    # Test rate limiting
    state.is_on_cooldown = False
    state.last_used = current_time - 0.05  # 50ms ago
    assert checker.check_availability(state, current_time) is False

    state.last_used = current_time - 0.11  # 110ms ago
    assert checker.check_availability(state, current_time) is True


# Test StateUpdater directly
def test_state_updater():
    updater = StateUpdater()
    state = ModelState()
    current_time = time.time()

    # Test record_success
    updater.record_success(state, current_time)
    assert state.last_used == current_time
    assert state.failure_count == 0
    assert state.last_error is None
    assert state.last_error_timestamp is None

    # Test record_failure
    updater.record_failure(state, 500, "Test error", current_time)
    assert state.failure_count == 1
    assert state.last_error == "Test error"
    assert state.last_error_timestamp == current_time

    # Test set_cooldown
    reset_time = current_time + 60
    updater.set_cooldown(state, reset_time)
    assert state.is_on_cooldown is True
    assert state.cooldown_until == reset_time
