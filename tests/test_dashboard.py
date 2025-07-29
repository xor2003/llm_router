import time

from fastapi.testclient import TestClient

from app.state import AvailabilityChecker, ModelState, ModelStateManager, StateUpdater
from main import app


def test_dashboard_displays_model_status():
    # Setup mock state manager with required dependencies
    availability_checker = AvailabilityChecker()
    state_updater = StateUpdater()
    state_manager = ModelStateManager(availability_checker, state_updater)

    # Initialize models for the test
    state_manager.initialize_models(["model1", "model2"])

    # Create mock model states
    model1 = ModelState()
    model1.last_used = time.time() - 3600  # 1 hour ago
    model1.is_on_cooldown = False
    model1.cooldown_until = 0
    model1.failure_count = 0
    model1.last_error = None
    model1.last_error_timestamp = None

    model2 = ModelState()
    model2.last_used = time.time() - 60  # 1 minute ago
    model2.is_on_cooldown = True
    model2.cooldown_until = time.time() + 300  # 5 minutes from now
    model2.failure_count = 3
    model2.last_error = "Rate limit exceeded"
    model2.last_error_timestamp = time.time()

    state_manager.states = {"model1": model1, "model2": model2}

    # Patch the dependency
    from app.dependencies import get_state_manager

    app.dependency_overrides[get_state_manager] = lambda: state_manager

    # Create test client
    client = TestClient(app)

    # Make request to dashboard
    response = client.get("/dashboard")

    # Verify response
    assert response.status_code == 200
    html = response.text

    # Check model1 status
    assert "model1" in html
    assert "Available" in html
    assert "Never" not in html  # Should show actual timestamp

    # Check model2 status
    assert "model2" in html
    assert "Rate Limited" in html
    assert "Rate limit exceeded" in html

    # Check table structure
    assert "<th>Model ID</th>" in html
    assert "<th>Status</th>" in html
    assert "<th>Last Used</th>" in html
    assert "<th>Cooldown Until</th>" in html
    assert "<th>Failure Count</th>" in html
    assert "<th>Last Error</th>" in html
    assert "<th>Error Timestamp</th>" in html

    # Check color coding classes
    assert 'class="available"' in html
    assert 'class="cooldown"' in html

    # Clean up
    app.dependency_overrides = {}


def test_dashboard_with_no_models():
    # Setup empty state manager with required dependencies
    availability_checker = AvailabilityChecker()
    state_updater = StateUpdater()
    state_manager = ModelStateManager(availability_checker, state_updater)
    state_manager.states = {}

    # Patch the dependency
    from app.dependencies import get_state_manager

    app.dependency_overrides[get_state_manager] = lambda: state_manager

    # Create test client
    client = TestClient(app)

    # Make request to dashboard
    response = client.get("/dashboard")

    # Verify response
    assert response.status_code == 200
    html = response.text

    # Check that the table is empty except for headers
    assert "<tr>" in html  # Header row exists
    assert "model1" not in html

    # Clean up
    app.dependency_overrides = {}
