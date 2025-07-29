import logging
import time


class ModelState:
    """Tracks state for a single model backend_model."""

    def __init__(self) -> None:
        self.last_used: float = 0.0
        self.is_on_cooldown: bool = False
        self.cooldown_until: float = 0.0
        self.failure_count: int = 0
        self.last_error: str | None = None
        self.last_error_timestamp: float | None = None


class AvailabilityChecker:
    """Handles availability checking logic"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def check_availability(self, state: ModelState, current_time: float) -> bool:
        """Check if a backend model is available for use."""
        # Check if on cooldown
        if state.is_on_cooldown:
            if current_time >= state.cooldown_until:
                state.is_on_cooldown = False
                state.failure_count = 0  # Reset failure count after cooldown
                self.logger.info("Cooldown ended for model")
            else:
                # Still on cooldown
                self.logger.info(
                    f"Model on cooldown until {state.cooldown_until}",
                )
                return False

        # Implement basic rate limiting - only apply if there was a previous request
        if state.last_used > 0:
            time_since_last_use = current_time - state.last_used
            if time_since_last_use < 0.1:  # 100ms minimum between requests
                return False

        return True


class StateUpdater:
    """Handles state modification logic"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def record_success(self, state: ModelState, current_time: float) -> None:
        """Record a successful request to a backend_model."""
        state.last_used = current_time
        state.failure_count = 0
        state.last_error = None
        state.last_error_timestamp = None
        self.logger.info("Recorded success for model")

    def record_failure(
        self,
        state: ModelState,
        status_code: int,
        error_message: str | None,
        current_time: float,
    ) -> None:
        """Record a failed request to a backend_model."""
        state.failure_count += 1
        state.last_error = error_message
        state.last_error_timestamp = current_time
        self.logger.warning(
            f"Recorded failure #{state.failure_count} (status: {status_code})",
        )

    def set_cooldown(self, state: ModelState, reset_time: float) -> None:
        """Set cooldown until specified reset time"""
        state.is_on_cooldown = True
        state.cooldown_until = reset_time
        self.logger.info(
            f"Set cooldown until {time.ctime(reset_time)}",
        )


class ModelStateManager:
    """Manages state for all model backend_models using dependency injection."""

    def __init__(self, availability_checker: AvailabilityChecker, state_updater: StateUpdater) -> None:
        self.logger = logging.getLogger(__name__)
        self.states: dict[str, ModelState] = {}
        self.availability_checker = availability_checker
        self.state_updater = state_updater

    def initialize_models(self, model_ids: list[str]) -> None:
        """Initialize state for multiple models at once"""
        for model_id in model_ids:
            if model_id not in self.states:
                self.states[model_id] = ModelState()
                self.logger.info(f"Initialized state for model: {model_id}")

    def _get_state(self, backend_model_id: str) -> ModelState:
        """Get or create state for a backend_model."""
        if backend_model_id not in self.states:
            self.states[backend_model_id] = ModelState()
        return self.states[backend_model_id]

    def is_available(self, backend_model_id: str) -> bool:
        """Check if a backend model is available for use."""
        state = self._get_state(backend_model_id)
        current_time = time.time()
        return self.availability_checker.check_availability(state, current_time)

    def record_success(self, backend_model_id: str) -> None:
        """Record a successful request to a backend_model."""
        state = self._get_state(backend_model_id)
        current_time = time.time()
        self.state_updater.record_success(state, current_time)

    def record_failure(
        self,
        backend_model_id: str,
        status_code: int,
        error_message: str | None = None,
    ) -> None:
        """Record a failed request to a backend_model."""
        state = self._get_state(backend_model_id)
        current_time = time.time()
        self.state_updater.record_failure(state, status_code, error_message, current_time)

        # For rate limit errors, set cooldown
        if status_code == 429:
            self.state_updater.set_cooldown(state, current_time + 60)  # 60s cooldown
