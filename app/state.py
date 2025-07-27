import logging
import time


class ModelState:
    """Tracks state for a single model backend_model."""

    def __init__(self):
        self.last_used: float = 0.0
        self.is_on_cooldown: bool = False
        self.cooldown_until: float = 0.0
        self.failure_count: int = 0


class ModelStateManager:
    """Manages state for all model backend_models."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.states: dict[str, ModelState] = {}

    def initialize_models(self, model_ids: list[str]):
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

        # Check if on cooldown
        if state.is_on_cooldown:
            if current_time >= state.cooldown_until:
                state.is_on_cooldown = False
                state.failure_count = 0  # Reset failure count after cooldown
                self.logger.info(f"Cooldown ended for {backend_model_id}")
            else:
                # Still on cooldown
                self.logger.info(
                    f"Model {backend_model_id} on cooldown until {state.cooldown_until}",
                )
                return False

        # Implement basic rate limiting
        time_since_last_use = current_time - state.last_used
        if time_since_last_use < 0.1:  # 100ms minimum between requests
            return False

        return True

    def record_success(self, backend_model_id: str):
        """Record a successful request to a backend_model."""
        state = self._get_state(backend_model_id)
        state.last_used = time.time()
        state.failure_count = 0
        self.logger.info(f"Recorded success for {backend_model_id}")

    def record_failure(self, backend_model_id: str, status_code: int):
        """Record a failed request to a backend_model."""
        state = self._get_state(backend_model_id)
        state.failure_count += 1
        self.logger.warning(
            f"Recorded failure #{state.failure_count} for {backend_model_id} (status: {status_code})",
        )

        # For rate limit errors, set cooldown
        if status_code == 429:
            self.set_cooldown(backend_model_id, 60)  # Default 60s cooldown

    def set_cooldown(self, backend_model_id: str, reset_time: float):
        """Set cooldown until specified reset time"""
        state = self._get_state(backend_model_id)
        current_time = time.time()

        # Calculate duration until reset time
        duration = max(reset_time - current_time, 1)  # At least 1 second

        state.is_on_cooldown = True
        state.cooldown_until = reset_time
        self.logger.info(
            f"Set cooldown for {backend_model_id} for {duration:.1f}s until {time.ctime(reset_time)}",
        )
