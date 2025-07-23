import time
import logging
from typing import Dict, Optional

class ModelState:
    """Tracks state for a single model deployment."""
    def __init__(self):
        self.last_used: float = 0.0
        self.is_on_cooldown: bool = False
        self.cooldown_until: float = 0.0
        self.failure_count: int = 0

class ModelStateManager:
    """Manages state for all model deployments."""
    def __init__(self):
        self.states: Dict[str, ModelState] = {}
        self.logger = logging.getLogger(__name__)

    def _get_state(self, deployment_id: str) -> ModelState:
        """Get or create state for a deployment."""
        if deployment_id not in self.states:
            self.states[deployment_id] = ModelState()
        return self.states[deployment_id]

    def is_available(self, deployment_id: str) -> bool:
        """Check if a deployment is available for use."""
        state = self._get_state(deployment_id)
        current_time = time.time()
        
        # Check if on cooldown
        if state.is_on_cooldown:
            if current_time >= state.cooldown_until:
                state.is_on_cooldown = False
                self.logger.info(f"Cooldown ended for {deployment_id}")
            else:
                # Still on cooldown
                self.logger.info(
                    f"Model {deployment_id} on cooldown until {state.cooldown_until}"
                )
                return False
        
        # Additional availability checks could be added here
        return True

    def record_success(self, deployment_id: str):
        """Record a successful request to a deployment."""
        state = self._get_state(deployment_id)
        state.last_used = time.time()
        state.failure_count = 0
        self.logger.info(f"Recorded success for {deployment_id}")

    def record_failure(self, deployment_id: str, status_code: int):
        """Record a failed request to a deployment."""
        state = self._get_state(deployment_id)
        state.failure_count += 1
        self.logger.warning(
            f"Recorded failure #{state.failure_count} for {deployment_id} (status: {status_code})"
        )
        
        # For rate limit errors, set cooldown
        if status_code == 429:
            self.set_cooldown(deployment_id, 60)  # Default 60s cooldown

    def set_cooldown(self, deployment_id: str, duration: float):
        """Set a cooldown period for a deployment."""
        state = self._get_state(deployment_id)
        state.is_on_cooldown = True
        state.cooldown_until = time.time() + duration
        self.logger.info(
            f"Set cooldown for {deployment_id} until {state.cooldown_until}"
        )
