# app/router.py
import logging

from app.config import AppConfig, BackendModel
from app.state import ModelStateManager
from app.utils.model_grouper import ModelGrouper

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, config: AppConfig, model_grouper: ModelGrouper):
        # Validate and filter models before building groups
        valid_models = []
        for model in config.model_list:
            if not all([model.id, model.group_name, model.model_name]):
                logger.error(
                    f"Model parameter missing for group {model.group_name}. Skipping.",
                )
                continue
            valid_models.append(model)

        (
            self.model_groups,
            self.model_map,
            self.group_counters,
            self.active_model_ids,
        ) = model_grouper.build_groups(valid_models)
        logger.info(f"Initialized groups: {list(self.model_groups.keys())}")

    def get_model_by_id(self, model_id: str) -> BackendModel | None:
        """Returns BackendModel object by its unique ID."""
        return self.model_map.get(model_id)

    # --- START NEW MAIN METHOD ---
    def get_active_or_failover_model(
        self,
        group_name: str,
        state_manager: ModelStateManager,
    ) -> BackendModel | None:
        """Returns current active model. If unavailable,
        selects next working model and makes it the new active one.
        """
        group = self.model_groups.get(group_name)
        if not group:
            return None

        # 1. Check current active model
        active_id = self.active_model_ids.get(group_name)
        if active_id:
            model = self.get_model_by_id(active_id)
            if model and state_manager.is_available(model.id):
                logger.debug(f"Using active model {model.id} for group {group_name}.")
                return model
            logger.warning(
                f"Active model {active_id} is unavailable. Failing over.",
            )
            self.active_model_ids[group_name] = None

        # 2. If no active model or it's unavailable - search for new one (Failover)
        # Try to find next available model, making no more than one full cycle
        for _ in range(len(group)):
            counter = self.group_counters[group_name]
            next_model = group[counter]

            # Shift counter for next time
            self.group_counters[group_name] = (counter + 1) % len(group)

            if state_manager.is_available(next_model.id):
                logger.info(
                    f"New active model for group {group_name} is {next_model.id}.",
                )
                self.active_model_ids[group_name] = next_model.id  # Make it the new active model
                return next_model

        logger.error(f"No available models found in group {group_name}.")
        return None
