import logging
import time

from .config import BackendModel, RouterSettings
from .state import ModelStateManager


class LLMRouter:
    """Отвечает за выбор доступной модели из группы."""

    def __init__(
        self,
        backend_models: list[BackendModel],
        state_manager: ModelStateManager,
        settings: RouterSettings,
    ):
        self._state_manager = state_manager
        self._model_groups: dict[str, list[BackendModel]] = {}
        self._group_counters: dict[str, int] = {}
        self._build_groups(backend_models)
        logging.info("Router initialized.")

    def _build_groups(self, backend_models: list[BackendModel]):
        logging.info("Building model groups...")
        logging.info(f"Total backend models: {len(backend_models)}")

        for model in backend_models:
            # Use the model's configured model_name as the group name
            group_name = model.model_name
            logging.info(
                f"Processing backend model: id={model.backend_model_id}, group_name={group_name}, backend model={model.litellm_params.model}",
            )

            if group_name not in self._model_groups:
                logging.info(f"Creating new group for model: {group_name}")
                self._model_groups[group_name] = []

            self._model_groups[group_name].append(model)
            logging.info(
                f"Added backend model to group '{group_name}': {model.backend_model_id}",
            )

        # Initialize counters for each group
        logging.info(f"Initialized groups: {list(self._model_groups.keys())}")
        for group_name in self._model_groups.keys():
            self._group_counters[group_name] = 0
            logging.info(f"Initialized counter for group: {group_name}")

    def get_next_backend_model(self, model_group: str, max_retries: int = 5, retry_delay: float = 0.1) -> BackendModel | None:
        """
        Finds the next available backend model in the group.
        Implements round-robin rotation with rate limit awareness.
        """
        models_in_group = self._model_groups.get(model_group, [])
        if not models_in_group:
            logging.warning(
                f"Attempted to get model from non-existent group: {model_group}"
            )
            return None

        start_index = self._group_counters[model_group]
        retry_count = 0

        while retry_count < max_retries:
            # Loop through to find available model
            for i in range(len(models_in_group)):
                current_index = (start_index + i) % len(models_in_group)
                model = models_in_group[current_index]

                if self._state_manager.is_available(model.backend_model_id):
                    logging.info(
                        f"Selected model: {model.model_name} from group {model_group}"
                    )
                    # Update counter for next request
                    self._group_counters[model_group] = (current_index + 1) % len(
                        models_in_group
                    )
                    return model

            if retry_count < max_retries - 1:
                logging.warning(
                    f"No available models in group {model_group}. "
                    f"Retrying in {retry_delay} sec. ({retry_count+1}/{max_retries})"
                )
                time.sleep(retry_delay)
                retry_count += 1
            else:
                break

        logging.error(f"No available models in group {model_group} after {max_retries} attempts.")
        return None
