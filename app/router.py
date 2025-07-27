import logging
from typing import Optional

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

        # Initialize state for all models
        model_ids = [
            model.id for group in self._model_groups.values() for model in group
        ]
        state_manager.initialize_models(model_ids)

        logging.info("Router initialized.")

    def _build_groups(self, backend_models: list[BackendModel]):
        logging.info("Building model groups...")
        logging.info(f"Total backend models: {len(backend_models)}")

        for model in backend_models:
            # Use the model's group_name as the group name
            group_name = model.group_name
            logging.info(
                f"Processing backend model: id={model.id}, group_name={group_name}, backend model={model.model_name}",
            )

            if group_name not in self._model_groups:
                logging.info(f"Creating new group for model: {group_name}")
                self._model_groups[group_name] = []

            self._model_groups[group_name].append(model)
            logging.info(
                f"Added backend model to group '{group_name}': {model.id}",
            )

        # Initialize counters for each group
        logging.info(f"Initialized groups: {list(self._model_groups.keys())}")
        for group_name in self._model_groups.keys():
            self._group_counters[group_name] = 0
            logging.info(f"Initialized counter for group: {group_name}")

    def get_next_backend_model(
        self,
        model_group: str,
        tools: Optional[list] = None,
        max_retries: int = 5,
        retry_delay: float = 0.1,
    ) -> BackendModel | None:
        """Finds the next available backend model in the group using round-robin.
        If tools are requested, it gives one-time preference to models known to support them.
        """
        models_in_group = self._model_groups.get(model_group, [])
        if not models_in_group:
            logging.warning(
                f"Attempted to get model from non-existent group: {model_group}",
            )
            return None

        start_index = self._group_counters.get(model_group, 0)
        preferred_order = self._get_preferred_order(models_in_group, tools)

        for i in range(len(preferred_order)):
            current_index = (start_index + i) % len(preferred_order)
            model = preferred_order[current_index]

            if self._state_manager.is_available(model.id):
                self._update_counter(model_group, current_index, len(preferred_order))
                return self._select_model(model, model_group)

        logging.error(
            f"No available models in group {model_group} after checking all options.",
        )
        return None

    def _get_preferred_order(
        self,
        models: list[BackendModel],
        tools: Optional[list],
    ) -> list[BackendModel]:
        """Returns models sorted by tool support if tools are requested"""
        if tools:
            return sorted(models, key=lambda m: m.supports_tools, reverse=True)
        return models

    def _update_counter(
        self,
        group_name: str,
        current_index: int,
        model_count: int,
    ) -> None:
        """Updates the round-robin counter for the group"""
        self._group_counters[group_name] = (current_index + 1) % model_count

    def _select_model(
        self,
        model: BackendModel,
        model_group: str,
    ) -> BackendModel | None:
        """Finalizes model selection and adds MCP support flag"""
        try:
            logging.info(
                f"Selected model: {model.model_name} from group {model_group}",
            )
            model.supports_mcp = True
            return model
        except Exception as e:
            logging.exception(f"Failed to select model {model.id}: {e!s}")
            return None
