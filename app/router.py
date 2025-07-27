import logging
import re
from typing import Any, Dict, Optional

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

        # Create a preferred order: tool-supporting models first, then the rest
        preferred_order = sorted(
            models_in_group,
            key=lambda m: m.supports_tools,
            reverse=True,
        )

        for i in range(len(preferred_order)):
            current_index = (start_index + i) % len(preferred_order)
            model = preferred_order[current_index]

            if self._state_manager.is_available(model.id):
                logging.info(
                    f"Selected model: {model.model_name} from group {model_group}",
                )
                # Update counter for the next request to ensure round-robin
                self._group_counters[model_group] = (current_index + 1) % len(
                    preferred_order,
                )

                try:
                    # Add MCP support flag to selected model
                    model.supports_mcp = True
                    return model
                except Exception as e:
                    logging.exception(f"Failed to select model {model.id}: {e!s}")
                    return None

        logging.error(
            f"No available models in group {model_group} after checking all options.",
        )
        return None

    def detect_tool_call(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Detects if the response content contains a tool call in XML format.
        If found, returns a structured tool call object.
        Otherwise, returns None.
        """
        # Simple regex to match XML tool calls
        tool_call_pattern = r"<(\w+)>(.*?)</\1>"
        match = re.search(tool_call_pattern, response_content, re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1)
        inner_content = match.group(2)

        # Extract parameters
        param_pattern = r"<(\w+)>(.*?)</\1>"
        param_matches = re.findall(param_pattern, inner_content, re.DOTALL)
        params = {name: value.strip() for name, value in param_matches}

        return {"tool_name": tool_name, "parameters": params, "raw_xml": match.group(0)}

    def translate_openai_tools_to_xml(self, tools: list) -> str:
        """Translates OpenAI-style tool definitions to XML format for MCP."""
        xml_tools = []
        for tool in tools:
            tool_name = tool["function"]["name"]
            params = tool["function"].get("parameters", {})
            xml_tool = f"<{tool_name}>\n"

            if params:
                for prop, details in params.get("properties", {}).items():
                    desc = details.get("description", "")
                    xml_tool += f"  <{prop}>{desc}</{prop}>\n"

            xml_tool += f"</{tool_name}>"
            xml_tools.append(xml_tool)

        return "\n\n".join(xml_tools)
