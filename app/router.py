import logging
from typing import Dict, List, Optional

from .config import ModelDeployment
from .state import ModelStateManager


class LLMRouter:
    """Отвечает за выбор доступной модели из группы."""

    def __init__(
        self, deployments: List[ModelDeployment], state_manager: ModelStateManager
    ):
        self._state_manager = state_manager
        self._model_groups: Dict[str, List[ModelDeployment]] = {}
        self._group_counters: Dict[str, int] = {}
        self._build_groups(deployments)
        logging.info("Маршрутизатор инициализирован.")

    def _build_groups(self, deployments: List[ModelDeployment]):
        for dep in deployments:
            if dep.model_name not in self._model_groups:
                self._model_groups[dep.deployment_id] = []
            self._model_groups[dep.deployment_id].append(dep)
        # Initialize counters for each group
        for group_name in self._model_groups.keys():
            self._group_counters[group_name] = 0

    def get_next_deployment(self, model_group: str) -> Optional[ModelDeployment]:
        """
        Находит следующее доступное развертывание в группе.
        Реализует простую ротацию (round-robin).
        """
        deployments_in_group = self._model_groups.get(model_group, [])
        if not deployments_in_group:
            logging.warning(
                f"Попытка получить модель из несуществующей группы: {model_group}"
            )
            return None

        start_index = self._group_counters[model_group]

        # Проходим по кругу один раз, чтобы найти доступную модель
        for i in range(len(deployments_in_group)):
            current_index = (start_index + i) % len(deployments_in_group)
            deployment = deployments_in_group[current_index]

            if self._state_manager.is_available(deployment.deployment_id):
                logging.info(
                    f"Выбрана модель: {deployment.deployment_id} из группы {model_group}"
                )
                # Обновляем счетчик для следующего запроса
                self._group_counters[model_group] = (current_index + 1) % len(
                    deployments_in_group
                )
                return deployment

        logging.error(f"В группе {model_group} нет доступных моделей.")
        return None
