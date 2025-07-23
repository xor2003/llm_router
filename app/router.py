import logging
import time
from typing import Dict, List, Optional

from .config import ModelDeployment, RouterSettings
from .state import ModelStateManager


class LLMRouter:
    """Отвечает за выбор доступной модели из группы."""

    def __init__(
        self, deployments: List[ModelDeployment], state_manager: ModelStateManager,
        settings: RouterSettings,
    ):
        self._state_manager = state_manager
        self._model_groups: Dict[str, List[ModelDeployment]] = {}
        self._group_counters: Dict[str, int] = {}
        self._build_groups(deployments)
        logging.info("Маршрутизатор инициализирован.")

    def _build_groups(self, deployments: List[ModelDeployment]):
        logging.info("Building model groups...")
        logging.info(f"Total deployments: {len(deployments)}")
        
        for dep in deployments:
            # Use the deployment's configured model_name as the group name
            group_name = dep.model_name
            logging.info(f"Processing deployment: id={dep.deployment_id}, group_name={group_name}, litellm_model={dep.litellm_params.model}")
            
            if group_name not in self._model_groups:
                logging.info(f"Creating new group for model: {group_name}")
                self._model_groups[group_name] = []
            
            self._model_groups[group_name].append(dep)
            logging.info(f"Added deployment to group '{group_name}': {dep.deployment_id}")
        
        # Initialize counters for each group
        logging.info(f"Initialized groups: {list(self._model_groups.keys())}")
        for group_name in self._model_groups.keys():
            self._group_counters[group_name] = 0
            logging.info(f"Initialized counter for group: {group_name}")

    def get_next_deployment(self, model_group: str, max_retries: int = 5, retry_delay: float = 0.1) -> Optional[ModelDeployment]:
        """
        Находит следующее доступное развертывание в группе.
        Реализует ротацию (round-robin) с учетом rate limits.
        При обнаружении 429 ошибки сразу переходит к следующей модели.
        """
        deployments_in_group = self._model_groups.get(model_group, [])
        if not deployments_in_group:
            logging.warning(
                f"Попытка получить модель из несуществующей группы: {model_group}"
            )
            return None

        start_index = self._group_counters[model_group]
        retry_count = 0

        while retry_count < max_retries:
            # Проходим по кругу один раз, чтобы найти доступную модель
            for i in range(len(deployments_in_group)):
                current_index = (start_index + i) % len(deployments_in_group)
                deployment = deployments_in_group[current_index]

                if self._state_manager.is_available(deployment.deployment_id):
                    logging.info(
                        f"Выбрана модель: {deployment.model_name} из группы {model_group}"
                    )
                    # Обновляем счетчик для следующего запроса
                    self._group_counters[model_group] = (current_index + 1) % len(
                        deployments_in_group
                    )
                    return deployment

            if retry_count < max_retries - 1:
                logging.warning(
                    f"В группе {model_group} нет доступных моделей. "
                    f"Повторная попытка через {retry_delay} сек. ({retry_count+1}/{max_retries})"
                )
                time.sleep(retry_delay)
                retry_count += 1
            else:
                break

        logging.error(f"В группе {model_group} нет доступных моделей после {max_retries} попыток.")
        return None
