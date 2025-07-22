import logging
import threading
import time
from typing import Dict

from .config import ModelDeployment


class ModelStateManager:
    """Потокобезопасный класс для управления состоянием и лимитами моделей."""

    def __init__(self, deployments: list[ModelDeployment]):
        self._state: Dict[str, Dict] = {}
        self._deployments = {d.deployment_id: d for d in deployments}
        self._lock = threading.Lock()
        self._initialize_state()
        logging.info("Менеджер состояния инициализирован.")

    def _initialize_state(self):
        logging.info(f"Initializing state for {len(self._deployments)} deployments.")
        for dep_id, deployment in self._deployments.items():
            logging.info(f"Initializing state for deployment: {deployment.deployment_id}")
            self._state[dep_id] = {
                "last_used": 0.0,
                "is_on_cooldown": False,
                "cooldown_until": 0.0,
            }

    def is_available(self, deployment_id: str) -> bool:
        """Проверяет, доступна ли модель с учетом лимитов."""
        with self._lock:
            logging.info(f"Checking availability for deployment: {deployment_id}")
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                logging.warning(f"Deployment {deployment_id} not found in deployments list")
                return False

            state = self._state[deployment_id]
            current_time = time.time()
            logging.info(f"Current state: {state}")

            # Проверка общего кулдауна (например, после ошибки 429)
            if state["is_on_cooldown"]:
                if current_time < state["cooldown_until"]:
                    logging.info(
                        f"Model {deployment_id} is on cooldown until {state['cooldown_until']} "
                        f"(current time: {current_time})"
                    )
                    return False
                else:
                    logging.info(
                        f"Cooldown expired for {deployment_id}. Resetting cooldown flag."
                    )
                    state["is_on_cooldown"] = False

            # Проверка RPM
            if deployment.litellm_params.rpm:
                cooldown_duration = 60.0 / deployment.litellm_params.rpm
                time_since_last_use = current_time - state["last_used"]
                logging.info(
                    f"RPM check: {deployment.litellm_params.rpm} RPM "
                    f"-> min interval: {cooldown_duration:.2f}s, "
                    f"time since last use: {time_since_last_use:.2f}s"
                )
                
                if time_since_last_use < cooldown_duration:
                    logging.info(
                        f"Model {deployment_id} is rate limited. "
                        f"Time since last use: {time_since_last_use:.2f} sec, "
                        f"required cooldown: {cooldown_duration:.2f} sec"
                    )
                    return False

            logging.info(f"Model {deployment_id} is available")
            return True

    def record_success(self, deployment_id: str):
        """Записывает успешное использование модели."""
        with self._lock:
            if deployment_id in self._state:
                self._state[deployment_id]["last_used"] = time.time()
                logging.debug(f"Успешный вызов для {deployment_id} записан.")

    def record_failure(self, deployment_id: str, status_code: int):
        """Записывает сбой и может отправить модель в 'кулдаун'."""
        with self._lock:
            if deployment_id in self._state and status_code == 429:
                cooldown_duration = 60  # секунд
                self._state[deployment_id]["is_on_cooldown"] = True
                self._state[deployment_id]["cooldown_until"] = (
                    time.time() + cooldown_duration
                )
                logging.warning(
                    f"Модель {deployment_id} получила статус 429. Отправлена в кулдаун на {cooldown_duration} сек."
                )
