import logging

from app.config import BackendModel

logger = logging.getLogger(__name__)


def build_model_groups(
    backend_models: list[BackendModel],
) -> dict[str, list[BackendModel]]:
    """Builds model groups from backend models.

    Args:
        backend_models: List of backend models to group

    Returns:
        Dictionary mapping group names to lists of backend models

    """
    model_groups: dict[str, list[BackendModel]] = {}
    logger.info("Building model groups...")
    logger.info(f"Total backend models: {len(backend_models)}")

    for model in backend_models:
        group_name = model.group_name
        logger.info(
            f"Processing backend model: id={model.id}, group_name={group_name}, backend model={model.model_name}",
        )

        if group_name not in model_groups:
            logger.info(f"Creating new group for model: {group_name}")
            model_groups[group_name] = []

        model_groups[group_name].append(model)
        logger.info(
            f"Added backend model to group '{group_name}': {model.id}",
        )

    logger.info(f"Initialized groups: {list(model_groups.keys())}")
    return model_groups
