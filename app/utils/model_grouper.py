# app/utils/model_grouper.py

from app.config import BackendModel


class ModelGrouper:
    def build_groups(
        self, model_list: list[BackendModel]
    ) -> tuple[dict[str, list[BackendModel]], dict[str, BackendModel], dict[str, int], dict[str, str | None]]:
        model_groups: dict[str, list[BackendModel]] = {}
        model_map: dict[str, BackendModel] = {}
        active_model_ids: dict[str, str | None] = {}
        group_counters: dict[str, int] = {}

        for model in model_list:
            model_map[model.id] = model
            if model.group_name not in model_groups:
                model_groups[model.group_name] = []
                active_model_ids[model.group_name] = None
            model_groups[model.group_name].append(model)

        for group_name in model_groups:
            group_counters[group_name] = 0

        return model_groups, model_map, group_counters, active_model_ids
