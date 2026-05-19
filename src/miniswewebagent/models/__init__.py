from __future__ import annotations

import copy
import importlib

from miniswewebagent import Model

_MODEL_MAPPING = {
    "phyagi": "miniswewebagent.models.phyagi_model.PhyagiModel",
    "azure_responses": "miniswewebagent.models.azure_responses_model.AzureResponsesModel",
    "openrouter": "miniswewebagent.models.openrouter_model.OpenRouterModel",
    "anthropic": "miniswewebagent.models.anthropic_model.AnthropicModel",
    "trapi_kimi": "miniswewebagent.models.trapi_kimi_model.TrapiKimiModel",
}


def get_model_class(spec: str) -> type[Model]:
    full_path = _MODEL_MAPPING.get(spec, spec)
    module_name, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_model(config: dict, *, default_type: str = "phyagi") -> Model:
    copied = copy.deepcopy(config)
    model_class = copied.pop("model_class", default_type)
    return get_model_class(model_class)(**copied)
