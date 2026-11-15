from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import Final

from cuawright.config.prompts import (
    OSWORLD_E_L1_DISCIPLINED_PROMPT_BUNDLE,
    OSWORLD_E_L1_DIRECT_PROMPT_BUNDLE,
    OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE,
    PromptBundle,
)

PROMPT_BUNDLE_REGISTRY: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "osworld-default": (
            "cuawright.config.prompts",
            "OSWORLD_DEFAULT_PROMPT_BUNDLE",
        ),
        "weavebench-default": (
            "cuawright.environments.weavebench.prompts",
            "WEAVEBENCH_DEFAULT_PROMPT_BUNDLE",
        ),
        "weavebench-linear": (
            "cuawright.environments.weavebench.prompts",
            "WEAVEBENCH_LINEAR_PROMPT_BUNDLE",
        ),
        "benchcad-full": (
            "cuawright.environments.benchcad.terminal_prompts",
            "BENCHCAD_FULL_PROMPT_BUNDLE",
        ),
        "benchcad-full-linear": (
            "cuawright.environments.benchcad.terminal_prompts",
            "BENCHCAD_FULL_LINEAR_PROMPT_BUNDLE",
        ),
        "benchcad-direct": (
            "cuawright.environments.benchcad.terminal_prompts",
            "BENCHCAD_DIRECT_PROMPT_BUNDLE",
        ),
        "cadgenbench-default": (
            "cuawright.environments.cadgenbench.terminal_prompts",
            "CADGENBENCH_DEFAULT_PROMPT_BUNDLE",
        ),
        "cadgenbench-linear": (
            "cuawright.environments.cadgenbench.terminal_prompts",
            "CADGENBENCH_LINEAR_PROMPT_BUNDLE",
        ),
        "odysseys-default": (
            "cuawright.environments.odysseys.prompts",
            "ODYSSEYS_DEFAULT_PROMPT_BUNDLE",
        ),
    }
)


def select_prompt_bundle(
    name: str,
    *,
    external_assets: Mapping[str, str] | None = None,
) -> PromptBundle:
    try:
        module_name, attribute = PROMPT_BUNDLE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown prompt bundle: {name!r}") from None
    template = getattr(import_module(module_name), attribute)
    if not isinstance(template, PromptBundle) or template.name != name:
        raise RuntimeError(f"prompt bundle registration drifted for {name!r}")
    values = external_assets or {}
    if template.unbound_asset_names or values:
        return template.bind_external_assets(values)
    return template


def select_osworld_prompt_bundle(
    *,
    no_subagents: bool,
    prompt_profile: str = "canonical",
) -> PromptBundle:
    if prompt_profile == "e-l1-direct":
        if not no_subagents:
            raise ValueError("e-l1-direct prompt requires no-subagents")
        return OSWORLD_E_L1_DIRECT_PROMPT_BUNDLE
    if prompt_profile == "e-l1-disciplined":
        if not no_subagents:
            raise ValueError("e-l1-disciplined prompt requires no-subagents")
        return OSWORLD_E_L1_DISCIPLINED_PROMPT_BUNDLE
    if prompt_profile != "canonical":
        raise ValueError(f"unknown OSWorld prompt profile: {prompt_profile!r}")
    if no_subagents:
        return OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE
    return select_prompt_bundle("osworld-default")


def select_weavebench_prompt_bundle(
    *,
    translated_system_prompt: str,
    no_subagents: bool = False,
) -> PromptBundle:
    return select_prompt_bundle(
        "weavebench-linear" if no_subagents else "weavebench-default",
        external_assets={
            "weavebench-external-translated-system": translated_system_prompt,
        },
    )


def select_benchcad_prompt_bundle(
    *,
    condition: str,
    direct_system_prompt: str | None = None,
    direct_user_prompt: str | None = None,
) -> PromptBundle:
    if condition == "direct":
        if direct_system_prompt is None or direct_user_prompt is None:
            raise ValueError("BenchCAD direct prompt sources are required")
        return select_prompt_bundle(
            "benchcad-direct",
            external_assets={
                "benchcad-direct-system": direct_system_prompt,
                "benchcad-direct-user": direct_user_prompt,
            },
        )
    if condition == "full-linear":
        return select_prompt_bundle("benchcad-full-linear")
    if condition != "full":
        raise ValueError(f"unsupported BenchCAD condition: {condition!r}")
    return select_prompt_bundle("benchcad-full")


def select_cadgenbench_prompt_bundle(
    *,
    no_subagents: bool = False,
) -> PromptBundle:
    return select_prompt_bundle(
        "cadgenbench-linear" if no_subagents else "cadgenbench-default"
    )


def select_odysseys_prompt_bundle() -> PromptBundle:
    return select_prompt_bundle("odysseys-default")
