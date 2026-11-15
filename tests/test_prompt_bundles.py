from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from cuawright.config.prompt_registry import (
    PROMPT_BUNDLE_REGISTRY,
    select_benchcad_prompt_bundle,
    select_cadgenbench_prompt_bundle,
    select_odysseys_prompt_bundle,
    select_osworld_prompt_bundle,
    select_prompt_bundle,
    select_weavebench_prompt_bundle,
)
from cuawright.config.prompts import (
    AGENT_WORKSPACE,
    E_L1_DISCIPLINED_ACTOR_INSTRUCTIONS,
    E_L1_DIRECT_ACTOR_INSTRUCTIONS,
    OSWORLD_DEFAULT_PROMPT_BUNDLE,
    OSWORLD_E_L1_DISCIPLINED_PROMPT_BUNDLE,
    OSWORLD_E_L1_DIRECT_PROMPT_BUNDLE,
    OSWORLD_INVENTORY_ACTOR_INSTRUCTIONS,
    OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE,
    OSWORLD_OVERLAY,
    BenchmarkOverlay,
    PromptAsset,
    compaction_user_prompt,
    compose_system_prompt,
    render_direct_prompt_bundle,
)
from cuawright.environments.benchcad.terminal_prompts import (
    BENCHCAD_FULL_LINEAR_PROMPT_BUNDLE,
    BENCHCAD_FULL_PROMPT_BUNDLE,
    BENCHCAD_TERMINAL_OVERLAY,
)
from cuawright.environments.cadgenbench.terminal_prompts import (
    CADGENBENCH_DEFAULT_PROMPT_BUNDLE,
    CADGENBENCH_LINEAR_PROMPT_BUNDLE,
    CADGENBENCH_TERMINAL_OVERLAY,
)
from cuawright.environments.weavebench.prompts import (
    WEAVEBENCH_EXTERNAL_SYSTEM_ASSET,
    WEAVEBENCH_LINEAR_PROMPT_BUNDLE,
    localized_linear_actor_instructions,
    localized_role_instructions,
    translate_system_prompt,
)

EXPECTED_SYSTEM_SHA256 = {
    "osworld-default": {
        "actor": "cedb9426a0c3cf10abc8a4b5d286787c01456b68361af33d4ade77966b52b4e5",
        "binder": "59ef1e30aa104478d6b130b98b987c90a5cebebf1ad18dc2a8c4500623cb7129",
        "critic": "e25ac24a05955f636db6b8ac659b6d4564f5f1a83c663fc7a29063caaf4adb9b",
        "verifier": "ba15a484da97aa0bc0d5ae9066a0a3447489384c8be32d7b4af485791fab8c65",
    },
    "osworld-no-subagents": {
        "actor": "fb5bcd089f47148d94dd5f8c0f78886dd5a55c7b7625ee83b71fee16d133fadd",
    },
    "benchcad-full": {
        "actor": "8f805d1528272d6be6de9f7344ad6d004528aaf9191089f201ab968e7e07f212",
        "binder": "cd312e8b1337c38a7098c73843aee15d97907cf104d486e60c764d14c8121223",
        "critic": "116b42ac06d4971badb837ee1f2b8766fbf77bd22f9f0da658da03df7c1ac748",
        "verifier": "484ccf98a9422f8e23eab24c47fb92357a12285e23f1927cf4cd32affd5de815",
    },
    "benchcad-full-linear": {
        "actor": "f573d578f1c7fce1748b975ca69b930595af7b308fe31ea3e6e6ad095bc0f346",
    },
    "cadgenbench-default": {
        "actor": "cdd447447b14c0fcb1dfcadc1334bb9866d745e3b75ce27628870c73276f2b44",
        "binder": "22c126ecdc53487a9bc38ff162a4e2b7f8875c490a40dc429c5d6ecbf5f71d79",
        "critic": "4fe2dbe4e411cc71cd60377683566584758d60b3e6e9c68465fe4ff75b5f6d0e",
        "verifier": "6407e911ab86e4d83ade9cc303a0f97a6f69ce11bbbe3636aeb3d199c40cb5a7",
    },
    "cadgenbench-linear": {
        "actor": "75ae0c936e9402d0dd3e270fc66820d081eb038f09a1fc35a70a5afcc6f8fba6",
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _weavebench_source_prompt() -> str:
    return (
        "Preserve this exact task contract.\n"
        "Built-in helper tools available in BOTH modes: `exec` (shell), "
        "`read`/`write`/`edit` (files), `process` (long-running jobs), "
        "`pdf` (structured PDF extraction including tables), `web_fetch` "
        "(HTTP), `memory_search` (recall prior context). Pick whichever is "
        "most direct for the sub-step.\n"
        "Working directory: `/tmp_workspace/`; Do NOT touch `/tmp_workspace/gt/`. "
        "The OpenClaw framework's own `## Workspace` section in the underlying "
        "system prompt has been configured to point at the SAME `/tmp_workspace/` "
        "path (via `agents.defaults.workspace`), so the two sources agree. "
        "Anything you write under `/home/user/.openclaw/...` is OpenClaw's "
        "internal agent state — invisible to the grader and effectively lost.\n"
        "Output discipline: verify required output "
        "(use `exec` like `ls -la <path>` or `read` a few of the created files).\n"
        'The sudo password is "password" when sudo is needed.\n'
        "The screen resolution is 1920x1080 pixels.\n"
        "An initial desktop screenshot was saved to /tmp/init_screenshot.png; "
        "you can read it with the `image` tool, or call `__computer__` with "
        "`actions:[]` (empty batch) to grab a fresh capture.\n"
        "- GUI is exposed via ONE function tool: `__computer__`. It runs one or "
        "more GUI actions and ALWAYS returns a fresh screenshot.\n"
        "- You MUST drive the GUI by calling `__computer__`.\n"
        "- DO NOT stop after a single observation from `__computer__`.\n"
        "- Write supporting artifacts via `exec`/`write`.\n"
        "- Coordinates are absolute screen pixels in the 1920x1080 system.\n"
        "=== GUI MODE ANTI-FABRICATION POLICY ===\n"
        "This run has a real Linux desktop available via the `__computer__` tool.\n"
        "Do NOT generate placeholder PNG files using `PIL.Image.new()`.\n"
        "  • `__computer__` — preferred. Every call already auto-screenshots\n"
        "    the resulting desktop; you can save that buffer to disk.\n"
        "If `pyautogui.screenshot()` or `__computer__` keeps failing, stop.\n"
    )


def test_registry_has_complete_independent_named_conditions() -> None:
    assert tuple(PROMPT_BUNDLE_REGISTRY) == (
        "osworld-default",
        "weavebench-default",
        "weavebench-linear",
        "benchcad-full",
        "benchcad-full-linear",
        "benchcad-direct",
        "cadgenbench-default",
        "cadgenbench-linear",
        "odysseys-default",
    )
    assert not any("no-subagents" in name for name in PROMPT_BUNDLE_REGISTRY)
    for name, (module_name, attribute) in PROMPT_BUNDLE_REGISTRY.items():
        assert name.split("-", 1)[0] in module_name or name.startswith("osworld")
        assert attribute.endswith("PROMPT_BUNDLE")
    with pytest.raises(TypeError):
        PROMPT_BUNDLE_REGISTRY["other"] = ("module", "attribute")
    with pytest.raises(FrozenInstanceError):
        OSWORLD_DEFAULT_PROMPT_BUNDLE.name = "other"


def test_e_l1_direct_prompt_is_short_and_has_no_specialists() -> None:
    bundle = select_osworld_prompt_bundle(
        no_subagents=True,
        prompt_profile="e-l1-direct",
    )
    prompt = compose_system_prompt("actor", bundle)

    assert bundle is OSWORLD_E_L1_DIRECT_PROMPT_BUNDLE
    assert prompt == E_L1_DIRECT_ACTOR_INSTRUCTIONS
    assert len(prompt) == 2_213
    for forbidden in ("Binder", "Critic", "Verifier", "subagent"):
        assert forbidden.lower() not in prompt.lower()
    with pytest.raises(ValueError, match="requires no-subagents"):
        select_osworld_prompt_bundle(
            no_subagents=False,
            prompt_profile="e-l1-direct",
        )


def test_e_l1_disciplined_prompt_adds_execution_controls() -> None:
    bundle = select_osworld_prompt_bundle(
        no_subagents=True,
        prompt_profile="e-l1-disciplined",
    )
    prompt = compose_system_prompt("actor", bundle)

    assert bundle is OSWORLD_E_L1_DISCIPLINED_PROMPT_BUNDLE
    assert prompt == E_L1_DISCIPLINED_ACTOR_INSTRUCTIONS
    assert prompt.startswith(E_L1_DIRECT_ACTOR_INSTRUCTIONS)
    assert "first 40 model calls" in prompt
    assert "at most two verification-and-repair cycles" in prompt
    assert "sanity-check units, ranges, and scale" in prompt
    for forbidden in ("Binder", "Critic", "Verifier", "subagent"):
        assert forbidden.lower() not in prompt.lower()
    with pytest.raises(ValueError, match="requires no-subagents"):
        select_osworld_prompt_bundle(
            no_subagents=False,
            prompt_profile="e-l1-disciplined",
        )


def test_no_subagents_defaults_to_inventory_actor() -> None:
    bundle = select_osworld_prompt_bundle(no_subagents=True)
    prompt = compose_system_prompt("actor", bundle)

    assert bundle is OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE
    assert prompt == OSWORLD_INVENTORY_ACTOR_INSTRUCTIONS
    assert "central, top-level requirement" in prompt
    assert "As a general guide, inventory whichever parts" in prompt
    assert "Accessibility.getFullAXTree" in prompt
    assert len(prompt.split()) == 311
    for forbidden in ("Binder", "Critic", "Verifier", "subagent"):
        assert forbidden.lower() not in prompt.lower()


@pytest.mark.parametrize(
    ("bundle", "overlay"),
    [
        (OSWORLD_DEFAULT_PROMPT_BUNDLE, OSWORLD_OVERLAY),
        (BENCHCAD_FULL_PROMPT_BUNDLE, BENCHCAD_TERMINAL_OVERLAY),
        (
            BENCHCAD_FULL_LINEAR_PROMPT_BUNDLE,
            BENCHCAD_TERMINAL_OVERLAY,
        ),
        (CADGENBENCH_DEFAULT_PROMPT_BUNDLE, CADGENBENCH_TERMINAL_OVERLAY),
        (
            CADGENBENCH_LINEAR_PROMPT_BUNDLE,
            CADGENBENCH_TERMINAL_OVERLAY,
        ),
    ],
)
def test_named_terminal_bundles_equal_legacy_composition(
    bundle,
    overlay,
) -> None:
    expected_hashes = EXPECTED_SYSTEM_SHA256[bundle.name]
    for role, expected_hash in expected_hashes.items():
        bundled = compose_system_prompt(role, bundle)
        legacy = compose_system_prompt(
            role,
            overlay,
            agent_topology=bundle.agent_topology,
        )
        assert bundled == legacy
        assert _sha256(bundled) == expected_hash
        assert bundle.shared_prefix is not None
        assert bundle.terminal_control is not None
        assert bundle.system_overlay is not None
        layers = (
            bundle.shared_prefix.render(),
            bundle.role_asset(role).render(workspace=bundle.workspace),
            bundle.terminal_control.render(workspace=bundle.workspace),
            bundle.system_overlay.render(),
        )
        assert bundled == "\n\n".join(layer for layer in layers if layer)
    assert compaction_user_prompt(
        agent_topology=bundle.agent_topology,
        prompt_bundle=bundle,
    ) == compaction_user_prompt(agent_topology=bundle.agent_topology)


def test_workspace_replacement_and_provider_suffix_remain_exact() -> None:
    workspace_asset = PromptAsset(
        "workspace-template",
        f"Cache path: {AGENT_WORKSPACE}/cache/{{invocation_id}}",
    )
    assert workspace_asset.render(workspace="/home/user/.agent") == (
        "Cache path: /home/user/.agent/cache/{invocation_id}"
    )
    system = compose_system_prompt("actor", OSWORLD_DEFAULT_PROMPT_BUNDLE)
    assert system == compose_system_prompt("actor", OSWORLD_OVERLAY)
    suffix = "Provider-specific stable suffix."
    with_suffix = compose_system_prompt(
        "actor",
        OSWORLD_DEFAULT_PROMPT_BUNDLE,
        parallel_tool_suffix=suffix,
    )
    assert with_suffix == system + "\n\n" + suffix
    assert OSWORLD_DEFAULT_PROMPT_BUNDLE.system_overlay.text == ""


def test_benchcad_direct_bundle_is_exact_passthrough_without_terminal_layers() -> None:
    system = "Direct system\nwith exact trailing bytes.\n"
    user = "Direct user prompt\n"
    bundle = select_benchcad_prompt_bundle(
        condition="direct",
        direct_system_prompt=system,
        direct_user_prompt=user,
    )
    assert render_direct_prompt_bundle(bundle) == (system, user)
    assert bundle.role_assets == ()
    assert bundle.shared_prefix is None
    assert bundle.terminal_control is None
    assert bundle.system_overlay is None
    assert bundle.compaction is None
    with pytest.raises(ValueError, match="not terminal-layered"):
        compose_system_prompt("actor", bundle)


@pytest.mark.parametrize(
    ("name", "topology", "expected_hashes"),
    [
        (
            "weavebench-default",
            "orchestrated",
            {
                "actor": "eb969f2e78de6cc4113bf84f806aebb2833cc9f61ddb55f8b022ec3c745dfc5b",
                "binder": "934f90a787726e6c39758a713fd14a0398b567d46628fdef4d1fe7838d24ffce",
                "critic": "ea052adc566dfe4f0d652e7dc415ea929e5689ffca9b7fe2d13bc1b77df475d9",
                "verifier": "63bdef3b927d64828211b0d41fa90912b91e432a74a434b8cbde29c9d400d7df",
            },
        ),
        (
            "weavebench-linear",
            "linear",
            {
                "actor": "dfb2a5bbb8aea765a34964b84a7c5d822ef17c73977ee41a11373cadc70df2b6",
            },
        ),
    ],
)
def test_weavebench_external_translation_is_its_own_bundle_boundary(
    name,
    topology,
    expected_hashes,
) -> None:
    translated = translate_system_prompt(
        _weavebench_source_prompt(),
        client_password="public-evaluation",
    )
    assert _sha256(translated) == (
        "572b7e5847f19fe44c9ec9c26b5ac5897cdd1d2b951eeac8e759139c6791ab20"
    )
    if topology == "linear":
        bundle = WEAVEBENCH_LINEAR_PROMPT_BUNDLE.bind_external_assets(
            {WEAVEBENCH_EXTERNAL_SYSTEM_ASSET: translated}
        )
    else:
        bundle = select_prompt_bundle(
            name,
            external_assets={WEAVEBENCH_EXTERNAL_SYSTEM_ASSET: translated},
        )
    legacy = BenchmarkOverlay(
        name="weavebench",
        system_prompt=translated,
        role_instructions=localized_role_instructions(),
        linear_actor_instructions=localized_linear_actor_instructions(),
        workspace="/home/user/.agent",
    )
    assert bundle.benchmark == "weavebench"
    assert bundle.system_overlay.name == WEAVEBENCH_EXTERNAL_SYSTEM_ASSET
    for role, expected_hash in expected_hashes.items():
        bundled = compose_system_prompt(role, bundle)
        assert bundled == compose_system_prompt(
            role,
            legacy,
            agent_topology=topology,
        )
        assert _sha256(bundled) == expected_hash


def test_condition_selectors_choose_once_without_cross_benchmark_fallback() -> None:
    assert select_osworld_prompt_bundle(no_subagents=False).name == ("osworld-default")
    assert select_osworld_prompt_bundle(no_subagents=True).name == (
        "osworld-no-subagents"
    )
    assert (
        select_benchcad_prompt_bundle(
            condition="full",
        ).name
        == "benchcad-full"
    )
    assert (
        select_benchcad_prompt_bundle(condition="full-linear").name
        == "benchcad-full-linear"
    )
    assert select_cadgenbench_prompt_bundle().name == "cadgenbench-default"
    assert (
        select_cadgenbench_prompt_bundle(no_subagents=True).name == "cadgenbench-linear"
    )
    assert select_odysseys_prompt_bundle().name == "odysseys-default"
    translated = translate_system_prompt(_weavebench_source_prompt())
    bundles = (
        select_osworld_prompt_bundle(no_subagents=False),
        select_osworld_prompt_bundle(no_subagents=True),
        select_weavebench_prompt_bundle(
            translated_system_prompt=translated,
        ),
        select_weavebench_prompt_bundle(
            translated_system_prompt=translated,
            no_subagents=True,
        ),
        select_benchcad_prompt_bundle(
            condition="full",
        ),
        select_benchcad_prompt_bundle(condition="full-linear"),
        select_benchcad_prompt_bundle(
            condition="direct",
            direct_system_prompt="system",
            direct_user_prompt="user",
        ),
        select_cadgenbench_prompt_bundle(),
        select_cadgenbench_prompt_bundle(no_subagents=True),
        select_odysseys_prompt_bundle(),
    )
    for bundle in bundles:
        assert bundle.benchmark in bundle.name
        for other in bundles:
            if bundle.benchmark != other.benchmark:
                assert bundle is not other


def test_unbound_external_assets_fail_closed() -> None:
    with pytest.raises(ValueError, match="external prompt assets"):
        select_prompt_bundle("weavebench-default")
    with pytest.raises(ValueError, match="external prompt assets"):
        select_prompt_bundle(
            "osworld-default",
            external_assets={"unexpected": "text"},
        )
    with pytest.raises(ValueError, match="requires external text"):
        PromptAsset("external", None).render(workspace="/workspace/.agent")
